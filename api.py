"""FastAPI service for the generated graph. Run: uvicorn api:app --reload."""
from __future__ import annotations
import json, os, re, asyncio, logging, hashlib
from pathlib import Path
from typing import Any
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env", override=False, encoding="utf-8-sig")
OUT = Path(os.getenv("GRAPH_OUT", str(BASE / "out")))
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", str(BASE / "frontend/dist")))
app = FastAPI(title="Money Graph API")
app.add_middleware(CORSMiddleware,
    allow_origins=[x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:4174,http://127.0.0.1:4174").split(",") if x.strip()],
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
    expose_headers=["X-Graph-Version"])

def graph_data():
    p = OUT / "graph.json"
    if not p.exists(): raise HTTPException(503, "graph.json is not built; run python pipeline.py")
    return json.loads(p.read_text(encoding="utf-8"))

def graph():
    d = graph_data(); g = nx.DiGraph()
    g.add_nodes_from(n["id"] if "id" in n else n["gid"] for n in d["nodes"])
    for e in d["links"]: g.add_edge(e["source"], e["target"], **e)
    return d, g

@app.get("/api/graph")
def get_graph():
    data = graph_data()
    version = data.get("meta", {}).get("build_id") or hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
    return JSONResponse(data, headers={"Cache-Control": "no-store", "X-Graph-Version": version})

@app.get("/api/health")
def health():
    return {"llm_configured": bool(os.getenv("OPENAI_API_KEY")),
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "graph_ready": (OUT / "graph.json").exists()}

@app.get("/api/node/{gid}")
def get_node(gid: int):
    d, g = graph(); node = next((n for n in d["nodes"] if int(n.get("gid", n.get("id"))) == gid), None)
    if node is None: raise HTTPException(404, "unknown gid")
    node = dict(node); node["senders"] = [g[u][gid] | {"gid": u} for u in g.predecessors(gid)]; node["receivers"] = [g[gid][v] | {"gid": v} for v in g.successors(gid)]
    return node

def tool(name: str, args: dict[str, Any]):
    d, g = graph(); nodes = {int(n.get("gid", n.get("id"))): n for n in d["nodes"]}
    gid = int(args.get("gid", 0))
    if name == "graph_top": return d["top"][:10]
    if name == "get_node": return get_node(gid)
    if name == "get_senders": return [nodes[x] for x in g.predecessors(gid)]
    if name == "get_receivers": return [nodes[x] for x in g.successors(gid)]
    if name == "common_receivers":
        ids = [int(x) for x in args.get("gids", [])]; sets = [set(g.successors(x)) for x in ids]
        return [nodes[x] for x in (set.intersection(*sets) if sets else set())]
    if name == "trace_downstream":
        depth = max(0, min(4, int(args.get("depth", 3)))); seen = {gid}; frontier = {gid}
        for _ in range(depth):
            frontier = set().union(*(set(g.successors(x)) for x in frontier)) - seen; seen |= frontier
        return [nodes[x] for x in seen if x != gid]
    if name == "cluster_summary": return next((c for c in d["clusters"] if c["cluster_id"] == int(args["cluster_id"])), {})
    raise ValueError(name)


def collect_gids(value: Any) -> set[int]:
    """Collect node ids from nested tool results for UI highlighting."""
    found: set[int] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"gid", "id", "source", "target"} and isinstance(item, (int, float)):
                found.add(int(item))
            else:
                found.update(collect_gids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(collect_gids(item))
    return found

class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    selected_gids: list[int] = Field(default_factory=list, max_length=100)
class Resilience(BaseModel): remove: list[int] = []

FUNCTIONS = [
    {"type": "function", "function": {"name": "graph_top", "description": "Top priority nodes with evidence", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_node", "description": "Return a node and its direct payments", "parameters": {"type": "object", "properties": {"gid": {"type": "integer"}}, "required": ["gid"]}}},
    {"type": "function", "function": {"name": "get_senders", "description": "Return direct senders", "parameters": {"type": "object", "properties": {"gid": {"type": "integer"}}, "required": ["gid"]}}},
    {"type": "function", "function": {"name": "get_receivers", "description": "Return direct receivers", "parameters": {"type": "object", "properties": {"gid": {"type": "integer"}}, "required": ["gid"]}}},
    {"type": "function", "function": {"name": "common_receivers", "description": "Find receivers common to several gids", "parameters": {"type": "object", "properties": {"gids": {"type": "array", "items": {"type": "integer"}}}, "required": ["gids"]}}},
    {"type": "function", "function": {"name": "trace_downstream", "description": "Trace downstream nodes", "parameters": {"type": "object", "properties": {"gid": {"type": "integer"}, "depth": {"type": "integer"}}, "required": ["gid"]}}},
    {"type": "function", "function": {"name": "cluster_summary", "description": "Summarize one cluster", "parameters": {"type": "object", "properties": {"cluster_id": {"type": "integer"}}, "required": ["cluster_id"]}}},
]

SYSTEM = ("Отвечай только по данным, полученным через инструменты. Указывай gid и цифры. "
          "Формулируй выводы как гипотезы и признаки, а не обвинения. "
          "В финальном ответе перечисли все затронутые gid.")

def selected_nodes(question, selected_gids):
    known = {int(n.get("gid", n.get("id"))) for n in graph_data()["nodes"]}
    explicit = list(dict.fromkeys(selected_gids))
    unknown = sorted(set(explicit) - known)
    if unknown:
        raise HTTPException(422, {"unknown_gids": unknown})
    return list(dict.fromkeys(explicit + [int(x) for x in re.findall(r"\b\d+\b", question) if int(x) in known]))


def raw_answer(steps, touched, reason):
    rows = []
    for step in steps:
        result = step["result"]
        if isinstance(result, dict) and "hypothesis" in result:
            rows.append(result["hypothesis"])
        items = result if isinstance(result, list) else [result]
        for item in items[:10]:
            if isinstance(item, dict) and ("gid" in item or "id" in item):
                rows.append(f"gid {item.get('gid', item.get('id'))}: {item.get('evidence', item.get('why', ''))}")
    return {"answer": "Локальный анализ (LLM недоступна). Гипотезы для проверки. " +
            ("; ".join(rows) if rows else "По заданным условиям совпадений не найдено."),
            "steps": list(steps), "highlight_gids": sorted(touched),
            "mode": "fallback", "llm_error": reason}


def fallback_answer(question: str, gids=None, reason="not_configured"):
    gids = selected_nodes(question, gids or [])
    q = question.lower()
    if len(gids) >= 2:
        name, args = "common_receivers", {"gids": gids}
    elif gids:
        name = ("get_senders" if any(x in q for x in ("отправител", "кто плат", "senders"))
                else "trace_downstream" if any(x in q for x in ("цепоч", "дальше", "downstream"))
                else "get_receivers" if any(x in q for x in ("получател", "кому", "receivers"))
                else "get_node")
        args = {"gid": gids[0]}
    else:
        match = re.search(r"(?:кластер|cluster)\s*(\d+)", q)
        name, args = ("cluster_summary", {"cluster_id": int(match[1])}) if match else ("graph_top", {})
    result = tool(name, args)
    return raw_answer([{"tool": name, "args": args, "result": result}],
                      set(gids) | collect_gids(result), reason)


async def model_answer(question, gids, steps, touched):
    from openai import AsyncOpenAI
    async with AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=20.0, max_retries=0) as client:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": question + "\nselected_gids: " + json.dumps(gids)}]
        for turn in range(6):
            response = await client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages, tools=FUNCTIONS, parallel_tool_calls=False,
                tool_choice="required" if not steps else "none" if turn == 5 else "auto")
            msg = response.choices[0].message
            if not msg.tool_calls:
                if not steps:
                    raise ValueError("ungrounded_answer")
                return {"answer": msg.content or "Недостаточно данных.", "steps": steps,
                        "highlight_gids": sorted(touched), "mode": "llm", "llm_error": None}
            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                if len(steps) >= 5:
                    raise ValueError("tool_limit")
                args = json.loads(call.function.arguments or "{}")
                try:
                    result = tool(call.function.name, args)
                except (KeyError, ValueError, nx.NetworkXError, HTTPException):
                    result = {"error": "Invalid tool arguments or unknown node"}
                steps.append({"tool": call.function.name, "args": args, "result": result})
                touched.update(collect_gids(result))
                touched.update(collect_gids(args))
                touched.update(int(x) for x in args.get("gids", []))
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(result, ensure_ascii=False, default=str)})
        raise ValueError("step_limit")


@app.post("/api/ask")
async def ask(req: Ask):
    gids = selected_nodes(req.question, req.selected_gids)
    if not os.getenv("OPENAI_API_KEY"):
        return fallback_answer(req.question, gids)
    steps, touched = [], set(gids)
    try:
        return await asyncio.wait_for(model_answer(req.question, gids, steps, touched), timeout=25)
    except Exception as exc:
        # Never return SDK exception text: it can contain sensitive request details.
        reason = type(exc).__name__
        logging.getLogger(__name__).warning("LLM request failed: %s", reason)
        return raw_answer(steps, touched, reason) if steps else fallback_answer(req.question, gids, reason)


@app.post("/api/resilience")
def resilience(req: Resilience):
    d, g = graph(); before_components = list(nx.weakly_connected_components(g)); total = sum(float(e.get("sum_kzt", 0)) for e in d["links"])
    removed_flow = sum(float(g[u][v].get("sum_kzt", 0)) for u, v in g.edges if u in req.remove or v in req.remove)
    before_largest = max((len(c) for c in before_components), default=0)
    g.remove_nodes_from(req.remove); after = nx.number_weakly_connected_components(g)
    after_largest = max((len(c) for c in nx.weakly_connected_components(g)), default=0)
    return {"before": {"components": len(before_components), "largest_component_size": before_largest, "n_nodes": len(d["nodes"])}, "after": {"components": after, "largest_component_size": after_largest, "n_nodes": g.number_of_nodes()}, "removed_gids": req.remove, "turnover_share_removed": removed_flow / total if total else 0.0}

@app.get("/", include_in_schema=False)
def index():
    p = FRONTEND_DIR / "index.html"
    if p.exists(): return FileResponse(p)
    return JSONResponse({"message": "API is running", "docs": "/docs"})


# API routes are declared first; the static mount handles the built frontend.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
