"""FastAPI service for the generated graph. Run: uvicorn api:app --reload."""
from __future__ import annotations
import json, os, re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

OUT = Path(os.getenv("GRAPH_OUT", "out"))
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", "frontend/dist"))
app = FastAPI(title="Money Graph API")

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
def get_graph(): return graph_data()

@app.get("/api/node/{gid}")
def get_node(gid: int):
    d, g = graph(); node = next((n for n in d["nodes"] if int(n.get("gid", n.get("id"))) == gid), None)
    if node is None: raise HTTPException(404, "unknown gid")
    node = dict(node); node["senders"] = [g[u][gid] | {"gid": u} for u in g.predecessors(gid)]; node["receivers"] = [g[gid][v] | {"gid": v} for v in g.successors(gid)]
    return node

def tool(name: str, args: dict[str, Any]):
    d, g = graph(); nodes = {int(n.get("gid", n.get("id"))): n for n in d["nodes"]}
    gid = int(args.get("gid", 0))
    if name == "get_node": return get_node(gid)
    if name == "get_senders": return [nodes[x] for x in g.predecessors(gid)]
    if name == "get_receivers": return [nodes[x] for x in g.successors(gid)]
    if name == "common_receivers":
        ids = [int(x) for x in args.get("gids", [])]; sets = [set(g.successors(x)) for x in ids]
        return [nodes[x] for x in (set.intersection(*sets) if sets else set())]
    if name == "trace_downstream":
        depth = int(args.get("depth", 3)); seen = {gid}; frontier = {gid}
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

class Ask(BaseModel): question: str
class Resilience(BaseModel): remove: list[int] = []

FUNCTIONS = [
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

def model_answer(question: str):
    """Run up to six OpenAI tool-calling turns; return None when unavailable."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, timeout=20.0, max_retries=0)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        steps, touched = [], set()
        for _ in range(6):
            response = client.chat.completions.create(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), messages=messages, tools=FUNCTIONS, tool_choice="auto", temperature=0)
            msg = response.choices[0].message
            if not msg.tool_calls:
                return {"answer": msg.content or "Недостаточно данных.", "steps": steps, "highlight_gids": sorted(touched)}
            messages.append(msg.model_dump() if hasattr(msg, "model_dump") else msg)
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                if "gid" in args: touched.add(int(args["gid"]))
                touched.update(int(x) for x in args.get("gids", []))
                result = tool(call.function.name, args)
                steps.append({"tool": call.function.name, "args": args, "result": result})
                touched.update(collect_gids(result))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False, default=str)})
        return {"answer": "Лимит анализа достигнут; смотрите результаты инструментов.", "steps": steps, "highlight_gids": sorted(touched)}
    except Exception:
        return None

@app.post("/api/ask")
def ask(req: Ask):
    # A five/six-step model loop is attempted with an execution timeout.
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        result = pool.submit(model_answer, req.question).result(timeout=25)
        if result is not None:
            return result
    except (TimeoutError, Exception):
        result = None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    # Deterministic fallback keeps the demo usable without a key or when the model is down.
    return fallback_answer(req.question)
    d, _ = graph(); q = req.question.lower(); gids = [int(n.get("gid", n.get("id"))) for n in d["nodes"] if str(n.get("gid", n.get("id"))) in q]
    touched = gids[:1]; steps = []
    if touched:
        result = tool("get_node", {"gid": touched[0]}); steps.append({"tool": "get_node", "result": result})
        answer = f"Гипотеза по gid {touched[0]}: роль {result.get('role')}, признаки: {result.get('evidence')}."
    else: answer = "Укажите gid в вопросе; ответ строится по данным графа."
    return {"answer": answer, "steps": steps[:6], "highlight_gids": touched}

def fallback_answer(question: str):
    """Return useful raw tool output when the LLM is unavailable."""
    d, _ = graph()
    known = {int(n.get("gid", n.get("id"))) for n in d["nodes"]}
    gids = [int(x) for x in re.findall(r"\d{6,}", question) if int(x) in known]
    gids = list(dict.fromkeys(gids))
    steps, touched = [], set(gids)
    if len(gids) >= 2:
        args = {"gids": gids}
        result = tool("common_receivers", args)
        steps.append({"tool": "common_receivers", "args": args, "result": result})
        touched.update(collect_gids(result))
        answer = "\u0413\u0438\u043f\u043e\u0442\u0435\u0437\u0430: \u043e\u0431\u0449\u0438\u0435 \u043f\u043e\u043b\u0443\u0447\u0430\u0442\u0435\u043b\u0438 \u0432\u043e\u0437\u0432\u0440\u0430\u0449\u0435\u043d\u044b common_receivers; \u043d\u0443\u0436\u043d\u0430 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u0430\u043d\u0430\u043b\u0438\u0442\u0438\u043a\u043e\u043c."
    elif gids:
        args = {"gid": gids[0]}
        result = tool("get_node", args)
        steps.append({"tool": "get_node", "args": args, "result": result})
        touched.update(collect_gids(result))
        answer = f"\u0413\u0438\u043f\u043e\u0442\u0435\u0437\u0430 \u043f\u043e gid {gids[0]}: \u0440\u043e\u043b\u044c {result.get('role')}, \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u0438: {result.get('evidence')}."
    else:
        result = d.get("top", [])[:10]
        steps.append({"tool": "graph_top", "result": result})
        touched.update(collect_gids(result))
        answer = "\u041c\u043e\u0434\u0435\u043b\u044c \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u0430; \u0432\u043e\u0437\u0432\u0440\u0430\u0449\u0451\u043d \u0441\u044b\u0440\u043e\u0439 top \u0438\u0437 graph.json."
    return {"answer": answer, "steps": steps[:6], "highlight_gids": sorted(touched)}


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
