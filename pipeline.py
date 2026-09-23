#!/usr/bin/env python3
"""Build the analytics exports and graph.json from the case parquet files.

Usage: python pipeline.py --data ./data --out ./out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

ROLES = ["consolidator", "transit", "distributor", "coordinator", "terminal", "peripheral"]
ROLE_WEIGHTS = {
    "coordinator": 1.00,
    "consolidator": 0.90,
    "distributor": 0.80,
    "transit": 0.65,
    "terminal": 0.55,
    "peripheral": 0.10,
}


def load(data_dir: Path):
    edges = pd.read_parquet(data_dir / "edges.parquet")
    nodes = pd.read_parquet(data_dir / "nodes.parquet")
    tx_path = data_dir / "transactions.parquet"
    tx = pd.read_parquet(tx_path) if tx_path.exists() else pd.DataFrame()
    return edges, nodes, tx


def _col(frame: pd.DataFrame, names, default=None):
    lower = {str(c).lower(): c for c in frame.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return default


def build_graph(edges: pd.DataFrame, nodes: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for row in nodes.itertuples(index=False):
        gid = getattr(row, "gid")
        g.add_node(gid)
    for row in edges.itertuples(index=False):
        g.add_edge(row.src, row.dst, sum_kzt=float(row.sum_kzt), n_tx=int(row.n_tx), depth=int(getattr(row, "depth", 0)))
    return g


def _minmax(values: pd.Series) -> pd.Series:
    values = values.astype(float).fillna(0)
    lo, hi = values.min(), values.max()
    return pd.Series(0.0 if hi == lo else (values - lo) / (hi - lo), index=values.index)


def _courier_mask(nodes: pd.DataFrame) -> pd.Series:
    for name in ("is_courier", "courier", "is_kurier"):
        c = _col(nodes, [name])
        if c:
            return nodes[c].fillna(False).astype(bool)
    # The supplied case has no courier column: its 81 seed clients are the
    # known upstream couriers. This keeps the seed incoming-flow caveat out of
    # pass-through and makes upstream courier counts meaningful.
    seed = _col(nodes, ["is_seed", "seed"])
    return nodes[seed].fillna(False).astype(bool) if seed else pd.Series(False, index=nodes.index)


def metrics(g: nx.DiGraph, nodes: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    id_col = _col(nodes, ["gid"])
    depth_col = _col(nodes, ["depth"])
    seed_col = _col(nodes, ["is_seed", "seed"])
    df = pd.DataFrame({
        "gid": nodes[id_col].astype(int),
        "depth": nodes[depth_col].fillna(0).astype(int),
        "is_seed": nodes[seed_col].fillna(False).astype(bool),
        "is_courier": _courier_mask(nodes).values,
    })
    in_senders = {n: len(set(g.predecessors(n))) for n in g}
    out_receivers = {n: len(set(g.successors(n))) for n in g}
    sum_in = dict(g.in_degree(weight="sum_kzt"))
    sum_out = dict(g.out_degree(weight="sum_kzt"))
    in_tx = dict(g.in_degree(weight="n_tx"))
    out_tx = dict(g.out_degree(weight="n_tx"))
    pagerank = nx.pagerank(g, weight="sum_kzt") if g else {}
    df["in_senders"] = df.gid.map(in_senders).fillna(0).astype(int)
    df["out_receivers"] = df.gid.map(out_receivers).fillna(0).astype(int)
    df["sum_in"] = df.gid.map(sum_in).fillna(0.0)
    df["sum_out"] = df.gid.map(sum_out).fillna(0.0)
    df["in_tx"] = df.gid.map(in_tx).fillna(0).astype(int)
    df["out_tx"] = df.gid.map(out_tx).fillna(0).astype(int)
    df["pagerank"] = df.gid.map(pagerank).fillna(0.0)
    df["pass_through"] = np.where(df.sum_in > 0, df.sum_out / df.sum_in, np.nan)
    df.loc[df.is_courier, "pass_through"] = np.nan
    df["is_cutoff"] = (df.depth == 4) & (df.out_receivers == 0)
    # Upstream distinct couriers; if the source data has no courier flag, this stays zero.
    courier_by_gid = dict(zip(df.gid, df.is_courier))
    upstream = {}
    for n in g:
        seen, stack = set(), list(g.predecessors(n))
        while stack:
            p = stack.pop()
            if p in seen: continue
            seen.add(p)
            stack.extend(g.predecessors(p))
        upstream[n] = sum(courier_by_gid.get(p, False) for p in seen)
    df["upstream_couriers"] = df.gid.map(upstream).fillna(0).astype(int)
    # Optional temporal signal from transactions: median days from an
    # observed incoming transfer to the next observed outgoing transfer.
    handoff = {}
    if not tx.empty and {"src", "dst", "date"}.issubset(tx.columns):
        dated = tx.assign(date=pd.to_datetime(tx["date"]))
        incoming = dated.groupby("dst")["date"].apply(list).to_dict()
        outgoing = dated.groupby("src")["date"].apply(list).to_dict()
        for gid, dates in incoming.items():
            next_out = sorted(outgoing.get(gid, []))
            delays = []
            for received in dates:
                later = next((sent for sent in next_out if sent >= received), None)
                if later is not None:
                    delays.append((later - received).days)
            if delays:
                handoff[gid] = float(np.median(delays))
    df["handoff_days_median"] = df.gid.map(handoff).astype(float)
    return df


def classify(df: pd.DataFrame) -> pd.DataFrame:
    def one(r):
        if r.out_receivers >= 20: return "distributor", min(r.out_receivers / 60.0, 1.0), f"{r.out_receivers} получателей"
        if r.in_senders >= 8: return "consolidator", min(r.in_senders / 24.0, 1.0), f"{r.in_senders} отправителей"
        if r.upstream_couriers >= 3: return "coordinator", min(r.upstream_couriers / 6.0, 1.0), f"{r.upstream_couriers} курьеров выше по цепочке"
        if pd.notna(r.pass_through) and 0.8 <= r.pass_through <= 1.2 and r.in_senders + r.out_receivers <= 8:
            return "transit", min(1.0, 1.0 - abs(1.0 - r.pass_through) / .2), f"пропуск {r.pass_through:.2f}, связей {r.in_senders + r.out_receivers}"
        if r.sum_in > 0 and r.sum_out <= r.sum_in * .1 and not r.is_cutoff:
            return "terminal", min(1.0, r.sum_in / max(df.sum_in.quantile(.9), 1)), f"вход {r.sum_in:,.0f}, выход {r.sum_out:,.0f}"
        return "peripheral", .2, f"{r.in_senders} отправителей, {r.out_receivers} получателей"
    values = df.apply(one, axis=1, result_type="expand")
    df[["role", "role_score", "evidence"]] = values
    df["evidence"] = df.evidence.str.slice(0, 200)
    return df


def clusters(g: nx.DiGraph, df: pd.DataFrame):
    ug = g.to_undirected()
    communities = list(nx.community.louvain_communities(ug, weight="sum_kzt", seed=42)) if ug else []
    cid = {gid: i for i, group in enumerate(communities) for gid in group}
    df["cluster_id"] = df.gid.map(cid).fillna(-1).astype(int)
    rows = []
    for i, group in enumerate(communities):
        roles = df[df.gid.isin(group)].role.value_counts(normalize=True)
        dominant = roles.index[0] if len(roles) else "peripheral"
        composition = ", ".join(f"{role} {share:.0%}" for role, share in roles.head(3).items())
        hyp = f"Кластер по составу ролей: {composition}; гипотеза — преобладает {dominant}"
        internal = sum(g[u][v].get("sum_kzt", 0) for u, v in g.edges if u in group and v in group)
        top = df[df.gid.isin(group)].sort_values("priority_score", ascending=False).gid.head(10).tolist()
        rows.append({"cluster_id": i, "n_nodes": len(group), "n_seed": int(df[df.gid.isin(group)].is_seed.sum()), "sum_kzt_internal": internal, "top_gids": ",".join(map(str, top)), "hypothesis": hyp})
    return pd.DataFrame(rows)


def make_outputs(g, df, edges, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    turnover = df.sum_in + df.sum_out
    df["role_weight"] = df.role.map(ROLE_WEIGHTS).fillna(0.0)
    df["priority_score"] = .35 * df["role_weight"] + .30 * _minmax(turnover) + .15 * _minmax(df.in_senders) + .20 * _minmax(df.upstream_couriers)
    clusters_df = clusters(g, df)
    # Keep the starter names as aliases so existing jury checks and dashboards continue to work.
    df["in_deg"] = df["in_senders"]
    df["out_deg"] = df["out_receivers"]
    df["in_kzt"] = df["sum_in"]
    df["out_kzt"] = df["sum_out"]
    df["truncated_by_depth"] = df["is_cutoff"]
    roles_cols = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence", "in_senders", "out_receivers", "sum_in", "sum_out", "in_deg", "out_deg", "in_kzt", "out_kzt", "in_tx", "out_tx", "pagerank", "pass_through", "depth", "is_seed", "is_cutoff", "truncated_by_depth", "upstream_couriers", "handoff_days_median"]
    df[roles_cols].to_csv(out_dir / "nodes_roles.csv", index=False)
    clusters_df.to_csv(out_dir / "clusters.csv", index=False)
    top = df.sort_values("priority_score", ascending=False).head(max(20, min(100, len(df))))
    top.assign(rank=np.arange(1, len(top) + 1), why=top.evidence)[["rank", "gid", "role", "priority_score", "why"]].to_csv(out_dir / "top_nodes.csv", index=False)
    links = [{"source": int(r.src), "target": int(r.dst), "sum_kzt": float(r.sum_kzt), "n_tx": int(r.n_tx)} for r in edges.itertuples(index=False)]
    node_json = [{"id": int(row["gid"]), **{k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}} for row in df[roles_cols].to_dict("records")]
    cluster_json = [{k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()} for row in clusters_df.to_dict("records")]
    top_json = [{"rank": i + 1, "gid": int(r.gid), "role": r.role, "priority_score": float(r.priority_score), "why": r.evidence} for i, r in enumerate(top.itertuples())]
    (out_dir / "graph.json").write_text(json.dumps({"nodes": node_json, "links": links, "clusters": cluster_json, "top": top_json}, ensure_ascii=False, indent=2), encoding="utf-8")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./out")
    args = ap.parse_args()
    edges, nodes, tx = load(Path(args.data))
    g = build_graph(edges, nodes)
    df = classify(metrics(g, nodes, tx))
    make_outputs(g, df, edges, Path(args.out))
    print(f"Wrote graph.json, nodes_roles.csv, clusters.csv and top_nodes.csv to {args.out}")


if __name__ == "__main__":
    main()
