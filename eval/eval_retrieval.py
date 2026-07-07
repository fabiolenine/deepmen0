#!/usr/bin/env python3
"""Retrieval evaluation harness for the Deep Mem0 synthetic corpus.

Subcommands
-----------
  audit  Read-only, no LLM: for every case, measure the rank of each expected
         target SEPARATELY in the dense retriever (Ollama embedding + Qdrant
         query API) and in BM25 sparse (fastembed, language-aware, same doc/query
         normalization as seed_corpus.py), then roll results up per category.
         This isolates *recall* per retriever before any fusion/reranking.

  run    End-to-end metrics (hit@k / MRR / precision / recall) against a running
         mem0-compatible MCP server (tool `search_memories`), saved as a dated
         JSON baseline for later comparison.

  compare  Diff two saved `run` results.

Cases and memories come from corpus_synthetic.json; expected targets are
referenced by slug and resolved to deterministic IDs with
uuid5(NAMESPACE_URL, "deepmem0:" + slug) — the same rule seed_corpus.py uses.

Environment (all optional):
  QDRANT_URL (http://localhost:6333)   OLLAMA_URL (http://localhost:11434)
  EMBED_MODEL (bge-m3)  EMBED_DIMS (1024)  BM25_LANGUAGE (portuguese)
  COLLECTION (deepmem0_demo)  USER_ID (demo)  MCP_URL (http://localhost:8081/mcp)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import requests

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
BM25_LANGUAGE = os.environ.get("BM25_LANGUAGE", "portuguese")
COLLECTION = os.environ.get("COLLECTION", "deepmem0_demo")
USER_ID = os.environ.get("USER_ID", "demo")
MCP_URL = os.environ.get("MCP_URL", "http://localhost:8081/mcp")

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(HERE, "corpus_synthetic.json")
BASELINES_DIR = os.path.join(HERE, "baselines")

K_VALUES = (1, 3, 5, 10)
_SEP = re.compile(r"[_\-]+")


def slug_to_id(slug: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"deepmem0:{slug}"))


def load_cases() -> list[dict]:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)
    cases = []
    for c in corpus["cases"]:
        cases.append({**c, "expected_ids": [slug_to_id(s) for s in c["expected_slugs"]]})
    return cases


def bm25_normalize(text: str) -> str:
    return _SEP.sub(" ", (text or "").lower())


def _metrics_for_case(ranked_ids: list, expected: set) -> dict:
    m = {}
    for k in K_VALUES:
        topk = ranked_ids[:k]
        rel = sum(1 for i in topk if i in expected)
        m[f"hit@{k}"] = 1.0 if rel else 0.0
        m[f"precision@{k}"] = rel / k
        m[f"recall@{k}"] = rel / len(expected) if expected else 0.0
    m["mrr"] = next((1.0 / r for r, i in enumerate(ranked_ids, 1) if i in expected), 0.0)
    return m


# --------------------------------------------------------------- audit -------
def _dense_embed(text: str) -> list:
    r = requests.post(
        f"{OLLAMA_URL}/api/embed", json={"model": EMBED_MODEL, "input": text}, timeout=120
    )
    r.raise_for_status()
    return r.json()["embeddings"][0]


_BM25_ENCODER = None


def _bm25_query_vector(query: str):
    global _BM25_ENCODER
    if _BM25_ENCODER is None:
        from fastembed import SparseTextEmbedding

        _BM25_ENCODER = SparseTextEmbedding(model_name="Qdrant/bm25", language=BM25_LANGUAGE)
    results = list(_BM25_ENCODER.embed([bm25_normalize(query)]))
    if not results:
        return None
    sv = results[0]
    return {"indices": sv.indices.tolist(), "values": sv.values.tolist()}


def _qdrant_rank(query_vector, expected: set, pool: int, using: str | None = None):
    body = {
        "query": query_vector,
        "limit": pool,
        "with_payload": False,
        "filter": {"must": [{"key": "user_id", "match": {"value": USER_ID}}]},
    }
    if using:
        body["using"] = using
    r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/query", json=body, timeout=60)
    r.raise_for_status()
    points = r.json()["result"]["points"]
    for rank, p in enumerate(points, start=1):
        if str(p["id"]) in expected:
            return rank
    return None


def cmd_audit(args):
    cases = load_cases()
    rows, by_cat = [], defaultdict(lambda: {"n": 0, "dense_top10": 0, "bm25_top10": 0})
    print(f"{'dense':>6s} {'bm25':>6s}  [{'category':<15s}] query")
    for c in cases:
        expected = set(c["expected_ids"])
        d = _qdrant_rank(_dense_embed(c["query"]), expected, args.pool)
        sv = _bm25_query_vector(c["query"])
        b = _qdrant_rank(sv, expected, args.pool, using="bm25") if sv else None
        cat = c.get("category", "?")
        agg = by_cat[cat]
        agg["n"] += 1
        agg["dense_top10"] += 1 if (d and d <= 10) else 0
        agg["bm25_top10"] += 1 if (b and b <= 10) else 0
        rows.append({"query": c["query"], "category": cat, "dense_rank": d, "bm25_rank": b})
        print(f"{str(d) if d else '-':>6s} {str(b) if b else '-':>6s}  [{cat:<15s}] {c['query'][:64]}")

    print("\nper-category recall@10 (per retriever, pre-fusion):")
    for cat, a in sorted(by_cat.items()):
        print(f"  {cat:<16s} n={a['n']:<3d} dense {a['dense_top10']}/{a['n']}  bm25 {a['bm25_top10']}/{a['n']}")

    os.makedirs(BASELINES_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(BASELINES_DIR, f"{stamp}_audit_{args.label or 'default'}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump({"_header": {"collection": COLLECTION, "embed_model": EMBED_MODEL,
                               "bm25_language": BM25_LANGUAGE, "pool": args.pool},
                   "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"saved to {dest}")


# ----------------------------------------------------------------- run -------
async def _mcp_search(query: str, flags: dict) -> list:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    call_args = {"query": query, "user_id": USER_ID, **flags}
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool("search_memories", call_args)
    text = "".join(c.text for c in res.content if getattr(c, "type", None) == "text")
    try:
        obj = json.loads(text)
    except Exception:
        return []
    if isinstance(obj, dict):
        for key in ("results", "memories", "data"):
            if isinstance(obj.get(key), list):
                return obj[key]
        return []
    return obj if isinstance(obj, list) else []


def cmd_run(args):
    cases = load_cases()
    flags = {"limit": args.limit}
    if args.rerank:
        flags["rerank"] = True
    per_case, agg = [], defaultdict(float)
    n = 0
    for c in cases:
        expected = set(c["expected_ids"])
        case_flags = dict(flags)
        for k, v in (c.get("filters") or {}).items():
            case_flags[k] = v
        try:
            results = asyncio.run(_mcp_search(c["query"], case_flags))
        except Exception as e:
            per_case.append({"query": c["query"], "error": str(e)})
            continue
        ranked = [r.get("id") for r in results if isinstance(r, dict)]
        met = _metrics_for_case(ranked, expected)
        for k, v in met.items():
            agg[k] += v
        n += 1
        per_case.append({"query": c["query"], "category": c.get("category"),
                         "returned_ids": ranked[: max(K_VALUES)], **met})
    aggregate = {k: round(v / n, 4) for k, v in agg.items()} if n else {}
    os.makedirs(BASELINES_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(BASELINES_DIR, f"{stamp}_{args.label}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump({"_header": {"label": args.label, "backend": MCP_URL, "collection": COLLECTION,
                               "flags": flags, "n_cases": n, "n_errors": len(per_case) - n},
                   "aggregate": aggregate, "cases": per_case}, f, ensure_ascii=False, indent=2)
    print(f"[{args.label}] n={n} errors={len(per_case)-n}")
    for k in ("hit@1", "hit@3", "hit@5", "hit@10", "mrr"):
        if k in aggregate:
            print(f"  {k:12s} {aggregate[k]}")
    print(f"saved to {dest}")


def cmd_compare(args):
    with open(args.baseline, encoding="utf-8") as f:
        a = json.load(f)["aggregate"]
    with open(args.candidate, encoding="utf-8") as f:
        b = json.load(f)["aggregate"]
    print(f"{'metric':14s} {'baseline':>10s} {'candidate':>10s} {'delta':>10s}")
    for k in sorted(set(a) | set(b)):
        print(f"{k:14s} {a.get(k, 0.0):>10.4f} {b.get(k, 0.0):>10.4f} {b.get(k, 0.0)-a.get(k, 0.0):>+10.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="per-retriever rank of expected targets (read-only)")
    a.add_argument("--pool", type=int, default=100)
    a.add_argument("--label")
    a.set_defaults(func=cmd_audit)

    r = sub.add_parser("run", help="end-to-end metrics via MCP server")
    r.add_argument("--label", required=True)
    r.add_argument("--limit", type=int, default=10)
    r.add_argument("--rerank", action="store_true")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="diff two saved run results")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
