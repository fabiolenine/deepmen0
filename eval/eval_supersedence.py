#!/usr/bin/env python3
"""DeepMem0 v0.3 supersedence scenario — the semantic-temporality proof.

Seeds versioned fact pairs (v1 superseded by v2) into a throwaway collection
and checks four properties:

  A. NORMAL SEARCH — the current fact (v2) outranks its superseded ancestor,
     which stays present in the pool (demoted, never excluded).
  B. NEVER LOST — a query phrased exclusively like v1 still finds v1.
  C. AS-OF — anchored between v1's creation and the supersession, v1 returns
     to the top: v2 didn't exist yet (record-time filter) and v1 carried no
     penalty at the anchor.
  D. CONTROL — on a corpus without any supersession, temporality ON == OFF.

Marking uses the real `_mark_superseded` production helper against a real
Qdrant, so the write path is exercised too; the LLM-detection leg is validated
separately in production smoke tests (this harness is deterministic, no LLM).

Environment: QDRANT_URL, OLLAMA_URL, EMBED_MODEL (bge-m3), EMBED_DIMS (1024),
MEM0_LANGUAGE (pt), LLM_MODEL (instantiated, never called).

Usage:
  python eval/eval_supersedence.py [--collection deepmem0_supersedence] [--rerank]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

os.environ.setdefault("MEM0_TELEMETRY", "False")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
EMBED_DIMS = int(os.environ.get("EMBED_DIMS", "1024"))
LANGUAGE = os.environ.get("MEM0_LANGUAGE", "pt")
USER_ID = "supersedence_demo"
# Controls live under their own scope so the D check compares pure untouched
# corpora — otherwise superseded pair members leak into control results and
# the (correct) penalty makes ON != OFF for the wrong reason.
CONTROL_USER_ID = "supersedence_demo_ctl"

# Versioned facts: v2 replaces v1 (same entity + attribute, new value).
PAIRS = [
    {
        "query": "qual banco de dados o atlas_ingest usa?",
        "v1": "O atlas_ingest usa um banco de dados MySQL para os eventos brutos.",
        "v2": "O atlas_ingest usa um banco de dados PostgreSQL para os eventos brutos.",
        "v1_only_query": "o atlas_ingest chegou a usar MySQL?",
    },
    {
        "query": "com que frequência rodam os backups do Orion?",
        "v1": "Os backups do Orion rodam semanalmente, aos domingos.",
        "v2": "Os backups do Orion rodam diariamente, às 03h.",
        "v1_only_query": "os backups do Orion eram semanais aos domingos?",
    },
    {
        "query": "qual é o limite de memória dos workers do hermes_fx?",
        "v1": "Os workers do hermes_fx têm limite de memória de 2 GiB.",
        "v2": "Os workers do hermes_fx têm limite de memória de 4 GiB.",
        "v1_only_query": "o limite dos workers do hermes_fx já foi 2 GiB?",
    },
]

CONTROL = [
    "A Vetorial Labs faz all-hands toda primeira sexta-feira do mês.",
    "O time de plataforma mantém plantão com rotação semanal.",
    "As chaves de API de staging expiram a cada 90 dias.",
    "A documentação interna fica num wiki com busca full-text.",
]

T0_DAYS_AGO = 30   # v1 created
T1_DAYS_AGO = 5    # v2 created + v1 superseded
ANCHOR_DAYS_AGO = 15  # as-of anchor between T0 and T1


def build_memory(rerank: bool, temporality_enabled: bool = True):
    from mem0 import Memory

    config = {
        "language": LANGUAGE,
        "llm": {
            "provider": "ollama",
            "config": {
                "model": os.environ.get("LLM_MODEL", "llama3.1"),
                "ollama_base_url": OLLAMA_URL,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": ARGS.collection,
                "url": QDRANT_URL,
                "embedding_model_dims": EMBED_DIMS,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": EMBED_MODEL,
                "embedding_dims": EMBED_DIMS,
                "ollama_base_url": OLLAMA_URL,
            },
        },
        "temporality": {"enabled": temporality_enabled},
    }
    if rerank:
        config["reranker"] = {
            "provider": "sentence_transformer",
            "config": {
                "model": os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
                "device": "cpu",
            },
        }
    return Memory.from_config(config)


def seed_and_supersede(memory) -> dict:
    """Seed v1/v2/control facts, backdate them, mark v1 superseded via the
    REAL production helper. Returns text -> memory_id."""
    from mem0.memory.main import _mark_superseded

    now = datetime.now(timezone.utc)
    t0 = (now - timedelta(days=T0_DAYS_AGO)).isoformat()
    t1 = (now - timedelta(days=T1_DAYS_AGO)).isoformat()

    ids = {}
    for pair in PAIRS:
        for text in (pair["v1"], pair["v2"]):
            result = memory.add(text, user_id=USER_ID, infer=False)
            ids[text] = result["results"][0]["id"]
    for text in CONTROL:
        result = memory.add(text, user_id=CONTROL_USER_ID, infer=False)
        ids[text] = result["results"][0]["id"]

    # Backdate: v1 born at T0, v2 born at T1, controls at T0.
    for pair in PAIRS:
        for text, born in ((pair["v1"], t0), (pair["v2"], t1)):
            mem_id = ids[text]
            payload = dict(memory.vector_store.get(vector_id=mem_id).payload)
            payload["created_at"] = born
            payload["updated_at"] = born
            memory.vector_store.update(vector_id=mem_id, payload=payload)
    for text in CONTROL:
        mem_id = ids[text]
        payload = dict(memory.vector_store.get(vector_id=mem_id).payload)
        payload["created_at"] = t0
        payload["updated_at"] = t0
        memory.vector_store.update(vector_id=mem_id, payload=payload)

    # Supersede via the production code path, then backdate superseded_at to T1
    # (the helper stamps "now"; the scenario needs the anchor between T0 and T1).
    for pair in PAIRS:
        marked = _mark_superseded(
            memory.vector_store, memory.db, ids[pair["v2"]], pair["v2"], [ids[pair["v1"]]]
        )
        assert marked == [ids[pair["v1"]]], f"marking failed for {pair['v1'][:40]!r}"
        payload = dict(memory.vector_store.get(vector_id=ids[pair["v1"]]).payload)
        payload["superseded_at"] = t1
        memory.vector_store.update(vector_id=ids[pair["v1"]], payload=payload)
    return ids


def ranked_ids(memory, query, k=5, as_of=None, user_id=USER_ID):
    kwargs = {"user_id": user_id, "top_k": k}
    if as_of:
        kwargs["as_of"] = as_of
    return [r["id"] for r in memory.search(query, **kwargs)["results"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="deepmem0_supersedence")
    parser.add_argument("--rerank", action="store_true")
    ARGS = parser.parse_args()

    print(f"collection={ARGS.collection} rerank={ARGS.rerank} embed={EMBED_MODEL}")

    memory_on = build_memory(ARGS.rerank, temporality_enabled=True)
    ids = seed_and_supersede(memory_on)
    print(f"seeded {len(ids)} memories ({len(PAIRS)} versioned pairs + {len(CONTROL)} controls)")

    anchor = (datetime.now(timezone.utc) - timedelta(days=ANCHOR_DAYS_AGO)).date().isoformat()
    failures = []

    # --- A. Normal search: v2 outranks v1; v1 still in the pool -------------
    a_wins, a_present = 0, 0
    for pair in PAIRS:
        ranked = ranked_ids(memory_on, pair["query"], k=5)
        v1_rank = ranked.index(ids[pair["v1"]]) if ids[pair["v1"]] in ranked else 99
        v2_rank = ranked.index(ids[pair["v2"]]) if ids[pair["v2"]] in ranked else 99
        if v2_rank < v1_rank:
            a_wins += 1
        if v1_rank < 99:
            a_present += 1
    print(f"[A] current fact outranks superseded: {a_wins}/{len(PAIRS)}; superseded still present: {a_present}/{len(PAIRS)}")
    if a_wins < len(PAIRS):
        failures.append("A")

    # --- B. Never lost: v1-only phrasing still finds v1 ---------------------
    b_found = 0
    for pair in PAIRS:
        ranked = ranked_ids(memory_on, pair["v1_only_query"], k=5)
        if ids[pair["v1"]] in ranked:
            b_found += 1
    print(f"[B] superseded fact still reachable by its own phrasing: {b_found}/{len(PAIRS)}")
    if b_found < len(PAIRS):
        failures.append("B")

    # --- C. As-of anchor between T0 and T1: v1 back on top ------------------
    c_wins = 0
    for pair in PAIRS:
        ranked = ranked_ids(memory_on, pair["query"], k=5, as_of=anchor)
        v2_absent = ids[pair["v2"]] not in ranked   # didn't exist at the anchor
        v1_top = bool(ranked) and ranked[0] == ids[pair["v1"]]
        if v1_top and v2_absent:
            c_wins += 1
        else:
            print(f"    [C-detail] {pair['query']!r}: v1_top={v1_top} v2_absent={v2_absent}")
    print(f"[C] as_of={anchor} restores the old fact: {c_wins}/{len(PAIRS)}")
    if c_wins < len(PAIRS):
        failures.append("C")

    # --- D. Control: no supersession involved -> ON == OFF ------------------
    memory_off = build_memory(ARGS.rerank, temporality_enabled=False)
    d_ok = 0
    control_queries = ["quando é o all-hands da Vetorial Labs?", "como funciona o plantão do time de plataforma?"]
    for query in control_queries:
        on = ranked_ids(memory_on, query, k=3, user_id=CONTROL_USER_ID)
        off = ranked_ids(memory_off, query, k=3, user_id=CONTROL_USER_ID)
        if on == off:
            d_ok += 1
    print(f"[D] control (no supersession): temporality ON == OFF on {d_ok}/{len(control_queries)} queries")
    if d_ok < len(control_queries):
        failures.append("D")

    print("RESULT:", "PASS" if not failures else f"FAIL {failures}")
    sys.exit(1 if failures else 0)
