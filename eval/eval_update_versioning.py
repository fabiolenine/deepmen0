#!/usr/bin/env python3
"""Update-versioning leak — the executable acceptance criterion (roadmap item).

Known limitation (docs/roadmap.md): in-place ``update()`` rewrites a memory's
content while preserving its original ``created_at`` — so an ``as_of`` anchor
that predates the update still sees the NEW content. Record-time queries leak
future knowledge into the past. This eval reproduces that leak deterministically
and is the acceptance criterion for the future "update versioning" work: today
it XFAILs by design; the day it passes, promote it to a hard gate.

Scenario per pair:
  1. add fact v1, backdate created_at to T0 (30 days ago);
  2. in-place ``memory.update(id, data=v2)`` NOW (created_at stays T0);
  3. ``search(as_of=anchor)`` with T0 < anchor < now — record-time truth at the
     anchor is v1.

Ternary verdict (each pair, then overall):
  LEAK        — anchored search returns the point with v2 content (the known
                limitation, reproduced).
  FIXED       — anchored search returns v1 content (update versioning works).
  UNEXPECTED  — the point vanished from anchored results, content matches
                neither version, or the search errored.

Exit codes:
  default:        0 if ALL pairs LEAK (XFAIL, known) or ALL pairs FIXED
                  (prints promotion instruction); 1 on any UNEXPECTED or a
                  mixed leak/fixed split (a partial fix is suspicious).
  --expect-fixed: 0 only if ALL pairs FIXED (the hard-gate mode for the
                  update-versioning implementation).

Isolation: throwaway collection, deleted at the end with verification — this
eval never touches a production collection.

Environment: QDRANT_URL, OLLAMA_URL, EMBED_MODEL (bge-m3), EMBED_DIMS (1024),
MEM0_LANGUAGE (pt), LLM_MODEL (instantiated, never called — infer=False).

Usage:
  python eval/eval_update_versioning.py [--collection deepmem0_update_versioning]
  python eval/eval_update_versioning.py --expect-fixed   # after the fix ships
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

os.environ.setdefault("MEM0_TELEMETRY", "False")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
EMBED_DIMS = int(os.environ.get("EMBED_DIMS", "1024"))
LANGUAGE = os.environ.get("MEM0_LANGUAGE", "pt")
USER_ID = "update_versioning_demo"

# v2 replaces v1 IN PLACE via update() — distinct value markers per version.
PAIRS = [
    {
        "query": "qual banco de dados o atlas_ingest usa?",
        "v1": "O atlas_ingest usa um banco de dados MySQL para os eventos brutos.",
        "v2": "O atlas_ingest usa um banco de dados PostgreSQL para os eventos brutos.",
        "v1_marker": "mysql",
        "v2_marker": "postgresql",
    },
    {
        "query": "com que frequência rodam os backups do Orion?",
        "v1": "Os backups do Orion rodam semanalmente, aos domingos.",
        "v2": "Os backups do Orion rodam diariamente, às 03h.",
        "v1_marker": "semanalmente",
        "v2_marker": "diariamente",
    },
    {
        "query": "qual é o limite de memória dos workers do hermes_fx?",
        "v1": "Os workers do hermes_fx têm limite de memória de 2 GiB.",
        "v2": "Os workers do hermes_fx têm limite de memória de 4 GiB.",
        "v1_marker": "2 gib",
        "v2_marker": "4 gib",
    },
]

T0_DAYS_AGO = 30      # v1 born
ANCHOR_DAYS_AGO = 15  # anchor between T0 and the update (now)


def build_memory():
    from mem0 import Memory

    return Memory.from_config({
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
        "temporality": {"enabled": True},
    })


def seed_and_update(memory) -> dict:
    """Add v1 backdated to T0, then in-place update() to v2 (created_at survives)."""
    now = datetime.now(timezone.utc)
    t0 = (now - timedelta(days=T0_DAYS_AGO)).isoformat()
    ids = {}
    for pair in PAIRS:
        result = memory.add(pair["v1"], user_id=USER_ID, infer=False)
        mem_id = result["results"][0]["id"]
        ids[pair["v1"]] = mem_id
        payload = dict(memory.vector_store.get(vector_id=mem_id).payload)
        payload["created_at"] = t0
        payload["updated_at"] = t0
        memory.vector_store.update(vector_id=mem_id, payload=payload)
        # the in-place update under test — production path, content becomes v2
        memory.update(mem_id, data=pair["v2"])
        stored = memory.vector_store.get(vector_id=mem_id).payload
        assert stored.get("created_at") == t0, (
            "premissa quebrou: update() não preservou created_at — o leak "
            "mecânico mudou; revisar o cenário")
    return ids


def cleanup(collection: str) -> bool:
    ok = True
    for cname in (collection, collection + "_entities"):
        try:
            requests.delete(f"{QDRANT_URL}/collections/{cname}", timeout=15)
            still = requests.get(f"{QDRANT_URL}/collections/{cname}", timeout=10)
            if still.status_code == 200 and still.json().get("result"):
                ok = False
                print(f"  !! cleanup FALHOU: {cname} ainda existe")
        except Exception as e:
            print(f"  !! cleanup indeterminado p/ {cname}: {e}")
            ok = False
    print(f"cleanup verificado: {'OK' if ok else 'FALHOU'} ({collection}[+_entities])")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="deepmem0_update_versioning")
    parser.add_argument("--expect-fixed", action="store_true",
                        help="modo gate: exige a correção (exit 1 com o leak presente)")
    ARGS = parser.parse_args()

    print(f"collection={ARGS.collection} embed={EMBED_MODEL}")
    memory = build_memory()
    verdicts = []
    try:
        ids = seed_and_update(memory)
        anchor = (datetime.now(timezone.utc)
                  - timedelta(days=ANCHOR_DAYS_AGO)).date().isoformat()
        print(f"seeded {len(ids)} facts; as_of anchor = {anchor} (v1 era a verdade)")

        for pair in PAIRS:
            mem_id = ids[pair["v1"]]
            try:
                results = memory.search(pair["query"], user_id=USER_ID,
                                        top_k=5, as_of=anchor)["results"]
            except Exception as e:
                verdicts.append(("UNEXPECTED", pair["query"], f"search error: {e}"))
                continue
            hit = next((r for r in results if r.get("id") == mem_id), None)
            if hit is None:
                verdicts.append(("UNEXPECTED", pair["query"],
                                 "ponto ausente dos resultados ancorados "
                                 "(as_of deveria incluí-lo: created_at=T0 < anchor)"))
                continue
            text = (hit.get("memory") or "").lower()
            if pair["v2_marker"] in text:
                verdicts.append(("LEAK", pair["query"],
                                 f"âncora {anchor} vê conteúdo v2 ({pair['v2_marker']!r})"))
            elif pair["v1_marker"] in text:
                verdicts.append(("FIXED", pair["query"],
                                 f"âncora vê v1 ({pair['v1_marker']!r}) — versioning OK"))
            else:
                verdicts.append(("UNEXPECTED", pair["query"],
                                 f"conteúdo não casa com v1 nem v2: {text[:80]!r}"))
    finally:
        cleanup_ok = cleanup(ARGS.collection)

    for verdict, query, detail in verdicts:
        print(f"  [{verdict:10s}] {query[:55]:55s} {detail}")

    kinds = {v for v, _, _ in verdicts}
    n = len(verdicts)
    if not cleanup_ok:
        print("RESULT: UNEXPECTED (cleanup falhou)")
        sys.exit(1)
    if kinds == {"LEAK"}:
        if ARGS.expect_fixed:
            print(f"RESULT: FAIL — leak presente em {n}/{n} (modo --expect-fixed)")
            sys.exit(1)
        print(f"RESULT: XFAIL — leak conhecido reproduzido em {n}/{n} "
              "(limitação documentada; critério de aceite do update versioning)")
        sys.exit(0)
    if kinds == {"FIXED"}:
        print(f"RESULT: PASS — correção verdadeira em {n}/{n}. "
              "Promova este eval a gate duro (--expect-fixed no CI/gate) e "
              "atualize docs/roadmap.md.")
        sys.exit(0)
    print(f"RESULT: UNEXPECTED — vereditos mistos/anômalos: "
          f"{[(v, q[:30]) for v, q, _ in verdicts]}")
    sys.exit(1)
