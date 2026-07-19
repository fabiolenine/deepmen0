#!/usr/bin/env python3
"""Scope-aware retrieval — executable spec (scoping ontology v1, routing step).

Like eval_update_versioning.py, this eval is written BEFORE the feature: today
scope routing does not exist, so routing scenarios report NOT_IMPLEMENTED
(XFAIL, exit 0). Once the routing step ships, ``--expect-implemented`` turns
this file into its hard gate. Invariants that must hold ALREADY (passive-field
era) are asserted unconditionally.

Contract under test (ontology v1, "scope first, similarity later"):
  - explicit ``memory_scope_filter`` selects allowed scopes (authoritative;
    never widens security filters); legacy ``memory_scope=null`` memories stay
    reachable during the transition (LEGACY_NEUTRAL) — a null memory is NEVER
    excluded by scope routing;
  - inferred routing for a decisive personal query EXCLUDES system_meta and
    eval_meta from the candidate set (score penalties are not enough — the
    fork's penalties act after the threshold and can only demote);
  - eval_meta is invisible to non-eval queries;
  - composite queries fan out into LANES with a minimum candidate budget per
    lane before the global rerank (union without diversity guarantees would
    let one scope dominate again).

Scenario cases declare allowed_ids / forbidden_ids / per-lane minimums — the
retrieval-compatibility layer of the synthetic eval package.

Verdicts: NOT_IMPLEMENTED (routing API absent; passive invariants hold) exit 0
| IMPLEMENTED (all scenario assertions pass) exit 0 + promote instruction |
UNEXPECTED (invariant broken, or routing present but violating a scenario)
exit 1. Throwaway collection, verified cleanup, no LLM (infer=False).

Environment: QDRANT_URL, OLLAMA_URL, EMBED_MODEL (bge-m3), EMBED_DIMS (1024).
"""
from __future__ import annotations

import argparse
import os
import sys

import requests

os.environ.setdefault("MEM0_TELEMETRY", "False")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
EMBED_DIMS = int(os.environ.get("EMBED_DIMS", "1024"))
USER_ID = "scoping_demo"

# synthetic labeled corpus: (key, text, memory_scope | None=legacy)
CORPUS = [
    ("user_langs", "Alice fala português nativo e espanhol intermediário.", "user_fact"),
    ("user_role", "Alice foi engenheira de dados na TransporteCo entre 2020 e 2024.", "user_fact"),
    ("sys_policy", "PREFERÊNCIA DO USUÁRIO: tratar o sistema de memória como armazenamento padrão de longo prazo para ler e gravar.", "system_meta"),
    ("sys_rank", "O sistema de memória rebaixa fatos supersedidos na fusão da busca.", "system_meta"),
    ("eval_score", "O golden set do sistema de memória está em 30 de 35 no hit@1.", "eval_meta"),
    ("proj_db", "O projeto Atlas usa um banco vetorial para embeddings de candles.", "project_meta"),
    ("proj_role", "No projeto Atlas, a Alice desenhou a camada de ingestão de ticks.", "project_meta"),
    ("legacy_fact", "O robô de cozinha da casa usa firmware versão 7.", None),  # legado sem scope
]

# routing scenarios (retrieval-compatibility layer): each declares the future
# API call plus allowed/forbidden ids and per-lane minimums.
SCENARIOS = [
    {
        "name": "explicit_filter_user_fact",
        "query": "preferências e fatos sobre a Alice",
        "kwargs": {"memory_scope_filter": ["user_fact"]},
        "forbidden": ["sys_policy", "sys_rank", "eval_score", "proj_db", "proj_role"],
        "allowed_extra": ["legacy_fact"],  # LEGACY_NEUTRAL: null nunca é excluído
        "expect_any": ["user_langs", "user_role"],
    },
    {
        "name": "personal_decisive_excludes_system",
        "query": "quais idiomas a Alice fala e em que nível?",
        "kwargs": {"query_scope_hint": "personal"},
        "forbidden": ["sys_policy", "sys_rank", "eval_score"],
        "expect_top1": "user_langs",
    },
    {
        "name": "eval_meta_invisible_outside_eval_intent",
        "query": "qual banco o projeto Atlas usa?",
        "kwargs": {"query_scope_hint": "project"},
        "forbidden": ["eval_score"],
        "expect_top1": "proj_db",
    },
    {
        "name": "composite_lanes_min_budget",
        "query": "qual foi o papel da Alice no projeto Atlas?",
        "kwargs": {"query_scope_hint": ["project", "personal"]},
        "forbidden": ["eval_score"],
        # lanes: pelo menos 1 candidato de CADA lane no top-k final
        "lane_min": {"project_meta": 1, "user_fact": 1},
    },
]


def build_memory():
    from mem0 import Memory

    return Memory.from_config({
        "language": "pt",
        "llm": {"provider": "ollama",
                "config": {"model": os.environ.get("LLM_MODEL", "llama3.1"),
                           "ollama_base_url": OLLAMA_URL}},
        "vector_store": {"provider": "qdrant",
                         "config": {"collection_name": ARGS.collection,
                                    "url": QDRANT_URL,
                                    "embedding_model_dims": EMBED_DIMS}},
        "embedder": {"provider": "ollama",
                     "config": {"model": EMBED_MODEL, "embedding_dims": EMBED_DIMS,
                                "ollama_base_url": OLLAMA_URL}},
    })


def cleanup(collection: str) -> bool:
    ok = True
    for cname in (collection, collection + "_entities"):
        try:
            requests.delete(f"{QDRANT_URL}/collections/{cname}", timeout=15)
            still = requests.get(f"{QDRANT_URL}/collections/{cname}", timeout=10)
            if still.status_code == 200 and still.json().get("result"):
                ok = False
        except Exception:
            ok = False
    print(f"cleanup verificado: {'OK' if ok else 'FALHOU'}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="deepmem0_scoping")
    parser.add_argument("--expect-implemented", action="store_true",
                        help="hard-gate mode for the routing step")
    ARGS = parser.parse_args()

    memory = build_memory()
    ids: dict[str, str] = {}
    scope_of: dict[str, str | None] = {}
    unexpected: list[str] = []
    verdicts: list[str] = []
    try:
        for key, text, scope in CORPUS:
            md = {"memory_scope": scope, "memory_scope_version": 1,
                  "memory_scope_source": "manual"} if scope else None
            r = memory.add(text, user_id=USER_ID, infer=False,
                           **({"metadata": md} if md else {}))
            mem_id = r["results"][0]["id"]
            ids[key] = mem_id
            scope_of[mem_id] = scope
        print(f"seeded {len(ids)} memories (incl. 1 legacy null)")

        # --- invariante que vale JÁ: legado null permanece alcançável --------
        res = memory.search("firmware do robô de cozinha", user_id=USER_ID, top_k=3)["results"]
        if ids["legacy_fact"] not in [x["id"] for x in res]:
            unexpected.append("legado null inalcançável já na era passiva")
        else:
            print("  [PASS          ] invariante: memória legada (null) alcançável hoje")

        # --- sonda canário: o routing existe DE VERDADE? ---------------------
        # Memory.search engole kwargs desconhecidos em silêncio (detectado na
        # 1ª execução deste eval) — TypeError não serve de detecção. A sonda é
        # determinística: filtro explícito só-eval_meta numa query não-eval;
        # se voltarem ids de OUTROS scopes (não-null), o filtro foi ignorado.
        # NOTA p/ o implementador do routing: kwargs desconhecidos ignorados em
        # silêncio são um risco de API (ilusão de filtro) — rejeitar ou
        # implementar, nunca engolir.
        probe = memory.search("banco vetorial do projeto Atlas", user_id=USER_ID,
                              top_k=5, memory_scope_filter=["eval_meta"])["results"]
        probe_scopes = {scope_of.get(x["id"]) for x in probe}
        routing_present = probe_scopes <= {"eval_meta", None}
        if not routing_present:
            print(f"  sonda canário: memory_scope_filter IGNORADO (voltaram scopes "
                  f"{sorted(s for s in probe_scopes if s)}) — routing ausente")

        # --- cenários de routing (spec do passo de retrieval) ----------------
        for sc in SCENARIOS:
            if not routing_present:
                verdicts.append("NOT_IMPLEMENTED")
                print(f"  [NOT_IMPLEMENTED] {sc['name']} ({list(sc['kwargs'])})")
                continue
            res = memory.search(sc["query"], user_id=USER_ID, top_k=5,
                                **sc["kwargs"])["results"]
            got = [x["id"] for x in res]
            bad = [k for k in sc.get("forbidden", []) if ids[k] in got]
            miss_any = (sc.get("expect_any")
                        and not any(ids[k] in got for k in sc["expect_any"]))
            top1_bad = (sc.get("expect_top1") and (not got or got[0] != ids[sc["expect_top1"]]))
            lane_bad = []
            for lane, minimum in (sc.get("lane_min") or {}).items():
                have = sum(1 for g in got if scope_of.get(g) == lane)
                if have < minimum:
                    lane_bad.append(f"{lane}<{minimum}")
            if bad or miss_any or top1_bad or lane_bad:
                verdicts.append("UNEXPECTED")
                unexpected.append(f"{sc['name']}: forbidden_presentes={bad} "
                                  f"miss_any={bool(miss_any)} top1_bad={bool(top1_bad)} "
                                  f"lanes={lane_bad}")
                print(f"  [UNEXPECTED     ] {sc['name']}")
            else:
                verdicts.append("IMPLEMENTED_OK")
                print(f"  [IMPLEMENTED_OK ] {sc['name']}")
    finally:
        if not cleanup(ARGS.collection):
            unexpected.append("cleanup falhou")

    if unexpected:
        print(f"RESULT: UNEXPECTED — {unexpected}")
        sys.exit(1)
    if verdicts and all(v == "IMPLEMENTED_OK" for v in verdicts):
        print("RESULT: IMPLEMENTED — routing satisfaz os cenários declarados. "
              "Promote this eval to the routing step's hard gate (--expect-implemented).")
        sys.exit(0)
    if ARGS.expect_implemented:
        print("RESULT: FAIL — routing ausente/incompleto (--expect-implemented)")
        sys.exit(1)
    print("RESULT: NOT_IMPLEMENTED (XFAIL) — executable spec for the routing step; "
          "passive invariants hold.")
    sys.exit(0)
