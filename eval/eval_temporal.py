#!/usr/bin/env python3
"""DeepMem0 v0.2 temporal scenario — the human-memory proof.

Seeds pairs of *equally-similar twin facts* (paraphrases of the same
underlying fact) into a throwaway collection, gives ONE twin of each pair a
lived reinforcement timeline (spread over the past month, exactly what T1/T2/T3
would have produced), and checks two things:

  A. NO REGRESSION — before any reinforcement exists, every memory carries the
     same fresh timeline, so dynamics ON and OFF must return the same top-1
     for every query.
  B. THE HUMAN-MEMORY EFFECT — after reinforcement, the reinforced twin must
     outrank its equally-similar unreinforced sibling.

The reinforcement timelines are written directly to the payload (identical in
shape to what `_reinforce_memory` produces) so a month of usage can be
simulated in seconds; the ranking math under test is untouched.

Requires a local Qdrant and an Ollama with the embedder model. Environment:
  QDRANT_URL (default http://localhost:6333), OLLAMA_URL, EMBED_MODEL (bge-m3),
  EMBED_DIMS (1024), MEM0_LANGUAGE (pt)

Usage:
  python eval/eval_temporal.py [--collection deepmem0_temporal] [--rerank]
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
USER_ID = "temporal_demo"

# Twin facts: same underlying fact, comparable lexical overlap with the query
# (this is what "equally similar" means — both twins carry the query's key
# terms and differ only in phrasing). The REINFORCED twin is always "hot".
PAIRS = [
    {
        "query": "como funciona o deploy do hermes_fx?",
        "cold": "O deploy do hermes_fx é feito por canary de 5% durante 24 horas antes do rollout completo.",
        "hot": "O deploy do hermes_fx usa canary de 5% por 24 horas e só então libera o rollout completo.",
    },
    {
        "query": "qual banco de dados o atlas_ingest usa?",
        "cold": "O atlas_ingest usa um banco de dados PostgreSQL particionado por dia para os eventos brutos.",
        "hot": "O banco de dados do atlas_ingest é um PostgreSQL com partições diárias para os eventos brutos.",
    },
    {
        "query": "politica de retries do boreal_app",
        "cold": "A política de retries do boreal_app é de três tentativas com backoff exponencial a partir de 2 segundos.",
        "hot": "No boreal_app a política de retries faz três tentativas, com backoff exponencial começando em 2 segundos.",
    },
    {
        "query": "quem aprova mudanças de schema no projeto Aurora?",
        "cold": "Mudanças de schema no Aurora precisam de aprovação do comitê de dados antes do merge.",
        "hot": "No Aurora, mudanças de schema exigem aprovação do comitê de dados antes de qualquer merge.",
    },
    {
        "query": "frequência dos backups do Orion",
        "cold": "Os backups completos do Orion rodam todo domingo às 03h, com incrementais diários.",
        "hot": "Os backups do Orion são completos aos domingos de madrugada e incrementais nos demais dias.",
    },
    {
        "query": "limite de memória dos workers do hermes_fx",
        "cold": "Cada worker do hermes_fx tem limite de 2 GiB de memória imposto pelo orquestrador.",
        "hot": "Os workers do hermes_fx têm limite de memória de 2 GiB definido pelo orquestrador.",
    },
]

# Design guard: when relevance is DECISIVE (one memory matches the query's
# exact terms, the other is a loose paraphrase), reinforcement must NOT
# overturn it — activation breaks near-ties, it does not defeat relevance.
GUARD = {
    "query": "qual é o timeout do healthcheck do boreal_app?",
    "cold": "O timeout do healthcheck do boreal_app é de 5 segundos.",
    "hot": "O boreal_app considera uma instância fora do ar após ficar sem resposta na sonda de saúde.",
}

DISTRACTORS = [
    "A Vetorial Labs faz all-hands toda primeira sexta-feira do mês.",
    "O time de plataforma mantém um canal de plantão com rotação semanal.",
    "A documentação interna fica num wiki com busca full-text.",
    "As chaves de API de staging expiram a cada 90 dias.",
]


def build_memory(rerank: bool, dynamics_enabled: bool = True):
    from mem0 import Memory

    config = {
        "language": LANGUAGE,
        # infer=False never calls the LLM, but Memory.__init__ instantiates it —
        # point it at the local Ollama so no cloud credentials are required.
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
                # production Qdrant now requires auth (locked 2026-07); pass the
                # key when present so this eval runs against the live instance.
                **({"api_key": os.environ["MEM0_QDRANT_API_KEY"]}
                   if os.environ.get("MEM0_QDRANT_API_KEY") else {}),
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
        "dynamics": {"enabled": dynamics_enabled},
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


def seed(memory) -> dict:
    """infer=False adds (no LLM); returns text -> memory_id."""
    ids = {}
    all_pairs = PAIRS + [GUARD]
    texts = [p["cold"] for p in all_pairs] + [p["hot"] for p in all_pairs] + DISTRACTORS
    for text in texts:
        result = memory.add(text, user_id=USER_ID, infer=False)
        ids[text] = result["results"][0]["id"]
    return ids


def backdate_and_reinforce(memory, ids) -> None:
    """Equalize both twins at 30 days old, then give the hot twin a lived
    timeline: reinforcements 7, 3 and 1 days ago (shape identical to what
    T1/T2/T3 write-backs produce)."""
    now = datetime.now(timezone.utc)
    born = (now - timedelta(days=30)).isoformat()

    def days_ago(d):
        return (now - timedelta(days=d)).isoformat()

    for pair in PAIRS + [GUARD]:
        for text, reinforced in ((pair["cold"], False), (pair["hot"], True)):
            mem_id = ids[text]
            point = memory.vector_store.get(vector_id=mem_id)
            payload = dict(point.payload)
            payload["created_at"] = born
            payload["updated_at"] = born
            if reinforced:
                payload["reinforced_at"] = [born, days_ago(7), days_ago(3), days_ago(1)]
                payload["access_count"] = 4
                payload["last_accessed"] = days_ago(1)
            else:
                payload["reinforced_at"] = [born]
                payload["access_count"] = 1
            memory.vector_store.update(vector_id=mem_id, payload=payload)


def top_ids(memory, query, k=5):
    results = memory.search(query, user_id=USER_ID, top_k=k)["results"]
    return [r["id"] for r in results]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="deepmem0_temporal")
    parser.add_argument("--rerank", action="store_true")
    ARGS = parser.parse_args()

    print(f"collection={ARGS.collection} rerank={ARGS.rerank} embed={EMBED_MODEL}")

    memory_on = build_memory(ARGS.rerank, dynamics_enabled=True)
    ids = seed(memory_on)
    print(f"seeded {len(ids)} memories ({len(PAIRS)} twin pairs + {len(DISTRACTORS)} distractors)")

    # --- A. No regression: fresh corpus, dynamics ON == OFF on every top-1 ---
    memory_off = build_memory(ARGS.rerank, dynamics_enabled=False)
    mismatches = []
    for pair in PAIRS:
        on = top_ids(memory_on, pair["query"], k=1)
        off = top_ids(memory_off, pair["query"], k=1)
        if on != off:
            mismatches.append(pair["query"])
    status = "OK" if not mismatches else f"FAIL {mismatches}"
    print(f"[A] no-regression (fresh corpus, on==off top-1): {status}")

    # --- B. Human-memory effect: reinforced twin outranks its sibling -------
    backdate_and_reinforce(memory_on, ids)
    wins, losses = 0, []
    for pair in PAIRS:
        ranked = top_ids(memory_on, pair["query"], k=5)
        hot_rank = ranked.index(ids[pair["hot"]]) if ids[pair["hot"]] in ranked else 99
        cold_rank = ranked.index(ids[pair["cold"]]) if ids[pair["cold"]] in ranked else 99
        if hot_rank < cold_rank:
            wins += 1
        else:
            losses.append((pair["query"], hot_rank, cold_rank))
    print(f"[B] reinforced twin wins: {wins}/{len(PAIRS)}")
    for query, hot_rank, cold_rank in losses:
        print(f"    lost: {query!r} (hot rank {hot_rank}, cold rank {cold_rank})")

    # --- C. Design guard: decisive relevance gap is NOT overturned ----------
    ranked = top_ids(memory_on, GUARD["query"], k=5)
    guard_ok = ranked and ranked[0] == ids[GUARD["cold"]]
    print(f"[C] decisive-gap guard (reinforced paraphrase must NOT beat exact match): {'OK' if guard_ok else 'FAIL ' + str(ranked[:2])}")

    # Sanity: with dynamics OFF the reinforced twin has no systematic edge.
    off_wins = 0
    for pair in PAIRS:
        ranked = top_ids(memory_off, pair["query"], k=5)
        hot_rank = ranked.index(ids[pair["hot"]]) if ids[pair["hot"]] in ranked else 99
        cold_rank = ranked.index(ids[pair["cold"]]) if ids[pair["cold"]] in ranked else 99
        if hot_rank < cold_rank:
            off_wins += 1
    print(f"[B-control] with dynamics OFF the reinforced twin wins {off_wins}/{len(PAIRS)} (similarity-only tie-breaks)")

    failed = bool(mismatches) or wins < len(PAIRS) or not guard_ok
    sys.exit(1 if failed else 0)
