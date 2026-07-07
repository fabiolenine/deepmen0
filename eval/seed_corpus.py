#!/usr/bin/env python3
"""Seed the synthetic evaluation corpus into a Qdrant collection.

Creates the collection with the same schema mem0 uses (unnamed dense vector,
cosine; named sparse vector `bm25` with IDF modifier) and upserts every memory
from corpus_synthetic.json with:
  - dense vector: Ollama embedding of the raw `data` text
  - bm25 sparse vector: fastembed BM25 (language-aware) over normalize(data),
    where normalize = lowercase + `_`/`-` -> space (MUST match the query-side
    prep of the pipeline under test, or token IDs diverge and BM25 scores zero)
  - payload: data, text_lemmatized (=normalized text), user_id, created_at,
    hash, plus the memory's metadata (importance/domain/memory_type/tags)

Point IDs are deterministic: uuid5(NAMESPACE_URL, "deepmem0:" + slug), so
re-running is idempotent and eval cases can resolve slugs to IDs independently.

Environment (all optional):
  QDRANT_URL   default http://localhost:6333
  OLLAMA_URL   default http://localhost:11434
  EMBED_MODEL  default bge-m3
  EMBED_DIMS   default 1024
  BM25_LANGUAGE default portuguese
  USER_ID      default demo

Usage:
  python eval/seed_corpus.py --collection deepmem0_demo [--recreate]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone

import requests

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
EMBED_DIMS = int(os.environ.get("EMBED_DIMS", "1024"))
BM25_LANGUAGE = os.environ.get("BM25_LANGUAGE", "portuguese")
USER_ID = os.environ.get("USER_ID", "demo")

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(HERE, "corpus_synthetic.json")

_SEP = re.compile(r"[_\-]+")


def slug_to_id(slug: str) -> str:
    """Deterministic point ID shared by seeder and eval harness."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"deepmem0:{slug}"))


def bm25_normalize(text: str) -> str:
    return _SEP.sub(" ", (text or "").lower())


def dense_embed_batch(texts: list[str]) -> list[list[float]]:
    r = requests.post(
        f"{OLLAMA_URL}/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=300
    )
    r.raise_for_status()
    embs = r.json().get("embeddings") or []
    if len(embs) != len(texts):
        raise RuntimeError(f"embedder returned {len(embs)} vectors for {len(texts)} texts")
    for e in embs:
        assert len(e) == EMBED_DIMS, f"dims={len(e)} != EMBED_DIMS={EMBED_DIMS}"
    return embs


def ensure_collection(collection: str, recreate: bool) -> None:
    if recreate:
        requests.delete(f"{QDRANT_URL}/collections/{collection}", timeout=30)
    r = requests.get(f"{QDRANT_URL}/collections/{collection}", timeout=10)
    if r.status_code == 200:
        return
    r = requests.put(
        f"{QDRANT_URL}/collections/{collection}",
        json={
            "vectors": {"size": EMBED_DIMS, "distance": "Cosine"},
            "sparse_vectors": {"bm25": {"modifier": "idf"}},
        },
        timeout=30,
    )
    r.raise_for_status()
    for field in ("user_id", "agent_id", "run_id"):
        requests.put(
            f"{QDRANT_URL}/collections/{collection}/index",
            json={"field_name": field, "field_schema": "keyword"},
            timeout=10,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", default=os.environ.get("COLLECTION", "deepmem0_demo"))
    ap.add_argument("--recreate", action="store_true", help="drop and recreate the collection first")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from fastembed import SparseTextEmbedding

    enc = SparseTextEmbedding(model_name="Qdrant/bm25", language=BM25_LANGUAGE)

    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)
    memories = corpus["memories"]

    ensure_collection(args.collection, args.recreate)
    now = datetime.now(timezone.utc).isoformat()

    written = 0
    for i in range(0, len(memories), args.batch):
        chunk = memories[i : i + args.batch]
        texts = [m["data"] for m in chunk]
        denses = dense_embed_batch(texts)
        norms = [bm25_normalize(t) for t in texts]
        sparses = list(enc.embed(norms))
        points = []
        for m, dense, norm, sp in zip(chunk, denses, norms, sparses):
            payload = {
                "data": m["data"],
                "text_lemmatized": norm,
                "user_id": USER_ID,
                "created_at": now,
                "updated_at": now,
                "hash": hashlib.md5(m["data"].encode()).hexdigest(),
                **(m.get("metadata") or {}),
            }
            points.append(
                {
                    "id": slug_to_id(m["slug"]),
                    "vector": {
                        "": dense,
                        "bm25": {"indices": sp.indices.tolist(), "values": sp.values.tolist()},
                    },
                    "payload": payload,
                }
            )
        r = requests.put(
            f"{QDRANT_URL}/collections/{args.collection}/points?wait=true",
            json={"points": points},
            timeout=120,
        )
        r.raise_for_status()
        written += len(points)
        print(f"  +{len(points)} (total {written}/{len(memories)})", flush=True)

    count = requests.post(
        f"{QDRANT_URL}/collections/{args.collection}/points/count", json={}, timeout=10
    ).json()["result"]["count"]
    print(f"DONE collection={args.collection} corpus={len(memories)} points_in_collection={count}")


if __name__ == "__main__":
    main()
