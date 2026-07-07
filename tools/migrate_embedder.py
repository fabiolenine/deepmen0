#!/usr/bin/env python3
"""Re-embed a mem0 Qdrant collection into a NEW collection (embedder/language cutover).

Changing the dense embedder (dimensions) or the BM25 language changes vector
dimensions / sparse token IDs — existing collections cannot be updated in place.
This tool migrates every point from a source collection into a freshly-created
destination collection (the source is kept untouched as rollback):

  - dense  = Ollama embedding of payload["data"] (raw text, same as mem0's add path)
  - sparse = fastembed BM25 (language-aware) over normalize(data), where
    normalize = lowercase + `_`/`-` -> space. Do NOT reuse a legacy
    `text_lemmatized` produced by an English lemmatizer — it silently drops
    snake_case tokens. The payload's text_lemmatized is rewritten accordingly.
  - payload preserved otherwise (classification, timestamps, hash, scopes).

Destination schema matches mem0's create_col: unnamed dense vector (cosine) +
named sparse vector `bm25` with IDF modifier + keyword indexes on scope fields.

Idempotent and resumable: points already present in the destination are skipped;
re-run to pick up a delta (e.g. memories added between migration and cutover).

Environment / args:
  QDRANT_URL (http://localhost:6333)  OLLAMA_URL (http://localhost:11434)
  EMBED_MODEL (bge-m3)  EMBED_DIMS (1024)  BM25_LANGUAGE (portuguese)

Usage:
  python tools/migrate_embedder.py --src old_collection --dst new_collection [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import requests
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, SparseVector, SparseVectorParams, VectorParams

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
EMBED_DIMS = int(os.environ.get("EMBED_DIMS", "1024"))
BM25_LANGUAGE = os.environ.get("BM25_LANGUAGE", "portuguese")

_SEP = re.compile(r"[_\-]+")


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


def ensure_dst(client: QdrantClient, src: str, dst: str) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if dst in existing:
        info = client.get_collection(dst)
        if info.config.params.vectors.size != EMBED_DIMS:
            raise SystemExit(f"destination '{dst}' exists with different dense size — aborting")
        if "bm25" not in (info.config.params.sparse_vectors or {}):
            raise SystemExit(f"destination '{dst}' exists without a 'bm25' sparse slot — aborting")
        return
    src_info = client.get_collection(src)
    on_disk = bool(getattr(src_info.config.params.vectors, "on_disk", None))
    client.create_collection(
        collection_name=dst,
        vectors_config=VectorParams(size=EMBED_DIMS, distance=models.Distance.COSINE, on_disk=on_disk),
        sparse_vectors_config={"bm25": SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field in ("user_id", "agent_id", "run_id", "actor_id"):
        try:
            client.create_payload_index(collection_name=dst, field_name=field, field_schema="keyword")
        except Exception:
            pass
    print(f"created '{dst}' (dense {EMBED_DIMS} cosine on_disk={on_disk} + bm25 idf)")


def existing_ids(client: QdrantClient, collection: str) -> set:
    ids, offset = set(), None
    while True:
        pts, offset = client.scroll(collection_name=collection, limit=512, offset=offset,
                                    with_payload=False, with_vectors=False)
        ids.update(str(p.id) for p in pts)
        if offset is None or not pts:
            break
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source collection (kept untouched)")
    ap.add_argument("--dst", required=True, help="destination collection (created if missing)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from fastembed import SparseTextEmbedding

    enc = SparseTextEmbedding(model_name="Qdrant/bm25", language=BM25_LANGUAGE)
    client = QdrantClient(url=QDRANT_URL, timeout=120)

    src_count = client.count(args.src).count
    done = set()
    if not args.dry_run:
        ensure_dst(client, args.src, args.dst)
        done = existing_ids(client, args.dst)
    print(f"src={args.src} ({src_count} pts)  dst={args.dst} (already migrated: {len(done)})  "
          f"embed={EMBED_MODEL}/{EMBED_DIMS}d  bm25={BM25_LANGUAGE}")

    seen = skipped = written = failed = 0
    t0 = time.time()
    offset, batch = None, []

    def flush():
        nonlocal written, failed, batch
        if not batch:
            return
        if args.dry_run:
            written += len(batch)
            batch = []
            return
        try:
            texts = [pl.get("data") or "" for _, pl in batch]
            denses = dense_embed_batch(texts)
            norms = [bm25_normalize(t) for t in texts]
            sparses = list(enc.embed(norms))
            points = []
            for (pid, pl), dense, norm, sp in zip(batch, denses, norms, sparses):
                payload = dict(pl)
                payload["text_lemmatized"] = norm
                sv = SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())
                points.append(PointStruct(id=pid, vector={"": dense, "bm25": sv}, payload=payload))
            client.upsert(collection_name=args.dst, points=points)
            written += len(points)
            print(f"  +{len(points)} (total {written})", flush=True)
        except Exception as e:
            failed += len(batch)
            print(f"[BATCHFAIL] {len(batch)} pts :: {e}", flush=True)
        batch = []

    while True:
        pts, offset = client.scroll(collection_name=args.src, limit=128, offset=offset,
                                    with_payload=True, with_vectors=False)
        for p in pts:
            seen += 1
            if str(p.id) in done:
                skipped += 1
                continue
            pl = p.payload or {}
            if not pl.get("data"):
                skipped += 1
                continue
            batch.append((p.id, pl))
            if len(batch) >= args.batch:
                flush()
        if offset is None or not pts:
            break

    flush()
    dst_count = client.count(args.dst).count if not args.dry_run else None
    print(f"DONE src={src_count} dst={dst_count} seen={seen} written={written} "
          f"skipped={skipped} failed={failed} in {time.time()-t0:.0f}s (dry_run={args.dry_run})")
    if not args.dry_run and dst_count != src_count:
        print("WARNING: counts diverge — re-run (idempotent) to pick up the remainder")


if __name__ == "__main__":
    main()
