#!/usr/bin/env python3
"""Backfill BM25 sparse vectors on legacy points of a mem0 Qdrant collection.

Points inserted before hybrid search existed carry only the dense vector; the
`bm25` named sparse vector is missing, so keyword search never finds them. This
tool encodes each point's text and ADDS the sparse vector via `update_vectors`
— never `upsert`, which would replace the whole vector set and drop the dense
vector.

Tokenization parity: the text is normalized exactly like the query side of a
language-aware pipeline (lowercase + `_`/`-` -> space) and encoded with the
same fastembed BM25 model/language. If your collection was built with a
different prep, re-index with tools/migrate_embedder.py instead.

Idempotent: points that already have a `bm25` vector are skipped.

Environment:
  QDRANT_URL (http://localhost:6333)  BM25_LANGUAGE (portuguese)

Usage:
  python tools/backfill_sparse.py --collection my_collection [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import re
import time

from qdrant_client import QdrantClient
from qdrant_client.models import PointVectors, SparseVector

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
BM25_LANGUAGE = os.environ.get("BM25_LANGUAGE", "portuguese")

_SEP = re.compile(r"[_\-]+")


def bm25_normalize(text: str) -> str:
    return _SEP.sub(" ", (text or "").lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    from fastembed import SparseTextEmbedding

    enc = SparseTextEmbedding(model_name="Qdrant/bm25", language=BM25_LANGUAGE)
    client = QdrantClient(url=QDRANT_URL, timeout=60)

    info = client.get_collection(args.collection)
    if "bm25" not in (info.config.params.sparse_vectors or {}):
        raise SystemExit(f"collection '{args.collection}' has no 'bm25' sparse slot — aborting "
                         "(create it or re-index with tools/migrate_embedder.py)")

    seen = skipped = written = failed = 0
    t0 = time.time()
    offset, pending = None, []

    def flush():
        nonlocal written, failed, pending
        if not pending:
            return
        if args.dry_run:
            written += len(pending)
            pending = []
            return
        try:
            client.update_vectors(collection_name=args.collection, points=pending)
            written += len(pending)
        except Exception as e:
            failed += len(pending)
            print(f"[UPDATEFAIL] {len(pending)} pts :: {e}", flush=True)
        pending = []

    while True:
        points, offset = client.scroll(collection_name=args.collection, limit=128, offset=offset,
                                       with_payload=True, with_vectors=True)
        for p in points:
            seen += 1
            if isinstance(p.vector, dict) and "bm25" in p.vector:
                skipped += 1
                continue
            text = (p.payload or {}).get("data") or ""
            if not text:
                skipped += 1
                continue
            try:
                sp = next(iter(enc.embed([bm25_normalize(text)])))
                sv = SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())
            except Exception as e:
                failed += 1
                print(f"[ENCFAIL] {p.id} :: {e}", flush=True)
                continue
            pending.append(PointVectors(id=p.id, vector={"bm25": sv}))
            if len(pending) >= args.batch:
                flush()
        if offset is None or not points:
            break

    flush()
    print(f"DONE seen={seen} written={written} skipped={skipped} failed={failed} "
          f"in {time.time()-t0:.0f}s (dry_run={args.dry_run})")


if __name__ == "__main__":
    main()
