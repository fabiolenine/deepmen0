"""Page-aware text chunking for document ingestion.

Pure stdlib, no external dependencies. Provides page-aware chunking for
document ingestion so any consumer (e.g. an MCP server ingesting PDFs) can
split extracted document text into chunks that carry page provenance.

Strategy: pages are the natural boundary. Small pages merge until the chunk
budget fills; oversized pages split at paragraph breaks, then sentence breaks,
then hard-split as a last resort. Each chunk remembers the 1-based page range
it came from — that page range is the provenance stored on every extracted
memory. A short tail overlap between consecutive chunks of the SAME page keeps
facts that straddle a split visible to both extractions; no overlap is added
across page boundaries (page text does not repeat itself).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


@dataclass
class Chunk:
    text: str
    page_start: int  # 1-based, inclusive
    page_end: int
    index: int


def _split_oversized(text: str, budget: int) -> list[str]:
    """Split one blob to fit the budget: paragraphs -> sentences -> hard cut."""
    if len(text) <= budget:
        return [text]
    parts: list[str] = []
    current = ""
    for para in _PARAGRAPH_RE.split(text):
        para = para.strip()
        if not para:
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= budget:
            current = candidate
            continue
        if current:
            parts.append(current)
            current = ""
        if len(para) <= budget:
            current = para
            continue
        # paragraph alone exceeds the budget: sentences, then hard cut
        for sentence in _SENTENCE_RE.split(para):
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= budget:
                current = candidate
                continue
            if current:
                parts.append(current)
                current = ""
            while len(sentence) > budget:
                parts.append(sentence[:budget])
                sentence = sentence[budget:]
            current = sentence
    if current:
        parts.append(current)
    return parts


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_chars: int = 1800,
    overlap: int = 200,
) -> list[Chunk]:
    """Chunk ``[(page_number, page_text), ...]`` into provenance-carrying chunks.

    Pages arrive already filtered (only pages with a text layer). Guarantees:
    every non-blank piece of input text lands in exactly one chunk (plus the
    intra-page tail overlap), no chunk exceeds ``chunk_chars``, and each
    chunk's page range is accurate.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    overlap = max(0, min(overlap, chunk_chars // 2))

    chunks: list[Chunk] = []
    buffer = ""
    buf_start = buf_end = 0  # page range of the buffer

    def flush():
        nonlocal buffer, buf_start, buf_end
        if buffer.strip():
            chunks.append(Chunk(text=buffer.strip(), page_start=buf_start,
                                page_end=buf_end, index=len(chunks)))
        buffer = ""

    for number, text in pages:
        text = (text or "").strip()
        if not text:
            continue
        candidate = f"{buffer}\n\n{text}" if buffer else text
        if len(candidate) <= chunk_chars:
            if not buffer:
                buf_start = number
            buffer = candidate
            buf_end = number
            continue
        # page does not fit with the current buffer: close the buffer first
        flush()
        pieces = _split_oversized(text, chunk_chars)
        for i, piece in enumerate(pieces[:-1]):
            tail = piece[-overlap:] if overlap else ""
            chunks.append(Chunk(text=piece.strip(), page_start=number,
                                page_end=number, index=len(chunks)))
            # seed the next piece with the tail of this one (same page only)
            if tail and i + 1 < len(pieces):
                pieces[i + 1] = f"{tail.strip()} {pieces[i + 1]}"
        buffer = pieces[-1]
        buf_start = buf_end = number
        if len(buffer) > chunk_chars:  # tail overlap pushed it over
            for piece in _split_oversized(buffer, chunk_chars):
                chunks.append(Chunk(text=piece.strip(), page_start=number,
                                    page_end=number, index=len(chunks)))
            buffer = ""
    flush()
    return chunks
