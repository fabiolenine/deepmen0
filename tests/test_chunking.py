"""Page-aware chunker: page-range provenance + budget guarantees."""

import pytest

from mem0.utils.chunking import Chunk, chunk_pages


def test_small_pages_merge_and_carry_page_range():
    pages = [(1, "alpha."), (2, "beta."), (3, "gamma.")]
    chunks = chunk_pages(pages, chunk_chars=1000, overlap=0)
    assert len(chunks) == 1
    c = chunks[0]
    assert isinstance(c, Chunk)
    assert (c.page_start, c.page_end) == (1, 3)
    assert "alpha" in c.text and "gamma" in c.text


def test_oversized_page_splits_within_budget():
    big = "sentence. " * 500  # ~5000 chars, one page
    chunks = chunk_pages([(7, big)], chunk_chars=800, overlap=0)
    assert len(chunks) > 1
    assert all(len(c.text) <= 800 for c in chunks)
    assert all((c.page_start, c.page_end) == (7, 7) for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_blank_pages_ignored():
    assert chunk_pages([(1, "   "), (2, "")]) == []


def test_non_positive_budget_rejected():
    with pytest.raises(ValueError):
        chunk_pages([(1, "x")], chunk_chars=0)
