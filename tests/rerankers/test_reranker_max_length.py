"""max_length flows from SentenceTransformerRerankerConfig into the CrossEncoder.

A lower max_length caps the tokens per query-document pair, sharply cutting CPU
cross-encoder latency on long documents. Default is None (model default), so
existing behavior is unchanged.
"""

from unittest.mock import MagicMock

import pytest

from mem0.configs.rerankers.sentence_transformer import SentenceTransformerRerankerConfig


@pytest.fixture
def mock_cross_encoder(monkeypatch):
    """Fake CrossEncoder so the reranker constructs without downloading a model."""
    import mem0.reranker.sentence_transformer_reranker as st

    fake_ce = MagicMock(name="CrossEncoder")
    monkeypatch.setattr(st, "CrossEncoder", fake_ce, raising=True)
    monkeypatch.setattr(st, "SENTENCE_TRANSFORMERS_AVAILABLE", True, raising=False)
    return st, fake_ce


def test_config_accepts_max_length_and_defaults_none():
    assert SentenceTransformerRerankerConfig(max_length=128).max_length == 128
    assert SentenceTransformerRerankerConfig().max_length is None


def test_max_length_passed_to_cross_encoder(mock_cross_encoder):
    st, fake_ce = mock_cross_encoder
    st.SentenceTransformerReranker(
        SentenceTransformerRerankerConfig(model="m", device="cpu", max_length=256)
    )
    _, kwargs = fake_ce.call_args
    assert kwargs.get("max_length") == 256


def test_default_max_length_is_none(mock_cross_encoder):
    st, fake_ce = mock_cross_encoder
    st.SentenceTransformerReranker(
        SentenceTransformerRerankerConfig(model="m", device="cpu")
    )
    _, kwargs = fake_ce.call_args
    assert kwargs.get("max_length") is None
