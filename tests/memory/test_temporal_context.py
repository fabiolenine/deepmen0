"""Deterministic tests for temporal_context ("conversation" | "document").

No LLM: we STUB generate_response and capture the exact system prompt it receives
and the payload handed to vector_store.insert. This proves the CONTRACT
(document mode disables Observation-Date resolution; conversation mode is
byte-identical to before; a valid event_date survives to the stored payload)
without depending on a 9B model's obedience — that is measured separately by
scripts/eval_doc_extraction.py. Uses unittest.mock only (no pytest-mock).
"""
import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from mem0.configs.prompts import (
    ADDITIVE_EXTRACTION_PROMPT,
    DOCUMENT_TEMPORAL_OVERRIDE,
    build_temporality_suffix,
)
from mem0.memory.main import Memory


@contextmanager
def _mocked_memory(llm_response):
    """A Memory with factories mocked; generate_response returns llm_response.
    Yields (memory, captured) where captured['system'] is the extraction system
    prompt and captured['insert'] holds vector_store.insert kwargs."""
    embedder = MagicMock()
    embedder.return_value.embed.return_value = [0.1, 0.2, 0.3]
    vstore = MagicMock()
    vstore.return_value.search.return_value = []
    llm = MagicMock()
    with patch("mem0.utils.factory.EmbedderFactory.create", embedder), \
         patch("mem0.utils.factory.VectorStoreFactory.create",
               side_effect=[vstore.return_value, MagicMock()]), \
         patch("mem0.utils.factory.LlmFactory.create", llm), \
         patch("mem0.memory.storage.SQLiteManager", MagicMock()):
        mem = Memory()
        mem.custom_instructions = None
        mem.db.get_last_messages = MagicMock(return_value=[])
        mem.db.save_messages = MagicMock()
        # the persist path uses embed_batch (not embed); return one vector per text
        mem.embedding_model.embed_batch = MagicMock(
            side_effect=lambda texts, *a, **k: [[0.1, 0.2, 0.3] for _ in texts])
        captured = {}

        def _gen(messages, **kw):
            captured["system"] = messages[0]["content"]
            return llm_response
        mem.llm.generate_response = MagicMock(side_effect=_gen)

        def _insert(vectors=None, ids=None, payloads=None, **kw):
            captured["insert"] = {"payloads": payloads}
        mem.vector_store.insert = MagicMock(side_effect=_insert)
        yield mem, captured


_MSG = [{"role": "user", "content": "The contract Sorocaba expires 17/10, R$ 812."}]


def test_document_context_appends_override():
    with _mocked_memory("{}") as (mem, cap):
        mem._add_to_vector_store(_MSG, {}, {}, True, temporal_context="document")
    assert "DOCUMENT MODE — TEMPORAL OVERRIDE" in cap["system"]
    assert cap["system"].endswith(DOCUMENT_TEMPORAL_OVERRIDE)


def test_conversation_context_has_no_override():
    with _mocked_memory("{}") as (mem, cap):
        mem._add_to_vector_store(_MSG, {}, {}, True, temporal_context="conversation")
    assert "TEMPORAL OVERRIDE" not in cap["system"]


def test_default_context_is_conversation():
    with _mocked_memory("{}") as (mem, cap):
        mem._add_to_vector_store(_MSG, {}, {}, True)  # no temporal_context
    assert "TEMPORAL OVERRIDE" not in cap["system"]


def test_conversation_system_prompt_byte_identical_to_pre_change():
    """Regression: the conversational extraction prompt must be exactly what it
    was before temporal_context existed (base + temporality suffix, no override)."""
    with _mocked_memory("{}") as (mem, cap):
        mem._add_to_vector_store(_MSG, {}, {}, True, temporal_context="conversation")
    expected = ADDITIVE_EXTRACTION_PROMPT + build_temporality_suffix(include_event_date=True)
    assert cap["system"] == expected


def test_event_date_survives_to_stored_payload():
    """The event_date emitted by extraction must reach the stored payload
    (the persistence chain the doc path found empty at 0/185). Exercises the
    real event_date handling; temporality is forced on so the block runs."""
    from types import SimpleNamespace
    resp = json.dumps({"memory": [
        {"id": "0", "text": "O contrato Sorocaba vence em setembro de 2027",
         "attributed_to": "document", "event_date": "2027-09-01"}
    ]})
    temp = SimpleNamespace(extract_event_date=True, superseded_penalty=0.2, enabled=True)
    with _mocked_memory(resp) as (mem, cap):
        with patch("mem0.memory.main._temporality_config", return_value=temp):
            mem._add_to_vector_store(_MSG, {}, {}, True, temporal_context="document")
    payloads = (cap.get("insert") or {}).get("payloads") or []
    assert payloads, "vector_store.insert was not called with payloads"
    assert any(p.get("event_date") == "2027-09-01" for p in payloads), \
        f"event_date lost before storage; payloads={payloads}"
