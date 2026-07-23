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


def test_document_mode_does_not_write_history_sync():
    """Write-gate: document mode must NOT call save_messages (the bleed source)."""
    with _mocked_memory("{}") as (mem, cap):
        mem._add_to_vector_store(_MSG, {}, {}, True, temporal_context="document")
    mem.db.save_messages.assert_not_called()


def test_conversation_mode_writes_history_sync():
    """Regression: conversation mode still saves to the message history."""
    with _mocked_memory("{}") as (mem, cap):
        mem._add_to_vector_store(_MSG, {}, {}, True, temporal_context="conversation")
    mem.db.save_messages.assert_called()


@contextmanager
def _mocked_async_memory(llm_response):
    """AsyncMemory with factories mocked (async path — the /critic-results gap)."""
    import asyncio
    from unittest.mock import AsyncMock
    from mem0.memory.main import AsyncMemory
    embedder = MagicMock()
    vstore = MagicMock()
    vstore.return_value.search.return_value = []
    llm = MagicMock()
    with patch("mem0.utils.factory.EmbedderFactory.create", embedder), \
         patch("mem0.utils.factory.VectorStoreFactory.create",
               side_effect=[vstore.return_value, MagicMock()]), \
         patch("mem0.utils.factory.LlmFactory.create", llm), \
         patch("mem0.memory.storage.SQLiteManager", MagicMock()):
        mem = AsyncMemory()
        mem.custom_instructions = None
        mem.db.get_last_messages = MagicMock(return_value=[])
        mem.db.save_messages = MagicMock()
        cap = {}

        def _gen(messages, **kw):
            cap["system"] = messages[0]["content"]
            return llm_response
        mem.llm.generate_response = MagicMock(side_effect=_gen)
        yield mem, cap


def test_document_mode_async_override_and_no_write():
    """Async path: override in system prompt AND save_messages gated (mirror of sync)."""
    import asyncio
    with _mocked_async_memory("{}") as (mem, cap):
        asyncio.run(mem._add_to_vector_store(_MSG, {}, {}, True, temporal_context="document"))
    assert "DOCUMENT MODE — TEMPORAL OVERRIDE" in cap["system"]
    mem.db.save_messages.assert_not_called()


def test_conversation_mode_async_writes_history():
    import asyncio
    with _mocked_async_memory("{}") as (mem, cap):
        asyncio.run(mem._add_to_vector_store(_MSG, {}, {}, True, temporal_context="conversation"))
    assert "TEMPORAL OVERRIDE" not in cap["system"]
    mem.db.save_messages.assert_called()


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


def test_invalid_temporal_context_fails_closed():
    """Typo não pode degradar silenciosamente p/ conversation (fail-open era o
    risco: o override de datas sumiria sem sinal)."""
    import pytest as _pytest
    with _mocked_memory("{}") as (mem, cap):
        for bad in ("Document", "doc", "", None, 42):
            with _pytest.raises((ValueError, TypeError)):
                mem.add(_MSG, user_id="u", temporal_context=bad)


def test_infer_event_date_from_text_table():
    """Post-parser determinístico (fase 3 do event_date): só data COMPLETA e única."""
    from mem0.utils.temporality import infer_event_date_from_text as f
    assert f("check-in em 18 de outubro de 2023") == "2023-10-18"
    assert f("reserva confirmada em 15/07/2023") == "2023-07-15"
    assert f("emitido em 15/07/23") == "2023-07-15"          # 2 dígitos = 20YY
    assert f("meeting on October 5, 2024") == "2024-10-05"   # EN
    assert f("prazo 2027-09-01 registrado") == "2027-09-01"  # ISO no texto
    assert f("retirada em 17 out às 22:00") is None          # SEM ano -> nunca inferir
    assert f("de 18 de outubro de 2023 a 20 de outubro de 2023 e 15/07/2023") is None  # múltiplas -> ambíguo
    assert f("check-in 18 de outubro de 2023 confirmado em 18/10/2023") == "2023-10-18"  # MESMA data 2 formas
    assert f("31/02/2023 inválida") is None                  # data impossível
    assert f("") is None and f(None) is None


def test_event_date_inferred_from_text_when_llm_omits_field():
    """O caso REAL do 0/185: LLM põe a data no texto, omite o campo. Em modo
    document com temporalidade ligada, o post-parser preenche."""
    from types import SimpleNamespace
    resp = json.dumps({"memory": [
        {"id": "0", "text": "A reserva do Hotel Qintara foi confirmada em 15/07/2023",
         "attributed_to": "document"}  # SEM event_date — como o qwen faz
    ]})
    temp = SimpleNamespace(extract_event_date=True, superseded_penalty=0.2, enabled=True)
    with _mocked_memory(resp) as (mem, cap):
        with patch("mem0.memory.main._temporality_config", return_value=temp):
            mem._add_to_vector_store(_MSG, {}, {}, True, temporal_context="document")
    payloads = (cap.get("insert") or {}).get("payloads") or []
    assert any(p.get("event_date") == "2023-07-15" for p in payloads), \
        f"post-parser não preencheu event_date; payloads={payloads}"


def test_event_date_not_inferred_in_conversation_mode():
    """Conservador: em conversa o fallback NÃO roda (resolução relativa é papel
    do LLM com Observation Date; não adivinhar por trás dele)."""
    from types import SimpleNamespace
    resp = json.dumps({"memory": [
        {"id": "0", "text": "Reunião confirmada em 15/07/2023", "attributed_to": "user"}
    ]})
    temp = SimpleNamespace(extract_event_date=True, superseded_penalty=0.2, enabled=True)
    with _mocked_memory(resp) as (mem, cap):
        with patch("mem0.memory.main._temporality_config", return_value=temp):
            mem._add_to_vector_store(_MSG, {}, {}, True, temporal_context="conversation")
    payloads = (cap.get("insert") or {}).get("payloads") or []
    assert payloads and all(not p.get("event_date") for p in payloads)


def test_event_date_text_wins_over_contradictory_llm_date():
    """Cross-validação (parecer): LLM emite data VÁLIDA-mas-ERRADA (ex.: ano
    corrente); em modo documento a data ESCRITA no texto vence."""
    from types import SimpleNamespace
    resp = json.dumps({"memory": [
        {"id": "0", "text": "A reserva foi confirmada em 15/07/2023",
         "attributed_to": "document", "event_date": "2026-07-15"}  # ERRADA
    ]})
    temp = SimpleNamespace(extract_event_date=True, superseded_penalty=0.2, enabled=True)
    with _mocked_memory(resp) as (mem, cap):
        with patch("mem0.memory.main._temporality_config", return_value=temp):
            mem._add_to_vector_store(_MSG, {}, {}, True, temporal_context="document")
    payloads = (cap.get("insert") or {}).get("payloads") or []
    assert any(p.get("event_date") == "2023-07-15" for p in payloads), \
        f"data do texto não venceu a do LLM; payloads={payloads}"


def test_event_date_llm_survives_when_text_ambiguous():
    """Com o texto AMBÍGUO (duas datas), a data do LLM fica (não há verdade única
    no texto para cross-validar)."""
    from types import SimpleNamespace
    resp = json.dumps({"memory": [
        {"id": "0", "text": "Check-in 18 de outubro de 2023 e check-out 20 de outubro de 2023",
         "attributed_to": "document", "event_date": "2023-10-18"}
    ]})
    temp = SimpleNamespace(extract_event_date=True, superseded_penalty=0.2, enabled=True)
    with _mocked_memory(resp) as (mem, cap):
        with patch("mem0.memory.main._temporality_config", return_value=temp):
            mem._add_to_vector_store(_MSG, {}, {}, True, temporal_context="document")
    payloads = (cap.get("insert") or {}).get("payloads") or []
    assert any(p.get("event_date") == "2023-10-18" for p in payloads)


def test_document_mode_does_not_read_history():
    """READ-side do bleed (gap do parecer): um canário no histórico CONVERSACIONAL
    não pode entrar no prompt de um add em modo document (get_last_messages
    desligado, não só save)."""
    CANARY = "CONVCANARY-8Q2Z"
    with _mocked_memory("{}") as (mem, cap):
        mem.db.get_last_messages = MagicMock(
            return_value=[{"role": "user", "content": f"segredo {CANARY}"}])
        captured_user = {}
        orig = mem.llm.generate_response.side_effect
        def _gen(messages, **kw):
            captured_user["user"] = next(
                (m["content"] for m in messages if m.get("role") == "user"), "")
            return orig(messages, **kw)
        mem.llm.generate_response = MagicMock(side_effect=_gen)
        mem._add_to_vector_store(_MSG, {}, {}, True, temporal_context="document")
        assert CANARY not in captured_user["user"], "doc-mode LEU o histórico conversacional!"
        mem.db.get_last_messages.assert_not_called()
        # controle: em conversation o histórico ENTRA
        mem._add_to_vector_store(_MSG, {}, {}, True, temporal_context="conversation")
        assert CANARY in captured_user["user"], "conversation deixou de ler o histórico (regressão)"
