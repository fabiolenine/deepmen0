"""DeepMem0 v0.6 event-date-aware ranking tests.

Two mechanisms: an explicit event-time window filter (event_from/event_to) and an
automatic query-anchor ranking signal (fusion boost + post-rerank tie-break). The
tie-break mirrors the ACT-R v0.2 discipline exactly: additive at fusion (pool
shaping), a BOUNDED tie-breaker post-rerank (never overturns a decisive reranker
gap). Pure units plus mocked-store search tests — no live infrastructure.
"""

import inspect
import os
from unittest.mock import Mock, patch

import pytest

from mem0.configs.base import (
    RERANK_TIE_BAND,
    MemoryConfig,
    MemoryDynamicsConfig,
    MemoryTemporalityConfig,
)
from mem0.memory.main import AsyncMemory, Memory, _apply_post_rerank_adjustments
from mem0.utils.scoring import score_and_rank
from mem0.utils.temporality import (
    _iter_full_date_matches,
    event_proximity,
    expand_event_window,
    infer_event_anchor_from_query,
    infer_event_date_from_text,
)


# --- reference implementation of the ORIGINAL infer_event_date_from_text, kept
# verbatim so the refactor onto _iter_full_date_matches is proven behavior-
# preserving (the eval_doc_extraction gate depends on this function). ----------
from mem0.utils.temporality import (
    _MONTHS,
    _ISO_TXT_RE,
    _NUM_DATE_RE,
    _NAME_DATE_RE,
    _NAME_DATE_MF_RE,
)
from datetime import datetime


def _original_infer(text):
    if not isinstance(text, str) or not text:
        return None
    found = set()
    for y, m, d in _ISO_TXT_RE.findall(text):
        found.add((int(y), int(m), int(d)))
    for d, m, y in _NUM_DATE_RE.findall(text):
        year = int(y) if len(y) == 4 else 2000 + int(y)
        found.add((year, int(m), int(d)))
    for d, mn, y in _NAME_DATE_RE.findall(text):
        mo = _MONTHS.get(mn.lower())
        if mo:
            found.add((int(y), mo, int(d)))
    for mn, d, y in _NAME_DATE_MF_RE.findall(text):
        mo = _MONTHS.get(mn.lower())
        if mo:
            found.add((int(y), mo, int(d)))
    valid = set()
    for y, m, d in found:
        try:
            datetime(year=y, month=m, day=d)
        except ValueError:
            continue
        valid.add((y, m, d))
    if len(valid) != 1:
        return None
    y, m, d = next(iter(valid))
    return f"{y:04d}-{m:02d}-{d:02d}"


class TestInferEventDateRegression:
    """The refactor MUST NOT change infer_event_date_from_text (shipped, gated)."""

    CORPUS = [
        "deploy em 17/10/2023 foi ok",
        "17 de outubro de 2023 aconteceu",
        "October 5, 2024 was the launch",
        "2023-10-17 e 2024-05-03 (duas datas distintas)",
        "reserva 3/5/85 antiga (20YY rule)",
        "1º de maio de 2024 feriado",
        "reunião marcada para 31/12/2025",
        "nenhuma data aqui",
        "",
        "em outubro de 2023 (month only)",
        "January 2025 report (month only)",
        "31/02/2023 data invalida",
        "2023-13-01 mes invalido",
        "duas vezes 10/10/2020 e 10/10/2020",  # same date twice -> one
        None,
        12345,
    ]

    @pytest.mark.parametrize("text", CORPUS)
    def test_byte_identical_to_original(self, text):
        assert infer_event_date_from_text(text) == _original_infer(text)


class TestEventProximity:
    W = ("2023-10-01", "2023-10-31")

    def test_inside_window_is_one(self):
        assert event_proximity(self.W, "2023-10-15", 30) == 1.0
        assert event_proximity(self.W, "2023-10-01", 30) == 1.0
        assert event_proximity(self.W, "2023-10-31", 30) == 1.0

    def test_linear_decay_outside(self):
        assert event_proximity(self.W, "2023-09-16", 30) == pytest.approx(0.5)
        assert event_proximity(self.W, "2023-11-15", 30) == pytest.approx(0.5)

    def test_beyond_window_is_zero(self):
        assert event_proximity(self.W, "2023-08-01", 30) == 0.0
        assert event_proximity(self.W, "2024-01-01", 30) == 0.0

    def test_missing_or_invalid_is_zero(self):
        for bad in (None, "", "garbage", "2023-13-40", 20231015, ["2023-10-15"]):
            assert event_proximity(self.W, bad, 30) == 0.0

    def test_time_suffix_normalized(self):
        assert event_proximity(self.W, "2023-10-15T10:30:00Z", 30) == 1.0

    def test_no_window_is_zero(self):
        assert event_proximity(None, "2023-10-15", 30) == 0.0

    def test_nonpositive_window_days_guarded(self):
        assert event_proximity(self.W, "2023-11-15", 0) == 0.0
        assert event_proximity(self.W, "2023-11-15", -5) == 0.0


class TestExpandEventWindow:
    def test_year(self):
        assert expand_event_window("2023", "2023") == ("2023-01-01", "2023-12-31")

    def test_month(self):
        assert expand_event_window("2023-10", "2023-10") == ("2023-10-01", "2023-10-31")

    def test_leap_february(self):
        assert expand_event_window("2024-02", "2024-02") == ("2024-02-01", "2024-02-29")
        assert expand_event_window("2023-02", "2023-02") == ("2023-02-01", "2023-02-28")

    def test_exact_day(self):
        assert expand_event_window("2023-10-17", "2023-10-17") == ("2023-10-17", "2023-10-17")

    def test_open_intervals(self):
        assert expand_event_window("2024", None) == ("2024-01-01", None)
        assert expand_event_window(None, "2024") == (None, "2024-12-31")
        assert expand_event_window(None, None) == (None, None)

    def test_mixed_granularity(self):
        assert expand_event_window("2023-06", "2023-12-25") == ("2023-06-01", "2023-12-25")

    def test_invalid_raises(self):
        for bad in ("garbage", "2023-13", "2023/10/17", "23-10-17", "2023-10-32"):
            with pytest.raises(ValueError):
                expand_event_window(bad, None)

    def test_from_after_to_raises(self):
        with pytest.raises(ValueError):
            expand_event_window("2023", "2022")
        with pytest.raises(ValueError):
            expand_event_window("2023-10-17", "2023-10-16")


class TestInferEventAnchor:
    def test_full_date_numeric(self):
        assert infer_event_anchor_from_query("deploy em 17/10/2023?") == ("2023-10-17", "2023-10-17")

    def test_full_date_iso(self):
        assert infer_event_anchor_from_query("o que houve em 2023-10-17") == ("2023-10-17", "2023-10-17")

    def test_month_year_pt(self):
        assert infer_event_anchor_from_query("a reserva de outubro de 2023") == ("2023-10-01", "2023-10-31")

    def test_month_year_en(self):
        assert infer_event_anchor_from_query("the October 2024 report") == ("2024-10-01", "2024-10-31")

    def test_month_year_numeric(self):
        assert infer_event_anchor_from_query("fatura de 10/2023") == ("2023-10-01", "2023-10-31")

    def test_full_date_not_double_counted_as_month(self):
        # "17 de outubro de 2023" is ONE full date; the nested "outubro de 2023"
        # must not count as a second expression (which would return None).
        assert infer_event_anchor_from_query("17 de outubro de 2023 foi o deploy") == (
            "2023-10-17",
            "2023-10-17",
        )

    def test_leap_month_window(self):
        assert infer_event_anchor_from_query("fevereiro de 2024") == ("2024-02-01", "2024-02-29")

    def test_bare_year_does_not_trigger(self):
        assert infer_event_anchor_from_query("o que aconteceu em 2023?") is None
        assert infer_event_anchor_from_query("versão 2023 do produto") is None

    def test_two_distinct_expressions_none(self):
        assert infer_event_anchor_from_query("entre 10/2023 e 05/2024") is None
        assert infer_event_anchor_from_query("de 17/10/2023 a 20/10/2023") is None

    def test_no_temporal_expression_none(self):
        assert infer_event_anchor_from_query("qual meu embedder favorito") is None
        assert infer_event_anchor_from_query("") is None
        assert infer_event_anchor_from_query(None) is None

    def test_same_date_repeated_collapses(self):
        assert infer_event_anchor_from_query("17/10/2023 e de novo 17/10/2023") == (
            "2023-10-17",
            "2023-10-17",
        )


class TestScoreAndRankEvent:
    CANDS = [
        {"id": "a", "score": 0.70, "payload": {}},
        {"id": "b", "score": 0.66, "payload": {}},
    ]

    def test_none_event_boosts_byte_identical(self):
        base = [r["score"] for r in score_and_rank(self.CANDS, {}, {}, 0.1, 10)]
        none = [r["score"] for r in score_and_rank(self.CANDS, {}, {}, 0.1, 10, event_boosts=None)]
        assert base == none

    def test_zero_weight_byte_identical(self):
        base = [r["score"] for r in score_and_rank(self.CANDS, {}, {}, 0.1, 10)]
        w0 = [
            r["score"]
            for r in score_and_rank(self.CANDS, {}, {}, 0.1, 10, event_boosts={"b": 1.0}, event_weight=0.0)
        ]
        assert base == w0

    def test_divisor_grows_only_when_event_present(self):
        # With an event boost on 'b', the divisor grows (all scores renormalize)
        # AND 'b' gets the additive term — enough to flip it above 'a'.
        out = score_and_rank(self.CANDS, {}, {}, 0.1, 10, event_boosts={"b": 1.0}, event_weight=0.15)
        assert [r["id"] for r in out] == ["b", "a"]

    def test_explain_exposes_event_boost(self):
        out = score_and_rank(
            self.CANDS, {}, {}, 0.1, 10, explain=True, event_boosts={"a": 0.5}, event_weight=0.15
        )
        details = {r["id"]: r["score_details"] for r in out}
        assert details["a"]["event_boost"] == pytest.approx(0.5 * 0.15)
        assert details["b"]["event_boost"] == 0.0


class TestEventPostRerank:
    """The tie-break must never overturn a decision the reranker made with margin."""

    def _doc(self, i, logit, event_date=None):
        meta = {"event_date": event_date} if event_date else {}
        return {"id": i, "rerank_score": logit, "metadata": meta}

    ANCHOR = ("2023-10-17", "2023-10-17")

    def test_within_band_proximity_wins(self):
        temp = MemoryTemporalityConfig()
        docs = [self._doc("x", 2.0, "2020-01-01"), self._doc("y", 2.0, "2023-10-17")]
        out = _apply_post_rerank_adjustments(docs, temp=temp, event_anchor=self.ANCHOR)
        assert [d["id"] for d in out] == ["y", "x"]

    def test_decisive_margin_not_overturned(self):
        # x is decisively higher on the reranker (>> tie band); y matches the
        # anchor exactly. y must NOT be promoted. This is the whole point.
        temp = MemoryTemporalityConfig()
        docs = [self._doc("x", 5.0, "2020-01-01"), self._doc("y", 1.0, "2023-10-17")]
        out = _apply_post_rerank_adjustments(docs, temp=temp, event_anchor=self.ANCHOR)
        assert out[0]["id"] == "x"

    def test_no_anchor_preserves_order(self):
        temp = MemoryTemporalityConfig()
        docs = [self._doc("x", 2.0, "2020-01-01"), self._doc("y", 2.0, "2023-10-17")]
        out = _apply_post_rerank_adjustments(docs, temp=temp, event_anchor=None)
        assert [d["id"] for d in out] == ["x", "y"]

    def test_equal_proximity_falls_through_to_activation(self):
        temp = MemoryTemporalityConfig()
        dyn = MemoryDynamicsConfig()
        # both match the anchor equally; the reinforced one should win on activation
        docs = [
            self._doc("x", 2.0, "2023-10-17"),
            self._doc("y", 2.0, "2023-10-17"),
        ]
        docs[1]["metadata"]["reinforced_at"] = ["2099-01-01T00:00:00+00:00"] * 3
        docs[1]["metadata"]["access_count"] = 3
        out = _apply_post_rerank_adjustments(docs, dyn=dyn, temp=temp, event_anchor=self.ANCHOR)
        assert out[0]["id"] == "y"

    def test_weight_zero_tie_break_still_active(self):
        # escape hatch: fusion off (weight=0) but tie-break still reorders ties
        temp = MemoryTemporalityConfig(event_ranking_weight=0)
        docs = [self._doc("x", 2.0, "2020-01-01"), self._doc("y", 2.0, "2023-10-17")]
        out = _apply_post_rerank_adjustments(docs, temp=temp, event_anchor=self.ANCHOR)
        assert out[0]["id"] == "y"

    def test_dyn_none_uses_shared_tie_band(self):
        temp = MemoryTemporalityConfig()
        docs = [self._doc("x", 2.0, "2020-01-01"), self._doc("y", 2.0, "2023-10-17")]
        out = _apply_post_rerank_adjustments(docs, dyn=None, temp=temp, event_anchor=self.ANCHOR)
        assert out[0]["id"] == "y"

    def test_dyn_tie_band_zero_disables_actr_reorder(self):
        # dyn.tie_band=0 disables the ACT-R tie-break (no event anchor here, so the
        # event band is not in play). A reinforced twin must NOT be promoted.
        dyn = MemoryDynamicsConfig(tie_band=0)
        docs = [self._doc("x", 2.0, None), self._doc("y", 2.0, None)]
        docs[1]["metadata"]["reinforced_at"] = ["2099-01-01T00:00:00+00:00"] * 3
        docs[1]["metadata"]["access_count"] = 3
        out = _apply_post_rerank_adjustments(docs, dyn=dyn, event_anchor=None)
        assert [d["id"] for d in out] == ["x", "y"]

    def test_event_ranking_off_no_effect(self):
        temp = MemoryTemporalityConfig(event_ranking=False)
        docs = [self._doc("x", 2.0, "2020-01-01"), self._doc("y", 2.0, "2023-10-17")]
        out = _apply_post_rerank_adjustments(docs, temp=temp, event_anchor=self.ANCHOR)
        assert [d["id"] for d in out] == ["x", "y"]

    def test_default_band_is_conservative(self):
        # At the DEFAULT band (0.002 = ACT-R), a pair the reranker separates by
        # ~0.005 sigmoid is NOT reordered — the conservative, no-overfit default.
        temp = MemoryTemporalityConfig()  # event_tie_band == RERANK_TIE_BAND
        docs = [self._doc("wrong", 2.05, "2020-01-01"), self._doc("right", 2.0, "2023-10-17")]
        out = _apply_post_rerank_adjustments(docs, temp=temp, event_anchor=self.ANCHOR)
        assert out[0]["id"] == "wrong"

    def test_widened_event_band_reorders_near_ties(self):
        # An explicitly WIDENED event band (post-calibration use) reorders a pair
        # the reranker separated by ~0.005 (< the widened band) toward the date.
        temp = MemoryTemporalityConfig(event_tie_band=0.05)
        docs = [self._doc("wrong", 2.05, "2020-01-01"), self._doc("right", 2.0, "2023-10-17")]
        out = _apply_post_rerank_adjustments(docs, temp=temp, event_anchor=self.ANCHOR)
        assert out[0]["id"] == "right"

    def test_widened_event_band_still_bounded(self):
        # Even a widened event band never overturns a decisive reranker margin.
        temp = MemoryTemporalityConfig(event_tie_band=0.05)
        docs = [self._doc("wrong", 3.5, "2020-01-01"), self._doc("right", 1.0, "2023-10-17")]
        out = _apply_post_rerank_adjustments(docs, temp=temp, event_anchor=self.ANCHOR)
        assert out[0]["id"] == "wrong"

    def test_widened_event_band_does_not_contaminate_activation(self):
        # REGRESSION (measured bug 2026-07-23): a widened EVENT band must NOT widen
        # the ACT-R activation window. Two UNDATED candidates on a DATED query,
        # reranker gap ~0.003 (> ACT-R 0.002, < event 0.05), the lower one heavily
        # reinforced: activation must NOT promote it (its window stays 0.002), even
        # though the event band is wide — the two tie-breaks are decoupled.
        temp = MemoryTemporalityConfig(event_tie_band=0.05)
        dyn = MemoryDynamicsConfig()
        docs = [self._doc("x", 2.03, None), self._doc("y", 2.0, None)]
        docs[1]["metadata"]["reinforced_at"] = ["2099-01-01T00:00:00+00:00"] * 5
        docs[1]["metadata"]["access_count"] = 5
        out = _apply_post_rerank_adjustments(docs, dyn=dyn, temp=temp, event_anchor=self.ANCHOR)
        assert [d["id"] for d in out] == ["x", "y"]

    def test_event_tie_band_zero_disables_event_reorder(self):
        temp = MemoryTemporalityConfig(event_tie_band=0)
        docs = [self._doc("wrong", 2.0, "2020-01-01"), self._doc("right", 2.0, "2023-10-17")]
        out = _apply_post_rerank_adjustments(docs, temp=temp, event_anchor=self.ANCHOR)
        assert out[0]["id"] == "wrong"  # band 0 → reranker order stands even on an exact tie

    def test_event_proximity_annotated_on_matching_doc(self):
        temp = MemoryTemporalityConfig()
        docs = [self._doc("y", 2.0, "2023-10-17"), self._doc("x", 2.0, None)]
        out = _apply_post_rerank_adjustments(docs, temp=temp, event_anchor=self.ANCHOR)
        by_id = {d["id"]: d for d in out}
        assert by_id["y"]["event_proximity"] == 1.0
        assert "event_proximity" not in by_id["x"]  # only set when > 0


class TestEventSupersessionFactorial:
    """Activating the event term (which grows the divisor) must not flip pairs
    that are UNRELATED to the anchor. Covers Codex critique point [2]."""

    def _cands(self):
        # c/d are an unrelated superseded/current pair with NO event_date; the
        # anchor matches 'm' elsewhere in the pool.
        return [
            {"id": "c", "score": 0.62, "payload": {"superseded_by": "d", "superseded_at": "2020-01-01T00:00:00+00:00"}},
            {"id": "d", "score": 0.60, "payload": {}},
            {"id": "m", "score": 0.55, "payload": {"event_date": "2023-10-17"}},
        ]

    def _order(self, **kw):
        penalties = {"c": 0.2}
        out = score_and_rank(self._cands(), {}, {}, 0.1, 10, penalties=penalties, **kw)
        return [r["id"] for r in out]

    def test_unrelated_pair_order_stable_with_event_active(self):
        # d must stay above c (c is superseded, penalized) whether or not the
        # event term is active for 'm'.
        without = self._order()
        with_event = self._order(event_boosts={"m": 1.0}, event_weight=0.15)
        assert without.index("d") < without.index("c")
        assert with_event.index("d") < with_event.index("c")

    def test_weight_zero_identical_to_no_event(self):
        assert self._order() == self._order(event_boosts={"m": 1.0}, event_weight=0.0)


class TestConfigBounds:
    def test_defaults(self):
        t = MemoryTemporalityConfig()
        assert t.event_ranking is True
        assert t.event_ranking_weight == 0.15
        assert t.event_window_days == 30
        assert t.event_tie_band == RERANK_TIE_BAND  # conservative default (0.002)

    def test_event_tie_band_negative_rejected(self):
        with pytest.raises(Exception):
            MemoryTemporalityConfig(event_tie_band=-0.01)

    def test_shared_tie_band_default(self):
        assert MemoryDynamicsConfig().tie_band == RERANK_TIE_BAND

    def test_negative_weight_rejected(self):
        with pytest.raises(Exception):
            MemoryTemporalityConfig(event_ranking_weight=-0.1)

    def test_nonpositive_window_days_rejected(self):
        for bad in (0, -5):
            with pytest.raises(Exception):
                MemoryTemporalityConfig(event_window_days=bad)

    def test_zero_weight_accepted(self):
        assert MemoryTemporalityConfig(event_ranking_weight=0).event_ranking_weight == 0.0


# --- search-level tests (mocked stores): filter injection, immutability, echo ---


@pytest.fixture(autouse=True)
def _openai_key():
    os.environ["OPENAI_API_KEY"] = "test"


def _make_memory():
    with (
        patch("mem0.utils.factory.EmbedderFactory") as mock_embedder,
        patch("mem0.memory.main.VectorStoreFactory") as mock_vs,
        patch("mem0.utils.factory.LlmFactory") as mock_llm,
        patch("mem0.memory.telemetry.capture_event"),
    ):
        mock_embedder.create.return_value = Mock()
        mock_vs.create.return_value = Mock()
        mock_vs.create.return_value.search.return_value = []
        mock_vs.create.return_value.keyword_search.return_value = None
        mock_llm.create.return_value = Mock()
        return Memory(MemoryConfig(version="v1.1"))


def _search(mem, **kw):
    mem.vector_store.search = Mock(return_value=[])
    mem.vector_store.keyword_search = Mock(return_value=None)
    mem.embedding_model.embed = Mock(return_value=[0.1, 0.2, 0.3])
    with (
        patch("mem0.memory.main.lemmatize_for_bm25", return_value="q"),
        patch("mem0.memory.main.extract_entities", return_value=[]),
    ):
        return mem.search("q", filters=kw.pop("filters", {"user_id": "u"}), **kw)


class TestSearchFilterInjection:
    def test_window_reaches_vector_and_keyword_search(self):
        mem = _make_memory()
        _search(mem, event_from="2023-10", event_to="2023-10")
        called = mem.vector_store.search.call_args.kwargs["filters"]
        assert called["event_date"] == {"gte": "2023-10-01", "lte": "2023-10-31"}
        kw_called = mem.vector_store.keyword_search.call_args.kwargs["filters"]
        assert kw_called["event_date"] == {"gte": "2023-10-01", "lte": "2023-10-31"}

    def test_open_interval_one_sided(self):
        mem = _make_memory()
        _search(mem, event_from="2024")
        assert mem.vector_store.search.call_args.kwargs["filters"]["event_date"] == {"gte": "2024-01-01"}

    def test_caller_filter_not_mutated(self):
        mem = _make_memory()
        caller = {"user_id": "u", "event_date": {"gte": "2020-01-01"}}
        _search(mem, filters=caller, event_to="2023-12")
        # the caller's nested dict must be untouched (only a fresh dict is stored)
        assert caller["event_date"] == {"gte": "2020-01-01"}

    def test_tightening_existing_bound(self):
        mem = _make_memory()
        caller = {"user_id": "u", "event_date": {"gte": "2020-01-01", "lte": "2025-12-31"}}
        _search(mem, filters=caller, event_from="2023", event_to="2023")
        stored = mem.vector_store.search.call_args.kwargs["filters"]["event_date"]
        assert stored["gte"] == "2023-01-01"  # tightened up from 2020
        assert stored["lte"] == "2023-12-31"  # tightened down from 2025

    def test_malformed_window_fails_fast(self):
        mem = _make_memory()
        with pytest.raises(ValueError):
            _search(mem, event_from="garbage")

    def test_composes_with_as_of(self):
        mem = _make_memory()
        _search(mem, as_of="2024-01-01", event_from="2023", event_to="2023")
        filters = mem.vector_store.search.call_args.kwargs["filters"]
        assert "created_at" in filters and "event_date" in filters


class TestSearchEcho:
    def test_event_filter_echoed(self):
        mem = _make_memory()
        r = _search(mem, event_from="2023-10", event_to="2023-10")
        assert r["event_filter"] == {"from": "2023-10-01", "to": "2023-10-31"}
        assert "event_anchor" not in r

    def test_event_anchor_echoed_from_query(self):
        mem = _make_memory()
        mem.vector_store.search = Mock(return_value=[])
        mem.vector_store.keyword_search = Mock(return_value=None)
        mem.embedding_model.embed = Mock(return_value=[0.1, 0.2, 0.3])
        with (
            patch("mem0.memory.main.lemmatize_for_bm25", return_value="q"),
            patch("mem0.memory.main.extract_entities", return_value=[]),
        ):
            r = mem.search("o deploy de 17/10/2023", filters={"user_id": "u"})
        assert r["event_anchor"] == {"from": "2023-10-17", "to": "2023-10-17"}
        assert "event_filter" not in r

    def test_explicit_window_suppresses_anchor(self):
        mem = _make_memory()
        mem.vector_store.search = Mock(return_value=[])
        mem.vector_store.keyword_search = Mock(return_value=None)
        mem.embedding_model.embed = Mock(return_value=[0.1, 0.2, 0.3])
        with (
            patch("mem0.memory.main.lemmatize_for_bm25", return_value="q"),
            patch("mem0.memory.main.extract_entities", return_value=[]),
        ):
            r = mem.search("o deploy de 17/10/2023", filters={"user_id": "u"}, event_from="2020")
        assert "event_filter" in r
        assert "event_anchor" not in r

    def test_no_temporal_query_no_echo(self):
        mem = _make_memory()
        r = _search(mem)
        assert "event_anchor" not in r and "event_filter" not in r


class TestAsyncParity:
    def test_async_signature_has_event_params(self):
        sig = inspect.signature(AsyncMemory.search)
        assert "event_from" in sig.parameters
        assert "event_to" in sig.parameters

    def test_sync_signature_has_event_params(self):
        sig = inspect.signature(Memory.search)
        assert "event_from" in sig.parameters
        assert "event_to" in sig.parameters

    def test_async_filter_injection_and_echo(self):
        # asyncio.run in a sync test (no pytest-asyncio dependency): proves the
        # async path detects the anchor, echoes it, and injects the same filter.
        import asyncio

        with (
            patch("mem0.utils.factory.EmbedderFactory") as mock_embedder,
            patch("mem0.memory.main.VectorStoreFactory") as mock_vs,
            patch("mem0.utils.factory.LlmFactory") as mock_llm,
            patch("mem0.memory.telemetry.capture_event"),
        ):
            mock_embedder.create.return_value = Mock()
            mock_vs.create.return_value = Mock()
            mock_vs.create.return_value.search.return_value = []
            mock_vs.create.return_value.keyword_search.return_value = None
            mock_llm.create.return_value = Mock()
            mem = AsyncMemory(MemoryConfig(version="v1.1"))
        mem.vector_store.search = Mock(return_value=[])
        mem.vector_store.keyword_search = Mock(return_value=None)
        mem.embedding_model.embed = Mock(return_value=[0.1, 0.2, 0.3])
        with (
            patch("mem0.memory.main.lemmatize_for_bm25", return_value="q"),
            patch("mem0.memory.main.extract_entities", return_value=[]),
        ):
            r = asyncio.run(mem.search("o deploy de 17/10/2023", filters={"user_id": "u"}))
        assert r["event_anchor"] == {"from": "2023-10-17", "to": "2023-10-17"}
        assert mem.vector_store.search.call_args.kwargs["filters"]["user_id"] == "u"
