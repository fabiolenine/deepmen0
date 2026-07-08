"""DeepMem0 v0.3 semantic temporality tests — pure units, no live infrastructure."""

from datetime import datetime, timedelta, timezone

import pytest

from mem0.configs.base import MemoryConfig, MemoryDynamicsConfig, MemoryTemporalityConfig
from mem0.configs.prompts import build_temporality_suffix
from mem0.memory.main import (
    _apply_post_rerank_adjustments,
    _mark_superseded,
    _temporality_config,
)
from mem0.utils.scoring import score_and_rank
from mem0.utils.temporality import (
    parse_as_of,
    parse_event_date,
    parse_supersedes_ids,
    superseded_penalty_applies,
)

NOW = datetime(2030, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
UUID_MAPPING = {"0": "uuid-zero", "1": "uuid-one", "2": "uuid-two"}


def iso_days_ago(d):
    return (NOW - timedelta(days=d)).isoformat()


class TestParseSupersedes:
    def test_valid_indices_resolve(self):
        assert parse_supersedes_ids(["0", "2"], UUID_MAPPING) == ["uuid-zero", "uuid-two"]

    def test_integer_indices_tolerated(self):
        assert parse_supersedes_ids([1], UUID_MAPPING) == ["uuid-one"]

    def test_hallucinated_index_discarded(self):
        assert parse_supersedes_ids(["7", "0"], UUID_MAPPING) == ["uuid-zero"]

    def test_garbage_input_is_empty(self):
        assert parse_supersedes_ids(None, UUID_MAPPING) == []
        assert parse_supersedes_ids("0", UUID_MAPPING) == []
        assert parse_supersedes_ids({"id": "0"}, UUID_MAPPING) == []
        assert parse_supersedes_ids([], UUID_MAPPING) == []
        assert parse_supersedes_ids([{"x": 1}, None], UUID_MAPPING) == []

    def test_duplicates_collapse(self):
        assert parse_supersedes_ids(["1", "1"], UUID_MAPPING) == ["uuid-one"]

    def test_empty_mapping_resolves_nothing(self):
        assert parse_supersedes_ids(["0"], {}) == []


class TestParseEventDate:
    def test_plain_date(self):
        assert parse_event_date("2026-03-15") == "2026-03-15"

    def test_full_iso_normalizes_to_date(self):
        assert parse_event_date("2026-03-15T10:30:00Z") == "2026-03-15"

    def test_garbage_is_none(self):
        for bad in (None, "", "march", "15/03/2026", "2026-13-40", 20260315, ["2026-03-15"]):
            assert parse_event_date(bad) is None


class TestParseAsOf:
    def test_plain_date_normalizes_to_end_of_day(self):
        iso, dt = parse_as_of("2026-03-15")
        assert dt.hour == 23 and dt.minute == 59
        assert dt.tzinfo is not None
        assert iso.startswith("2026-03-15T23:59:59")

    def test_full_iso_respected(self):
        iso, dt = parse_as_of("2026-03-15T08:00:00+00:00")
        assert dt.hour == 8

    def test_naive_datetime_assumed_utc(self):
        _, dt = parse_as_of("2026-03-15T08:00:00")
        assert dt.tzinfo is not None

    def test_invalid_raises(self):
        for bad in ("not-a-date", "", None, "2026-13-01"):
            with pytest.raises(ValueError):
                parse_as_of(bad)


class TestSupersededPenaltyApplies:
    def test_current_fact_is_never_penalized(self):
        assert superseded_penalty_applies({}) is False
        assert superseded_penalty_applies({"data": "x"}) is False

    def test_superseded_without_anchor_is_penalized(self):
        payload = {"superseded_by": "uuid-new", "superseded_at": iso_days_ago(3)}
        assert superseded_penalty_applies(payload) is True

    def test_superseded_after_anchor_was_current_then(self):
        payload = {"superseded_by": "uuid-new", "superseded_at": iso_days_ago(3)}
        anchor = NOW - timedelta(days=10)
        assert superseded_penalty_applies(payload, as_of=anchor) is False

    def test_superseded_before_anchor_stays_penalized(self):
        payload = {"superseded_by": "uuid-new", "superseded_at": iso_days_ago(10)}
        anchor = NOW - timedelta(days=3)
        assert superseded_penalty_applies(payload, as_of=anchor) is True

    def test_boundary_equality_is_penalized(self):
        ts = iso_days_ago(5)
        payload = {"superseded_by": "uuid-new", "superseded_at": ts}
        assert superseded_penalty_applies(payload, as_of=NOW - timedelta(days=5)) is True

    def test_malformed_superseded_at_is_conservative(self):
        payload = {"superseded_by": "uuid-new", "superseded_at": "garbage"}
        assert superseded_penalty_applies(payload, as_of=NOW) is True


class TestPenaltyInFusion:
    CANDIDATES = [
        {"id": "old", "score": 0.80, "payload": {"data": "fact v1"}},
        {"id": "new", "score": 0.80, "payload": {"data": "fact v2"}},
    ]

    def test_penalty_demotes_but_never_excludes(self):
        ranked = score_and_rank(
            self.CANDIDATES, {}, {}, 0.1, 2, penalties={"old": 0.2}
        )
        assert [r["id"] for r in ranked] == ["new", "old"]
        assert len(ranked) == 2  # still present

    def test_penalty_floors_at_zero(self):
        ranked = score_and_rank(self.CANDIDATES, {}, {}, 0.1, 2, penalties={"old": 5.0})
        by_id = {r["id"]: r["score"] for r in ranked}
        assert by_id["old"] == 0.0

    def test_none_penalties_is_backward_compatible(self):
        legacy = score_and_rank(self.CANDIDATES, {}, {}, 0.1, 2)
        explicit = score_and_rank(self.CANDIDATES, {}, {}, 0.1, 2, penalties=None)
        assert [r["score"] for r in legacy] == [r["score"] for r in explicit]

    def test_explain_exposes_penalty(self):
        ranked = score_and_rank(
            self.CANDIDATES, {}, {}, 0.1, 2, explain=True, penalties={"old": 0.2}
        )
        details = {r["id"]: r["score_details"] for r in ranked}
        assert details["old"]["superseded_penalty"] == 0.2
        assert details["new"]["superseded_penalty"] == 0.0

    def test_threshold_gate_untouched_by_penalty(self):
        # gate acts on the semantic score BEFORE penalties: a penalized
        # candidate above the gate stays; one below the gate is out anyway.
        ranked = score_and_rank(self.CANDIDATES, {}, {}, 0.5, 2, penalties={"old": 0.9})
        assert {r["id"] for r in ranked} == {"old", "new"}


class TestPostRerankAdjustments:
    def make_docs(self):
        def real_days_ago(d):
            return (datetime.now(timezone.utc) - timedelta(days=d)).isoformat()

        return [
            {
                "id": "old",
                "memory": "orion backups run weekly",
                "created_at": real_days_ago(60),
                "rerank_score": 2.10,
                "metadata": {
                    "superseded_by": "new",
                    "superseded_at": real_days_ago(5),
                },
            },
            {
                "id": "new",
                "memory": "orion backups run daily",
                "created_at": real_days_ago(5),
                "rerank_score": 2.00,
                "metadata": {},
            },
        ]

    def test_penalty_demotes_superseded_after_rerank(self):
        temp = MemoryTemporalityConfig()
        ordered = _apply_post_rerank_adjustments(self.make_docs(), temp=temp)
        assert [d["id"] for d in ordered] == ["new", "old"]
        assert ordered[1]["superseded_penalty"] == temp.superseded_penalty

    def test_as_of_before_supersession_waives_penalty(self):
        temp = MemoryTemporalityConfig()
        anchor = datetime.now(timezone.utc) - timedelta(days=30)
        ordered = _apply_post_rerank_adjustments(self.make_docs(), temp=temp, as_of=anchor)
        assert ordered[0]["id"] == "old"  # was current at the anchor; higher rerank wins
        assert "superseded_penalty" not in ordered[0]

    def test_activation_and_penalty_compose_in_single_sort(self):
        docs = self.make_docs()
        # give the superseded doc a strong reinforcement history: activation
        # must soften but not cancel the demotion in the same sort
        docs[0]["metadata"]["reinforced_at"] = [
            (datetime.now(timezone.utc) - timedelta(days=d)).isoformat() for d in (3, 2, 1)
        ]
        docs[0]["metadata"]["access_count"] = 3
        dyn = MemoryDynamicsConfig()
        temp = MemoryTemporalityConfig()
        ordered = _apply_post_rerank_adjustments(docs, dyn=dyn, temp=temp)
        top = ordered[0]
        assert top["id"] == "new"
        assert ordered[1].get("activation", 0) > 0  # both adjustments visible

    def test_disabled_everything_keeps_order(self):
        ordered = _apply_post_rerank_adjustments(self.make_docs(), dyn=None, temp=None)
        assert [d["id"] for d in ordered] == ["old", "new"]


class FakeVectorStore:
    def __init__(self, points):
        self.points = points  # id -> payload
        self.updates = []

    def get(self, vector_id):
        payload = self.points.get(vector_id)
        if payload is None:
            return None
        return type("P", (), {"id": vector_id, "payload": payload})()

    def update(self, vector_id, vector=None, payload=None):
        self.updates.append((vector_id, payload))
        self.points[vector_id] = payload


class FakeDB:
    def __init__(self):
        self.history = []

    def add_history(self, memory_id, old_memory, new_memory, event, **kwargs):
        self.history.append((memory_id, old_memory, new_memory, event))


class TestMarkSuperseded:
    def test_marks_and_records_history(self):
        store = FakeVectorStore({"old-1": {"data": "v1", "created_at": iso_days_ago(30), "domain": "infra"}})
        db = FakeDB()
        marked = _mark_superseded(store, db, "new-1", "v2", ["old-1"])
        assert marked == [("old-1", "new-1")]
        _, payload = store.updates[0]
        assert payload["superseded_by"] == "new-1"
        assert "superseded_at" in payload
        assert payload["data"] == "v1" and payload["domain"] == "infra"  # full merge
        assert db.history == [("old-1", "v1", "v2", "SUPERSEDED")]

    def test_first_marking_wins(self):
        store = FakeVectorStore(
            {"old-1": {"data": "v1", "superseded_by": "earlier", "superseded_at": iso_days_ago(9)}}
        )
        db = FakeDB()
        assert _mark_superseded(store, db, "new-1", "v2", ["old-1"]) == []
        assert store.updates == [] and db.history == []

    def test_self_reference_and_missing_are_skipped(self):
        store = FakeVectorStore({"a": {"data": "x"}})
        db = FakeDB()
        assert _mark_superseded(store, db, "a", "x2", ["a", "ghost"]) == []

    def test_store_failure_never_raises(self):
        class ExplodingStore:
            def get(self, vector_id):
                raise RuntimeError("boom")

        assert _mark_superseded(ExplodingStore(), FakeDB(), "n", "t", ["o"]) == []


class TestConfigSurface:
    def test_defaults(self):
        temp = MemoryConfig().temporality
        assert temp.enabled is True
        assert temp.superseded_penalty == 0.2
        assert temp.extract_event_date is True

    def test_disabled_resolves_to_none(self):
        config = MemoryConfig(temporality=MemoryTemporalityConfig(enabled=False))
        assert _temporality_config(config) is None
        assert _temporality_config(MemoryConfig()) is not None

    def test_from_dict_config(self):
        config = MemoryConfig(**{"temporality": {"superseded_penalty": 0.35, "extract_event_date": False}})
        assert config.temporality.superseded_penalty == 0.35
        assert config.temporality.extract_event_date is False
        assert config.temporality.enabled is True

    def test_prompt_suffix_gating(self):
        with_dates = build_temporality_suffix(include_event_date=True)
        without_dates = build_temporality_suffix(include_event_date=False)
        assert "supersedes" in with_dates and "event_date" in with_dates
        assert "supersedes" in without_dates and "event_date" not in without_dates
