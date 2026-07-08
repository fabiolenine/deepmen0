"""DeepMem0 v0.4 ingest-time decoupling tests — pure units, no live infrastructure.

Async ingestion separates a fact's submission time from its processing time.
Two contracts keep record-time semantics honest across that gap:

1. A caller-provided ``created_at`` in metadata is canonical — the pipeline
   must never overwrite it with processing time (the queue worker injects the
   enqueue timestamp so as_of anchors and supersession direction stay true).
2. Supersession direction honors record time: a queued fact that reaches the
   store AFTER a newer fact about the same subject is born superseded by it,
   never the other way around (``supersession_inverted``).
"""

from datetime import datetime, timedelta, timezone

from mem0.memory.main import Memory, _mark_superseded
from mem0.utils.temporality import supersession_inverted

NOW = datetime(2030, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def iso_days_ago(d):
    return (NOW - timedelta(days=d)).isoformat()


class TestSupersessionInverted:
    def test_arriving_older_inverts(self):
        assert supersession_inverted(iso_days_ago(5), iso_days_ago(1)) is True

    def test_arriving_newer_stays_forward(self):
        assert supersession_inverted(iso_days_ago(1), iso_days_ago(5)) is False

    def test_equal_timestamps_stay_forward(self):
        ts = iso_days_ago(3)
        assert supersession_inverted(ts, ts) is False

    def test_missing_or_malformed_stays_forward(self):
        assert supersession_inverted(None, iso_days_ago(1)) is False
        assert supersession_inverted(iso_days_ago(1), None) is False
        assert supersession_inverted("not-a-date", iso_days_ago(1)) is False
        assert supersession_inverted(iso_days_ago(1), "not-a-date") is False


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

    def insert(self, vectors, ids, payloads):
        for mid, pay in zip(ids, payloads):
            self.points[mid] = pay


class FakeDB:
    def __init__(self):
        self.history = []

    def add_history(self, memory_id, old_memory, new_memory, event, **kwargs):
        self.history.append((memory_id, old_memory, new_memory, event))


class TestBornSuperseded:
    def test_late_arrival_is_born_superseded(self):
        # existing fact is FRESHER than the arriving one -> newcomer demoted
        store = FakeVectorStore({
            "old-1": {"data": "v2 fresh", "created_at": iso_days_ago(1)},
            "new-1": {"data": "v1 stale", "created_at": iso_days_ago(5)},
        })
        db = FakeDB()
        marked = _mark_superseded(store, db, "new-1", "v1 stale", ["old-1"], new_created_at=iso_days_ago(5))
        assert marked == [("new-1", "old-1")]
        assert store.points["new-1"]["superseded_by"] == "old-1"
        assert "superseded_at" in store.points["new-1"]
        assert "superseded_by" not in store.points["old-1"]  # fresher truth untouched
        assert db.history == [("new-1", "v1 stale", "v2 fresh", "SUPERSEDED")]

    def test_fresh_arrival_keeps_forward_direction(self):
        store = FakeVectorStore({
            "old-1": {"data": "v1", "created_at": iso_days_ago(5)},
            "new-1": {"data": "v2", "created_at": iso_days_ago(1)},
        })
        db = FakeDB()
        marked = _mark_superseded(store, db, "new-1", "v2", ["old-1"], new_created_at=iso_days_ago(1))
        assert marked == [("old-1", "new-1")]
        assert store.points["old-1"]["superseded_by"] == "new-1"
        assert "superseded_by" not in store.points["new-1"]

    def test_no_created_at_keeps_v03_behavior(self):
        store = FakeVectorStore({
            "old-1": {"data": "v1", "created_at": iso_days_ago(1)},
            "new-1": {"data": "v2", "created_at": iso_days_ago(5)},
        })
        marked = _mark_superseded(store, FakeDB(), "new-1", "v2", ["old-1"])
        assert marked == [("old-1", "new-1")]

    def test_first_marking_wins_for_newcomer(self):
        # two fresher existing facts: the newcomer is settled by the first,
        # never re-marked by the second
        store = FakeVectorStore({
            "old-1": {"data": "fresh a", "created_at": iso_days_ago(1)},
            "old-2": {"data": "fresh b", "created_at": iso_days_ago(2)},
            "new-1": {"data": "stale", "created_at": iso_days_ago(9)},
        })
        db = FakeDB()
        marked = _mark_superseded(
            store, db, "new-1", "stale", ["old-1", "old-2"], new_created_at=iso_days_ago(9)
        )
        assert marked == [("new-1", "old-1")]
        assert store.points["new-1"]["superseded_by"] == "old-1"
        assert "superseded_by" not in store.points["old-2"]

    def test_mixed_directions_build_a_chain(self):
        # newcomer is fresher than old-2 but staler than old-1:
        # old-2 -> new-1 -> old-1 emerges in one pass
        store = FakeVectorStore({
            "old-1": {"data": "freshest", "created_at": iso_days_ago(1)},
            "old-2": {"data": "oldest", "created_at": iso_days_ago(9)},
            "new-1": {"data": "middle", "created_at": iso_days_ago(5)},
        })
        db = FakeDB()
        marked = _mark_superseded(
            store, db, "new-1", "middle", ["old-1", "old-2"], new_created_at=iso_days_ago(5)
        )
        assert ("new-1", "old-1") in marked and ("old-2", "new-1") in marked
        assert store.points["new-1"]["superseded_by"] == "old-1"
        assert store.points["old-2"]["superseded_by"] == "new-1"

    def test_already_superseded_newcomer_not_remarked(self):
        store = FakeVectorStore({
            "old-1": {"data": "fresh", "created_at": iso_days_ago(1)},
            "new-1": {"data": "stale", "created_at": iso_days_ago(9),
                      "superseded_by": "earlier", "superseded_at": iso_days_ago(3)},
        })
        db = FakeDB()
        assert _mark_superseded(store, db, "new-1", "stale", ["old-1"], new_created_at=iso_days_ago(9)) == []
        assert store.updates == [] and db.history == []


class FakeEmbedder:
    def embed(self, data, memory_action=None):
        return [0.0, 1.0]


class TestCreatedAtOverride:
    """metadata['created_at'] is canonical — the pipeline never overwrites it."""

    def _bare_memory(self):
        from mem0.configs.base import MemoryConfig

        mem = Memory.__new__(Memory)
        mem.config = MemoryConfig()
        mem.embedding_model = FakeEmbedder()
        mem.vector_store = FakeVectorStore({})
        mem.db = FakeDB()
        return mem

    def test_create_memory_honors_provided_created_at(self):
        mem = self._bare_memory()
        submitted = iso_days_ago(2)
        memory_id = mem._create_memory("fact", {}, metadata={"created_at": submitted, "task_id": "tsk_x"})
        payload = mem.vector_store.points[memory_id]
        assert payload["created_at"] == submitted
        assert payload["updated_at"] == submitted
        assert payload["task_id"] == "tsk_x"  # provenance flows into the payload

    def test_create_memory_defaults_to_now_without_override(self):
        mem = self._bare_memory()
        memory_id = mem._create_memory("fact", {}, metadata={})
        created = datetime.fromisoformat(mem.vector_store.points[memory_id]["created_at"])
        assert abs((datetime.now(timezone.utc) - created).total_seconds()) < 60
