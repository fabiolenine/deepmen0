"""DeepMem0 v0.2 human-memory dynamics tests — pure units, no live infrastructure."""

from datetime import datetime, timedelta, timezone

from mem0.configs.base import MemoryConfig, MemoryDynamicsConfig
from mem0.memory.main import (
    _apply_activation_post_rerank,
    _dynamics_config,
    _reinforce_memory,
)
from mem0.utils.dynamics import (
    activation_boost,
    base_level_activation,
    boost_from_payload,
    reinforcement_fields,
    should_reinforce,
)
from mem0.utils.scoring import score_and_rank

NOW = datetime(2030, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def hours_ago(h):
    return (NOW - timedelta(hours=h)).isoformat()


class TestBaseLevelActivation:
    def test_no_history_is_neutral(self):
        assert base_level_activation(None, now=NOW) is None
        assert base_level_activation([], now=NOW) is None
        assert activation_boost(None) == 0.0

    def test_recency_raises_activation(self):
        recent = base_level_activation([hours_ago(1)], now=NOW)
        stale = base_level_activation([hours_ago(720)], now=NOW)
        assert recent > stale

    def test_frequency_raises_activation(self):
        once = base_level_activation([hours_ago(24)], now=NOW)
        thrice = base_level_activation([hours_ago(72), hours_ago(48), hours_ago(24)], now=NOW)
        assert thrice > once

    def test_decay_is_implicit_with_passing_time(self):
        history = [hours_ago(2)]
        earlier = base_level_activation(history, now=NOW)
        later = base_level_activation(history, now=NOW + timedelta(days=30))
        assert later < earlier

    def test_petrov_tail_counts_trimmed_reinforcements(self):
        history = [hours_ago(48), hours_ago(24)]
        exact = base_level_activation(history, access_count=2, now=NOW)
        with_tail = base_level_activation(
            history, access_count=50, now=NOW, first_seen=hours_ago(24 * 365)
        )
        assert with_tail > exact

    def test_future_or_immediate_timestamps_are_clamped(self):
        activation = base_level_activation([NOW.isoformat(), (NOW + timedelta(hours=1)).isoformat()], now=NOW)
        assert activation is not None
        assert activation < 10  # finite, clamped — not an unbounded spike

    def test_malformed_timestamps_are_ignored(self):
        assert base_level_activation(["not-a-date", 42], now=NOW) is None
        ok = base_level_activation(["not-a-date", hours_ago(5)], now=NOW)
        assert ok is not None

    def test_boost_is_bounded(self):
        for h in (0.01, 1, 24, 24 * 365):
            b = activation_boost(base_level_activation([hours_ago(h)], now=NOW))
            assert 0.0 < b < 1.0

    def test_boost_from_payload_reads_dynamics_fields(self):
        payload = {
            "created_at": hours_ago(1000),
            "reinforced_at": [hours_ago(48), hours_ago(2)],
            "access_count": 7,
        }
        assert boost_from_payload(payload, now=NOW) > 0.0
        assert boost_from_payload({"created_at": hours_ago(2)}, now=NOW) == 0.0


class TestReinforcementWindow:
    def test_no_history_reinforces(self):
        assert should_reinforce({}, now=NOW) is True

    def test_inside_window_is_suppressed(self):
        payload = {"reinforced_at": [hours_ago(0.5)]}
        assert should_reinforce(payload, now=NOW, window_seconds=3600) is False

    def test_outside_window_reinforces(self):
        payload = {"reinforced_at": [hours_ago(1.5)]}
        assert should_reinforce(payload, now=NOW, window_seconds=3600) is True

    def test_zero_window_disables_suppression(self):
        payload = {"reinforced_at": [NOW.isoformat()]}
        assert should_reinforce(payload, now=NOW, window_seconds=0) is True


class TestReinforcementFields:
    def test_legacy_memory_adopts_created_at(self):
        payload = {"created_at": hours_ago(240), "data": "hermes_fx uses walk-forward validation"}
        fields = reinforcement_fields(payload, now=NOW)
        assert fields["reinforced_at"] == [hours_ago(240), NOW.isoformat()]
        assert fields["access_count"] == 2
        assert fields["last_accessed"] == NOW.isoformat()

    def test_history_is_bounded_but_count_is_not(self):
        payload = {
            "reinforced_at": [hours_ago(h) for h in range(20, 10, -1)],
            "access_count": 40,
        }
        fields = reinforcement_fields(payload, now=NOW, max_timestamps=10)
        assert len(fields["reinforced_at"]) == 10
        assert fields["reinforced_at"][-1] == NOW.isoformat()
        assert fields["access_count"] == 41

    def test_creation_is_neutral_until_first_reinforcement(self):
        # Option B: a freshly created memory (no dynamics fields) is neutral,
        # exactly like the legacy corpus — no new-vs-old bias.
        fresh = {"data": "boreal_app ships weekly", "created_at": hours_ago(0)}
        assert boost_from_payload(fresh, now=NOW) == 0.0
        # First reinforcement adopts created_at, yielding a two-event history.
        fields = reinforcement_fields(fresh, now=NOW)
        assert fields["reinforced_at"] == [hours_ago(0), NOW.isoformat()]
        assert fields["access_count"] == 2
        assert boost_from_payload({**fresh, **fields}, now=NOW) > 0.0


class TestActivationInFusion:
    CANDIDATES = [
        {"id": "aaa", "score": 0.80, "payload": {"data": "fact A"}},
        {"id": "bbb", "score": 0.80, "payload": {"data": "fact B"}},
    ]

    def test_activation_breaks_ties(self):
        ranked = score_and_rank(
            semantic_results=self.CANDIDATES,
            bm25_scores={},
            entity_boosts={},
            threshold=0.1,
            top_k=2,
            activation_boosts={"bbb": 0.9},
            activation_weight=0.15,
        )
        assert ranked[0]["id"] == "bbb"

    def test_no_boosts_is_backward_compatible(self):
        legacy = score_and_rank(self.CANDIDATES, {}, {}, 0.1, 2)
        explicit = score_and_rank(
            self.CANDIDATES, {}, {}, 0.1, 2, activation_boosts={}, activation_weight=0.15
        )
        assert [r["score"] for r in legacy] == [r["score"] for r in explicit]

    def test_explain_exposes_activation(self):
        ranked = score_and_rank(
            self.CANDIDATES, {}, {}, 0.1, 2, explain=True,
            activation_boosts={"bbb": 1.0}, activation_weight=0.2,
        )
        by_id = {r["id"]: r["score_details"] for r in ranked}
        assert by_id["bbb"]["activation_boost"] == 0.2
        assert by_id["aaa"]["activation_boost"] == 0.0


class TestActivationPostRerank:
    def make_docs(self):
        # _apply_activation_post_rerank reads the real clock, so these
        # timestamps must be relative to the actual now (not the fixed NOW).
        def real_hours_ago(h):
            return (datetime.now(timezone.utc) - timedelta(hours=h)).isoformat()

        return [
            {
                "id": "cold",
                "memory": "atlas_ingest retries three times",
                "created_at": real_hours_ago(500),
                "rerank_score": 2.05,
                "metadata": {},
            },
            {
                "id": "hot",
                "memory": "atlas_ingest retries thrice",
                "created_at": real_hours_ago(500),
                "rerank_score": 2.00,
                "metadata": {
                    "reinforced_at": [real_hours_ago(30), real_hours_ago(4)],
                    "access_count": 6,
                },
            },
        ]

    def test_reinforced_memory_wins_near_tie(self):
        dyn = MemoryDynamicsConfig()
        ordered = _apply_activation_post_rerank(self.make_docs(), dyn)
        assert ordered[0]["id"] == "hot"
        assert ordered[0]["activation"] > 0

    def test_decisive_rerank_gap_is_not_overturned(self):
        docs = self.make_docs()
        docs[0]["rerank_score"] = 8.0  # sigmoid ~1.0 vs ~0.88: gap > weight
        dyn = MemoryDynamicsConfig()
        ordered = _apply_activation_post_rerank(docs, dyn)
        assert ordered[0]["id"] == "cold"

    def test_zero_weight_keeps_reranked_order(self):
        dyn = MemoryDynamicsConfig(weight=0.0)
        ordered = _apply_activation_post_rerank(self.make_docs(), dyn)
        assert [d["id"] for d in ordered] == ["cold", "hot"]


class FakeVectorStore:
    def __init__(self):
        self.updates = []

    def update(self, vector_id, vector=None, payload=None):
        self.updates.append((vector_id, payload))


class TestReinforceMemory:
    def test_writes_full_merged_payload(self):
        store = FakeVectorStore()
        dyn = MemoryDynamicsConfig()
        payload = {"data": "orion pipeline", "created_at": hours_ago(72), "domain": "infra"}
        assert _reinforce_memory(store, dyn, "mem-1", payload) is True
        vector_id, written = store.updates[0]
        assert vector_id == "mem-1"
        assert written["data"] == "orion pipeline"  # non-dynamics keys preserved
        assert written["domain"] == "infra"
        assert written["access_count"] == 2

    def test_window_suppresses_write(self):
        store = FakeVectorStore()
        dyn = MemoryDynamicsConfig()
        payload = {"reinforced_at": [datetime.now(timezone.utc).isoformat()]}
        assert _reinforce_memory(store, dyn, "mem-1", payload) is False
        assert store.updates == []

    def test_store_failure_never_raises(self):
        class ExplodingStore:
            def update(self, **kwargs):
                raise RuntimeError("boom")

        dyn = MemoryDynamicsConfig()
        assert _reinforce_memory(ExplodingStore(), dyn, "mem-1", {"data": "x"}) is False


class TestConfigSurface:
    def test_defaults(self):
        dyn = MemoryConfig().dynamics
        assert dyn.enabled is True
        assert dyn.decay == 0.5
        assert dyn.weight == 0.15
        assert dyn.reinforcement_window == 3600
        assert dyn.max_timestamps == 10
        assert dyn.reinforce_on_search is False

    def test_disabled_dynamics_resolves_to_none(self):
        config = MemoryConfig(dynamics=MemoryDynamicsConfig(enabled=False))
        assert _dynamics_config(config) is None
        assert _dynamics_config(MemoryConfig()) is not None

    def test_from_dict_config(self):
        config = MemoryConfig(**{"dynamics": {"weight": 0.3, "reinforcement_window": 7200}})
        assert config.dynamics.weight == 0.3
        assert config.dynamics.reinforcement_window == 7200
        assert config.dynamics.enabled is True
