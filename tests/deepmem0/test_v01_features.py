"""DeepMem0 v0.1 feature tests — pure units, no live infrastructure needed."""

import sys

import pytest

from mem0.configs.base import MemoryConfig
from mem0.configs.vector_stores.qdrant import QdrantConfig
from mem0.memory.main import _apply_metadata_post_filters, _extract_top_level_entity_params
from mem0.utils.languages import resolve_bm25_language
from mem0.utils.lemmatization import lemmatize_for_bm25, normalize_compounds


class TestLanguageResolution:
    def test_iso_codes_map_to_snowball(self):
        assert resolve_bm25_language("pt") == "portuguese"
        assert resolve_bm25_language("en") == "english"
        assert resolve_bm25_language("es") == "spanish"

    def test_snowball_names_pass_through(self):
        assert resolve_bm25_language("portuguese") == "portuguese"

    def test_none_defaults_to_english(self):
        assert resolve_bm25_language(None) == "english"


class TestCompoundNormalization:
    def test_snake_case_splits(self):
        assert normalize_compounds("feature_store_v2") == "feature store v2"

    def test_kebab_case_splits(self):
        assert normalize_compounds("walk-forward") == "walk forward"

    def test_empty_and_none_are_safe(self):
        assert normalize_compounds("") == ""
        assert normalize_compounds(None) == ""


class TestLemmatizeForBm25:
    def test_non_english_returns_normalized_lowercase(self):
        # PT path must not touch spaCy at all (EN lemmatizer is noise on PT)
        out = lemmatize_for_bm25("Hermes_FX walk-forward CONFIGURAÇÃO", language="pt")
        assert out == "hermes fx walk forward configuração"

    def test_snake_case_survives_english_path(self, monkeypatch):
        # Even with spaCy unavailable, the EN path must keep compound parts
        # (upstream dropped any token containing '_' from the BM25 index).
        import mem0.utils.lemmatization as lemma_mod

        monkeypatch.setattr(
            "mem0.utils.spacy_models.get_nlp_lemma", lambda: None
        )
        out = lemma_mod.lemmatize_for_bm25("the feature_store_v2 rollout", language="en")
        assert "feature" in out and "store" in out and "v2" in out

    @pytest.mark.skipif(
        not pytest.importorskip("spacy").util.is_package("en_core_web_sm"),
        reason="en_core_web_sm not installed",
    )
    def test_english_path_lemmatizes_and_keeps_compounds(self):
        out = lemmatize_for_bm25("the feature_store_v2 was migrating", language="en")
        assert "feature" in out and "store" in out and "v2" in out
        assert "migrate" in out  # lemmatization still active for EN


class TestConfigSurface:
    def test_language_default_is_en(self):
        assert MemoryConfig().language == "en"

    def test_language_accepts_pt(self):
        assert MemoryConfig(language="pt").language == "pt"

    def test_rerank_pool_default(self):
        assert MemoryConfig().rerank_pool == 20

    def test_qdrant_config_accepts_language(self):
        cfg = QdrantConfig(path="/tmp/q", language="pt")
        assert cfg.language == "pt"

    def test_unknown_config_keys_are_ignored(self):
        # apps may pass 'language' to upstream mem0ai too; here it is real,
        # but arbitrary extra keys must not explode either
        cfg = MemoryConfig(**{"language": "pt", "version": "v1.1"})
        assert cfg.language == "pt"


class TestScopeKwargsSugar:
    def test_extracts_only_entity_params(self):
        kwargs = {"user_id": "u1", "run_id": "r1", "threshold": 0.2}
        scope = _extract_top_level_entity_params(kwargs)
        assert scope == {"user_id": "u1", "run_id": "r1"}
        assert kwargs == {"threshold": 0.2}  # popped

    def test_empty_when_absent(self):
        kwargs = {"threshold": 0.2}
        assert _extract_top_level_entity_params(kwargs) == {}


class TestMetadataPostFilters:
    MEMS = [
        {"id": "a", "metadata": {"importance": 0.9, "domain": "trading", "memory_type": "procedural"}},
        {"id": "b", "metadata": {"importance": 0.5, "domain": "data"}},
        {"id": "c", "metadata": {}},
        {"id": "d", "metadata": {"importance": 0.7, "domain": "trading"}},
    ]

    def test_noop_without_criteria(self):
        assert _apply_metadata_post_filters(self.MEMS) is self.MEMS

    def test_min_importance_excludes_unclassified(self):
        out = _apply_metadata_post_filters(self.MEMS, min_importance=0.6)
        assert [m["id"] for m in out] == ["a", "d"]

    def test_domain_filter(self):
        out = _apply_metadata_post_filters(self.MEMS, domain="trading")
        assert [m["id"] for m in out] == ["a", "d"]

    def test_memory_type_filter(self):
        out = _apply_metadata_post_filters(self.MEMS, memory_type="procedural")
        assert [m["id"] for m in out] == ["a"]

    def test_sort_by_importance(self):
        out = _apply_metadata_post_filters(self.MEMS, sort_by_importance=True)
        assert [m["id"] for m in out][:3] == ["a", "d", "b"]


class TestRebrand:
    def test_deepmem0_marker(self):
        import mem0

        assert getattr(mem0, "__deepmem0__", False) is True

    def test_notices_are_disabled(self, capsys):
        from mem0.memory import notices

        assert notices.NOTICES_DISABLED is True
        # must return instantly without touching its arguments
        notices.display_first_run_notice(None, "sync", "search")
        notices.display_temporal_usage_notice(None, "sync", "search", "query", "test")
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_search_rerank_defaults_to_none(self):
        import inspect

        from mem0.memory.main import Memory

        sig = inspect.signature(Memory.search)
        assert sig.parameters["rerank"].default is None
        assert "min_importance" in sig.parameters
        assert "sort_by_importance" in sig.parameters
