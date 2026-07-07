# Deep Mem0 — Roadmap

Deep Mem0 is a fork of [mem0ai/mem0](https://github.com/mem0ai/mem0) **v2.0.7** (Apache-2.0).
Scope: (1) **first-class Portuguese** in addition to English, and (2) **human-memory dynamics**
(frequency + recency) via ACT-R base-level activation.

## Motivation (measured)

On a private, mostly-Portuguese 35-query golden set, the stock OSS hybrid pipeline scored
hit@1 0.60 / MRR 0.63 with 12 recall misses. A per-query audit traced the misses to an
English-only pipeline in four layers:

1. BM25 sparse encoder instantiated without `language` → English stemmer/stopwords.
2. spaCy `en_core_web_sm` lemmatization — noise on Portuguese, and `lemma.isalnum()` silently
   **drops tokens containing `_`** (snake_case identifiers vanish from the keyword index).
3. English-centric default dense embedder (targets ranked 95–351; a multilingual embedder moved
   the same targets to ranks 1–3).
4. Bilingual zh/en cross-encoder reranker, weak on Portuguese.

Additionally, the OSS fusion builds its candidate set **from the dense retriever only** (BM25 can
only boost, never introduce candidates), and `Memory.search` defaults to `rerank=False`, so a
configured reranker silently never runs for clients that don't opt in per call.

Fixing all of the above (multilingual embedder + language-aware BM25 + multilingual reranker +
reranker over-fetch) measured **hit@1 0.886 / MRR 0.929 / 1 miss** on the same set. Those changes,
currently validated as runtime patches, get baked into this fork as first-class code.

Note: upstream's `reference_date` / `decay` parameters are paid-platform stubs that raise in OSS —
human-memory dynamics here is greenfield, no collision.

## Phase 0 — Repo hygiene (gate before anything goes public)

- Clean history: fork re-rooted at upstream v2.0.7, single initial commit, project-dedicated git
  identity. No personal data in code, fixtures or history.
- `LICENSE` (Apache-2.0, from upstream) + `NOTICE` (attribution to Mem0).
- Synthetic PT+EN evaluation corpus (`eval/corpus_synthetic.json`) replacing the private golden
  set; harness (`eval/`) reproducible against any Qdrant + Ollama.
- README with identity, scope, quickstart, measured motivation.

## Phase 1 — Portuguese first-class (v0.1)

- `language: str = "en"` field in `MemoryConfig` (`mem0/configs/base.py`), wired through:
  - **BM25**: `SparseTextEmbedding(model_name, language=cfg)` in `mem0/vector_stores/qdrant.py`,
    plus tokenization normalization (`_`/`-` → space, lowercase) applied **identically to documents
    and queries** (diverging preps silently zero BM25 — token IDs stop matching).
  - **Lemmatization**: language-parameterized spaCy model (`pt_core_news_sm` for PT) in
    `mem0/utils/spacy_models.py`, with fail-safe fallback baked in (no `sys.exit` on missing model —
    degrade to raw text instead of crashing the server per query).
  - **Extraction**: enable the dormant `use_input_language=True` of
    `generate_additive_extraction_prompt` so facts are extracted in the input language.
- Multilingual defaults: dense embedder `bge-m3` (1024d), reranker `BAAI/bge-reranker-v2-m3`
  (CPU, 0 VRAM), both provider-agnostic and overridable.
- Reranker **over-fetch**: fetch `max(2*limit, pool)` fused candidates, rerank, cut back to
  `limit` (measured: hit@1 0.857→0.886, one extra recall). `rerank` defaultable via config.
- Metadata-aware search filters (`min_importance` / `domain` / `memory_type` /
  `sort_by_importance`) as optional `search` parameters.
- **Re-index migration tool** (`tools/migrate_embedder.py`): changing `language` or embedder
  changes token IDs / dimensions — existing collections must be re-embedded into a new collection
  (old kept as rollback). Idempotent, resumable.
- Proof: harness numbers on the synthetic corpus, PT vs EN pipeline, published in the repo.

## Phase 2 — Human-memory dynamics (v0.2)

Model: ACT-R **base-level activation** — `B_i = ln( Σ_j Δt_j^{-d} )` over each memory's
reinforcement timestamps (`d ≈ 0.5`), a single term capturing both frequency (how many
reinforcements) and recency (how recent). Used as a ranking signal.

**Activation is derived, not stored.** What persists is the event history (timestamps); the
activation value is computed lazily **at query time**, only for the over-fetched candidate pool
(~20 items — pure arithmetic, microseconds). There is no batch decay job and no persisted weight
to refresh: as wall-clock time passes, every `Δt` grows and activation falls on its own, with
zero writes. A stored weight would be stale the moment it was written and would require periodic
full-collection rescans; a lazy value is exact at the only moment it matters — ranking.

Writes happen only at **reinforcement triggers**:

- **T1 — re-encounter on add** (synchronous): upstream dedups an already-known fact by hash as a
  silent no-op; that exact spot becomes the hook — `access_count += 1`, append to `reinforced_at`
  via `vector_store.update`. Strongest signal ("the user said this again").
- **T2 — LLM-decided UPDATE** of an existing memory: the fact evolved, therefore it is alive —
  counts as a reinforcement on the same chain.
- **T3 — hit on search** (optional, off by default): a memory returned in the **final top-k**
  (not the over-fetch pool) gets an async fire-and-forget bump — never blocking the hot path,
  best-effort by design.
- A **reinforcement dedup window** (same fact hash within a configurable interval counts once)
  protects against client retries — e.g. an MCP client that times out and re-sends an `add` must
  not double-count.

- Payload fields: `access_count`, `reinforced_at` (timestamps), `last_accessed`. Bounded growth:
  keep only the most recent **K** timestamps (default ~10) plus the total count; older
  reinforcements fold into the standard ACT-R hybrid approximation (Petrov 2006) so payload size
  stays O(K) regardless of memory age.
- **Reinforcement on re-encounter**: upstream dedups identical facts by hash as a silent no-op;
  that exact spot becomes the reinforcement hook (increment + timestamp via `vector_store.update`).
- **Reinforcement on access** (optional): bump on search hit — async/best-effort only, never
  blocking the hot path.
- **Activation in scoring**: `score_and_rank` (pure, stable function) gains an optional
  `activation_boosts` dict alongside the existing `bm25_scores`/`entity_boosts`, weight
  configurable. Alternative integration as a pluggable `BaseReranker` for zero core impact.
- **Forgetting is non-destructive**: below-threshold activation deprioritizes or archives,
  never deletes by default.
- Proof: temporal scenario in the harness — reinforced facts must outrank equally-similar
  unreinforced competitors, with no regression on the non-temporal baseline.

## Phase 3 — Semantic temporality (v0.3)

Where v0.2 makes the **usage** timeline govern ranking, v0.3 makes the **content** timeline a
first-class dimension:

- **Fact validity & supersedence**: when a fact changes ("the embedder *was* X, is *now* Y"),
  record an explicit supersedence chain (previous value, valid-from/valid-to) instead of two
  unrelated memories. Foundation: the existing history table (ADD/UPDATE/DELETE per memory) plus
  the v0.2 reinforcement hook, which already intercepts the exact dedup/update point.
- **As-of queries**: "what did I know last March?" — search with a temporal anchor that filters
  or re-scores candidates by validity interval.
- **Event-time anchoring**: extract dates mentioned in the content so facts can be anchored to
  when they *happened*, not when they were stored (upstream's `reference_date` is a paid-platform
  stub — greenfield here, same as `decay`).
- Non-goal: destructive rewrites. Superseded facts remain queryable as history.

## Out of core (companion repo)

MCP server, LLM-based metadata classifier and observability emitters live in a companion project
(they bind to specific local infra); the fork core stays a clean library.

## Definition of Done

- **v0.1**: `language="pt"` end-to-end; PT gains measured and versioned on the synthetic corpus;
  zero personal data in the repo; LICENSE + NOTICE; README.
- **v0.2**: activation influencing ranking, measured on the temporal scenario without regressing
  the non-temporal baseline; reinforcement write-back idempotent and concurrency-safe.
- **v0.3**: supersedence chain recorded on update; as-of search anchor working end-to-end;
  superseded facts retrievable as history, never silently lost.
