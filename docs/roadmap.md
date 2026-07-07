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

## Phase 2 — Human-memory dynamics (v0.2) — SHIPPED

Measured on the temporal scenario (`eval/eval_temporal.py`, twin near-duplicate facts where one
twin carries a lived reinforcement timeline): reinforced twin outranks its equally-similar
sibling **6/6** with dynamics on (control without dynamics: 2/6 fusion, 3/6 reranked); a
decisively more relevant match is **not** overturned by reinforcement; and on a fresh corpus
(equal timelines) dynamics ON == OFF on every query — enabling the feature never reprices an
existing corpus.

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
- A **reinforcement window** (`reinforcement_window`, default **1 hour**, `0` disables): after a
  memory is reinforced, further re-encounters or hits on the *same* memory within the window have
  **no reinforcement effect** — at most one reinforcement per memory per window, across all
  triggers (a content UPDATE still applies; only the reinforcement bookkeeping is suppressed).
  This absorbs client retries (an MCP client that times out and re-sends an `add` must not
  double-count) and approximates the ACT-R **spacing effect**: massed repetition within the hour
  adds nothing; spaced repetition does.

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

## Phase 3 — Semantic temporality (v0.3) — SHIPPED

Where v0.2 makes the **usage** timeline govern ranking, v0.3 makes the **content** timeline a
first-class dimension.

Measured on the supersedence scenario (`eval/eval_supersedence.py`, versioned fact pairs marked
through the real production write path): the current fact outranks its superseded ancestor **3/3**
on both ranking paths (fusion and reranked); the superseded fact stays reachable by its own
phrasing **3/3** (demoted, never excluded); an `as_of` anchor between creation and supersession
restores the old fact **3/3** (the replacement is cut by the record-time filter and the old fact
carries no penalty at the anchor); and an untouched corpus ranks identically with temporality on
or off.

How it shipped:

- **Supersession detected at extraction** — the same LLM call that already runs on every add
  (its prompt already taught "Contradiction"/"Updated preference" for linking) now emits an
  optional `supersedes: [<existing-memory-index>]`, resolved through the previously dead
  `uuid_mapping` (hallucinated indices discarded). Explicit `update()` keeps its own chain in the
  history log. Marking is deferred until the new fact is persisted, writes the full merged
  payload, never re-marks (first marking wins → chains A→B→C emerge), and lands a `SUPERSEDED`
  event in the history table (free-text event column — zero schema migration).
- **Demotion, not deletion**: `superseded_penalty` (default 0.2 on the normalized score) applied
  at fusion (`score_and_rank` gained a generic `penalties` dict) and after the reranker — in a
  single sort combined with the ACT-R activation, so neither adjustment discards the other.
- **`as_of` anchors**: record-time filter `created_at <= anchor` injected into the Qdrant filter
  for both dense and keyword legs (auto-detected `DatetimeRange`; plain dates normalize to
  end-of-day), plus anchor-aware penalty waiving. New `datetime` payload indexes on
  `created_at`/`superseded_at` (created online on existing collections at startup).
- **`event_date`** (optional, extraction-time): ISO date when the text clearly anchors *when* a
  fact happened — event-time recorded and exposed; ranking use is future work.
- Known limitation: in-place `update()` content is visible to earlier anchors (update versioning
  is future work; the history table keeps the audit trail).

## Out of core (companion repo)

MCP server, LLM-based metadata classifier and observability emitters live in a companion project
(they bind to specific local infra); the fork core stays a clean library.

## Definition of Done

- **v0.1**: `language="pt"` end-to-end; PT gains measured and versioned on the synthetic corpus;
  zero personal data in the repo; LICENSE + NOTICE; README.
- **v0.2**: activation influencing ranking, measured on the temporal scenario without regressing
  the non-temporal baseline; reinforcement write-back idempotent and concurrency-safe. **DONE**
  (6/6 reinforced-twin wins on both ranking paths; decisive-gap guard holds; fresh-corpus
  ON == OFF; write-back full-payload merge + windowed, failures never raise into the hot path).
- **v0.3**: supersedence chain recorded on update; as-of search anchor working end-to-end;
  superseded facts retrievable as history, never silently lost. **DONE** (3/3 on all supersedence
  scenario checks on both ranking paths; untouched corpus unchanged; marking write path
  full-merge + first-marking-wins, failures never reach the hot path).
