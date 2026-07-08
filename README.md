# Deep Mem0

**Memory for AI agents that speaks Portuguese and remembers like a human — facts on an evolving timeline.**

An open-source project by **Alpha Quant AI**.

Deep Mem0 is an open-source fork of [Mem0](https://github.com/mem0ai/mem0) (v2.0.7, Apache-2.0)
focused on two things the upstream OSS core does not do today:

1. **First-class multilingual retrieval — Portuguese first.** Mem0's hybrid search pipeline is
   English end-to-end (English BM25 stemmer/stopwords, English spaCy lemmatization, English-centric
   default embedder). On a Portuguese corpus this silently destroys recall — e.g. the English
   lemmatizer *drops* snake_case tokens entirely, so a memory mentioning `feature_store_v2` becomes
   unfindable by keyword. Deep Mem0 makes language a **config field** and wires it through every
   layer: BM25 sparse encoding, lemmatization/normalization, extraction prompt, embedder and
   reranker defaults.

2. **Human-like memorization (ACT-R).** Deep Mem0 memorizes the way people do: every fact lives
   on an **evolving timeline** — each time it is re-encountered or used it gets *reinforced* and
   surfaces more readily, so what matters to you *now* is what you recall first. This is modeled
   with the ACT-R **base-level activation** equation — `B_i = ln(Σ Δt_j^{-d})` over each fact's
   reinforcement history — used as a ranking signal: frequency and recency of use shape relevance.
   Nothing is ever silently deleted; stale facts simply yield the spotlight to living ones.
   In upstream OSS these concepts exist only as paid-platform stubs; here they are real, local
   and open.

## Why (measured motivation)

This fork was born from running a self-hosted Mem0 on a real, mostly-Portuguese memory corpus.
Fixing the language pipeline and the ranking produced, on a private 35-query golden set:

| configuration | hit@1 | MRR | misses@10 |
|---|---|---|---|
| stock hybrid pipeline (EN, base reranker) | 0.60 | 0.63 | 12/35 |
| multilingual rebuild (bge-m3 + PT BM25 + v2-m3 reranker + over-fetch) | **0.886** | **0.929** | **1/35** |

That private golden set cannot be published (it is made of real personal memories), so this repo
ships a **synthetic PT+EN corpus** (`eval/corpus_synthetic.json`) exercising the same failure
modes — Portuguese morphology, snake_case compounds, cross-lingual queries, paraphrase without
lexical overlap, near-duplicates — plus the evaluation harness to reproduce the numbers on it.

## Deep Mem0 vs. Mem0 OSS — what actually changes

Portuguese scores are the headline, but not the whole story. Every row below was verified against
the upstream 2.0.7 source; v0.1 items are already validated in a production deployment (as runtime
patches) and are being baked into this fork as first-class code.

| Capability | Mem0 OSS 2.0.7 | Deep Mem0 |
|---|---|---|
| Retrieval language | English hardcoded in three layers (BM25 stemmer/stopwords, spaCy `en_core_web_sm`, EN-centric embedder default) | `language` config field wired through BM25, lemmatization, extraction prompt and model defaults *(v0.1)* |
| snake_case identifiers | Silently **dropped from the BM25 index** by the EN lemmatizer (`lemma.isalnum()` rejects `_`) — `feature_store_v2` becomes unfindable by keyword | Preserved via identical doc+query normalization (`_`/`-` → space) *(v0.1)* |
| Missing spaCy model | `spacy.cli.download()` calls `sys.exit(1)` → **crashes the whole process on every search** | Graceful degradation to raw text; flagged, never fatal *(v0.1)* |
| Reranking | Interface exists, but `Memory.search` defaults `rerank=False` — a configured reranker **silently never runs** unless every caller opts in | Enabled by config default; multilingual cross-encoder on CPU (0 VRAM) *(v0.1)* |
| Reranker pool | Reranks only the fused top-k — targets buried by fusion are unrecoverable | **Over-fetch** `max(2·limit, 20)` then cut back — measured +0.03 hit@1 / −1 miss *(v0.1)* |
| Hybrid fusion | Candidate set built **from the dense retriever only**; BM25 can boost but never introduce a candidate | Mitigated today by over-fetch + rerank; true candidate union on the roadmap |
| Frequency / recency (human memory) | **Paid platform only** — `decay` and `reference_date` raise errors in OSS | ACT-R base-level activation as a ranking signal, reinforcement timeline per memory, open source *(v0.2, shipped)* |
| Fact evolution over time | Updated facts overwrite in place or coexist as unrelated near-duplicates; `reference_date` is a paid stub | Supersession chains detected at extraction ("was X, now Y" links old → new), superseded facts demoted-never-deleted, `as_of` time-travel search, optional `event_date` per fact *(v0.3, shipped)* |
| Deferred / queued ingestion | `add()` blocks the caller for the full LLM extraction; processing time silently becomes the fact's record time | Record-time contracts for async pipelines: a caller-supplied `created_at` (the submission time) is honored end-to-end, and supersession direction arbitrates by record time — a queued fact that lost the race to a fresher write is **born superseded**, never demoting newer truth *(v0.4, shipped)* |
| Search-time metadata filters | Scope and payload filters | Plus `min_importance`, `domain`, `memory_type`, `sort_by_importance` *(v0.1)* |
| Ops tooling | — | Re-index migration (embedder/language cutover), BM25 backfill, eval harness + synthetic PT/EN corpus |

## Benchmark on the synthetic corpus (reproducible)

Measured with this repo's harness on the shipped synthetic corpus (44 memories, 28 queries,
Portuguese-majority with English and cross-lingual cases), all self-hosted on the same machine:

| pipeline | hit@1 | hit@5 | MRR | misses@10 | p50 |
|---|---|---|---|---|---|
| Mem0 OSS-style (EN BM25 + EN lemmatizer + EN-centric embedder, fused) | 0.750 | 0.964 | 0.833 | 1/28 | 0.2 s |
| Deep Mem0 — language fix only (bge-m3 + PT BM25, fused) | 0.857 | **1.000** | 0.906 | **0/28** | 1.2 s |
| Deep Mem0 — full (+ multilingual reranker, over-fetch 20) | **0.964** | 0.964 | **0.968** | **0/28** | 5.0 s |

Honest notes:
- The language fix **alone** already eliminates all misses at fusion-level latency; the CPU
  cross-encoder buys the last hit@1 points at a latency cost — it is a dial, not a requirement.
- The stand-in for upstream's embedder is `nomic-embed-text` (local, English-centric), since the
  actual OSS default (OpenAI) requires an external API; the BM25 and lemmatization paths are
  exactly the stock code.
- This corpus is small (44 points). On a real 498-point production corpus the same comparison was
  **hit@1 0.60 → 0.886** — the gap widens as the corpus grows and distractors multiply.
- Latency measured on 2015-era Xeons with the reranker on CPU; reproduce with
  `eval/seed_corpus.py` + `eval/eval_retrieval.py`.

## Status

**v0.4 shipped** — ingest-time decoupling. Queueing an add separates *when a fact was said* from
*when the pipeline processed it*, and the core now keeps those honest: `metadata["created_at"]`
supplied by the caller (canonically the submission time) is preserved through both the inference
and raw paths — `as_of` anchors and history keep working across queue delays — and supersession
marking arbitrates by record time: when the extraction LLM links an arriving fact to an existing
one whose `created_at` is *newer*, the direction inverts and the newcomer lands already marked
`superseded_by` the fresher fact (born superseded), so an out-of-order arrival can never demote
newer truth. Forward marking, first-marking-wins immutability and the SUPERSEDED history event are
unchanged; mixed timestamps build the chain in a single pass. This is the core half of an
asynchronous ingestion pipeline — the queue/worker themselves belong to whatever serves the API
(an MCP server, a job runner), which only needs to stamp `created_at` at enqueue time.

**v0.3** — semantic temporality is live. Facts now live on a *content* timeline: when a
new fact **replaces** an old one ("Atlas used MySQL" → "Atlas uses PostgreSQL"), the extraction
LLM — in the same call that already runs on every add — marks the supersession; the old memory
gains `superseded_by`/`superseded_at`, the new one records `supersedes`, and the chain lands in
the history log as a `SUPERSEDED` event. Superseded facts are **demoted, never deleted or
excluded**: a configurable ranking penalty (default 0.2, applied both at fusion and after the
reranker, in a single sort combined with the ACT-R activation) makes the current fact win while
the old one stays reachable. `search(..., as_of="2026-03-15")` restores the world as it was —
memories created after the anchor are filtered out (a Qdrant datetime index backs the filter) and
a memory superseded only *after* the anchor carries no penalty there. Extracted facts may also
carry an `event_date` (when the text clearly anchors *when* something happened — event-time,
distinct from record-time). Proof lives in `eval/eval_supersedence.py`: current fact outranks its
superseded ancestor 3/3 on both ranking paths, the old fact stays reachable by its own phrasing
3/3, the as-of anchor restores it 3/3, and an untouched corpus ranks identically with the feature
on or off.

**v0.2** — human-memory dynamics. Memories accrue a reinforcement timeline
(`reinforced_at` + `access_count`); re-encountering a fact on `add` (upstream's silent hash-dedup
no-op became the hook), updating it, or — opt-in — retrieving it reinforces that timeline, at most
once per memory per window (default 1 h; absorbs client retries and approximates the ACT-R spacing
effect). At query time the ACT-R base-level activation `B_i = ln(Σ Δt_j^{-d})` is computed lazily
over the candidate pool and blended into ranking twice: as an additive term in hybrid fusion and
as a tie-breaker on top of the cross-encoder after reranking. No batch decay jobs, no stored
weights going stale — time passing lowers activation by itself. **A memory is neutral until its
first reinforcement** — creation alone does not put it on the timeline — so activation measures
re-encounters, not first presentations, and neither the existing corpus nor fresh adds are
repriced until they are actually re-used (measured on a production corpus: birth-time activation
let long, freshly written notes steal top-1 from unrelated queries — hit@1 dropped 0.857 → 0.714
until the neutral-at-creation rule restored it). Proof lives in `eval/eval_temporal.py`:
reinforced twins outrank their equally-similar unreinforced siblings 6/6 (control without
dynamics: 2/6), a decisively more relevant match is *not* overturned by reinforcement, and on a
fresh corpus dynamics ON == OFF.

**v0.1** — first-class multilingual retrieval: `language` in `MemoryConfig` (wired through BM25,
lemmatization and the extraction prompt), snake_case-safe BM25 indexing, fail-safe spaCy
loading, reranker on-by-default with an over-fetched pool, metadata-aware search filters.
Validated in a self-hosted production deployment before release (numbers below).

Next, per `docs/roadmap.md`: update versioning (in-place updates visible to as-of anchors),
event-date-aware ranking, and richer temporal queries over the supersession graph.

## API

```python
from mem0 import Memory

memory = Memory.from_config({
    "language": "pt",                       # v0.1: wired through BM25, lemmatizer, prompts
    "embedder": {"provider": "ollama", "config": {"model": "bge-m3", "embedding_dims": 1024}},
    "reranker": {"provider": "sentence_transformer",
                 "config": {"model": "BAAI/bge-reranker-v2-m3", "device": "cpu"}},
    "vector_store": {"provider": "qdrant", "config": {"collection_name": "memories"}},
    "dynamics": {                           # v0.2: human-memory dynamics (all optional)
        "enabled": True,                    # on by default
        "weight": 0.15,                     # activation's share of the ranking
        "reinforcement_window": 3600,       # >=1 reinforcement/memory/hour, all triggers
        "reinforce_on_search": False,       # T3: opt-in, async, never blocks the hot path
    },
    "temporality": {                        # v0.3: semantic temporality (all optional)
        "enabled": True,                    # on by default
        "superseded_penalty": 0.2,          # demotion for replaced facts (never exclusion)
        "extract_event_date": True,         # optional event_date per extracted fact
    },
})

memory.add("O deploy do auth_service_v3 é feito por canary de 5% durante 24h.", user_id="demo")
memory.search("como fazemos deploy do serviço de autenticação?", user_id="demo", rerank=True)

# v0.3: a later add that contradicts a stored fact supersedes it automatically
# (detected by the extraction LLM); the old fact is demoted, never deleted.
memory.add("Mudamos: o deploy do auth_service_v3 agora é blue-green, sem canary.", user_id="demo")

# time travel: what did we know / what held on that date?
memory.search("como fazemos deploy?", user_id="demo", as_of="2026-03-15")

# audit trail of a memory (ADD / UPDATE / SUPERSEDED / DELETE)
memory.history("<memory-id>")
```

## Evaluation harness

```bash
# 1. seed the synthetic corpus into a local Qdrant (throwaway collection)
python eval/seed_corpus.py --collection deepmem0_demo

# 2. per-retriever sanity: rank of each expected target in dense and BM25, no fusion
python eval/eval_retrieval.py audit

# 3. end-to-end metrics against a running mem0-compatible MCP server
python eval/eval_retrieval.py run --label my_run --mcp http://localhost:8081/mcp
```

## Use it as a Claude skill (memory that fills itself)

`skill/deepmem0-memory.skill` is a Claude Skill that makes Claude use these memory
tools *proactively and selectively* — pulling context at the start of a technical
chat and saving only durable facts, under a strict save/discard filter. In Claude
Code its own trigger is enough; in Claude Desktop (chat), pair it with a short
always-on custom instruction so it fires every relevant turn — progressive
disclosure: the instruction fires, the skill supplies the *how*. Setup and the
copy-paste rubric live in [`skill/README.md`](skill/README.md).

## Relationship to upstream

Deep Mem0 tracks Mem0 v2.0.7 as its base and stays intentionally close to upstream design
(same config surface, same vector-store contract, additive scoring changes in pure functions).
It is **not** affiliated with Mem0.ai. Attribution in `NOTICE`; license Apache-2.0 in `LICENSE`.

---

## Português

**Memória para agentes de IA que fala português e memoriza como um humano: fatos numa linha do
tempo evolutiva.**

O Deep Mem0 é um fork open-source do [Mem0](https://github.com/mem0ai/mem0) com dois focos que o
core OSS não cobre: (1) **retrieval multilíngue de primeira classe, começando pelo português** —
o pipeline híbrido do Mem0 é inglês de ponta a ponta (stemmer/stopwords do BM25, lematização
spaCy, embedder default), o que destrói o recall em corpus PT de forma silenciosa; aqui o idioma
vira **campo de configuração** propagado por todas as camadas; e (2) **memorização como a humana**
— cada fato vive numa linha do tempo que evolui com o uso: reencontrar ou usar um fato o
**reforça** (frequência + recência via ativação base-level do **ACT-R**), então o que importa
agora é o que vem à tona primeiro. Nada é apagado silenciosamente: fatos parados apenas cedem o
palco aos que estão vivos.

A motivação é medida: num golden set privado de 35 consultas majoritariamente em português,
corrigir o pipeline de idioma levou hit@1 de 0.60 para **0.886** e MRR de 0.63 para **0.929**
(misses de 12 para 1). Como o golden set real não pode ser publicado, este repositório inclui um
**corpus sintético PT+EN** com os mesmos modos de falha e o harness para reproduzir os números.

E o score em português é a manchete, não a história toda: a tabela **"Deep Mem0 vs. Mem0 OSS"**
acima resume tudo que muda — snake_case preservado no índice BM25, servidor que não cai quando o
modelo spaCy falta, reranker que realmente roda por default (no OSS o `rerank=False` faz um
reranker configurado nunca executar), over-fetch do pool de rerank, filtros de metadata na busca,
ferramentas de migração/avaliação e — o diferencial — frequência/recência de memória humana em
código aberto, algo que no upstream existe só na plataforma paga.

Roadmap: **v0.1** português de primeira classe; **v0.2** dinâmica de memória humana (ACT-R);
**v0.3** temporalidade semântica (validade e supersedência de fatos, consultas as-of).
Detalhes em `docs/roadmap.md`. Licença Apache-2.0; atribuição ao Mem0 em `NOTICE`.

**Usar como skill do Claude:** `skill/deepmem0-memory.skill` faz o Claude usar essas
ferramentas de memória de forma proativa e seletiva — recupera contexto no início de
uma conversa técnica e salva só o que é durável, com filtro rigoroso. No Claude Code o
gatilho da própria skill basta; no Claude Desktop (chat), combine-a com uma instrução
curta sempre-ativa (progressive disclosure: a instrução dispara, a skill fornece o
*como*). Setup e a rubrica para copiar estão em [`skill/README.md`](skill/README.md).

Um projeto open-source da **Alpha Quant AI**.
