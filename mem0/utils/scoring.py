"""
Scoring utilities for hybrid retrieval.

Provides:
- **BM25 normalization**: Sigmoid normalization of raw BM25 scores to [0, 1].
- **BM25 parameter selection**: Query-length-adaptive sigmoid parameters.
- **Additive scoring**: Combined scoring with semantic + BM25 + entity boost.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def get_bm25_params(query: str, *, lemmatized: Optional[str] = None) -> tuple:
    """Get BM25 sigmoid parameters based on query length.

    Longer queries tend to have higher raw BM25 scores, so we adjust
    the sigmoid midpoint and steepness accordingly.

    Returns:
        (midpoint, steepness) for sigmoid normalization.
    """
    if lemmatized is None:
        from mem0.utils.lemmatization import lemmatize_for_bm25

        lemmatized = lemmatize_for_bm25(query)
    num_terms = len(lemmatized.split()) if lemmatized else 1

    if num_terms <= 3:
        return 5.0, 0.7
    elif num_terms <= 6:
        return 7.0, 0.6
    elif num_terms <= 9:
        return 9.0, 0.5
    elif num_terms <= 15:
        return 10.0, 0.5
    else:
        return 12.0, 0.5


def normalize_bm25(raw_score: float, midpoint: float, steepness: float) -> float:
    """Normalize BM25 score to [0, 1] using logistic sigmoid.

    Args:
        raw_score: Raw BM25 score (unbounded, typically 0-20+).
        midpoint: Score at which sigmoid outputs 0.5.
        steepness: Controls how quickly sigmoid transitions.

    Returns:
        Normalized score in range [0, 1].
    """
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))


ENTITY_BOOST_WEIGHT = 0.5
ACTIVATION_BOOST_WEIGHT = 0.15
EVENT_BOOST_WEIGHT = 0.15


def score_and_rank(
    semantic_results: List[Dict[str, Any]],
    bm25_scores: Dict[str, float],
    entity_boosts: Dict[str, float],
    threshold: float,
    top_k: int,
    explain: bool = False,
    activation_boosts: Optional[Dict[str, float]] = None,
    activation_weight: float = ACTIVATION_BOOST_WEIGHT,
    penalties: Optional[Dict[str, float]] = None,
    event_boosts: Optional[Dict[str, float]] = None,
    event_weight: float = EVENT_BOOST_WEIGHT,
) -> List[Dict[str, Any]]:
    """Score candidates additively and return top-k results.

    For each candidate:
        semantic_score is taken from the result's score field.
        combined = (semantic + bm25 + entity_boost + activation + event) / max_possible
        combined = max(combined - penalty, 0.0)   # DeepMem0 v0.3

    Threshold gates the semantic score BEFORE combining -- candidates
    below the threshold are excluded even if BM25/entity would boost them.
    Penalties act on the FINAL normalized score, after the gate — they demote,
    they can never exclude.

    The divisor adapts based on which signals are active:
        - Semantic only: max_possible = 1.0
        - Semantic + BM25: max_possible = 2.0
        - Semantic + BM25 + entity: max_possible = 2.5
        - Semantic + entity (no BM25): max_possible = 1.5
        - + activation (DeepMem0 v0.2): max_possible += activation_weight
        - + event proximity (DeepMem0 v0.6): max_possible += event_weight

    Args:
        semantic_results: Candidate memories from vector search.
        bm25_scores: Normalized keyword scores keyed by memory ID.
        entity_boosts: Entity-link boosts keyed by memory ID.
        threshold: Minimum semantic score required before hybrid scoring.
        top_k: Maximum number of results to return.
        explain: Include score_details in each result when true.
        activation_boosts: DeepMem0 v0.2 — ACT-R activation boosts in [0, 1]
            keyed by memory ID. Memories absent from the dict are neutral.
        activation_weight: Weight of the activation term.
        penalties: DeepMem0 v0.3 — score penalties (e.g. superseded facts)
            keyed by memory ID, subtracted from the final normalized score.
        event_boosts: DeepMem0 v0.6 — event-time proximity boosts in [0, 1]
            keyed by memory ID (a query names a date, matching event_dates score
            higher). Memories absent from the dict are neutral. An empty/None dict
            leaves the divisor unchanged, so a query with no temporal anchor is a
            no-op here.
        event_weight: Weight of the event-proximity term. 0 disables it.

    Returns:
        List of scored result dicts sorted by combined score descending.
    """
    has_bm25 = bool(bm25_scores)
    has_entity = bool(entity_boosts)
    has_activation = bool(activation_boosts) and activation_weight > 0
    has_event = bool(event_boosts) and event_weight > 0

    max_possible = 1.0
    if has_bm25:
        max_possible += 1.0
    if has_entity:
        max_possible += ENTITY_BOOST_WEIGHT
    if has_activation:
        max_possible += activation_weight
    if has_event:
        max_possible += event_weight

    scored: List[Dict[str, Any]] = []

    for result in semantic_results:
        mem_id = result.get("id")
        if mem_id is None:
            continue

        semantic_score = result.get("score") or 0.0
        if semantic_score < threshold:
            continue

        mem_id_str = str(mem_id)
        bm25_score = bm25_scores.get(mem_id_str, 0.0)
        entity_boost = entity_boosts.get(mem_id_str, 0.0)
        activation = (activation_boosts.get(mem_id_str, 0.0) * activation_weight) if has_activation else 0.0
        event = (event_boosts.get(mem_id_str, 0.0) * event_weight) if has_event else 0.0

        raw_combined = semantic_score + bm25_score + entity_boost + activation + event
        combined = min(raw_combined / max_possible, 1.0)
        penalty = (penalties or {}).get(mem_id_str, 0.0)
        if penalty:
            combined = max(combined - penalty, 0.0)

        scored_result = {
            "id": mem_id_str,
            "score": combined,
            "payload": result.get("payload"),
        }
        if explain:
            scored_result["score_details"] = {
                "semantic_score": semantic_score,
                "bm25_score": bm25_score,
                "entity_boost": entity_boost,
                "activation_boost": activation,
                "event_boost": event,
                "superseded_penalty": penalty,
                "raw_score": raw_combined,
                "max_possible_score": max_possible,
                "final_score": combined,
                "threshold": threshold,
            }
        scored.append(scored_result)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
