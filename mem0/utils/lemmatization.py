"""
BM25 lemmatization for consistent keyword matching.

Uses spaCy's lemmatizer for better handling of:
- Verb forms: attending/attends/attended -> attend
- Comparatives/superlatives: older/oldest -> old
- Plurals: memories -> memory
- Avoids over-stemming: organization != organize

Also includes original -ing forms alongside lemmas to handle cases
where spaCy's context-dependent lemmatization produces inconsistent
results (e.g., "meeting" as noun vs verb -> different lemmas).
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_COMPOUND_SEP_RE = re.compile(r"[_\-]+")


def normalize_compounds(text: str) -> str:
    """Split snake_case/kebab-case compounds into plain words.

    DeepMem0 fix: upstream fed compounds straight into spaCy, whose
    ``lemma.isalnum()`` filter silently DROPPED any token containing ``_`` —
    identifiers like ``feature_store_v2`` vanished from the BM25 index
    entirely. Splitting first preserves every part, for every language, and
    must be applied identically to documents and queries.
    """
    return _COMPOUND_SEP_RE.sub(" ", text or "")


def lemmatize_for_bm25(text: str, language: str = "en") -> str:
    """Normalize text for BM25 matching.

    English keeps the upstream spaCy lemmatization (on compound-normalized
    text). Other languages return lowercased normalized text and delegate
    morphology to the BM25 encoder's language-aware Snowball stemmer — the
    pipeline DeepMem0 validated on a Portuguese corpus (the English
    lemmatizer is noise, or worse, on non-English text).

    Falls back to the normalized text if spaCy is unavailable.
    """
    text = normalize_compounds(text)

    if (language or "en").strip().lower() not in ("en", "english"):
        return text.lower()

    from mem0.utils.spacy_models import get_nlp_lemma

    nlp = get_nlp_lemma()
    if nlp is None:
        return text

    doc = nlp(text.lower())
    tokens = []

    for token in doc:
        if token.is_punct or token.is_stop:
            continue

        lemma = token.lemma_
        if lemma.isalnum():
            tokens.append(lemma)

        # Also add original if it ends in -ing and differs from lemma.
        # This handles noun/verb ambiguity (meeting/meet, attending/attend).
        if token.text.endswith("ing") and token.text != lemma and token.text.isalnum():
            tokens.append(token.text)

    return " ".join(tokens)
