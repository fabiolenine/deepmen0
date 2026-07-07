"""Language resolution helpers for DeepMem0's multilingual retrieval.

MemoryConfig.language takes an ISO 639-1 code (e.g. "pt", "en"). The BM25
sparse encoder (fastembed) expects Snowball stemmer names ("portuguese",
"english"); this module maps between the two. Unknown values pass through
unchanged so full Snowball names are also accepted.
"""

ISO_TO_SNOWBALL = {
    "ar": "arabic",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "hu": "hungarian",
    "it": "italian",
    "nb": "norwegian",
    "no": "norwegian",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "es": "spanish",
    "sv": "swedish",
    "tr": "turkish",
}


def resolve_bm25_language(language: str | None) -> str:
    """Resolve an ISO code (or Snowball name) to a fastembed BM25 language."""
    lang = (language or "en").strip().lower()
    return ISO_TO_SNOWBALL.get(lang, lang)
