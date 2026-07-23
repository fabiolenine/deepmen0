"""
DeepMem0 v0.3 — semantic temporality (fact supersession + as-of anchors).

Where v0.2 puts the *usage* timeline into ranking (ACT-R activation), v0.3
makes the *content* timeline a first-class dimension. When a new fact REPLACES
an old one ("the embedder WAS X, is NOW Y"), the old memory is marked
superseded instead of lingering as an equal competitor:

- the OLD memory gains ``superseded_by`` (UUID of the replacing memory) and
  ``superseded_at`` (record-time of the supersession, immutable — the first
  marking wins, so chains A -> B -> C emerge naturally);
- the NEW memory gains ``supersedes`` (list of replaced UUIDs) for reverse
  auditing;
- extracted facts may also carry ``event_date`` (ISO date) when the text
  clearly anchors WHEN the fact happened — event-time, distinct from the
  record-time ``created_at``.

Nothing is destructive: a superseded memory is never deleted or excluded from
search — it is *demoted* by a configurable ranking penalty, and an ``as_of``
anchor restores the world as it was ("what did I know on that date?"):
memories created after the anchor are filtered out, and a memory superseded
only AFTER the anchor was still current then, so its penalty is waived.

All parsing here is deliberately fail-open for LLM-produced fields (a bad
``supersedes`` index or ``event_date`` is discarded, never raised) and
fail-fast for caller-provided parameters (an invalid ``as_of`` raises).
"""

from __future__ import annotations

import calendar
import logging
import re
from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional, Tuple

from mem0.utils.dynamics import _parse_ts

logger = logging.getLogger(__name__)

FIELD_SUPERSEDED_BY = "superseded_by"
FIELD_SUPERSEDED_AT = "superseded_at"
FIELD_SUPERSEDES = "supersedes"
FIELD_EVENT_DATE = "event_date"

TEMPORALITY_FIELDS = (FIELD_SUPERSEDED_BY, FIELD_SUPERSEDED_AT, FIELD_SUPERSEDES, FIELD_EVENT_DATE)

_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})([T ].*)?$")


def parse_supersedes_ids(raw: Any, uuid_mapping: Dict[str, str]) -> List[str]:
    """Resolve LLM-emitted ``supersedes`` indices to real memory UUIDs.

    The LLM sees existing memories under sequential string ids ("0", "1", ...);
    ``uuid_mapping`` maps those back to real UUIDs. Anything that does not
    resolve (hallucinated index, wrong type, non-list input) is silently
    discarded — a bad mark must never block or distort the add.
    """
    if not raw or not isinstance(raw, list) or not uuid_mapping:
        return []
    resolved: List[str] = []
    for item in raw:
        if isinstance(item, (str, int)):
            key = str(item).strip()
            real_id = uuid_mapping.get(key)
            if real_id and real_id not in resolved:
                resolved.append(real_id)
            elif real_id is None:
                logger.debug(f"Discarding unresolvable supersedes id from LLM: {item!r}")
    return resolved


def parse_event_date(raw: Any) -> Optional[str]:
    """Normalize an LLM-emitted event date to ``YYYY-MM-DD``; None if unusable."""
    if not isinstance(raw, str):
        return None
    match = _DATE_RE.match(raw.strip())
    if not match:
        return None
    date_part = match.group(1)
    try:
        datetime.strptime(date_part, "%Y-%m-%d")
    except ValueError:
        return None
    return date_part


# --- deterministic event_date fallback (measured need: the small extractor puts
# the date in the fact TEXT but omits the structured field — 0/185 on document
# ingestion even with the suffix in the prompt). Conservative by design: only a
# FULL, unambiguous date counts; if the text has zero or MULTIPLE distinct full
# dates, return None (never guess). Year-less dates never infer a year. --------

_MONTHS = {
    # pt
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4, "maio": 5,
    "junho": 6, "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10,
    "novembro": 11, "dezembro": 12,
    # en
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_ISO_TXT_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_NUM_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})\b")
_NAME_DATE_RE = re.compile(
    r"\b(\d{1,2})(?:º|o)?\s+(?:de\s+)?([A-Za-zçÇ]+)(?:\s+(?:de|of|,)?\s*(\d{4}))\b",
    re.IGNORECASE,
)
# EN month-first: "October 5, 2024" / "October 5 2024"
_NAME_DATE_MF_RE = re.compile(
    r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
    re.IGNORECASE,
)
# month + year, NO day (v0.6 query anchor): "outubro de 2023", "October 2023", "10/2023"
_NAME_MONTH_YEAR_RE = re.compile(
    r"\b(?:de\s+|of\s+)?([A-Za-zçÇ]+)\s+(?:de\s+|of\s+)?(\d{4})\b",
    re.IGNORECASE,
)
_NUM_MONTH_YEAR_RE = re.compile(r"\b(\d{1,2})/(\d{4})\b")


def _iter_full_date_matches(text: str) -> List[Tuple[Tuple[int, int], Tuple[int, int, int]]]:
    """Single source of full-date extraction: yield (span, (y, m, d)) for every
    valid full date in the text (day+month+year). Two-digit years -> 20YY.

    Both ``infer_event_date_from_text`` (persistence) and
    ``infer_event_anchor_from_query`` (query anchoring) build on this so the two
    never diverge. Order-preserving list (not a set) because the query anchor
    needs spans to suppress month-year matches nested inside a full date.
    """
    out: List[Tuple[Tuple[int, int], Tuple[int, int, int]]] = []
    for m_ in _ISO_TXT_RE.finditer(text):
        out.append((m_.span(), (int(m_.group(1)), int(m_.group(2)), int(m_.group(3)))))
    for m_ in _NUM_DATE_RE.finditer(text):
        d, mo, yy = m_.group(1), m_.group(2), m_.group(3)
        year = int(yy) if len(yy) == 4 else 2000 + int(yy)
        out.append((m_.span(), (year, int(mo), int(d))))
    for m_ in _NAME_DATE_RE.finditer(text):
        month = _MONTHS.get(m_.group(2).lower())
        if month:
            out.append((m_.span(), (int(m_.group(3)), month, int(m_.group(1)))))
    for m_ in _NAME_DATE_MF_RE.finditer(text):
        month = _MONTHS.get(m_.group(1).lower())
        if month:
            out.append((m_.span(), (int(m_.group(3)), month, int(m_.group(2)))))
    valid: List[Tuple[Tuple[int, int], Tuple[int, int, int]]] = []
    for span, (y, m, d) in out:
        try:
            datetime(year=y, month=m, day=d)
        except ValueError:
            continue
        valid.append((span, (y, m, d)))
    return valid


def infer_event_date_from_text(text: Any) -> Optional[str]:
    """Extract ONE unambiguous full date (day+month+year) from a fact's text.

    Returns ``YYYY-MM-DD`` when the text contains exactly one distinct full
    date; ``None`` when it has none or several different ones (ambiguous —
    which event would it anchor?). Two-digit years follow the documented
    20YY rule. Purely deterministic: complements the LLM's event_date when the
    model wrote the date into the sentence but skipped the structured field.
    """
    if not isinstance(text, str) or not text:
        return None
    valid = {v for _, v in _iter_full_date_matches(text)}
    if len(valid) != 1:
        return None
    y, m, d = next(iter(valid))
    return f"{y:04d}-{m:02d}-{d:02d}"


def _spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1])


def _iter_month_year_matches(
    text: str, day_spans: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Yield distinct (year, month) for month+year expressions NOT nested inside
    a full date (e.g. the "outubro de 2023" span of "17 de outubro de 2023" is
    dropped, so it does not count as a second temporal expression)."""
    seen: set = set()
    for m_ in _NAME_MONTH_YEAR_RE.finditer(text):
        month = _MONTHS.get(m_.group(1).lower())
        if not month:
            continue
        if any(_spans_overlap(m_.span(), ds) for ds in day_spans):
            continue
        seen.add((int(m_.group(2)), month))
    for m_ in _NUM_MONTH_YEAR_RE.finditer(text):
        month = int(m_.group(1))
        if not (1 <= month <= 12):
            continue
        if any(_spans_overlap(m_.span(), ds) for ds in day_spans):
            continue
        seen.add((int(m_.group(2)), month))
    return list(seen)


def infer_event_anchor_from_query(text: Any) -> Optional[Tuple[str, str]]:
    """Detect a single temporal expression in a QUERY and return an event-time
    window ``(from_iso, to_iso)`` for ranking.

    Accepts a FULL date (-> that day, ``[d, d]``) or a MONTH+YEAR (-> that whole
    month, ``[1st, last]``). A bare year does NOT trigger (too weak — a query
    that merely mentions a year is rarely asking about event-time). Conservative
    like ``infer_event_date_from_text``: exactly ONE distinct temporal
    expression, else ``None`` (never guesses which one to anchor on).
    """
    if not isinstance(text, str) or not text:
        return None
    day_matches = _iter_full_date_matches(text)
    day_vals = {v for _, v in day_matches}
    month_vals = _iter_month_year_matches(text, [s for s, _ in day_matches])
    if len(day_vals) + len(month_vals) != 1:
        return None
    if day_vals:
        y, m, d = next(iter(day_vals))
        iso = f"{y:04d}-{m:02d}-{d:02d}"
        return (iso, iso)
    y, m = month_vals[0]
    last = calendar.monthrange(y, m)[1]
    return (f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{last:02d}")


def _expand_partial_date(value: str, end: bool) -> str:
    """Expand a partial ISO date to a full ``YYYY-MM-DD`` day (start or end of
    the implied range). ``end`` picks the last day of the year/month; otherwise
    the first. Fail-fast ValueError on anything but YYYY / YYYY-MM / YYYY-MM-DD.
    """
    v = value.strip()
    if re.fullmatch(r"\d{4}", v):
        datetime(int(v), 1, 1)
        return f"{v}-12-31" if end else f"{v}-01-01"
    if re.fullmatch(r"\d{4}-\d{2}", v):
        y, mo = int(v[:4]), int(v[5:7])
        datetime(y, mo, 1)
        last = calendar.monthrange(y, mo)[1]
        return f"{y:04d}-{mo:02d}-{last:02d}" if end else f"{y:04d}-{mo:02d}-01"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
        datetime.strptime(v, "%Y-%m-%d")
        return v
    raise ValueError(f"Invalid event date {value!r}: expected YYYY, YYYY-MM, or YYYY-MM-DD")


def expand_event_window(
    event_from: Optional[str] = None, event_to: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Expand caller-provided partial dates into an inclusive ``[from, to]`` day
    window (either side may be None = open interval). Fail-fast on unparsable
    input or ``from > to`` — mirrors ``parse_as_of`` for caller parameters.
    """
    lo = _expand_partial_date(event_from, end=False) if event_from is not None else None
    hi = _expand_partial_date(event_to, end=True) if event_to is not None else None
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"event_from ({lo}) is after event_to ({hi})")
    return lo, hi


def event_proximity(
    window: Optional[Tuple[str, str]], event_date: Any, window_days: int = 30
) -> float:
    """Linear proximity of a fact's ``event_date`` to an anchor window.

    1.0 inside the window; decaying linearly to 0.0 at ``window_days`` beyond the
    nearest edge; 0.0 further out. Missing/invalid event_date or window, or a
    non-positive ``window_days``, all yield 0.0 (neutral — never raises).
    """
    if not window or window_days is None or window_days <= 0:
        return 0.0
    iso = parse_event_date(event_date)
    if iso is None:
        return 0.0
    try:
        e = datetime.strptime(iso, "%Y-%m-%d").date()
        lo = datetime.strptime(window[0], "%Y-%m-%d").date()
        hi = datetime.strptime(window[1], "%Y-%m-%d").date()
    except (ValueError, TypeError, IndexError):
        return 0.0
    if lo <= e <= hi:
        return 1.0
    delta = (lo - e).days if e < lo else (e - hi).days
    return max(0.0, 1.0 - delta / window_days)


def parse_as_of(value: Any) -> Tuple[str, datetime]:
    """Parse a caller-provided ``as_of`` anchor. Fail-fast on bad input.

    Accepts a plain date (``YYYY-MM-DD``) or a full ISO-8601 datetime. A plain
    date is normalized to the END of that day (UTC): "what did I know on
    2026-03-15" includes the whole day. Naive datetimes are assumed UTC.

    Returns ``(iso_string_for_filtering, datetime_for_penalty_logic)``.
    """
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.isoformat(), dt
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"as_of must be an ISO date or datetime string, got: {value!r}")
    text = value.strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            day = datetime.strptime(text, "%Y-%m-%d").date()
            dt = datetime.combine(day, time.max, tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError(f"Invalid as_of value {value!r}: {e}") from None
    return dt.isoformat(), dt


def supersession_inverted(new_created_at: Any, old_created_at: Any) -> bool:
    """Whether an arriving fact should be born-superseded by an existing one.

    Asynchronous ingestion decouples submission time from processing time: a
    fact can reach the store AFTER a newer fact about the same subject was
    already persisted (a queued item overtaken by a direct write). The default
    marking direction — "what arrives supersedes what exists" — assumes
    arrival order equals truth order, which a queue breaks. When the arriving
    memory's record time (``created_at``, canonically its submission time)
    strictly predates the existing memory's, the direction inverts: the
    newcomer is born superseded and the existing fact stays current. A
    missing or unparsable timestamp on either side keeps the forward
    direction (the pre-queue behavior).
    """
    new_dt = _parse_ts(new_created_at)
    old_dt = _parse_ts(old_created_at)
    if new_dt is None or old_dt is None:
        return False
    return new_dt < old_dt


def superseded_penalty_applies(payload: Dict[str, Any], as_of: Optional[datetime] = None) -> bool:
    """Whether a memory should carry the superseded ranking penalty.

    Without an anchor, any superseded memory is demoted. With an ``as_of``
    anchor, a memory superseded only AFTER the anchor was still the current
    fact at that time, so its penalty is waived; superseded at or before the
    anchor stays demoted. A missing/unparsable ``superseded_at`` on a marked
    memory demotes conservatively.
    """
    if not payload or not payload.get(FIELD_SUPERSEDED_BY):
        return False
    if as_of is None:
        return True
    superseded_at = _parse_ts(payload.get(FIELD_SUPERSEDED_AT))
    if superseded_at is None:
        return True
    return superseded_at <= as_of
