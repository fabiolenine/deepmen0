"""
DeepMem0 v0.2 — human-memory dynamics (ACT-R base-level activation).

Every memory lives on an evolving timeline: each re-encounter or use appends a
reinforcement timestamp. Relevance then carries an activation term

    B_i = ln( sum_j  dt_j^(-d) )

over that timeline (dt_j = DAYS since reinforcement j, d ~= 0.5), a single
quantity capturing both frequency (how many reinforcements) and recency (how
recent they are).

The unit of dt only shifts B by a constant, so choosing it fixes the sigmoid's
operating point. Days center it on memory-corpus timescales: "reinforced once,
today" sits exactly at boost 0.5; a month-old untouched fact ~0.15; a fact
reinforced repeatedly over the last week ~0.7+. dt is clamped to >= 1 day, so
sub-day recency is deliberately flat — a brand-new memory cannot out-activate
a genuinely reinforced one, and same-day repetition is frequency (bounded by
the reinforcement window), not recency.

Activation is DERIVED, never stored. What persists is the event history
(``reinforced_at`` timestamps + ``access_count``); the value is computed lazily
at query time, only for the candidates being ranked. There is no batch decay
job and no persisted weight to refresh — as wall-clock time passes every dt
grows and activation falls on its own, with zero writes.

Memories without a reinforcement history (e.g. created before v0.2) are
NEUTRAL: they receive no boost and no penalty, so enabling dynamics never
reprices an existing corpus. A memory joins the timeline on its first
reinforcement, adopting its ``created_at`` as the first encounter.

Bounded growth: only the most recent ``max_timestamps`` reinforcements are
retained verbatim; the older tail is folded into the standard ACT-R hybrid
approximation (Petrov, 2006) using the total count and the memory's age, so
payload size stays O(K) regardless of how old or busy a memory is.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

FIELD_REINFORCED_AT = "reinforced_at"
FIELD_ACCESS_COUNT = "access_count"
FIELD_LAST_ACCESSED = "last_accessed"

DYNAMICS_FIELDS = (FIELD_REINFORCED_AT, FIELD_ACCESS_COUNT, FIELD_LAST_ACCESSED)

# dt is measured in days, clamped to >= 1: within the first day every memory
# is equally "today", so freshness alone cannot dominate real reinforcement.
_MIN_AGE_DAYS = 1.0


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: Any) -> Optional[datetime]:
    """Tolerant ISO-8601 parse; naive datetimes are assumed UTC."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _age_days(ts: datetime, now: datetime) -> float:
    return max((now - ts).total_seconds() / 86400.0, _MIN_AGE_DAYS)


def init_dynamics_fields(metadata: Dict[str, Any], created_at: Optional[str] = None) -> None:
    """Initialize the timeline on create: creation is the first encounter."""
    ts = created_at or utcnow().isoformat()
    metadata.setdefault(FIELD_REINFORCED_AT, [ts])
    metadata.setdefault(FIELD_ACCESS_COUNT, 1)


def base_level_activation(
    reinforced_at: Optional[List[Any]],
    access_count: Optional[int] = None,
    *,
    now: Optional[datetime] = None,
    decay: float = 0.5,
    first_seen: Any = None,
) -> Optional[float]:
    """ACT-R base-level activation over a reinforcement timeline.

    Exact sum over the retained timestamps; if ``access_count`` exceeds the
    retained count, the trimmed tail is approximated by spreading the missing
    reinforcements uniformly between ``first_seen`` and the oldest retained
    timestamp (Petrov 2006 hybrid approximation).

    Returns None when there is no usable history (the memory is neutral).
    """
    now = now or utcnow()
    ages = sorted(
        _age_days(ts, now)
        for ts in (_parse_ts(v) for v in (reinforced_at or []))
        if ts is not None
    )
    if not ages:
        return None

    total = sum(age ** -decay for age in ages)

    n = access_count if isinstance(access_count, int) and access_count > 0 else len(ages)
    missing = n - len(ages)
    if missing > 0:
        first = _parse_ts(first_seen)
        t_first = _age_days(first, now) if first is not None else ages[-1]
        t_oldest = ages[-1]
        if t_first > t_oldest and decay != 1.0:
            # integral-mean of t^-d over [t_oldest, t_first]
            tail_mean = (t_first ** (1.0 - decay) - t_oldest ** (1.0 - decay)) / (
                (1.0 - decay) * (t_first - t_oldest)
            )
        else:
            tail_mean = t_oldest ** -decay
        total += missing * tail_mean

    if total <= 0:
        return None
    return math.log(total)


def activation_boost(activation: Optional[float]) -> float:
    """Squash activation (-inf, ~ln n] into (0, 1); None (no history) -> 0."""
    if activation is None:
        return 0.0
    return 1.0 / (1.0 + math.exp(-activation))


def boost_from_payload(
    payload: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
    decay: float = 0.5,
) -> float:
    """Activation boost in [0, 1) from a memory payload; 0 when no history."""
    if not payload or not payload.get(FIELD_REINFORCED_AT):
        return 0.0
    activation = base_level_activation(
        payload.get(FIELD_REINFORCED_AT),
        payload.get(FIELD_ACCESS_COUNT),
        now=now,
        decay=decay,
        first_seen=payload.get("created_at"),
    )
    return activation_boost(activation)


def should_reinforce(
    payload: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
    window_seconds: int = 3600,
) -> bool:
    """At most one reinforcement per memory per window, across all triggers.

    Inside the window, re-encounters and hits have NO reinforcement effect —
    this absorbs client retries (a timed-out MCP client re-sending an add must
    not double-count) and approximates the ACT-R spacing effect: massed
    repetition adds nothing, spaced repetition does. ``window_seconds <= 0``
    disables the window.
    """
    if window_seconds <= 0:
        return True
    history = payload.get(FIELD_REINFORCED_AT) or []
    last = _parse_ts(history[-1]) if history else None
    if last is None:
        return True
    now = now or utcnow()
    return (now - last).total_seconds() >= window_seconds


def reinforcement_fields(
    payload: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
    max_timestamps: int = 10,
) -> Dict[str, Any]:
    """Updated dynamics fields for one reinforcement event.

    A memory created before v0.2 (no history) joins the timeline here,
    adopting its ``created_at`` as the first encounter. The returned dict
    contains ONLY the dynamics fields, ready to be merged into the payload.
    """
    now = now or utcnow()
    history = list(payload.get(FIELD_REINFORCED_AT) or [])
    count = payload.get(FIELD_ACCESS_COUNT)
    count = count if isinstance(count, int) and count > 0 else len(history)

    if not history:
        created = payload.get("created_at")
        if created and _parse_ts(created) is not None:
            history = [created]
            count = max(count, 1)

    now_iso = now.isoformat()
    history.append(now_iso)
    count += 1
    if max_timestamps > 0 and len(history) > max_timestamps:
        history = history[-max_timestamps:]

    return {
        FIELD_REINFORCED_AT: history,
        FIELD_ACCESS_COUNT: count,
        FIELD_LAST_ACCESSED: now_iso,
    }
