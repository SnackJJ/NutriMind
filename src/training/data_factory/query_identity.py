"""Unique query identity — count and rank speech (nutrimind-pilot/003).

The unit is a normalized user query, not a trajectory, teacher attempt,
GRPO group, optimizer step, or token. Two surface realizations of the same
identity count as one.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Mapping, Sequence

__all__ = [
    "OPD_UNIQUE",
    "REALIZATION_RANK",
    "RL_PILOT_UNIQUE",
    "SFT_COLD_START_UNIQUE",
    "UniqueQueryIndex",
    "query_identity",
    "select_realization",
    "unique_caps",
]

SFT_COLD_START_UNIQUE = 100
RL_PILOT_UNIQUE = 200
OPD_UNIQUE = 0

# Selection order when several surface realizations are sampled (ticket 003).
REALIZATION_RANK = ("consistency", "unique_bind", "diversity", "collision")

_TRAILING = re.compile(r"[" + re.escape(string.punctuation) + r"\s]+$")


def query_identity(query: str) -> str:
    """Casefold, collapse whitespace, strip trailing punctuation."""
    collapsed = re.sub(r"\s+", " ", (query or "").casefold()).strip()
    return _TRAILING.sub("", collapsed)


def unique_caps(family_target_n: Mapping[str, int], budget: int) -> dict[str, int]:
    """Coverage-balanced unique-query caps from production ``target_n`` mix."""
    names = [name for name, target in family_target_n.items() if target > 0]
    if not names or budget <= 0:
        return {name: 0 for name in family_target_n}
    total = sum(family_target_n[name] for name in names)
    raw = {name: family_target_n[name] / total * budget for name in names}
    caps = {name: int(raw[name]) for name in names}
    while sum(caps.values()) < budget:
        name = max(names, key=lambda item: raw[item] - caps[item])
        caps[name] += 1
    return caps


def realization_sort_key(
    candidate: Mapping,
    *,
    seen_identities: Sequence[str] = (),
) -> tuple:
    """Lower is better. Order: consistency → unique bind → diversity → collision."""
    identity = query_identity(str(candidate.get("query") or ""))
    seen = set(seen_identities)
    consistency = 0 if candidate.get("consistency_ok") else 1
    unique_bind = 0 if candidate.get("unique_bind") else 1
    diversity = 0 if identity and identity not in seen else 1
    collision = 0
    if (
        candidate.get("exam_collision")
        or candidate.get("verbatim_collision")
        or candidate.get("semantic_collision")
    ):
        collision = 1
    return (consistency, unique_bind, diversity, collision)


def select_realization(
    candidates: Sequence[Mapping],
    *,
    seen_identities: Sequence[str] = (),
) -> Mapping:
    """Pick one realization by ``REALIZATION_RANK``."""
    if not candidates:
        raise ValueError("select_realization requires at least one candidate")
    return min(
        candidates,
        key=lambda row: realization_sort_key(row, seen_identities=seen_identities),
    )


class UniqueQueryIndex:
    """First-seen unique query identities, partitioned by family."""

    def __init__(self) -> None:
        self._first: dict[str, str | None] = {}
        self.by_family: Counter[str] = Counter()

    def add(self, query: str, *, family: str, task_id: str | None = None) -> bool:
        """Record ``query``. Return True iff this identity is new."""
        identity = query_identity(query)
        if not identity or identity in self._first:
            return False
        self._first[identity] = task_id
        self.by_family[family] += 1
        return True

    def __len__(self) -> int:
        return len(self._first)

    def identities(self) -> frozenset[str]:
        return frozenset(self._first)
