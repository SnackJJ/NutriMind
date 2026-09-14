"""RL unique-query pool, expansion rule, OPD selector (nutrimind-pilot/006).

Consumes factory TaskPackage-shaped mappings. Does not author tasks and
does not train OPD. Reward map stays Pass=1 / Fail=0 / Indeterminate excluded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.training.data_factory.query_identity import (
    OPD_UNIQUE,
    RL_PILOT_UNIQUE,
    UniqueQueryIndex,
    query_identity,
)
from src.training.rl.reward import REWARD_MAP

__all__ = [
    "OPD_UNIQUE",
    "RL_PILOT_UNIQUE",
    "build_rl_unique_query_pool",
    "pool_expansion_needed",
    "select_opd_queries",
]


def _package_query(package: Mapping) -> str:
    if "query" in package:
        return str(package["query"] or "")
    task = package.get("task") or {}
    if isinstance(task, Mapping):
        return str(task.get("query") or "")
    return ""


def build_rl_unique_query_pool(
    packages: Sequence[Mapping],
    *,
    sft_identities: Sequence[str] = (),
    size: int = RL_PILOT_UNIQUE,
    held_out_fraction: float = 0.25,
) -> dict:
    """Select ~``size`` unique query identities from TaskPackages.

    Overlap with SFT is explicit. A held-out set is carved from identities
    not in ``sft_identities``.
    """
    sft = {query_identity(item) for item in sft_identities if query_identity(item)}
    index = UniqueQueryIndex()
    overlap: list[dict] = []
    fresh: list[dict] = []
    for package in packages:
        query = _package_query(package)
        family = str(package.get("family") or "")
        if not index.add(query, family=family, task_id=str(package.get("task_id") or "")):
            continue
        row = {
            "task_id": package.get("task_id"),
            "family": family,
            "query": query,
            "identity": query_identity(query),
        }
        if row["identity"] in sft:
            overlap.append(row)
        else:
            fresh.append(row)
        if len(index) >= size:
            break

    held_n = max(1, int(round(len(fresh) * held_out_fraction))) if fresh else 0
    held_out = fresh[:held_n]
    train = overlap + fresh[held_n:]
    return {
        "unique_query_count": len(index),
        "target": size,
        "overlap_with_sft": overlap,
        "train": train,
        "held_out": held_out,
        "reward_map": dict(REWARD_MAP),
    }


def pool_expansion_needed(diagnostics: Mapping) -> bool:
    """Expand only on listed shortages — not because the training curve is noisy."""
    mixed = int(diagnostics.get("mixed_reward_groups") or 0)
    noisy = bool(diagnostics.get("noisy_curve"))
    if noisy and mixed > 0 and not diagnostics.get("train_up_held_out_flat"):
        if not diagnostics.get("too_few_mixed_groups"):
            if not diagnostics.get("empty_family_coverage"):
                if not diagnostics.get("empty_composite_coverage"):
                    if not diagnostics.get("verifier_shortcut"):
                        return False
    if diagnostics.get("too_few_mixed_groups"):
        return True
    if diagnostics.get("empty_family_coverage") or diagnostics.get(
        "empty_composite_coverage"
    ):
        return True
    if diagnostics.get("train_up_held_out_flat"):
        return True
    if diagnostics.get("verifier_shortcut"):
        return True
    return False


def select_opd_queries(
    states: Sequence[Mapping],
    *,
    budget: int = OPD_UNIQUE,
) -> list[Mapping]:
    """OPD unique-query budget starts at 0. Ambiguity / disagreement is refused."""
    if budget <= 0:
        return []
    kept: list[Mapping] = []
    for state in states:
        if state.get("catalog_ambiguous"):
            continue
        if state.get("teacher_status") != state.get("verifier_status"):
            continue
        kept.append(state)
        if len(kept) >= budget:
            break
    return kept
