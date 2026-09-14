"""SFT cold-start go/no-go on held-out unique queries (nutrimind-pilot/005).

SFT loss is not an input. Verdict is insufficient or usable for an RL pilot.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

__all__ = ["cold_start_go_no_go"]

_INVALID_OR_NO_FINISH = 0.1


def _rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return num / den


def cold_start_go_no_go(
    rows: Sequence[Mapping],
    *,
    group_size: int,
) -> dict:
    """Report protocol health on held-out unique-query episodes.

    Each row: ``family``, ``schema_valid``, ``finished``, ``execution``,
    ``oracle_exec``, ``status`` (pass/fail/indeterminate), ``recovery_positive``,
    ``group_id``. ``composite_type`` optional.
    """
    n = len(rows)
    schema_ok = sum(1 for row in rows if row.get("schema_valid"))
    finished = sum(1 for row in rows if row.get("finished"))
    exec_ok = sum(1 for row in rows if row.get("execution") == "ok")
    oracle_ok = sum(1 for row in rows if row.get("oracle_exec") == "ok")
    recovery = sum(1 for row in rows if row.get("recovery_positive"))
    recovery_applicable = sum(
        1 for row in rows if "recovery_positive" in row
    )

    by_family: dict[str, list[str]] = defaultdict(list)
    composite_types: set[str] = set()
    for row in rows:
        family = str(row.get("family") or "")
        by_family[family].append(str(row.get("status") or ""))
        ctype = row.get("composite_type")
        if ctype:
            composite_types.add(str(ctype))

    pass_at_1 = {}
    mixed_families = 0
    for family, statuses in by_family.items():
        n_pass = sum(1 for status in statuses if status == "pass")
        n_fail = sum(1 for status in statuses if status == "fail")
        denom = n_pass + n_fail
        pass_at_1[family] = (n_pass / denom) if denom else None
        if n_pass and n_fail:
            mixed_families += 1

    groups: dict[object, list[float]] = defaultdict(list)
    for row in rows:
        status = row.get("status")
        if status == "pass":
            reward = 1.0
        elif status == "fail":
            reward = 0.0
        else:
            continue
        groups[row.get("group_id")].append(reward)
    sized = [rewards for rewards in groups.values() if len(rewards) >= group_size]
    mixed_groups = sum(1 for rewards in sized if 0.0 in rewards and 1.0 in rewards)
    mixed_group_rate = _rate(mixed_groups, len(sized)) if sized else 0.0

    schema_rate = _rate(schema_ok, n) or 0.0
    finish_rate = _rate(finished, n) or 0.0
    almost_dead = n == 0 or schema_rate < _INVALID_OR_NO_FINISH or finish_rate < _INVALID_OR_NO_FINISH
    all_zero_groups = bool(sized) and mixed_groups == 0
    no_groups = n > 0 and not sized
    insufficient = almost_dead or all_zero_groups or no_groups
    usable = (not insufficient) and mixed_families >= 2

    if insufficient:
        verdict = "insufficient"
    elif usable:
        verdict = "usable"
    else:
        verdict = "insufficient"

    return {
        "verdict": verdict,
        "schema_tool_call_validity": schema_rate,
        "finish_rate": finish_rate,
        "no_finish_rate": 1.0 - finish_rate if n else None,
        "execution_health": _rate(exec_ok, n),
        "oracle_reconstruction_health": _rate(oracle_ok, n),
        "pass_at_1_by_family": dict(pass_at_1),
        "families_covered": sorted(by_family),
        "composite_types_covered": sorted(composite_types),
        "recovery_positive_rate": _rate(recovery, recovery_applicable),
        "mixed_reward_group_rate": mixed_group_rate,
        "planned_g": group_size,
        "mixed_pass_fail_families": mixed_families,
    }
