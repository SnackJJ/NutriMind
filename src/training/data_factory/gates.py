"""Seam 2 — the pure validation boundary (ticket 005, spec §11).

``GateContext.from_exam(exam_tasks)`` precomputes the exam corpus facts ONCE per
run (normalized queries, ``semantic_key``s, update slot values); ``run(task, ctx)``
is a pure function of ``(task, ctx)`` — same inputs, same ``GateResult``, no I/O,
no module-level state, no implicit exam load.

Gate order (spec §11, first failure wins):

1. ``gate.verbatim_query_collision``  — normalized query equals an exam query's.
2. ``gate.semantic_key_collision``    — ``semantic_key(task)`` equals an exam key.
3. ``gate.slot_value_overlaps_exam``  — update-leg slot values (added allergens,
   changed weight / goal) reused from an exam update item. The profile-diff is
   empty for every non-update shape, so the check naturally no-ops there.
4. ``gate.stage_a``                   — ``stage_a_code_gate(task)`` non-empty.
5. ``gate.draft_invalid``             — ``validate_draft(task)`` non-empty, minus
   the 3-leg composite false-positive allow-list (spec §22.9: the frozen exam
   item ``adr24-comp-8255`` trips ``"update oracle ledger is missing"`` too; the
   allow-list is scoped to ``family == "composite"``).
6. ``gate.unachievable``              — ``check_achievable`` unreachable. This is
   an **authoring bug, not a model failure**: ``build`` routes this code to
   ``rejects/indeterminate.jsonl`` (status ``indeterminate``), never to a plain
   gate drop (spec §11).
"""

from __future__ import annotations

import dataclasses
import re
import string
from collections.abc import Iterable

from nutrienv.bench.achievable import check_achievable
from nutrienv.bench.pipeline.review_harness import stage_a_code_gate
from nutrienv.bench.validator import semantic_key, validate_draft

from .concepts import GateResult

__all__ = [
    "GateContext",
    "normalize_query",
    "run",
    "GATE_ORDER",
]

VERBATIM = "gate.verbatim_query_collision"
SEMANTIC = "gate.semantic_key_collision"
SLOT = "gate.slot_value_overlaps_exam"
STAGE_A = "gate.stage_a"
DRAFT = "gate.draft_invalid"
UNACHIEVABLE = "gate.unachievable"

GATE_ORDER = (VERBATIM, SEMANTIC, SLOT, STAGE_A, DRAFT, UNACHIEVABLE)

# spec §22.9 gate policy for the 3-leg shape: validate_draft may return exactly
# this known false-positive (the frozen exam item adr24-comp-8255 trips it too).
_THREE_LEG_DRAFT_FP = "update oracle ledger is missing"


def normalize_query(query: str) -> str:
    """Casefold + collapse whitespace + strip trailing punctuation (spec §11).

    The trailing strip removes any run of whitespace and punctuation together
    ("... allergies. !!!" → "... allergies"), so purely cosmetic tails cannot
    dodge a verbatim match. Near-duplicates that are not verbatim after this
    normalization are deliberately allowed.
    """
    collapsed = re.sub(r"\s+", " ", query.casefold()).strip()
    return _TRAILING_NOISE.sub("", collapsed)


_TRAILING_NOISE = re.compile(r"[" + re.escape(string.punctuation) + r"\s]+$")


def _profile_diff_slot_values(task) -> set[str]:
    """Slot values an update leg changed: added allergens, new weight, new goal.

    Applied uniformly: a task with no profile-bearing oracle (or whose expected
    profile equals ``s0``) yields the empty set, so non-update shapes never trip
    gate 3. Exam-side, the same extraction yields the values the frozen exam's
    update items used (allergens egg / milk / peanut / shellfish / tree_nut at
    rev 203d807).
    """
    values: set[str] = set()
    if task.oracle.sub_oracles:
        profiles = [s.profile for s in task.oracle.sub_oracles if s.profile is not None]
    else:
        profiles = [task.oracle.profile] if task.oracle.profile is not None else []
    base = task.s0.profile
    for expected in profiles:
        values.update(set(expected.allergies) - set(base.allergies))
        if expected.weight_kg != base.weight_kg:
            values.add(str(expected.weight_kg))
        goal_before = (
            base.plan_preset.get("goal") if isinstance(base.plan_preset, dict) else None
        )
        goal_after = (
            expected.plan_preset.get("goal")
            if isinstance(expected.plan_preset, dict)
            else None
        )
        if goal_after != goal_before and goal_after is not None:
            values.add(str(goal_after))
    return values


@dataclasses.dataclass(frozen=True)
class GateContext:
    """Precomputed exam-corpus facts — built once, threaded through ``run``.

    ``from_exam`` accepts any iterable of ``Task``s (the loaded 63 in production,
    a hand-picked list in tests). It never touches the exam split file itself.
    """

    normalized_exam_queries: frozenset[str]
    exam_semantic_keys: frozenset[tuple]
    exam_update_slot_values: frozenset[str]

    @classmethod
    def from_exam(cls, exam_tasks: Iterable) -> "GateContext":
        tasks = list(exam_tasks)
        return cls(
            normalized_exam_queries=frozenset(
                normalize_query(t.query) for t in tasks
            ),
            exam_semantic_keys=frozenset(semantic_key(t) for t in tasks),
            exam_update_slot_values=frozenset(
                value for t in tasks for value in _profile_diff_slot_values(t)
            ),
        )


def run(task, ctx: GateContext) -> GateResult:
    """Validate one authored ``Task``; ordered, first failure wins (spec §11)."""
    # 1 — verbatim query collision (normalized)
    normalized = normalize_query(task.query)
    if normalized in ctx.normalized_exam_queries:
        return GateResult(
            keep=False,
            failure_code=VERBATIM,
            reason_detail=f"normalized query {normalized!r} matches an exam query",
        )

    # 2 — semantic_key collision (used as-is, spec §11 note on its branches)
    key = semantic_key(task)
    if key in ctx.exam_semantic_keys:
        return GateResult(
            keep=False,
            failure_code=SEMANTIC,
            reason_detail=f"semantic_key {key!r} matches an exam task",
        )

    # 3 — update slot-value overlap with the exam
    overlap = _profile_diff_slot_values(task) & ctx.exam_update_slot_values
    if overlap:
        return GateResult(
            keep=False,
            failure_code=SLOT,
            reason_detail=f"update slot values overlap the exam: {sorted(overlap)}",
        )

    # 4 — stage A code gate
    stage_a_reasons = stage_a_code_gate(task)
    if stage_a_reasons:
        return GateResult(
            keep=False,
            failure_code=STAGE_A,
            reason_detail=f"stage_a_code_gate: {stage_a_reasons}",
        )

    # 5 — draft validity (3-leg composite false-positive allow-listed)
    issues = validate_draft(task)
    if task.family == "composite":
        issues = [issue for issue in issues if issue != _THREE_LEG_DRAFT_FP]
    if issues:
        return GateResult(
            keep=False,
            failure_code=DRAFT,
            reason_detail=f"validate_draft: {issues}",
        )

    # 6 — achievability (authoring bug → build routes this as indeterminate)
    report = check_achievable([task])
    if task.id in report.unreachable:
        return GateResult(
            keep=False,
            failure_code=UNACHIEVABLE,
            reason_detail=(
                "check_achievable: task unreachable — an authoring bug, not a "
                "model failure; route as indeterminate (spec §11)"
            ),
        )

    return GateResult(keep=True)


def rejects_record(result: GateResult, task, *, intent: dict | None = None) -> dict:
    """The ``rejects/gate.jsonl`` line shape for a gate drop (spec §9.4).

    ``gate.unachievable`` carries ``status="indeterminate"`` so the build routes
    it to ``rejects/indeterminate.jsonl``; every other gate drop is a plain
    pre-verification drop (``status="dropped"``).
    """
    if result.keep:
        raise ValueError("no reject record for a kept task")
    return {
        "schema_version": "nutrimind-v2-reject/1",
        "task_id": getattr(task, "id", None),
        "stage": "gate",
        "status": "indeterminate" if result.failure_code == UNACHIEVABLE else "dropped",
        "failure_codes": [result.failure_code] if result.failure_code else [],
        "reason_detail": result.reason_detail,
        "query": getattr(task, "query", None),
        "intent": intent,
    }
