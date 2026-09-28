"""Exam reporting: with-reasoning vs tools-only, pass@1 and pass@k (ticket 010).

A dirty exam never produces a number (ticket 005 gate).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from src.training.data_factory.concepts import EpisodeResult, TurnMeta
from src.training.rl.exam_gate import (
    ExamGateError,
    assert_exam_byte_identical,
    pinned_exam_blob,
)
from src.training.rl.prompt import prompt_for_package

__all__ = [
    "exam_report",
    "pass_at_k",
    "prompt_for_package",
    "tools_only_episode",
]


def tools_only_episode(episode: EpisodeResult) -> EpisodeResult:
    """Strip ``reasoning_content``; leave ``tool_calls`` unchanged."""
    turns: list[TurnMeta] = []
    for turn in episode.turns:
        turns.append(dataclasses.replace(turn, reasoning_content=None))
    return dataclasses.replace(episode, turns=turns)


def pass_at_k(statuses: Sequence[str], k: int) -> float:
    """1.0 if any of the first ``k`` statuses is ``pass``, else 0.0."""
    if k < 1:
        raise ValueError("k must be >= 1")
    window = list(statuses)[:k]
    return 1.0 if any(status == "pass" for status in window) else 0.0


def _pass_stats(statuses: Sequence[str], k: int) -> dict:
    return {
        "pass_at_1": pass_at_k(statuses, 1),
        "pass_at_k": pass_at_k(statuses, k),
        "k": k,
        "n": len(statuses),
    }


def exam_report(
    *,
    task_results: Sequence[dict],
    k: int,
    exam_path=None,
    expected_rev: str | None = None,
) -> dict:
    """Build the exam report. ``task_results`` items carry ``statuses``.

    Raises :class:`ExamGateError` before any number if the exam is dirty.
    """
    assert_exam_byte_identical(exam_path, expected_rev=expected_rev)
    with_reasoning = []
    tools_only = []
    for row in task_results:
        statuses = list(row["statuses"])
        with_reasoning.append(_pass_stats(statuses, k))
        tools_only.append(_pass_stats(list(row.get("tools_only_statuses") or statuses), k))
    n = len(task_results)
    def _mean(rows, key):
        if n == 0:
            return None
        return sum(row[key] for row in rows) / n

    return {
        "exam_revision": pinned_exam_blob(),
        "k": k,
        "n_tasks": n,
        "with_reasoning": {
            "pass_at_1": _mean(with_reasoning, "pass_at_1"),
            "pass_at_k": _mean(with_reasoning, "pass_at_k"),
        },
        "tools_only": {
            "pass_at_1": _mean(tools_only, "pass_at_1"),
            "pass_at_k": _mean(tools_only, "pass_at_k"),
        },
    }
