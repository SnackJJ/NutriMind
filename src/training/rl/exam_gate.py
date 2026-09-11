"""Frozen v1.0 exam gate (RL spec D8 / ticket 005).

Call ``assert_exam_byte_identical`` (or ``before_eval_rollout``) before any
eval rollout. Training / difficulty / mini-exam val must pass their task ids
through ``assert_disjoint_from_exam`` so the 63 never enter those loops.

This is a stage module: it imports nutrienv at function level where needed.
"""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


class ExamGateError(RuntimeError):
    """The exam pin check failed, or a train-loop id leaked onto the exam."""


def _lab_root() -> pathlib.Path:
    import nutrienv

    return pathlib.Path(nutrienv.__file__).resolve().parents[2]


def _git_output(args: list[str], *, cwd: pathlib.Path | None = None) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExamGateError(f"cannot resolve exam git blob: {exc}") from exc


def pinned_exam_blob() -> str:
    """Git blob of the exam file at the installed lab HEAD (the pin)."""
    from nutrienv.bench import EXAM_SPLIT_PATH

    root = _lab_root()
    rel = pathlib.Path(EXAM_SPLIT_PATH).resolve().relative_to(root).as_posix()
    return _git_output(["git", "rev-parse", f"HEAD:{rel}"], cwd=root)


def exam_file_blob(path: pathlib.Path) -> str:
    return _git_output(["git", "hash-object", str(path)])


def assert_exam_byte_identical(exam_path: pathlib.Path | str | None = None) -> None:
    """Abort unless ``exam_path`` equals the committed v1.0 blob at the pin."""
    from nutrienv.bench import EXAM_SPLIT_PATH

    path = pathlib.Path(exam_path) if exam_path is not None else pathlib.Path(EXAM_SPLIT_PATH)
    working = exam_file_blob(path)
    pinned = pinned_exam_blob()
    if working != pinned:
        raise ExamGateError(
            f"exam file {path} blob {working} != pin blob {pinned}; "
            "eval aborts before any rollout"
        )


def before_eval_rollout(
    rollout: Callable[[], T],
    *,
    exam_path: pathlib.Path | str | None = None,
) -> T:
    """Run ``rollout`` only after the exam pin check succeeds."""
    assert_exam_byte_identical(exam_path)
    return rollout()


def exam_task_ids() -> frozenset[str]:
    from nutrienv.bench import load_exam

    return frozenset(task.id for task in load_exam())


def assert_disjoint_from_exam(task_ids: Iterable[str], *, loop: str) -> None:
    """Raise if any id belongs to the frozen 63. ``loop`` names the caller."""
    overlap = set(task_ids) & exam_task_ids()
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise ExamGateError(
            f"{loop} task ids overlap the v1.0 exam ({len(overlap)} ids): {sample}"
        )
