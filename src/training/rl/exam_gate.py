"""Frozen v1.0 exam gate (RL spec D8 / ticket 005).

Call ``assert_exam_byte_identical`` (or ``before_eval_rollout``) before any
eval rollout. Training / difficulty / mini-exam val must pass their task ids
through ``assert_disjoint_from_exam`` so the 63 never enter those loops.

This is a stage module: it imports nutrienv at function level where needed.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")

_CONFIG = pathlib.Path(__file__).resolve().parents[3] / "configs" / "data_factory.yaml"


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


def assert_lab_at_rev(expected_rev: str | None = None) -> str:
    """Abort unless the installed lab HEAD is ``expected_rev``; return HEAD.

    ``expected_rev`` must be the full 40-hex SHA (no prefixes). ``None`` means
    the ADR-012 pin, ``nutrienv.rev`` in ``configs/data_factory.yaml``.
    """
    if expected_rev is None:
        from src.training.data_factory.config import load_config

        expected_rev = load_config(_CONFIG).nutrienv_rev
    if not re.fullmatch(r"[0-9a-f]{40}", expected_rev):
        raise ExamGateError(f"expected_rev {expected_rev!r} is not a full 40-hex SHA")
    root = _lab_root()
    head = _git_output(["git", "rev-parse", "HEAD"], cwd=root)
    if head != expected_rev:
        raise ExamGateError(
            f"NutriEnv tree at {root} is at HEAD {head}, expected rev {expected_rev}; "
            "eval aborts before any rollout"
        )
    _assert_clean_tree(root)
    return head


def _assert_clean_tree(root: pathlib.Path) -> None:
    """Reject a pin whose tracked files are modified.

    The HEAD check alone is not enough: an uncommitted edit to the scorer or to
    the catalog changes the score while ``rev-parse HEAD`` still matches, and the
    report still looks normal.
    """
    dirty = _git_output(["git", "status", "--porcelain", "-uno"], cwd=root)
    if dirty:
        raise ExamGateError(
            f"NutriEnv tree at {root} has uncommitted tracked changes "
            f"({dirty.splitlines()[0]}); eval aborts before any rollout"
        )


def pinned_exam_blob() -> str:
    """Git blob of the exam file at the installed lab HEAD (the pin)."""
    from nutrienv.bench import EXAM_SPLIT_PATH

    root = _lab_root()
    rel = pathlib.Path(EXAM_SPLIT_PATH).resolve().relative_to(root).as_posix()
    return _git_output(["git", "rev-parse", f"HEAD:{rel}"], cwd=root)


def exam_file_blob(path: pathlib.Path) -> str:
    return _git_output(["git", "hash-object", str(path)])


def assert_exam_byte_identical(
    exam_path: pathlib.Path | str | None = None,
    *,
    expected_rev: str | None = None,
) -> None:
    """Abort unless the pin HEAD is clean and ``exam_path`` equals its exam blob."""
    assert_lab_at_rev(expected_rev)
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
    expected_rev: str | None = None,
) -> T:
    """Run ``rollout`` only after the exam pin check succeeds."""
    assert_exam_byte_identical(exam_path, expected_rev=expected_rev)
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
