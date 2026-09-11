"""RL ticket 005 — exam byte gate + TRAIN_ROSTER isolation from the 63."""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench import EXAM_SPLIT_PATH, load_exam  # noqa: E402

from src.training.data_factory.roster_train import TRAIN_ROSTER  # noqa: E402
from src.training.rl.exam_gate import (  # noqa: E402
    ExamGateError,
    assert_disjoint_from_exam,
    assert_exam_byte_identical,
    before_eval_rollout,
    exam_task_ids,
)


def test_unmodified_pin_proceeds():
    assert_exam_byte_identical()
    called: list[bool] = []
    result = before_eval_rollout(lambda: called.append(True) or "ok")
    assert result == "ok"
    assert called == [True]


def test_mutated_exam_fails_before_rollout(tmp_path: pathlib.Path):
    mutated = tmp_path / "nutrienv-v1.0.json"
    mutated.write_bytes(pathlib.Path(EXAM_SPLIT_PATH).read_bytes() + b"\n")
    called: list[bool] = []

    def rollout() -> str:
        called.append(True)
        return "rolled"

    with pytest.raises(ExamGateError, match="eval aborts before any rollout"):
        before_eval_rollout(rollout, exam_path=mutated)
    assert called == []


def test_train_difficulty_mini_exam_ids_disjoint_from_exam():
    exam = exam_task_ids()
    assert len(exam) == 63
    assert exam == frozenset(task.id for task in load_exam())

    train_users = {person.user_id for person in TRAIN_ROSTER}
    exam_users = {task.s0.profile.user_id for task in load_exam()}
    assert train_users.isdisjoint(exam_users)
    assert all(user_id.startswith("train-") for user_id in train_users)
    assert all(user_id.startswith("roster-") for user_id in exam_users)

    # Factory-shaped ids used by train / difficulty / mini-exam val loops.
    loop_ids = {
        f"{family}--{family}--{person.user_id}--{seed:06d}"
        for person in TRAIN_ROSTER
        for family in ("log", "recommend", "evaluate", "update", "composite")
        for seed in (1, 30, 191)
    }
    assert loop_ids.isdisjoint(exam)
    assert_disjoint_from_exam(loop_ids, loop="train")
    assert_disjoint_from_exam(loop_ids, loop="difficulty")
    assert_disjoint_from_exam(loop_ids, loop="mini-exam")

    with pytest.raises(ExamGateError, match="train task ids overlap"):
        assert_disjoint_from_exam(exam, loop="train")
