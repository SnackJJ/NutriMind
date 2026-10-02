"""The evaluate clarification is teacher-only: exam and RL rollouts must not see it."""

from __future__ import annotations

import types

import pytest

pytest.importorskip("nutrienv")

from nutrienv.bench import load_split  # noqa: E402

from src.training.data_factory.rollout_fc import rollout_tool_call  # noqa: E402

HINT = "Clarification for evaluate tasks"


def _evaluate_task():
    import nutrienv.bench as bench

    for task in load_split(bench.EXAM_SPLIT_PATH):
        if task.family == "evaluate":
            return task
    pytest.skip("exam has no evaluate task")


def _run(task, **kwargs):
    seen: list[str] = []

    def complete(request):
        seen.append(request["messages"][0]["content"])
        return {
            "content": "",
            "reasoning_content": None,
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    rollout_tool_call(task, teacher_complete=complete, catalog=task.s0.catalog, **kwargs)
    return seen


def test_exam_rollout_sees_the_lab_prompt_only():
    seen = _run(_evaluate_task())
    assert seen and all(HINT not in s for s in seen)


def test_data_factory_teacher_gets_the_clarification():
    seen = _run(_evaluate_task(), evaluate_hint=True)
    assert seen and all(HINT in s for s in seen)


SAFETY = "Clarification for calorie-target requests"


def test_safety_clarification_is_teacher_only():
    task = _evaluate_task()
    assert all(SAFETY not in s for s in _run(task))
    assert all(SAFETY in s for s in _run(task, safety_hint=True))


def test_build_gives_the_safety_clarification_to_refuse_packages_only(monkeypatch):
    """_teacher_stage keys the hint on the package's steps."""
    import inspect

    from src.training.data_factory import build

    source = inspect.getsource(build._teacher_stage)
    assert 'safety_hint="refuse" in package.steps' in source
