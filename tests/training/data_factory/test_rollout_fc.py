"""Ticket 025 — teacher rollout via the lab FC loop, injected completion."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench import Scorer  # noqa: E402
from nutrienv.harness.runner import FINISH_OPS  # noqa: E402

from src.training.data_factory.rollout_fc import (  # noqa: E402
    NUTRIENV_TOOLS,
    TOOL_SYSTEM_PROMPT,
    ScriptedFCTeacher,
    rollout_tool_call,
)


from tests.training.data_factory import _fixtures as fx  # noqa: E402


def _call(name: str, args: dict, *, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def log_script(task, *, grams_scale=1.0, finish=True):
    turns = []
    for index, row in enumerate(task.oracle.ledger_tail, start=1):
        args = {
            "food_id": row.food_id,
            "grams": round(row.grams * grams_scale, 2),
            "eaten_at": row.eaten_at,
        }
        turns.append((
            "log the spoken lunch row",
            [_call("log_meal", args, call_id=f"call_{index}")],
        ))
    if finish:
        turns.append((
            "all logged, finishing",
            [_call("done", {}, call_id=f"call_{len(turns) + 1}")],
        ))
    return turns


@pytest.fixture(scope="module")
def catalog():
    return fx.gold_catalog()


@pytest.fixture(scope="module")
def log_task(catalog):
    return fx.make_log_task(catalog, fx.first_person(), seed=30)


def test_schema_is_the_lab_schema():
    assert NUTRIENV_TOOLS and TOOL_SYSTEM_PROMPT
    names = {t["function"]["name"] for t in NUTRIENV_TOOLS}
    assert "log_meal" in names
    assert FINISH_OPS


def test_scripted_pass_episode(catalog, log_task):
    teacher = ScriptedFCTeacher(log_script(log_task))
    episode = rollout_tool_call(log_task, teacher_complete=teacher, catalog=catalog)
    assert episode.reached_finish is True
    assert episode.error is None
    assert episode.turns
    assert all(t.raw_action_text is None for t in episode.turns)
    assert all(t.fallback_used is False for t in episode.turns)
    assert episode.turns[0].tool_calls and episode.turns[0].tool_call_id
    assert episode.turns[0].executed_op["op"] == "log_meal"
    assert episode.turns[0].reasoning_content == "log the spoken lunch row"
    assert episode.turns[0].content is None
    finish = episode.turns[-1]
    assert finish.tool_calls[0]["function"]["name"] in FINISH_OPS
    assert finish.executed_op is None
    assert Scorer().score(episode.end_state, log_task.oracle)["passed"] is True


def test_scripted_fail_episode(catalog, log_task):
    teacher = ScriptedFCTeacher(log_script(log_task, grams_scale=1.2))
    episode = rollout_tool_call(log_task, teacher_complete=teacher, catalog=catalog)
    assert Scorer().score(episode.end_state, log_task.oracle)["passed"] is False


def test_no_finish_exhausts_budget(catalog, log_task):
    teacher = ScriptedFCTeacher(log_script(log_task, finish=False))
    episode = rollout_tool_call(log_task, teacher_complete=teacher, catalog=catalog)
    assert episode.reached_finish is False
    assert episode.error is None
    assert any(t.tool_calls == [] and t.executed_op is None for t in episode.turns)


def test_no_tool_call_turn_is_distinct(catalog, log_task):
    teacher = ScriptedFCTeacher([
        ("talking instead of acting", []),
        ("all logged, finishing", [_call("done", {}, call_id="call_f")]),
    ])
    episode = rollout_tool_call(log_task, teacher_complete=teacher, catalog=catalog)
    first = episode.turns[0]
    assert first.tool_calls == []
    assert first.executed_op is None
    assert first.reasoning_content == "talking instead of acting"
    assert first.executed_op != {"op": "get_profile"}


def test_invalid_tool_is_stepped_as_error_observation(catalog, log_task):
    teacher = ScriptedFCTeacher([
        ("bad tool", [_call("not_a_tool", {}, call_id="call_bad")]),
        ("give up", [_call("done", {}, call_id="call_f")]),
    ])
    episode = rollout_tool_call(log_task, teacher_complete=teacher, catalog=catalog)
    first = episode.turns[0]
    assert first.executed_op == {"op": "not_a_tool"}
    assert first.observation is not None
    assert "unknown_op" in first.observation or "error" in first.observation


def test_second_tool_call_in_one_turn_is_not_executed(catalog, log_task):
    row = log_task.oracle.ledger_tail[0]
    log_args = {
        "food_id": row.food_id,
        "grams": row.grams,
        "eaten_at": row.eaten_at,
    }
    teacher = ScriptedFCTeacher([
        (
            "two calls, only first",
            [
                _call("log_meal", log_args, call_id="call_a"),
                _call("get_profile", {}, call_id="call_b"),
            ],
        ),
        ("finish", [_call("done", {}, call_id="call_f")]),
    ])
    episode = rollout_tool_call(log_task, teacher_complete=teacher, catalog=catalog)
    first = episode.turns[0]
    assert len(first.tool_calls) == 2
    assert first.executed_op["op"] == "log_meal"
    assert all(
        t.executed_op is None or t.executed_op.get("op") != "get_profile"
        for t in episode.turns
    )


def test_reasoning_not_collapsed_into_content(catalog, log_task):
    teacher = ScriptedFCTeacher(log_script(log_task))
    episode = rollout_tool_call(log_task, teacher_complete=teacher, catalog=catalog)
    for turn in episode.turns:
        if turn.reasoning_content:
            assert turn.content is None
            assert turn.reasoning_content not in (turn.content or "")


def test_reuses_lab_loop_not_a_copy(catalog, log_task, monkeypatch):
    calls: list[int] = []
    real = lab_loop

    def wrapped(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(
        "src.training.data_factory.rollout_fc.run_episode_tool_call", wrapped
    )
    teacher = ScriptedFCTeacher(log_script(log_task))
    rollout_tool_call(log_task, teacher_complete=teacher, catalog=catalog)
    assert calls == [1]


def test_no_network_without_flag(catalog, log_task, monkeypatch):
    monkeypatch.delenv("NUTRIMIND_ALLOW_NETWORK", raising=False)

    def forbidden(_request):
        raise RuntimeError(
            "real network disabled: set NUTRIMIND_ALLOW_NETWORK=1 to "
            "call the ark endpoint"
        )

    episode = rollout_tool_call(
        log_task, teacher_complete=forbidden, catalog=catalog
    )
    assert episode.error is not None
    assert "NUTRIMIND_ALLOW_NETWORK" in episode.error


def test_parallel_tool_calls_rejected(catalog, log_task):
    teacher = ScriptedFCTeacher([])
    with pytest.raises(ValueError, match="parallel_tool_calls"):
        rollout_tool_call(
            log_task,
            teacher_complete=teacher,
            catalog=catalog,
            parallel_tool_calls=True,
        )
