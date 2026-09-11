"""RL ticket 001 — student_rollout seam, scripted FC policy, no network."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from src.training.data_factory.materialize import RunContext, catalog_digest, materialize  # noqa: E402
from src.training.data_factory.rollout_fc import ScriptedFCTeacher  # noqa: E402
from src.training.rl.rollout import student_rollout  # noqa: E402

from tests.training.data_factory import _fixtures as fx  # noqa: E402


def _call(name: str, args: dict, *, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


@pytest.fixture(scope="module")
def catalog():
    return fx.gold_catalog()


@pytest.fixture(scope="module")
def log_task(catalog):
    return fx.make_log_task(catalog, fx.first_person(), seed=30)


@pytest.fixture(scope="module")
def package(catalog, log_task):
    return materialize(
        log_task,
        RunContext(
            catalog=catalog,
            catalog_sha=catalog_digest(catalog),
            nutrienv_rev="0ee68eaa6c246e8079915761c95fc986c53d4979",
            nutrimind_rev="d" * 40,
            config_sha="0" * 64,
            seed=30,
            built_at="2026-09-11T12:00:00+00:00",
        ),
    )


def _pass_script(task):
    turns = []
    for index, row in enumerate(task.oracle.ledger_tail, start=1):
        args = {
            "food_id": row.food_id,
            "grams": row.grams,
            "eaten_at": row.eaten_at,
        }
        turns.append(("log", [_call("log_meal", args, call_id=f"c{index}")]))
    turns.append(("done", [_call("done", {}, call_id="cf")]))
    return turns


def test_k_episodes_independent(catalog, log_task, package):
    teacher = ScriptedFCTeacher(_pass_script(log_task) * 2)
    spec = {"complete": teacher, "catalog": catalog, "parallel_tool_calls": False}
    episodes = student_rollout(spec, package, k=2, seed=7)
    assert len(episodes) == 2
    assert episodes[0].reached_finish and episodes[1].reached_finish
    assert episodes[0].end_state is not episodes[1].end_state


def test_step_budget_distinct_from_finish(catalog, log_task, package):
    teacher = ScriptedFCTeacher(_pass_script(log_task)[:-1])  # no finish
    spec = {"complete": teacher, "catalog": catalog}
    (episode,) = student_rollout(spec, package, k=1, seed=0)
    assert episode.reached_finish is False
    assert episode.error is None


def test_no_tool_call_distinct(catalog, log_task, package):
    teacher = ScriptedFCTeacher([
        ("talk", []),
        ("done", [_call("done", {}, call_id="cf")]),
    ])
    spec = {"complete": teacher, "catalog": catalog}
    (episode,) = student_rollout(spec, package, k=1, seed=0)
    assert episode.turns[0].tool_calls == []
    assert episode.turns[0].executed_op is None


def test_second_tool_call_not_executed(catalog, log_task, package):
    row = log_task.oracle.ledger_tail[0]
    args = {"food_id": row.food_id, "grams": row.grams, "eaten_at": row.eaten_at}
    teacher = ScriptedFCTeacher([
        (
            "two",
            [
                _call("log_meal", args, call_id="a"),
                _call("get_profile", {}, call_id="b"),
            ],
        ),
        ("done", [_call("done", {}, call_id="cf")]),
    ])
    spec = {"complete": teacher, "catalog": catalog}
    (episode,) = student_rollout(spec, package, k=1, seed=0)
    assert episode.turns[0].executed_op["op"] == "log_meal"
    assert all(
        t.executed_op is None or t.executed_op.get("op") != "get_profile"
        for t in episode.turns
    )


def test_error_observation_preserved(catalog, log_task, package):
    teacher = ScriptedFCTeacher([
        ("bad", [_call("not_a_tool", {}, call_id="x")]),
        ("done", [_call("done", {}, call_id="cf")]),
    ])
    spec = {"complete": teacher, "catalog": catalog}
    (episode,) = student_rollout(spec, package, k=1, seed=0)
    assert episode.turns[0].observation
    assert "error" in episode.turns[0].observation


def test_parallel_rejected(catalog, package):
    spec = {"complete": ScriptedFCTeacher([]), "parallel_tool_calls": True}
    with pytest.raises(ValueError, match="parallel_tool_calls"):
        student_rollout(spec, package, k=1, seed=0)
