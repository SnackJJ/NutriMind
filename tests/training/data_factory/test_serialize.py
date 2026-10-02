"""Ticket 026 — serialize seam: native tool-calling v2 SFT records (spec §9.2).

Offline: Pass episodes come from the ticket-025 scripted FC teacher through
the lab loop. Ticket 008's text-op tests are superseded.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench import Scorer  # noqa: E402
from nutrienv.harness.runner import FINISH_OPS  # noqa: E402
from nutrienv.harness.tools_schema import TOOL_SYSTEM_PROMPT  # noqa: E402

from src.training.data_factory import materialize as mz  # noqa: E402
from src.training.data_factory import serialize as sz  # noqa: E402
from src.training.data_factory.config import load_config  # noqa: E402
from src.training.data_factory.concepts import (  # noqa: E402
    EpisodeResult,
    VerificationResult,
)
from src.training.data_factory.materialize import RunContext  # noqa: E402
from src.training.data_factory.rollout_fc import (  # noqa: E402
    ScriptedFCTeacher,
    rollout_tool_call,
)
from src.training.data_factory.serialize import SerializeError  # noqa: E402

from tests.training.data_factory import _fixtures as fx  # noqa: E402


@pytest.fixture(scope="module")
def catalog():
    return fx.gold_catalog()


@pytest.fixture(scope="module")
def config():
    return load_config("configs/data_factory.yaml")


@pytest.fixture(scope="module")
def log_task(catalog):
    return fx.make_log_task(catalog, fx.first_person(), seed=30)


def _call(name: str, args: dict, *, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def fc_log_script(task, *, finish=True):
    turns = []
    for index, row in enumerate(task.oracle.ledger_tail, start=1):
        args = {
            "food_id": row.food_id,
            "grams": row.grams,
            "eaten_at": row.eaten_at,
        }
        turns.append((
            "I should log my lunch.",
            [_call("log_meal", args, call_id=f"call_{index}")],
        ))
    if finish:
        turns.append((
            "All logged, finishing.",
            [_call("done", {}, call_id=f"call_{len(turns) + 1}")],
        ))
    return turns


def _pass_verification(package) -> VerificationResult:
    return VerificationResult(
        status="pass",
        execution="ok",
        oracle_exec="ok",
        scorer="pass",
        reward=1.0,
        oracle_version=package.oracle.oracle_version,
        rubric_version="v2-r1",
        reward_version="v2-r1",
    )


@pytest.fixture(scope="module")
def pass_artifacts(catalog, config, log_task):
    """(package, episode, verification) for one scripted Pass log episode."""
    episode = rollout_tool_call(
        log_task,
        teacher_complete=ScriptedFCTeacher(fc_log_script(log_task)),
        catalog=catalog,
    )
    assert Scorer().score(episode.end_state, log_task.oracle)["passed"] is True
    package = mz.materialize(
        log_task,
        RunContext(
            catalog=catalog,
            catalog_sha=mz.catalog_digest(catalog),
            nutrienv_rev=config.nutrienv_rev,
            nutrimind_rev="d" * 40,
            config_sha="0" * 64,
            seed=30,
            built_at="2026-09-09T12:00:00+00:00",
        ),
    )
    return package, episode, _pass_verification(package)


# --------------------------------------------------------------------------- #
# the happy record
# --------------------------------------------------------------------------- #


def test_pass_episode_record_structure(pass_artifacts, config):
    package, episode, verification = pass_artifacts
    record = sz.serialize(
        package, episode, verification, config=config, accepted_from_attempt=1
    )

    assert record["schema_version"] == "nutrimind-v2-sft/1"
    assert record["task_key"] == package.task_key
    assert record["task_id"] == package.task_id
    assert record["task_package_ref"] == f"task_packages/{package.task_id}.json"
    assert record["accepted_from_attempt"] == 1

    messages, segments, train_on = (
        record["messages"], record["segments"], record["train_on"]
    )
    assert len(messages) == len(segments) == len(train_on)
    assert segments[-1] == "final"
    assert train_on == [s in ("step", "final") for s in segments]

    # system once, lab TOOL_SYSTEM_PROMPT; then the Task turn
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == TOOL_SYSTEM_PROMPT
    assert messages[1] == {"role": "user", "content": f"Task:\n{package.query}"}
    assert set(segments) <= {"system", "task", "step", "tool", "final"}
    assert "observation" not in segments
    assert train_on == [s in ("step", "final") for s in segments]
    assert all(
        flag is False or seg in ("step", "final")
        for flag, seg in zip(train_on, segments)
    )

    assistants = [m for m in messages if m["role"] == "assistant"]
    tools = [m for m in messages if m["role"] == "tool"]
    assert len(assistants) == len(episode.turns)
    assert len(tools) == len(episode.turns) - 1
    for message, turn in zip(assistants, episode.turns):
        assert message.get("tool_calls")
        assert message.get("content") is None
        assert message.get("reasoning_content") == turn.reasoning_content
        name = message["tool_calls"][0]["function"]["name"]
        if turn is episode.turns[-1]:
            assert name in FINISH_OPS
        else:
            assert name == (turn.executed_op or {}).get("op")
    last = messages[-1]
    assert last["role"] == "assistant"
    assert last["tool_calls"][0]["function"]["name"] in FINISH_OPS

    for message in messages:
        if message["role"] == "assistant":
            blob = (message.get("reasoning_content") or "") + json.dumps(
                message.get("tool_calls")
            )
            assert not any(
                marker in blob
                for marker in ("<tool_call>", "<think>", "<|im_start|>")
            )

    # determinism: pure function of its inputs
    again = sz.serialize(
        package, episode, verification, config=config, accepted_from_attempt=1
    )
    assert again == record


def test_meta_version_block_complete(pass_artifacts, config):
    package, episode, verification = pass_artifacts
    record = sz.serialize(package, episode, verification, config=config)
    meta = record["meta"]
    assert meta["family"] == "log"
    assert meta["steps"] == ["log"]
    assert meta["tier"] == ""
    assert meta["persona"] == episode.task.persona
    assert meta["batch"] == 1
    assert meta["seed"] == 30
    assert meta["teacher"] == config.teacher.model_id
    assert meta["teacher_params"] == {
        "thinking": {"type": "enabled"},
        "temperature_first": 0.0,
        "temperature_retry": 0.7,
    }
    assert meta["expander"] == config.expander.model_id
    assert meta["verification"]["status"] == "pass"
    assert meta["verification"]["reward"] == 1.0
    assert meta["oracle_version"] == package.oracle.oracle_version
    assert meta["rubric_version"] == "v2-r1"
    assert meta["reward_version"] == "v2-r1"
    assert meta["environment_version"] == f"nutrienv-{config.nutrienv_rev[:7]}"
    assert meta["task_schema_version"] == package.schema_version
    assert meta["catalog_sha"] == package.catalog.catalog_sha
    assert meta["nutrienv_rev"] == config.nutrienv_rev
    assert meta["nutrimind_rev"] == "d" * 40
    assert meta["n_steps"] == len(episode.turns)
    assert meta["n_turns_without_plan"] == 0
    assert meta["plan_truncation"] == "chars4"


# --------------------------------------------------------------------------- #
# serialize-edge failures
# --------------------------------------------------------------------------- #


def test_empty_episode(pass_artifacts, config):
    package, _, verification = pass_artifacts
    empty = EpisodeResult(task=package and None)
    with pytest.raises(SerializeError, match="serialize.empty_episode"):
        sz.serialize(package, empty, verification, config=config)


def test_last_turn_not_finish(catalog, config, log_task):
    episode = rollout_tool_call(
        log_task,
        teacher_complete=ScriptedFCTeacher(fc_log_script(log_task, finish=False)),
        catalog=catalog,
    )
    assert episode.reached_finish is False
    package = mz.materialize(
        log_task,
        RunContext(catalog=catalog, catalog_sha=mz.catalog_digest(catalog),
                   nutrienv_rev=config.nutrienv_rev, nutrimind_rev="d" * 40,
                   config_sha="0" * 64, seed=30),
    )
    with pytest.raises(SerializeError, match="serialize.last_turn_not_finish"):
        sz.serialize(package, episode, _pass_verification(package), config=config)


def test_missing_observation(pass_artifacts, config):
    package, episode, verification = pass_artifacts
    broken = dataclasses.replace(
        episode,
        turns=[
            dataclasses.replace(episode.turns[0], observation=None),
            *episode.turns[1:],
        ],
    )
    with pytest.raises(SerializeError, match="serialize.missing_observation"):
        sz.serialize(package, broken, verification, config=config)


def test_too_long(pass_artifacts, config):
    package, episode, verification = pass_artifacts
    tiny = dataclasses.replace(config, max_seq_tokens=5)
    with pytest.raises(SerializeError, match="serialize.too_long"):
        sz.serialize(package, episode, verification, config=tiny)


def test_no_plan_any_turn(pass_artifacts, config):
    package, episode, verification = pass_artifacts
    planless = dataclasses.replace(
        episode,
        turns=[
            dataclasses.replace(t, reasoning_content=None) for t in episode.turns
        ],
    )
    with pytest.raises(SerializeError, match="serialize.no_plan_any_turn"):
        sz.serialize(package, planless, verification, config=config)


def test_one_turn_without_plan_tolerated(pass_artifacts, config):
    package, episode, verification = pass_artifacts
    partial = dataclasses.replace(
        episode,
        turns=[
            dataclasses.replace(episode.turns[0], reasoning_content=None),
            *episode.turns[1:],
        ],
    )
    record = sz.serialize(package, partial, verification, config=config)
    assert record["meta"]["n_turns_without_plan"] == 1
    first_assistant = next(m for m in record["messages"] if m["role"] == "assistant")
    assert first_assistant["content"] is None
    assert first_assistant.get("reasoning_content") is None
    assert first_assistant["tool_calls"]


def test_text_op_without_tool_calls_is_not_accepted(pass_artifacts, config):
    """Retired text-op blob (plan + {\"op\"}) with no tool_calls is rejected."""
    package, episode, verification = pass_artifacts
    fabricated = dataclasses.replace(
        episode,
        turns=[
            dataclasses.replace(
                episode.turns[0],
                tool_calls=[],
                tool_call_id=None,
                content='I should log my lunch.\n{"op": "log_meal"}',
            ),
            *episode.turns[1:],
        ],
    )
    with pytest.raises(SerializeError, match="serialize.no_tool_calls"):
        sz.serialize(package, fabricated, verification, config=config)


# --------------------------------------------------------------------------- #
# plan truncation
# --------------------------------------------------------------------------- #


def test_plan_truncated_chars_heuristic(pass_artifacts, config):
    package, episode, verification = pass_artifacts
    small = dataclasses.replace(config, plan_max_tokens=3)  # 3 * 4 = 12 chars
    long_plan = "Plan: first search the catalog for oatmeal then log carefully."
    padded = dataclasses.replace(
        episode,
        turns=[
            dataclasses.replace(episode.turns[0], reasoning_content=long_plan),
            *episode.turns[1:],
        ],
    )
    record = sz.serialize(package, padded, verification, config=small)
    first_assistant = next(m for m in record["messages"] if m["role"] == "assistant")
    assert first_assistant["reasoning_content"] == long_plan[:12]
    assert first_assistant["tool_calls"]
    assert record["meta"]["plan_truncation"] == "chars4"


class WordTokenizer:
    """Fake student tokenizer: whitespace 'tokens'."""

    def encode(self, text):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


def test_plan_truncated_token_exact(pass_artifacts, config):
    package, episode, verification = pass_artifacts
    small = dataclasses.replace(config, plan_max_tokens=3)
    long_plan = "one two three four five six seven"
    padded = dataclasses.replace(
        episode,
        turns=[
            dataclasses.replace(episode.turns[0], reasoning_content=long_plan),
            *episode.turns[1:],
        ],
    )
    record = sz.serialize(
        package, padded, verification, config=small, tokenizer=WordTokenizer()
    )
    first_assistant = next(m for m in record["messages"] if m["role"] == "assistant")
    assert first_assistant["reasoning_content"] == "one two three"
    assert first_assistant["tool_calls"]
    assert record["meta"]["plan_truncation"] == "token"


def test_v1_markers_stripped_from_plan(pass_artifacts, config):
    package, episode, verification = pass_artifacts
    marked = dataclasses.replace(
        episode,
        turns=[
            dataclasses.replace(
                episode.turns[0],
                reasoning_content="<think>chain</think> Log the meal now.",
            ),
            *episode.turns[1:],
        ],
    )
    record = sz.serialize(package, marked, verification, config=config)
    plan = next(m for m in record["messages"] if m["role"] == "assistant")[
        "reasoning_content"
    ]
    assert "<think>" not in plan and "</think>" not in plan
    assert plan.startswith("chain Log the meal now.")


def test_fc_record_has_tool_observations(pass_artifacts, config):
    package, episode, verification = pass_artifacts
    record = sz.serialize(package, episode, verification, config=config)
    tools = [m for m in record["messages"] if m["role"] == "tool"]
    assert tools
    assert all(m.get("tool_call_id") for m in tools)
    assert all(m.get("content") for m in tools)


# --------------------------------------------------------------------------- #
# validate_record — the loader-mirror structural codes
# --------------------------------------------------------------------------- #


def _base_record(pass_artifacts, config):
    package, episode, verification = pass_artifacts
    return sz.serialize(package, episode, verification, config=config)


def test_validate_no_system_turn(pass_artifacts, config):
    record = _base_record(pass_artifacts, config)
    del record["messages"][0]
    del record["segments"][0]
    del record["train_on"][0]
    with pytest.raises(SerializeError, match="serialize.no_system_turn"):
        sz.validate_record(record)


def test_validate_consecutive_assistant(pass_artifacts, config):
    record = _base_record(pass_artifacts, config)
    record["messages"].insert(
        4,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("done", {}, call_id="dup")],
        },
    )
    record["segments"].insert(4, "step")
    record["train_on"].insert(4, True)
    with pytest.raises(SerializeError, match="serialize.consecutive_assistant"):
        sz.validate_record(record)


def test_validate_turn_count_mismatch(pass_artifacts, config):
    record = _base_record(pass_artifacts, config)
    record["segments"].pop()
    with pytest.raises(SerializeError, match="serialize.turn_count_mismatch"):
        sz.validate_record(record)


def test_validate_empty_record():
    with pytest.raises(SerializeError, match="serialize.empty_episode"):
        sz.validate_record({"messages": [], "segments": [], "train_on": []})


def test_validate_v1_marker_rejected(pass_artifacts, config):
    record = _base_record(pass_artifacts, config)
    asst = next(m for m in record["messages"] if m["role"] == "assistant")
    asst["reasoning_content"] = "<think>x</think>" + (asst.get("reasoning_content") or "")
    with pytest.raises(SerializeError, match="serialize.v1_marker"):
        sz.validate_record(record)


def test_hand_in_turn_takes_the_final_plan_budget(pass_artifacts, config):
    package, episode, verification = pass_artifacts
    budgeted = dataclasses.replace(config, plan_max_tokens=2, final_plan_max_tokens=5)
    long_plan = "one two three four five six seven"
    padded = dataclasses.replace(
        episode,
        turns=[dataclasses.replace(turn, reasoning_content=long_plan)
               for turn in episode.turns],
    )
    record = sz.serialize(
        package, padded, verification, config=budgeted, tokenizer=WordTokenizer()
    )
    plans = [m["reasoning_content"] for m in record["messages"] if m["role"] == "assistant"]
    assert len(plans) >= 2
    assert plans[:-1] == ["one two"] * (len(plans) - 1)
    assert plans[-1] == "one two three four five"
    assert record["meta"]["final_plan_max_tokens"] == 5
