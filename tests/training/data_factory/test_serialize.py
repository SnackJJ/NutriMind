"""Ticket 008 — serialize seam: the v2 SFT record (Seam 4, spec §9.2/§19.4).

Offline: Pass episodes come from the ticket-009 scripted teacher through the
real env, packages from ticket 006, verification from ticket 007 — the full
shared-type chain — then ``serialize`` runs against the real
``configs/data_factory.yaml`` knobs.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.harness.react import react_manual  # noqa: E402

from src.training.data_factory import materialize as mz  # noqa: E402
from src.training.data_factory import serialize as sz  # noqa: E402
from src.training.data_factory import verify as vf  # noqa: E402
from src.training.data_factory.config import load_config  # noqa: E402
from src.training.data_factory.concepts import EpisodeResult, TurnMeta  # noqa: E402
from src.training.data_factory.materialize import RunContext  # noqa: E402
from src.training.data_factory.rollout import (  # noqa: E402
    ScriptedTeacher,
    TeacherReActHarness,
    rollout,
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


@pytest.fixture(scope="module")
def pass_artifacts(catalog, config, log_task):
    """(package, episode, verification) for one scripted Pass log episode."""
    turns = []
    for row in log_task.oracle.ledger_tail:
        action = {
            "op": "log_meal", "food_id": row.food_id,
            "grams": row.grams, "eaten_at": row.eaten_at,
        }
        turns.append((json.dumps(action), "I should log my lunch."))
    turns.append(('{"op": "done"}', "All logged, finishing."))
    episode = rollout(
        TeacherReActHarness(teacher_complete=ScriptedTeacher(turns)), log_task
    )
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
    verification = vf.verify(package, episode)
    assert verification.status == "pass"
    return package, episode, verification


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

    # system once, frozen v2 manual; then the Task turn
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == react_manual("v2")
    assert messages[1] == {"role": "user", "content": f"Task:\n{package.query}"}

    # observations carry the step-budget line and the capped env observation
    observation_messages = [
        m for m, s in zip(messages, segments) if s == "observation"
    ]
    assert all(
        m["content"].startswith("Step budget: ")
        and "\nObservation:\n" in m["content"]
        for m in observation_messages
    )

    # assistant content = plan + op_json; op_json parses back to the executed op
    assistant_pairs = [
        (m, turn)
        for m, turn in zip(messages[3::2], episode.turns)
    ]
    for message, turn in assistant_pairs:
        plan, _, op_json = message["content"].rpartition("\n")
        assert json.loads(op_json) == turn.executed_op
        if turn.reasoning_content:
            assert plan == turn.reasoning_content

    # no v1 markers anywhere in assistant content
    for message in messages:
        if message["role"] == "assistant":
            assert not any(
                marker in message["content"]
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
    script = [('{"op": "get_profile"}', "checking.")] * 12
    episode = rollout(
        TeacherReActHarness(teacher_complete=ScriptedTeacher(script)), log_task
    )
    assert episode.reached_finish is False
    package = mz.materialize(
        log_task,
        RunContext(catalog=catalog, catalog_sha=mz.catalog_digest(catalog),
                   nutrienv_rev=config.nutrienv_rev, nutrimind_rev="d" * 40,
                   config_sha="0" * 64, seed=30),
    )
    verification = vf.verify(package, episode)
    with pytest.raises(SerializeError, match="serialize.last_turn_not_finish"):
        sz.serialize(package, episode, verification, config=config)


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
    # that turn's content is the bare op_json and still parses
    first_assistant = record["messages"][3]
    assert first_assistant["content"].startswith('{"op"')
    assert json.loads(first_assistant["content"]) == episode.turns[0].executed_op


def test_invalid_op_turn_produces_no_record(pass_artifacts, config):
    """A turn whose raw text does not re-parse to the executed op — never
    serialized (verify already routes these to teacher_invalid_op)."""
    package, episode, verification = pass_artifacts
    fabricated = dataclasses.replace(
        episode,
        turns=[
            dataclasses.replace(
                episode.turns[0], raw_action_text="no action json at all"
            ),
            *episode.turns[1:],
        ],
    )
    with pytest.raises(SerializeError, match="serialize.invalid_op_turn"):
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
    first_assistant = record["messages"][3]["content"]
    plan, _, op_json = first_assistant.rpartition("\n")
    assert plan == long_plan[:12]
    assert json.loads(op_json) == episode.turns[0].executed_op
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
    plan, _, op_json = record["messages"][3]["content"].rpartition("\n")
    assert plan == "one two three"  # exactly 3 tokens
    assert json.loads(op_json) == episode.turns[0].executed_op
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
    content = record["messages"][3]["content"]
    # the MARKERS are stripped (spec §9.2); the plan text between them remains
    assert "<think>" not in content and "</think>" not in content
    assert content.startswith("chain Log the meal now.\n{")


def test_reset_observation_fallback(pass_artifacts, config):
    """Episodes recorded without reset_observation (older episodes): serialize
    re-derives it deterministically from the episode's task s0."""
    package, episode, verification = pass_artifacts
    legacy = dataclasses.replace(episode, reset_observation=None)
    record = sz.serialize(package, legacy, verification, config=config)
    first_observation = record["messages"][2]["content"]
    assert "\nObservation:\n" in first_observation
    # and equals the freshly reset observation of the same s0
    from nutrienv.env import NutriEnv

    fresh = NutriEnv().reset(episode.task.s0)
    assert json.dumps(fresh, default=str)[:6000] in first_observation


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
        4, {"role": "assistant", "content": '{"op": "done"}'}
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
    record["messages"][3]["content"] = "<think>x</think>\n" + record["messages"][3]["content"]
    with pytest.raises(SerializeError, match="serialize.v1_marker"):
        sz.validate_record(record)
