"""Tests for zero-waste requirements and evaluate success rate improvements:
- Anti-overproduction: intents past target_n skipped_quota_met.
- Deterministic early stopping: 2 consecutive identical failures stop remaining attempts.
- Infra error exponential backoff and non-consumption of teacher_k.
- Budget stop taking effect before the next call.
- Evaluate safe snapping and quantity formatting (12 seeds reproduction).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from nutrienv.world.catalog_store import load_catalog

from src.training.data_factory.author import _author_evaluate, _format_evaluate_food_phrase
from src.training.data_factory.build import (
    TokenMeter,
    _canonical_submission,
    _extract_submission,
    _is_identical_failure,
    _is_infra_error,
    _teacher_stage,
    build,
)
from src.training.data_factory.concepts import AttemptRecord, EpisodeResult, TurnMeta
from src.training.data_factory.config import DataFactoryConfig, FamilyConfig, Pricing, TeacherConfig, TokenRates, load_config
from src.training.data_factory.synthetic import synth_expander
from src.training.data_factory.verify import VerificationResult

CONFIG_PATH = pathlib.Path("configs/data_factory.yaml")


def _make_vr(status: str, failure_codes: list[str] | None = None):
    return VerificationResult(
        status=status,
        execution="ok",
        oracle_exec="ok",
        scorer=status,
        reward=1.0 if status == "pass" else 0.0,
        oracle_version="v2",
        rubric_version="v2-r1",
        reward_version="v2-r1",
        failure_codes=list(failure_codes or []),
        evidence=[],
    )


def test_is_infra_error():
    assert _is_infra_error("provider request failed: HTTP 429")
    assert _is_infra_error("provider request failed: HTTP 500")
    assert _is_infra_error("Step 1 tool call API failure: ToolCallInfraError")
    assert _is_infra_error("connection reset by peer")
    assert _is_infra_error("read timed out")
    assert not _is_infra_error(None)
    assert not _is_infra_error("task_fail: wrong_goal")


def test_is_identical_failure():
    # Attempt 1: wrong_goal, submit_plan reject
    ep1 = EpisodeResult(
        end_state=None,
        turns=[TurnMeta(executed_op={"op": "submit_plan", "verdict": "reject", "reasons": ["kcal_lo"], "items": []})],
        reached_finish=True,
    )
    v1 = _make_vr("fail", ["task_fail", "wrong_goal"])
    a1 = AttemptRecord(attempt_id="t--1", episode=ep1, verification=v1)

    # Attempt 2: same failure
    ep2 = EpisodeResult(
        end_state=None,
        turns=[TurnMeta(executed_op={"op": "submit_plan", "verdict": "reject", "reasons": ["kcal_lo"], "items": []})],
        reached_finish=True,
    )
    v2 = _make_vr("fail", ["task_fail", "wrong_goal"])
    a2 = AttemptRecord(attempt_id="t--2", episode=ep2, verification=v2)

    assert _is_identical_failure(a1, a2)

    # Attempt 3: different submission
    ep3 = EpisodeResult(
        end_state=None,
        turns=[TurnMeta(executed_op={"op": "submit_plan", "verdict": "reject", "reasons": ["kcal_lo", "protein_g_lo"], "items": []})],
        reached_finish=True,
    )
    v3 = _make_vr("fail", ["task_fail", "wrong_goal"])
    a3 = AttemptRecord(attempt_id="t--3", episode=ep3, verification=v3)

    assert not _is_identical_failure(a1, a3)

    # Attempt 4: pass
    v4 = _make_vr("pass", [])
    a4 = AttemptRecord(attempt_id="t--4", episode=ep1, verification=v4)
    assert not _is_identical_failure(a1, a4)


def test_early_stopping_in_teacher_stage(tmp_path):
    catalog = load_catalog()
    config = load_config(CONFIG_PATH)
    family_cfg = FamilyConfig(target_n=1, teacher_k=6, over_generate_x=1.0, gram_anchor=False)
    package = MagicMock()
    package.task_id = "test-early-stop-001"
    task = MagicMock()
    task.id = package.task_id
    task.family = "evaluate"

    call_count = 0

    def mock_rollout(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return EpisodeResult(
            end_state=None,
            turns=[TurnMeta(executed_op={"op": "submit_plan", "verdict": "reject", "reasons": ["kcal_lo"], "items": []})],
            reached_finish=True,
        )

    mock_verify = MagicMock()
    mock_verify.verify.return_value = _make_vr("fail", ["task_fail", "wrong_goal"])

    with patch("src.training.data_factory.build.rollout_tool_call", side_effect=mock_rollout), \
         patch("src.training.data_factory.build.verify_mod", mock_verify):
        cache = _teacher_stage(
            package,
            task,
            config=config,
            family_cfg=family_cfg,
            teacher_complete=lambda req: {},
            catalog=catalog,
            out=tmp_path,
        )

    # Should have stopped after 2 attempts, not running all 6!
    assert call_count == 2
    assert len(cache.attempts) == 2
    assert cache.selected_attempt is None


def test_infra_retry_not_consuming_teacher_k(tmp_path):
    catalog = load_catalog()
    config = load_config(CONFIG_PATH)
    family_cfg = FamilyConfig(target_n=1, teacher_k=2, over_generate_x=1.0, gram_anchor=False)
    package = MagicMock()
    package.task_id = "test-infra-retry-001"
    task = MagicMock()
    task.id = package.task_id
    task.family = "evaluate"

    call_count = 0

    def mock_rollout(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return EpisodeResult(
                end_state=None,
                turns=[],
                reached_finish=False,
                error="teacher error: ToolCallInfraError: HTTP 429",
            )
        return EpisodeResult(
            end_state=None,
            turns=[TurnMeta(executed_op={"op": "submit_plan", "verdict": "accept", "items": [{"food_id": "2705681", "grams": 100}]})],
            reached_finish=True,
        )

    mock_verify = MagicMock()
    mock_verify.verify.return_value = _make_vr("pass", [])

    with patch("src.training.data_factory.build.rollout_tool_call", side_effect=mock_rollout), \
         patch("src.training.data_factory.build.verify_mod", mock_verify), \
         patch("time.sleep"):
        cache = _teacher_stage(
            package,
            task,
            config=config,
            family_cfg=family_cfg,
            teacher_complete=lambda req: {},
            catalog=catalog,
            out=tmp_path,
        )

    assert call_count == 3
    assert len(cache.attempts) == 1
    assert cache.selected_attempt == 0


def test_safe_snapping_and_eval_rewriter_all_12_seeds():
    catalog = load_catalog()
    intents_file = pathlib.Path("data/student/pilot-20260929-cc50/intents/evaluate.jsonl")
    if not intents_file.is_file():
        pytest.skip("pilot-20260929-cc50 intents not found")

    with open(intents_file) as f:
        intents = [json.loads(line) for line in f]

    synth = synth_expander(catalog)

    accepted_count = 0
    for intent in intents:
        task, reject = _author_evaluate(intent, catalog=catalog, expander=synth)
        assert reject is None, f"Seed {intent['seed']} was rejected: {reject}"
        assert task is not None
        accepted_count += 1
        assert "some " not in task.query, f"Seed {intent['seed']} query still has 'some': {task.query}"

    assert accepted_count == 12


def test_format_evaluate_food_phrase():
    catalog = load_catalog()
    phrase1 = _format_evaluate_food_phrase("2705681", 390.0, "named_measure", catalog)
    assert "1.5 cups" in phrase1
    phrase2 = _format_evaluate_food_phrase("2705681", 390.0, "explicit_grams", catalog)
    assert "390 g" in phrase2


def test_anti_overproduction_skipped_quota_met(tmp_path):
    """When target_n is reached, queued intents for that family skip teacher."""
    base_config = load_config(CONFIG_PATH)
    log_cfg = FamilyConfig(target_n=1, teacher_k=1, over_generate_x=3.0, gram_anchor=False)
    config = dataclasses.replace(
        base_config,
        families={"log": log_cfg},
        max_intents=10,
        output_dir=str(tmp_path),
    )

    catalog = load_catalog()
    expander = synth_expander(catalog)

    teacher_calls = 0
    def mock_teacher_complete(request):
        nonlocal teacher_calls
        teacher_calls += 1
        return {
            "content": "",
            "reasoning_content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "submit_plan", "arguments": '{"items": []}'}}],
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

    with patch("src.training.data_factory.build.rollout_tool_call") as mock_rollout, \
         patch("src.training.data_factory.build.verify_mod.verify") as mock_verify:
        mock_rollout.return_value = EpisodeResult(
            end_state=None,
            turns=[
                TurnMeta(
                    reasoning_content="I should log this item to the diary.",
                    executed_op={"op": "submit_plan", "items": []},
                    tool_calls=[{"id": "c1", "type": "function", "function": {"name": "submit_plan", "arguments": '{"items": []}'}}],
                )
            ],
            reached_finish=True,
        )
        mock_verify.return_value = _make_vr("pass", [])

        manifest = build(
            config,
            expander=expander,
            teacher_complete=mock_teacher_complete,
            output_dir=tmp_path,
        )

    assert manifest["counts"]["accepted"] == 1
    assert manifest["counts"]["skipped_quota_met"] >= 1
    assert mock_rollout.call_count == 1


def test_budget_stop_takes_effect_before_next_call(tmp_path):
    """When usd_budget is reached with on_budget: stop, no further calls occur."""
    base_config = load_config(CONFIG_PATH)
    log_cfg = FamilyConfig(target_n=5, teacher_k=1, over_generate_x=2.0, gram_anchor=False)
    config = dataclasses.replace(
        base_config,
        families={"log": log_cfg},
        usd_budget=0.00001,
        on_budget="stop",
        max_intents=10,
        output_dir=str(tmp_path),
    )

    catalog = load_catalog()
    expander = synth_expander(catalog)

    rollout_calls = 0
    def mock_rollout(*args, **kwargs):
        nonlocal rollout_calls
        rollout_calls += 1
        return EpisodeResult(
            end_state=None,
            turns=[
                TurnMeta(
                    reasoning_content="I should log this item to the diary.",
                    executed_op={"op": "submit_plan", "items": []},
                    tool_calls=[{"id": "c1", "type": "function", "function": {"name": "submit_plan", "arguments": '{"items": []}'}}],
                )
            ],
            reached_finish=True,
        )

    from unittest.mock import PropertyMock

    with patch("src.training.data_factory.build.rollout_tool_call", side_effect=mock_rollout), \
         patch("src.training.data_factory.build.verify_mod.verify", return_value=_make_vr("pass", [])), \
         patch.object(TokenMeter, "total", new_callable=PropertyMock, return_value=1000000), \
         patch("src.training.data_factory.build._est_usd", return_value=1.0):

        manifest = build(
            config,
            expander=expander,
            teacher_complete=lambda r: {},
            output_dir=tmp_path,
        )

    assert manifest["status"] == "stopped_budget"
    assert manifest["cost"]["budget_stopped"] is True
    assert manifest["counts"]["accepted"] < 5


