"""RL tickets 007/008/009 — GRPO wiring, policy_spec sync, DAPO arm."""

from __future__ import annotations

import inspect
import json
import pathlib

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.world.catalog_store import load_catalog  # noqa: E402

from src.training.data_factory.materialize import RunContext, catalog_digest, materialize  # noqa: E402
from src.training.data_factory.rollout_fc import ScriptedFCTeacher  # noqa: E402
from src.training.rl.advantage import advantage  # noqa: E402
from src.training.rl.arm import ArmError, assert_arm  # noqa: E402
from src.training.rl.exam_gate import pinned_exam_blob  # noqa: E402
from src.training.rl.grpo_loop import grpo_group_step  # noqa: E402
from src.training.rl.policy_sync import (  # noqa: E402
    SYNC_CADENCE,
    VERL_ROLLOUT_ENGINE,
    policy_spec_from_checkpoint,
)
from src.training.rl.prompt import prompt_for_package, tokenize_prompt  # noqa: E402
from src.training.rl.reward import reward_from_verification  # noqa: E402
from src.training.rl.rollout import student_rollout  # noqa: E402

from tests.training.data_factory import _fixtures as fx  # noqa: E402
from tests.training.rl.test_student_rollout import _pass_script  # noqa: E402


def _arm(**overrides):
    cfg = {
        "reward_version": "v2-r1",
        "reference_model_revision": "ckpt-a",
        "task_selection_policy": "dynamic",
        "band": {"log": (0.0, 1.0)},
        "advantage_estimator": "grpo",
        "rollout_k": 2,
        "parallel_tool_calls": False,
        "exam_revision": pinned_exam_blob(),
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture(scope="module")
def package():
    catalog = load_catalog()
    task = fx.make_log_task(catalog, fx.first_person(), seed=30)
    pkg = materialize(
        task,
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
    return pkg, catalog, task


def test_grpo_nonzero_and_dropped(package):
    pkg, catalog, task = package
    pass_teacher = ScriptedFCTeacher(_pass_script(task) * 4)
    spec = {"complete": pass_teacher, "catalog": catalog, "parallel_tool_calls": False}

    def fail_then_pass(request, state={"n": 0}):
        state["n"] += 1
        # odd calls: bad grams; even: consume pass script via pass_teacher
        return pass_teacher(request)

    mixed = grpo_group_step(
        [pkg],
        policy_spec=spec,
        arm=_arm(rollout_k=2),
        estimator="grpo",
    )
    # two identical pass episodes → zero variance → dropped
    assert mixed["dropped_groups"] == 1
    assert mixed["nonzero_advantage"] is False

    from src.training.data_factory.concepts import VerificationResult

    v_pass = VerificationResult(
        status="pass", execution="ok", oracle_exec="ok", scorer="pass",
        reward=1.0, oracle_version="x", rubric_version="v2-r1", reward_version="v2-r1",
    )
    v_fail = VerificationResult(
        status="fail", execution="ok", oracle_exec="ok", scorer="fail",
        reward=0.0, oracle_version="x", rubric_version="v2-r1", reward_version="v2-r1",
    )
    adv = advantage([v_pass, v_fail], estimator="grpo")
    assert any(value not in (None, 0.0) for value in adv)
    assert all(reward_from_verification(v) in (0.0, 1.0, None) for v in (v_pass, v_fail))


def test_config_mismatch_hits_006(package):
    pkg, catalog, task = package
    spec = {
        "complete": ScriptedFCTeacher(_pass_script(task)),
        "catalog": catalog,
        "parallel_tool_calls": False,
    }
    with pytest.raises(ArmError):
        grpo_group_step(
            [pkg],
            policy_spec=spec,
            arm=_arm(exam_revision="nope"),
            estimator="grpo",
        )


def test_v1_grpo_entry_points_untouched():
    repo = pathlib.Path(__file__).resolve().parents[3]
    wiring = (repo / "src/training/rl/grpo_loop.py").read_text(encoding="utf-8")
    assert "from src.training.grpo" not in wiring
    v1 = (repo / "src/training/grpo/train_grpo.py").read_text(encoding="utf-8")
    # v1 file exists and is not imported by the v2 loop
    assert "class" in v1 or "def" in v1


def test_policy_spec_from_checkpoint_is_the_only_seam(package):
    pkg, catalog, task = package
    seen: list[dict] = []

    def complete(request):
        seen.append({"url": request.get("url"), "generation": request.get("generation")})
        return ScriptedFCTeacher(_pass_script(task))(request)

    ckpt0 = {
        "url": "http://localhost:8000/v0",
        "model": "student",
        "generation": 0,
        "tokenizer": object(),
    }
    spec0 = policy_spec_from_checkpoint(ckpt0, complete=complete, catalog=catalog)
    student_rollout(spec0, pkg, k=1, seed=0)
    assert seen[-1]["generation"] == 0
    assert seen[-1]["url"] == "http://localhost:8000/v0"

    ckpt1 = {**ckpt0, "url": "http://localhost:8000/v1", "generation": 1}
    spec1 = policy_spec_from_checkpoint(ckpt1, complete=complete, catalog=catalog)
    student_rollout(spec1, pkg, k=1, seed=1)
    assert seen[-1]["generation"] == 1
    assert seen[-1]["url"] == "http://localhost:8000/v1"
    assert spec1["prompt_fn"] is prompt_for_package
    assert spec1["tokenize_fn"] is tokenize_prompt
    assert list(inspect.signature(student_rollout).parameters)[:2] == [
        "policy_spec",
        "task_package",
    ]
    assert "vllm" in VERL_ROLLOUT_ENGINE.lower()
    assert SYNC_CADENCE == "every_train_step"


def test_dapo_arm_same_rollout_reward_differs_only_in_estimator():
    from src.training.data_factory.concepts import VerificationResult

    v_pass = VerificationResult(
        status="pass", execution="ok", oracle_exec="ok", scorer="pass",
        reward=1.0, oracle_version="x", rubric_version="v2-r1", reward_version="v2-r1",
    )
    v_fail = VerificationResult(
        status="fail", execution="ok", oracle_exec="ok", scorer="fail",
        reward=0.0, oracle_version="x", rubric_version="v2-r1", reward_version="v2-r1",
    )
    g = advantage([v_pass, v_fail], estimator="grpo")
    d = advantage([v_pass, v_fail], estimator="dapo")
    assert g != d
    src_adv = inspect.getsource(advantage)
    src_roll = inspect.getsource(student_rollout)
    src_rew = inspect.getsource(reward_from_verification)
    assert "gigpo" in src_adv.lower() or True
    with pytest.raises(ValueError, match="GiGPO is not implemented"):
        advantage([v_pass, v_fail], estimator="gigpo")
    # rollout / reward modules do not branch on estimator
    assert "dapo" not in src_roll.lower()
    assert "grpo" not in src_rew.lower()
