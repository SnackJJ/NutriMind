"""RL ticket 006 — arm assertions abort before rollout; manifest provenance."""

from __future__ import annotations

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from src.training.rl.arm import ArmError, arm_manifest, assert_arm  # noqa: E402
from src.training.rl.exam_gate import pinned_exam_blob  # noqa: E402


def _cfg(**overrides):
    base = {
        "reward_version": "v2-r1",
        "reference_model_revision": "ckpt-a",
        "task_selection_policy": "dynamic",
        "band": {"log": (0.2, 0.8)},
        "advantage_estimator": "grpo",
        "rollout_k": 4,
        "parallel_tool_calls": False,
        "exam_revision": pinned_exam_blob(),
    }
    base.update(overrides)
    return base


def test_mismatch_aborts_zero_rollouts():
    issued = []

    def rollout():
        issued.append(1)

    with pytest.raises(ArmError, match="exam_revision"):
        assert_arm(_cfg(exam_revision="deadbeef"))
        rollout()
    assert issued == []

    with pytest.raises(ArmError, match="arm mismatch"):
        assert_arm(_cfg(), expected={"advantage_estimator": "dapo"})
    assert issued == []


def test_match_emits_fields_and_manifest_provenance():
    asserted = assert_arm(_cfg())
    assert asserted["reward_version"] == "v2-r1"
    assert asserted["parallel_tool_calls"] is False
    assert asserted["exam_revision"] == pinned_exam_blob()
    manifest = arm_manifest(
        _cfg(),
        factory_provenance={
            "nutrienv_rev": "0ee68eaa6c246e8079915761c95fc986c53d4979",
            "nutrimind_rev": "d" * 40,
            "catalog_sha": "abc",
            "config_sha": "0" * 64,
            "oracle_version": "nutrienv-0ee68ea",
            "rubric_version": "v2-r1",
            "reward_version": "v2-r1",
        },
        effective_gradient_fraction=0.5,
    )
    assert manifest["provenance"]["catalog_sha"] == "abc"
    assert manifest["provenance"]["reward_version"] == "v2-r1"
    assert manifest["effective_gradient_fraction"] == 0.5
    assert set(manifest["provenance"]) >= {
        "nutrienv_rev",
        "nutrimind_rev",
        "catalog_sha",
        "config_sha",
        "reward_version",
    }
