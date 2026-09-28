"""Arm startup assertions + manifest provenance (RL ticket 006 / spec D9)."""

from __future__ import annotations

from src.training.rl.exam_gate import assert_lab_at_rev, pinned_exam_blob

__all__ = ["ArmConfig", "ArmError", "assert_arm", "arm_manifest"]

_REQUIRED = (
    "reward_version",
    "reference_model_revision",
    "task_selection_policy",
    "band",
    "advantage_estimator",
    "rollout_k",
    "parallel_tool_calls",
    "exam_revision",
)


class ArmError(RuntimeError):
    """Arm configuration mismatch — abort before rollout spend."""


class ArmConfig(dict):
    """Typed-enough mapping for the asserted fields."""


def assert_arm(config: dict, *, expected: dict | None = None) -> dict:
    """Print/assert arm fields. Mismatch raises before any rollout."""
    missing = [key for key in _REQUIRED if key not in config]
    if missing:
        raise ArmError(f"arm config missing {missing}")
    if config["parallel_tool_calls"] is not False:
        raise ArmError("parallel_tool_calls must be false")
    if int(config["rollout_k"]) < 1:
        raise ArmError("rollout_k must be >= 1")
    assert_lab_at_rev()
    live_exam = pinned_exam_blob()
    if config["exam_revision"] != live_exam:
        raise ArmError(
            f"exam_revision {config['exam_revision']!r} != pin {live_exam!r}"
        )
    if expected is not None:
        for key, value in expected.items():
            if config.get(key) != value:
                raise ArmError(f"arm mismatch {key}: {config.get(key)!r} != {value!r}")
    return {key: config[key] for key in _REQUIRED}


def arm_manifest(config: dict, *, factory_provenance: dict, effective_gradient_fraction: float) -> dict:
    """Diffable against factory run_manifest provenance plus gradient fraction."""
    asserted = assert_arm(config)
    return {
        "schema_version": "nutrimind-rl-arm/1",
        "arm": asserted,
        "provenance": {
            "nutrienv_rev": factory_provenance.get("nutrienv_rev"),
            "nutrimind_rev": factory_provenance.get("nutrimind_rev"),
            "catalog_sha": factory_provenance.get("catalog_sha"),
            "config_sha": factory_provenance.get("config_sha"),
            "oracle_version": factory_provenance.get("oracle_version"),
            "rubric_version": factory_provenance.get("rubric_version"),
            "reward_version": factory_provenance.get("reward_version")
            or asserted["reward_version"],
        },
        "effective_gradient_fraction": effective_gradient_fraction,
    }
