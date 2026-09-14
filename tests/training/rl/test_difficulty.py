"""RL ticket 004 — p̂ excludes indeterminate; bands are checkpoint-bound; drop zero-var."""

from __future__ import annotations

import pytest

from src.training.data_factory.concepts import VerificationResult
from src.training.rl.difficulty import (
    DifficultyTable,
    drop_zero_variance_group,
    hat_p,
    per_family_hat_p,
)
from src.training.rl.reward import reward_from_verification


def _vr(status: str) -> VerificationResult:
    reward = {"pass": 1.0, "fail": 0.0, "indeterminate": None}[status]
    return VerificationResult(
        status=status,
        execution="ok" if status != "indeterminate" else "no_finish",
        oracle_exec="ok",
        scorer="pass" if status == "pass" else "fail" if status == "fail" else None,
        reward=reward,
        oracle_version="x",
        rubric_version="v2-r1",
        reward_version="v2-r1",
    )


def test_hat_p_excludes_indeterminate():
    assert hat_p(["pass", "fail", "indeterminate"]) == 0.5
    assert hat_p(["indeterminate", "indeterminate"]) is None
    assert hat_p(["pass", "pass", "fail"]) == pytest.approx(2 / 3)


def test_band_refused_for_other_checkpoint():
    table = DifficultyTable("ckpt-a", {"log": (0.2, 0.8), "recommend": (0.1, 0.5)})
    assert table.band_for("log", checkpoint_hash="ckpt-a") == (0.2, 0.8)
    with pytest.raises(ValueError, match="refused"):
        table.band_for("log", checkpoint_hash="ckpt-b")
    agg = per_family_hat_p(
        [
            {"family": "log", "statuses": ["pass", "fail"]},
            {"family": "log", "statuses": ["pass"]},
            {"family": "recommend", "statuses": ["fail", "fail"]},
        ]
    )
    assert agg["log"] == pytest.approx(2 / 3)
    assert agg["recommend"] == 0.0


def test_zero_variance_group_dropped_not_resampled():
    all_pass = [_vr("pass"), _vr("pass"), _vr("indeterminate")]
    assert drop_zero_variance_group(all_pass) is True
    mixed = [_vr("pass"), _vr("fail")]
    assert drop_zero_variance_group(mixed) is False
    too_few = [_vr("pass"), _vr("indeterminate")]
    assert drop_zero_variance_group(too_few) is True
    # drop means draw the next batch — the function does not take a resample flag
    import inspect

    assert "resample" not in inspect.getsource(drop_zero_variance_group)
