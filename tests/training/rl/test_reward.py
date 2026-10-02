"""RL ticket 003 — reward is a closed-table projection of the verifier."""

from __future__ import annotations

import inspect

import pytest

from src.training.data_factory.concepts import VerificationResult
from src.training.rl.reward import REWARD_MAP, reward_from_verification


def _vr(status: str) -> VerificationResult:
    return VerificationResult(
        status=status,
        execution="ok",
        oracle_exec="ok",
        scorer="pass" if status == "pass" else "fail" if status == "fail" else None,
        reward=REWARD_MAP[status] if status in REWARD_MAP else 0.0,
        oracle_version="x",
        rubric_version="v2-r1",
        reward_version="v2-r1",
    )


def test_dependency_surface_is_only_verification():
    sig = inspect.signature(reward_from_verification)
    assert list(sig.parameters) == ["verification"]
    src = inspect.getsource(reward_from_verification)
    for banned in ("length", "tier", "format", "tool_call", "episode", "query"):
        assert banned not in src


def test_closed_table_and_indeterminate_not_mapped_to_number():
    assert reward_from_verification(_vr("pass")) == 1.0
    assert reward_from_verification(_vr("fail")) == 0.0
    assert reward_from_verification(_vr("indeterminate")) is None
    assert REWARD_MAP["indeterminate"] is None
    with pytest.raises(ValueError, match="unknown verification status"):
        reward_from_verification(_vr("maybe"))
