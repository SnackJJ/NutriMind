"""GRPO / DAPO advantage over ticket-003 rewards (RL tickets 007 / 009)."""

from __future__ import annotations

from collections.abc import Sequence

from src.training.data_factory.concepts import VerificationResult
from src.training.rl.difficulty import drop_zero_variance_group
from src.training.rl.reward import reward_from_verification

__all__ = ["DroppedGroup", "advantage", "grpo_advantage", "dapo_advantage"]


class DroppedGroup(Exception):
    """Zero-variance or <2 valid samples — draw the next batch, do not resample."""


def _valid_rewards(verifications: Sequence[VerificationResult]) -> list[float]:
    if drop_zero_variance_group(verifications):
        raise DroppedGroup("zero-variance or <2 valid rewards")
    return [reward_from_verification(item) for item in verifications if reward_from_verification(item) is not None]


def grpo_advantage(verifications: Sequence[VerificationResult]) -> list[float]:
    """Mean-centered GRPO advantages over valid (non-indeterminate) rewards."""
    valid = _valid_rewards(verifications)
    mean = sum(valid) / len(valid)
    out = []
    for item in verifications:
        reward = reward_from_verification(item)
        out.append((reward - mean) if reward is not None else None)
    return out


def dapo_advantage(verifications: Sequence[VerificationResult]) -> list[float]:
    """DAPO-style token-level-ready advantages: (r - mean) / std of valid rewards."""
    valid = _valid_rewards(verifications)
    mean = sum(valid) / len(valid)
    var = sum((value - mean) ** 2 for value in valid) / len(valid)
    std = var ** 0.5
    out = []
    for item in verifications:
        reward = reward_from_verification(item)
        if reward is None:
            out.append(None)
        else:
            out.append((reward - mean) / std)
    return out


def advantage(verifications: Sequence[VerificationResult], *, estimator: str) -> list:
    if estimator == "grpo":
        return grpo_advantage(verifications)
    if estimator == "dapo":
        return dapo_advantage(verifications)
    if estimator == "gigpo":
        raise ValueError("GiGPO is not implemented")
    raise ValueError(f"unknown advantage estimator {estimator!r}")
