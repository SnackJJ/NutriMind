"""Measured p̂ per checkpoint, per-family band, zero-variance drop (RL ticket 004)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from src.training.rl.reward import reward_from_verification

__all__ = [
    "DifficultyTable",
    "drop_zero_variance_group",
    "hat_p",
    "per_family_hat_p",
]


def hat_p(statuses: Sequence[str]) -> float | None:
    """n_pass / (n_pass + n_fail). Indeterminate is excluded from both."""
    n_pass = sum(1 for status in statuses if status == "pass")
    n_fail = sum(1 for status in statuses if status == "fail")
    denom = n_pass + n_fail
    if denom == 0:
        return None
    return n_pass / denom


def per_family_hat_p(rows: Sequence[dict]) -> dict[str, float | None]:
    """Aggregate ``hat_p`` per family. Each row: ``family``, ``statuses``."""
    by_family: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].extend(row["statuses"])
    return {family: hat_p(statuses) for family, statuses in by_family.items()}


class DifficultyTable:
    """Sweet-spot bands keyed by checkpoint hash. Cross-checkpoint reuse raises."""

    def __init__(self, checkpoint_hash: str, bands: dict[str, tuple[float, float]]):
        self.checkpoint_hash = checkpoint_hash
        self.bands = dict(bands)

    def band_for(self, family: str, *, checkpoint_hash: str) -> tuple[float, float]:
        if checkpoint_hash != self.checkpoint_hash:
            raise ValueError(
                f"band measured under checkpoint {self.checkpoint_hash} "
                f"refused for {checkpoint_hash}"
            )
        if family not in self.bands:
            raise KeyError(f"no band for family {family!r}")
        return self.bands[family]


def drop_zero_variance_group(verifications: Sequence) -> bool:
    """True → drop the group; caller draws the next batch of tasks, not extra k."""
    rewards = [reward_from_verification(item) for item in verifications]
    valid = [value for value in rewards if value is not None]
    if len(valid) < 2:
        return True
    return len(set(valid)) < 2
