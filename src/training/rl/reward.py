"""RL reward is a closed-table projection of the tri-state verifier (ticket 003).

``pass → 1.0``, ``fail → 0.0``, ``indeterminate → None`` (excluded from the
group and from ``p̂``). No other fields. An unknown status raises.
"""

from __future__ import annotations

from src.training.data_factory.concepts import VerificationResult

__all__ = ["REWARD_MAP", "reward_from_verification"]

REWARD_MAP: dict[str, float | None] = {
    "pass": 1.0,
    "fail": 0.0,
    "indeterminate": None,
}


def reward_from_verification(verification: VerificationResult) -> float | None:
    """Project ``verification.status`` through ``REWARD_MAP``. Unknown raises."""
    status = verification.status
    if status not in REWARD_MAP:
        raise ValueError(
            f"unknown verification status {status!r}; "
            f"closed table is {sorted(REWARD_MAP)}"
        )
    return REWARD_MAP[status]
