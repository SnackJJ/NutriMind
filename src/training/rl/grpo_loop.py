"""veRL GRPO wiring — testable, injected policy_spec (RL ticket 007).

Does not extend v1 GRPO / GiGPO trainers. Checkpoint→URL glue is ticket 008.
This loop consumes D3 rollouts + ticket-003 rewards.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from src.training.data_factory.concepts import TaskPackage
from src.training.data_factory.verify import verify
from src.training.rl.advantage import DroppedGroup, advantage
from src.training.rl.arm import assert_arm
from src.training.rl.rollout import student_rollout

__all__ = ["grpo_group_step"]


def grpo_group_step(
    packages: Sequence[TaskPackage],
    *,
    policy_spec: dict,
    arm: dict,
    estimator: str = "grpo",
    verify_fn: Callable = verify,
) -> dict:
    """One scripted/dry group. Injected ``policy_spec`` (no weight sync here)."""
    assert_arm(arm)
    if arm["advantage_estimator"] != estimator:
        raise ValueError("arm advantage_estimator does not match estimator argument")
    issued = 0
    results = []
    dropped = 0
    for package in packages:
        episodes = student_rollout(
            policy_spec, package, k=int(arm["rollout_k"]), seed=0
        )
        issued += len(episodes)
        verifications = [verify_fn(package, episode) for episode in episodes]
        try:
            adv = advantage(verifications, estimator=estimator)
        except DroppedGroup:
            dropped += 1
            results.append({"task_id": package.task_id, "dropped": True, "advantages": []})
            continue
        results.append(
            {
                "task_id": package.task_id,
                "dropped": False,
                "advantages": adv,
            }
        )
    nonzero = any(
        value not in (None, 0.0)
        for row in results
        if not row["dropped"]
        for value in row["advantages"]
    )
    return {
        "rollouts_issued": issued,
        "dropped_groups": dropped,
        "nonzero_advantage": nonzero,
        "groups": results,
    }
