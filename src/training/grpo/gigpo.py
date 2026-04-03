"""
GiGPO Algorithm Prototype (DEPRECATED for training).

This was an independent implementation for algorithm understanding.
For actual GiGPO training, use verl-agent:
    ./scripts/run_verl_agent_gigpo.sh

Known issues in this prototype:
1. get_action_at_step only uses tool_name without args (step advantage diluted)
2. get_token_level_advantages doesn't account for injected_response offset
3. Not connected to any training loop

Kept for reference and interview discussion.

---

GiGPO (Group-in-Group Policy Optimization) Implementation.

Adds step-level credit assignment on top of GRPO by finding anchor states
where rollouts diverge and computing step-level advantages based on
downstream success rates.

Algorithm:
    GRPO:  advantage = (reward - mean) / std  → trajectory-level only
    GiGPO:
      Layer 1 (group-level): Same as GRPO
      Layer 2 (step-level): Find anchor states where rollouts diverge
           → Steps leading to higher downstream success get positive advantage
           → Steps leading to failure get negative advantage
      Final: advantage = group_advantage × step_advantage

See phase4_grpo.md Task 4.4 for design details.
"""

import json
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.training.grpo.environment import (
    NutriMindEnv as _NutriMindEnv,
    RolloutTrajectory,
    RolloutStep,
    TaskMetadata,
    compute_state_key,
)
from src.training.grpo.reward import RewardBreakdown

_SYSTEM_PROMPT = _NutriMindEnv.SYSTEM_PROMPT


@dataclass
class StepAdvantage:
    """Advantage computed for a single step."""

    step_idx: int
    group_advantage: float  # From GRPO (trajectory-level)
    step_advantage: float  # From GiGPO (step-level)
    combined_advantage: float  # group × step
    anchor_state_key: Optional[str] = None
    num_rollouts_at_anchor: int = 0


@dataclass
class AnchorState:
    """
    An anchor state where multiple rollouts share the same context
    but may take different actions.
    """

    state_key: str
    rollout_indices: List[int]  # Which rollouts reached this state
    actions_taken: Dict[int, str]  # rollout_idx -> action (tool_name or "final_answer")
    downstream_rewards: Dict[int, float]  # rollout_idx -> final reward


@dataclass
class GiGPOResult:
    """Complete GiGPO computation result for a rollout group."""

    prompt: str
    num_rollouts: int
    trajectories: List[RolloutTrajectory]
    rewards: List[float]
    group_advantages: List[float]  # GRPO-style trajectory advantages
    step_advantages: List[List[StepAdvantage]]  # Per-step advantages
    anchor_states: Dict[str, AnchorState]


def compute_group_advantages(rewards: List[float], eps: float = 1e-8) -> List[float]:
    """
    Compute GRPO-style group advantages (trajectory-level).

    advantage = (reward - mean(rewards)) / std(rewards)

    Args:
        rewards: List of trajectory rewards
        eps: Small constant for numerical stability

    Returns:
        List of normalized advantages
    """
    rewards_array = np.array(rewards)
    mean = np.mean(rewards_array)
    std = np.std(rewards_array)

    if std < eps:
        # All rewards nearly identical - no useful signal
        return [0.0] * len(rewards)

    return ((rewards_array - mean) / (std + eps)).tolist()


def build_conversation_at_step(
    trajectory: RolloutTrajectory,
    step_idx: int,
    system_prompt: str,
) -> List[Dict[str, str]]:
    """
    Build conversation history up to (but not including) a specific step.

    This represents the state BEFORE the model made its decision at step_idx.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": trajectory.prompt},
    ]

    for i, step in enumerate(trajectory.steps):
        if i >= step_idx:
            break
        messages.append({"role": "assistant", "content": step.model_output})
        if step.injected_response:
            messages.append({"role": "user", "content": step.injected_response})

    return messages


def get_action_at_step(step: RolloutStep) -> str:
    """Extract the action taken at a step (tool name or 'final_answer')."""
    if step.action_type == "tool_call" and step.tool_execution:
        return step.tool_execution.tool_name
    elif step.action_type == "final_answer":
        return "final_answer"
    else:
        return f"parse_error:{step.step_idx}"


class GiGPOComputer:
    """
    Computes GiGPO step-level advantages for a group of rollouts.

    Key algorithm:
    1. Identify anchor states (same context, different actions)
    2. For each anchor state, compute action → downstream_reward mapping
    3. Step advantage = normalized success rate of chosen action at anchor
    4. Combined advantage = group_advantage × step_advantage
    """

    SYSTEM_PROMPT = _SYSTEM_PROMPT

    def __init__(
        self,
        discount_factor: float = 0.99,
        step_advantage_weight: float = 1.0,
    ):
        """
        Initialize GiGPO computer.

        Args:
            discount_factor: Gamma for discounting future rewards (not used in current impl)
            step_advantage_weight: Weight for step-level advantage (vs group-level)
        """
        self.discount_factor = discount_factor
        self.step_advantage_weight = step_advantage_weight

    def compute(
        self,
        trajectories: List[RolloutTrajectory],
        rewards: List[float],
        task_metadata: TaskMetadata,
    ) -> GiGPOResult:
        """
        Compute GiGPO advantages for a group of rollouts.

        Args:
            trajectories: List of completed rollout trajectories
            rewards: Corresponding reward for each trajectory
            task_metadata: Metadata about the task

        Returns:
            GiGPOResult with all computed advantages
        """
        num_rollouts = len(trajectories)
        prompt = task_metadata.query

        # Step 1: Compute group-level advantages (GRPO)
        group_advantages = compute_group_advantages(rewards)

        # Step 2: Find anchor states
        anchor_states = self._find_anchor_states(trajectories, rewards)

        # Step 3: Compute step-level advantages
        step_advantages = []
        for rollout_idx, trajectory in enumerate(trajectories):
            rollout_step_advs = self._compute_step_advantages_for_rollout(
                rollout_idx=rollout_idx,
                trajectory=trajectory,
                group_advantage=group_advantages[rollout_idx],
                anchor_states=anchor_states,
            )
            step_advantages.append(rollout_step_advs)

        return GiGPOResult(
            prompt=prompt,
            num_rollouts=num_rollouts,
            trajectories=trajectories,
            rewards=rewards,
            group_advantages=group_advantages,
            step_advantages=step_advantages,
            anchor_states=anchor_states,
        )

    def _find_anchor_states(
        self,
        trajectories: List[RolloutTrajectory],
        rewards: List[float],
    ) -> Dict[str, AnchorState]:
        """
        Find anchor states where rollouts diverge.

        An anchor state is a conversation context that multiple rollouts share
        before taking (potentially different) actions.
        """
        # Map: state_key -> (rollout_indices, actions, rewards)
        state_info: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"indices": [], "actions": {}, "rewards": {}}
        )

        for rollout_idx, trajectory in enumerate(trajectories):
            for step_idx, step in enumerate(trajectory.steps):
                # Compute state key BEFORE this step
                messages = build_conversation_at_step(
                    trajectory, step_idx, self.SYSTEM_PROMPT
                )
                state_key = compute_state_key(messages)

                action = get_action_at_step(step)

                state_info[state_key]["indices"].append(rollout_idx)
                state_info[state_key]["actions"][rollout_idx] = action
                state_info[state_key]["rewards"][rollout_idx] = rewards[rollout_idx]

        # Filter to states with multiple rollouts (actual anchor points)
        anchor_states = {}
        for state_key, info in state_info.items():
            if len(info["indices"]) > 1:
                anchor_states[state_key] = AnchorState(
                    state_key=state_key,
                    rollout_indices=info["indices"],
                    actions_taken=info["actions"],
                    downstream_rewards=info["rewards"],
                )

        return anchor_states

    def _compute_step_advantages_for_rollout(
        self,
        rollout_idx: int,
        trajectory: RolloutTrajectory,
        group_advantage: float,
        anchor_states: Dict[str, AnchorState],
    ) -> List[StepAdvantage]:
        """
        Compute step-level advantages for a single rollout.

        For each step:
        1. Find if this step is at an anchor state
        2. If yes, compute advantage based on action's relative success rate
        3. Combined = group_advantage × step_advantage
        """
        step_advs = []

        for step_idx, step in enumerate(trajectory.steps):
            # Compute state key for this step
            messages = build_conversation_at_step(
                trajectory, step_idx, self.SYSTEM_PROMPT
            )
            state_key = compute_state_key(messages)

            if state_key in anchor_states:
                # This is an anchor state - compute step advantage
                anchor = anchor_states[state_key]
                step_adv = self._compute_action_advantage(
                    rollout_idx=rollout_idx,
                    anchor=anchor,
                )

                step_advs.append(
                    StepAdvantage(
                        step_idx=step_idx,
                        group_advantage=group_advantage,
                        step_advantage=step_adv,
                        combined_advantage=group_advantage * step_adv * self.step_advantage_weight,
                        anchor_state_key=state_key,
                        num_rollouts_at_anchor=len(anchor.rollout_indices),
                    )
                )
            else:
                # Not an anchor state - use group advantage only
                step_advs.append(
                    StepAdvantage(
                        step_idx=step_idx,
                        group_advantage=group_advantage,
                        step_advantage=1.0,  # Neutral
                        combined_advantage=group_advantage,
                        anchor_state_key=None,
                        num_rollouts_at_anchor=1,
                    )
                )

        return step_advs

    def _compute_action_advantage(
        self,
        rollout_idx: int,
        anchor: AnchorState,
    ) -> float:
        """
        Compute step advantage for an action at an anchor state.

        The advantage is based on how well the action performs compared to
        other actions taken at the same anchor state.

        Method:
        1. Group rollouts by action
        2. Compute average reward for each action group
        3. Normalize: (action_reward - mean) / std
        """
        # Group rewards by action
        action_rewards: Dict[str, List[float]] = defaultdict(list)
        for idx in anchor.rollout_indices:
            action = anchor.actions_taken[idx]
            reward = anchor.downstream_rewards[idx]
            action_rewards[action].append(reward)

        # Compute average reward per action
        action_avg_rewards = {
            action: np.mean(rewards) for action, rewards in action_rewards.items()
        }

        # Get this rollout's action and its avg reward
        my_action = anchor.actions_taken[rollout_idx]
        my_action_avg = action_avg_rewards[my_action]

        # Compute normalized advantage
        all_avgs = list(action_avg_rewards.values())
        mean_avg = np.mean(all_avgs)
        std_avg = np.std(all_avgs)

        if std_avg < 1e-8:
            # All actions have same average reward
            return 1.0

        # Normalize to roughly [-1, 1] range, then shift to [0, 2] for multiplication
        normalized = (my_action_avg - mean_avg) / std_avg
        # Scale to [0.5, 1.5] range (centered at 1.0 = neutral)
        step_advantage = 1.0 + 0.5 * np.tanh(normalized)

        return float(step_advantage)


def compute_gigpo_advantages(
    trajectories: List[RolloutTrajectory],
    rewards: List[float],
    task_metadata: TaskMetadata,
    discount_factor: float = 0.99,
) -> GiGPOResult:
    """
    Convenience function to compute GiGPO advantages.

    Args:
        trajectories: List of rollout trajectories
        rewards: Corresponding rewards
        task_metadata: Task metadata
        discount_factor: Discount for future rewards

    Returns:
        GiGPOResult with computed advantages
    """
    computer = GiGPOComputer(discount_factor=discount_factor)
    return computer.compute(trajectories, rewards, task_metadata)


def get_token_level_advantages(
    gigpo_result: GiGPOResult,
    tokenizer,
) -> List[List[Tuple[int, float]]]:
    """
    Map step-level advantages to token-level for training.

    Each token in the model's output at a step gets the same advantage
    as computed for that step.

    Args:
        gigpo_result: Computed GiGPO result
        tokenizer: Tokenizer for the model

    Returns:
        List of (token_idx, advantage) pairs per rollout
    """
    token_advantages = []

    for rollout_idx, trajectory in enumerate(gigpo_result.trajectories):
        rollout_token_advs = []
        step_advs = gigpo_result.step_advantages[rollout_idx]

        token_offset = 0
        for step_idx, step in enumerate(trajectory.steps):
            # Get advantage for this step
            if step_idx < len(step_advs):
                advantage = step_advs[step_idx].combined_advantage
            else:
                advantage = gigpo_result.group_advantages[rollout_idx]

            # Tokenize the model output for this step
            tokens = tokenizer.encode(step.model_output, add_special_tokens=False)

            for i, token_id in enumerate(tokens):
                rollout_token_advs.append((token_offset + i, advantage))

            token_offset += len(tokens)
            # Also account for injected response tokens (not trained on)

        token_advantages.append(rollout_token_advs)

    return token_advantages


# ============================================================================
# Comparison Utilities
# ============================================================================


def compare_grpo_vs_gigpo(
    trajectories: List[RolloutTrajectory],
    rewards: List[float],
    task_metadata: TaskMetadata,
) -> Dict[str, Any]:
    """
    Compare GRPO and GiGPO advantages for analysis.

    Returns statistics about how step-level credit assignment differs
    from trajectory-level.
    """
    # Compute GRPO advantages
    grpo_advantages = compute_group_advantages(rewards)

    # Compute GiGPO advantages
    gigpo_result = compute_gigpo_advantages(trajectories, rewards, task_metadata)

    # Analyze differences
    stats = {
        "num_rollouts": len(trajectories),
        "num_anchor_states": len(gigpo_result.anchor_states),
        "grpo_advantages": grpo_advantages,
        "gigpo_group_advantages": gigpo_result.group_advantages,
        "anchor_state_sizes": [
            len(anchor.rollout_indices)
            for anchor in gigpo_result.anchor_states.values()
        ],
        "step_advantage_variance": [],
    }

    # Compute variance in step advantages per rollout
    for rollout_step_advs in gigpo_result.step_advantages:
        if rollout_step_advs:
            advs = [sa.step_advantage for sa in rollout_step_advs]
            stats["step_advantage_variance"].append(float(np.var(advs)))
        else:
            stats["step_advantage_variance"].append(0.0)

    # Summary stats
    all_step_advs = []
    for rollout_step_advs in gigpo_result.step_advantages:
        for sa in rollout_step_advs:
            all_step_advs.append(sa.step_advantage)

    if all_step_advs:
        stats["step_advantage_mean"] = float(np.mean(all_step_advs))
        stats["step_advantage_std"] = float(np.std(all_step_advs))
        stats["step_advantage_min"] = float(np.min(all_step_advs))
        stats["step_advantage_max"] = float(np.max(all_step_advs))

    return stats
