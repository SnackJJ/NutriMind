"""
veRL-agent EnvManager for NutriMind.

Adapts NutriMindEnv to verl-agent's StepEnvManager interface,
enabling GiGPO training with step-independent rollout.

This module bridges the NutriMind environment to verl-agent's GiGPO implementation,
which provides proper step-level credit assignment that the prototype gigpo.py lacks.

Usage:
    # In verl_agent_gigpo.yaml:
    env_manager:
      class_name: "src.training.grpo.verl_agent_env.NutriMindStepEnvManager"

Requirements:
    pip install verl-agent
    # Or: pip install git+https://github.com/langfengQ/verl-agent.git
"""

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from src.training.grpo.environment import (
    NutriMindEnv,
    DeterministicToolCache,
    RolloutTrajectory,
    TaskMetadata,
)
from src.training.grpo.reward import (
    reward_v1,
    reward_v2,
    RewardBreakdown,
)
from src.orchestrator.orchestrator import TOOL_REGISTRY

logger = logging.getLogger(__name__)


# Try to import verl-agent base class
try:
    from verl_agent.env_manager.step_env_manager import StepEnvManager
    VERL_AGENT_AVAILABLE = True
except ImportError:
    # Create a stub base class for development without verl-agent
    class StepEnvManager:
        """Stub base class when verl-agent is not installed."""
        def __init__(self, config: Dict[str, Any]):
            self.config = config

    VERL_AGENT_AVAILABLE = False
    logger.warning(
        "verl-agent not installed. NutriMindStepEnvManager will not work for training. "
        "Install with: pip install verl-agent"
    )


class NutriMindStepEnvManager(StepEnvManager):
    """
    Maps NutriMind environment to verl-agent's step-independent rollout.

    Key responsibilities:
    1. build_text_obs() — Construct input for each step
    2. step() — Execute tool calls, return observation + reward
    3. reset() — Initialize environment from env_state snapshot

    For GiGPO, this enables proper step-level credit assignment by:
    - Tracking anchor states where rollouts diverge
    - Computing step-level advantages based on downstream success
    - Combining with trajectory-level GRPO advantages

    Note: NutriMind has max 6 tool rounds, so we use full history
    (no need for context compression like long-horizon tasks).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the environment manager.

        Args:
            config: Configuration containing:
                - max_tool_rounds: Maximum tool call rounds (default: 6)
                - reward_version: "v1" or "v2" (default: "v2")
                - group_size: Number of rollouts per prompt group
        """
        super().__init__(config)

        self.max_tool_rounds = config.get("max_tool_rounds", 6)
        self.reward_version = config.get("reward_version", "v2")
        self.group_size = config.get("group_size", 8)

        # Tool registry from orchestrator
        self.tool_registry = TOOL_REGISTRY

        # Current environment instance (set in reset())
        self.env: Optional[NutriMindEnv] = None
        self.task_meta: Optional[TaskMetadata] = None

        # Per-group shared cache for deterministic tool execution
        self._group_caches: Dict[str, DeterministicToolCache] = {}

        logger.info(
            f"NutriMindStepEnvManager initialized: "
            f"max_rounds={self.max_tool_rounds}, reward={self.reward_version}"
        )

    def reset(
        self,
        prompt_data: Dict[str, Any],
        group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Initialize environment for a new rollout.

        Called by verl-agent before starting rollout generation.

        Args:
            prompt_data: Dict containing:
                - prompt: List of messages (system + user)
                - extra_info: Contains interaction_kwargs with env_state
            group_id: Optional group identifier for cache sharing

        Returns:
            Initial observation dict
        """
        # Extract env_state from prompt data
        extra_info = prompt_data.get("extra_info", {})
        interaction_kwargs = extra_info.get("interaction_kwargs", {})
        env_state = interaction_kwargs.get("env_state", {})
        tier = interaction_kwargs.get("tier", "T1")
        difficulty = interaction_kwargs.get("difficulty", "medium")

        # Get or create group cache for deterministic rollouts
        if group_id is not None:
            if group_id not in self._group_caches:
                self._group_caches[group_id] = DeterministicToolCache()
            tool_cache = self._group_caches[group_id]
        else:
            tool_cache = DeterministicToolCache()

        # Create NutriMindEnv instance
        self.env = NutriMindEnv(
            tool_registry=self.tool_registry,
            max_tool_rounds=self.max_tool_rounds,
            tool_cache=tool_cache,
            user_state_snapshot=env_state,
        )

        # Extract user query from prompt
        prompt_messages = prompt_data.get("prompt", [])
        user_query = ""
        for msg in prompt_messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if not content.startswith("<tool_response>"):
                    user_query = content
                    break

        # Reset environment
        initial_messages = self.env.reset(user_query)

        # Create task metadata
        self.task_meta = TaskMetadata(
            query=user_query,
            tier=tier,
            difficulty=difficulty,
            expected_tools=[],  # Not pre-annotated for RL
            optimal_steps=self._get_optimal_steps(tier),
        )

        return {
            "messages": initial_messages,
            "query": user_query,
            "tier": tier,
        }

    def step(
        self,
        action: str,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step: process model output, execute tools if needed.

        Called by verl-agent after each model generation.

        Args:
            action: Model-generated text (may contain tool_call or final answer)

        Returns:
            Tuple of:
            - observation: Dict with updated messages
            - reward: Step reward (0 during rollout, final reward at termination)
            - done: Whether rollout is complete
            - info: Additional step information
        """
        if self.env is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Step the environment
        messages, done, info = self.env.step(action)

        # Compute reward only at termination
        reward = 0.0
        if done:
            trajectory = self.env.get_trajectory()
            reward = self._compute_reward(trajectory)
            info["final_reward"] = reward

        observation = {
            "messages": messages,
            "done": done,
        }

        return observation, reward, done, info

    def build_text_obs(
        self,
        history: List[Dict[str, str]],
        step_idx: int,
    ) -> List[Dict[str, str]]:
        """
        Construct text input for the model at a given step.

        verl-agent's step-independent rollout core:
        Can construct customized input instead of full history concatenation.

        For NutriMind (max 6 rounds), we use full history since
        context length won't explode. For longer-horizon tasks,
        this could implement context compression.

        Args:
            history: Full conversation history up to this point
            step_idx: Current step index

        Returns:
            Messages to feed to the model
        """
        # NutriMind: 6 rounds max, full history is fine
        return history

    def get_trajectory(self) -> Optional[RolloutTrajectory]:
        """Get the current rollout trajectory."""
        if self.env is None:
            return None
        return self.env.get_trajectory()

    def cleanup_group(self, group_id: str) -> None:
        """Clean up cache for a completed group."""
        if group_id in self._group_caches:
            del self._group_caches[group_id]

    def _compute_reward(self, trajectory: RolloutTrajectory) -> float:
        """Compute reward using configured version."""
        if self.task_meta is None:
            return 0.0

        if self.reward_version == "v1":
            breakdown = reward_v1(trajectory, self.task_meta)
        else:
            breakdown = reward_v2(trajectory, self.task_meta)

        return breakdown.total

    def _get_optimal_steps(self, tier: str) -> int:
        """Map tier to expected optimal tool call count."""
        if tier.startswith("T0") or tier.startswith("T4"):
            return 0
        elif tier.startswith("T1"):
            return 1
        elif tier.startswith("T2") or tier == "error_recovery":
            return 2
        elif tier.startswith("T3"):
            return 3
        else:
            return 1


def create_env_manager(config: Dict[str, Any]) -> NutriMindStepEnvManager:
    """Factory function for verl-agent to create environment manager."""
    return NutriMindStepEnvManager(config)
