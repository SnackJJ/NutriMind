"""
veRL Interaction for NutriMind Multi-Turn Tool Calling.

This module implements veRL's BaseInteraction interface to bridge with
NutriMindEnv for environment-in-the-loop GRPO training.

Key responsibilities:
1. Manage per-instance environment state (env_state)
2. Execute tools via NutriMindEnv when model outputs <tool_call>
3. Compute rewards using existing reward_v1/v2/v3 functions
4. Handle multi-turn conversation flow with response mask awareness

Usage:
    # veRL loads this via configs/verl_interaction.yaml
    interaction = NutriMindInteraction(config)
    instance_id = await interaction.start_interaction(env_state=..., tier=...)
    should_stop, response, reward, meta = await interaction.generate_response(instance_id, messages)
"""

import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from src.training.grpo.environment import (
    NutriMindEnv,
    DeterministicToolCache,
    RolloutTrajectory,
    TaskMetadata,
)
from src.training.grpo.reward import (
    reward_v1,
    reward_v2,
    reward_v3,
    RewardBreakdown,
)
from src.orchestrator.orchestrator import TOOL_REGISTRY
from src.orchestrator.tool_parser import (
    ToolParser,
    format_tool_response,
    format_error_response,
)

logger = logging.getLogger(__name__)


class NutriMindInteraction:
    """
    veRL Interaction for NutriMind multi-turn tool calling.

    Bridges veRL's async interaction system with NutriMindEnv.

    Handles:
    - Per-instance env_state restoration for deterministic rollouts
    - Tool execution with mocked stateful tools
    - Reward computation using existing reward_v1/v2/v3
    - Termination detection (final answer / max rounds / parse errors)

    Response Mask Awareness:
    - Model-generated tokens (think, tool_call, final answer): mask=1
    - Tool responses injected by environment: mask=0
    - veRL handles masking; we just return the response text
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NutriMind interaction.

        Args:
            config: Configuration dict containing:
                - reward_version: "v1", "v2", or "v3"
                - max_tool_rounds: Maximum tool call rounds (default: 6)
        """
        self.name = "nutrimind"
        self._config = config
        self._reward_version = config.get("reward_version", "v2")
        self._max_tool_rounds = config.get("max_tool_rounds", 6)

        # Per-instance state storage
        self._instances: Dict[str, Dict[str, Any]] = {}

        # Shared tool parser
        self._parser = ToolParser(validate_tool_name=True)

        logger.info(
            f"NutriMindInteraction initialized: "
            f"reward={self._reward_version}, max_rounds={self._max_tool_rounds}"
        )

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        env_state: Optional[Dict[str, Any]] = None,
        tier: str = "T1",
        difficulty: str = "medium",
        **kwargs,
    ) -> str:
        """
        Initialize a new interaction session.

        Called by veRL before starting rollout generation for a prompt.
        Creates a fresh NutriMindEnv with the provided env_state snapshot.

        Args:
            instance_id: Optional ID for this instance (generated if None)
            env_state: User state snapshot (user_profile, meals_today, etc.)
            tier: Task tier (T1, T2, T3, T4, etc.)
            difficulty: Task difficulty (easy, medium, hard)
            **kwargs: Additional arguments (ignored)

        Returns:
            Instance ID for tracking this session
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Create per-instance tool cache for deterministic rollouts
        # All rollouts in the same group share the cache for consistency
        tool_cache = DeterministicToolCache()

        # Create NutriMindEnv with env_state snapshot
        env = NutriMindEnv(
            tool_registry=TOOL_REGISTRY,
            max_tool_rounds=self._max_tool_rounds,
            tool_cache=tool_cache,
            user_state_snapshot=env_state,
        )

        # Store instance state
        self._instances[instance_id] = {
            "env": env,
            "env_state": env_state,
            "tier": tier,
            "difficulty": difficulty,
            "query": None,  # Set when first user message is processed
            "tool_calls": [],
            "final_answer": None,
            "reward": 0.0,
            "terminated": False,
        }

        logger.debug(f"Started interaction {instance_id}: tier={tier}")
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """
        Process model output and execute tools if needed.

        Called by veRL after each model generation step.
        Parses the model output, executes tools if requested,
        and returns the environment response.

        Args:
            instance_id: Instance ID from start_interaction
            messages: Conversation history including latest assistant message

        Returns:
            Tuple of:
            - should_terminate: Whether to stop generation
            - response_text: Text to inject (tool response or empty)
            - turn_reward: Reward for this turn (0 during rollout, final at end)
            - metadata: Additional info about this step
        """
        if instance_id not in self._instances:
            raise ValueError(f"Unknown instance: {instance_id}")

        inst = self._instances[instance_id]

        if inst["terminated"]:
            return True, "", inst["reward"], {"already_terminated": True}

        # Extract the query from user message if not set
        if inst["query"] is None:
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    # Skip tool responses
                    if not content.startswith("<tool_response>"):
                        inst["query"] = content
                        break

        # Get the latest assistant message
        assistant_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_content = msg.get("content", "")
                break

        if not assistant_content:
            # No assistant message yet, continue generation
            return False, "", 0.0, {"waiting_for_generation": True}

        # Parse the assistant output
        parsed = self._parser.parse(assistant_content)

        if parsed.type == "final_answer":
            # Trajectory complete - model gave final answer
            inst["final_answer"] = parsed.content
            inst["terminated"] = True

            # Compute final reward
            reward = await self.calculate_score(instance_id)
            inst["reward"] = reward

            return True, "", reward, {
                "action": "final_answer",
                "answer_length": len(parsed.content or ""),
            }

        elif parsed.type == "tool_call":
            # Execute tool and return response
            tool_name = parsed.tool_call.name
            tool_args = parsed.tool_call.arguments

            # Initialize env if needed (first tool call)
            env = inst["env"]
            if env._trajectory is None:
                query = inst["query"] or "Unknown query"
                env.reset(query)

            # Step the environment with this tool call
            _, done, info = env.step(assistant_content)

            # Get tool response from trajectory
            trajectory = env.get_trajectory()
            last_step = trajectory.steps[-1] if trajectory.steps else None
            tool_response = last_step.injected_response if last_step else ""

            inst["tool_calls"].append({
                "name": tool_name,
                "args": tool_args,
                "success": info.get("tool_success", False),
            })

            if done:
                # Max rounds reached
                inst["terminated"] = True
                reward = await self.calculate_score(instance_id)
                inst["reward"] = reward

                return True, tool_response, reward, {
                    "action": "tool_call",
                    "tool": tool_name,
                    "truncated": True,
                    "termination_reason": "max_rounds",
                }

            # Continue generation after tool response
            return False, tool_response, 0.0, {
                "action": "tool_call",
                "tool": tool_name,
                "tool_success": info.get("tool_success", False),
            }

        else:
            # Parse error - inject error message
            error_msg = f"Invalid format: {parsed.error_message}"

            # Still step the environment to track this
            env = inst["env"]
            if env._trajectory is None:
                query = inst["query"] or "Unknown query"
                env.reset(query)

            _, done, info = env.step(assistant_content)

            if done:
                inst["terminated"] = True
                reward = await self.calculate_score(instance_id)
                inst["reward"] = reward
                return True, error_msg, reward, {
                    "action": "parse_error",
                    "error": parsed.error_message,
                    "truncated": True,
                }

            # Let model retry
            return False, error_msg, -0.1, {
                "action": "parse_error",
                "error": parsed.error_message,
            }

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """
        Calculate final reward for the trajectory.

        Uses existing reward_v1/v2/v3 functions from reward.py.

        Args:
            instance_id: Instance ID

        Returns:
            Final reward score in [0, 1]
        """
        if instance_id not in self._instances:
            raise ValueError(f"Unknown instance: {instance_id}")

        inst = self._instances[instance_id]
        env = inst["env"]

        # Get trajectory
        if env._trajectory is None:
            # No tools called, final answer was direct
            # Create minimal trajectory
            query = inst["query"] or "Unknown query"
            env.reset(query)

        trajectory = env.get_trajectory()

        # Build TaskMetadata
        task_meta = TaskMetadata(
            query=trajectory.prompt or inst["query"] or "",
            tier=inst["tier"],
            difficulty=inst["difficulty"],
            expected_tools=[],  # Not pre-annotated
            optimal_steps=self._get_optimal_steps(inst["tier"]),
        )

        # Compute reward using selected version
        if self._reward_version == "v1":
            breakdown = reward_v1(trajectory, task_meta)
        elif self._reward_version == "v2":
            breakdown = reward_v2(trajectory, task_meta)
        elif self._reward_version == "v3":
            # v3 needs LLM judge which we don't have in veRL context.
            # Using the training model as judge would introduce bias (model judging itself).
            # Fall back to v2 for deterministic rule-based reward.
            logger.warning(
                "reward v3 (LLM-Judge) not supported in veRL training. "
                "Falling back to v2. To use v3, run with train.py + external judge."
            )
            breakdown = reward_v2(trajectory, task_meta)
        else:
            breakdown = reward_v2(trajectory, task_meta)

        logger.debug(
            f"Instance {instance_id}: reward={breakdown.total:.3f} "
            f"(format={breakdown.r_format:.2f}, tool={breakdown.r_tool_selection:.2f}, "
            f"outcome={breakdown.r_outcome:.2f})"
        )

        return breakdown.total

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """
        Clean up instance resources after rollout is complete.

        Args:
            instance_id: Instance ID to finalize
        """
        if instance_id in self._instances:
            del self._instances[instance_id]
            logger.debug(f"Finalized interaction {instance_id}")

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

    def get_instance_info(self, instance_id: str) -> Dict[str, Any]:
        """Get current state info for an instance (for debugging)."""
        if instance_id not in self._instances:
            return {"error": "Unknown instance"}

        inst = self._instances[instance_id]
        return {
            "tier": inst["tier"],
            "difficulty": inst["difficulty"],
            "query": inst["query"],
            "tool_calls": inst["tool_calls"],
            "final_answer": inst["final_answer"],
            "reward": inst["reward"],
            "terminated": inst["terminated"],
        }


# Factory function for veRL
def create_interaction(config: Dict[str, Any]) -> NutriMindInteraction:
    """Factory function for veRL to create interaction instance."""
    return NutriMindInteraction(config)
