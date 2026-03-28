"""
Multi-Turn Rollout Environment for GRPO/GiGPO Training.

This module wraps the NutriMind orchestrator as a GRPO-compatible environment
that supports environment-in-the-loop rollouts with tool execution.

Key features:
- Pause generation at </tool_call> token
- Execute real tools and inject responses
- Support deterministic rollouts for GiGPO anchor state detection
- State snapshot/restore for isolated rollouts

See phase4_grpo.md Task 4.1 for design details.
"""

import json
import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.orchestrator.tool_parser import (
    ToolParser,
    ParseResult,
    format_tool_response,
    format_error_response,
    VALID_TOOLS,
)


@dataclass
class ToolExecutionResult:
    """Result of a single tool execution."""

    tool_name: str
    tool_args: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


@dataclass
class RolloutStep:
    """A single step in a multi-turn rollout."""

    step_idx: int
    model_output: str  # Raw model generation
    think_content: Optional[str]  # Extracted <think> block
    action_type: str  # "tool_call", "final_answer", "parse_error"
    tool_execution: Optional[ToolExecutionResult] = None
    injected_response: Optional[str] = None  # Response injected into context


@dataclass
class RolloutTrajectory:
    """Complete trajectory from a single rollout."""

    prompt: str
    steps: List[RolloutStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    terminated: bool = False
    termination_reason: str = ""  # "final_answer", "max_rounds", "max_tokens"
    total_tool_calls: int = 0
    total_tokens_generated: int = 0

    def get_tools_called(self) -> List[str]:
        """Return list of tool names called in order."""
        return [
            step.tool_execution.tool_name
            for step in self.steps
            if step.tool_execution is not None
        ]

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Reconstruct conversation history from trajectory."""
        history = []
        for step in self.steps:
            # Assistant turn
            history.append({"role": "assistant", "content": step.model_output})
            # Tool response (if any)
            if step.injected_response:
                history.append({"role": "user", "content": step.injected_response})
        return history


@dataclass
class TaskMetadata:
    """Metadata for a GRPO prompt/task."""

    query: str
    tier: str  # T0, T1, T2, T3, T4, error-recovery
    expected_tools: List[str] = field(default_factory=list)
    optimal_steps: int = 1
    ground_truth: Optional[Dict[str, Any]] = None
    branch_condition: Optional[Dict[str, Any]] = None  # For T3 tasks
    difficulty: str = "medium"  # easy, medium, hard


class DeterministicToolCache:
    """
    Cache tool execution results for deterministic rollouts.

    For GiGPO anchor state detection, tools must return identical results
    for identical inputs across all rollouts in the same group.
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get_cache_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Generate deterministic cache key for tool call."""
        args_str = json.dumps(args, sort_keys=True, ensure_ascii=False)
        return f"{tool_name}:{args_str}"

    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached result, or None if not cached."""
        key = self.get_cache_key(tool_name, args)
        return self._cache.get(key)

    def set(self, tool_name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Cache a tool execution result."""
        key = self.get_cache_key(tool_name, args)
        self._cache[key] = deepcopy(result)

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()


class NutriMindEnv:
    """
    Multi-turn rollout environment for NutriMind GRPO training.

    Wraps the orchestrator as a GRPO-compatible environment:
    1. Model generates text
    2. Detect </tool_call> -> pause generation
    3. Parse tool_call -> execute real tool -> get tool_response
    4. Inject tool_response into context
    5. Model continues generating
    6. Repeat until final answer or max_rounds

    For GiGPO, supports deterministic rollouts via:
    - User state snapshots (isolated per rollout)
    - Tool result caching (shared within rollout group)
    """

    # System prompt for rollout generation
    SYSTEM_PROMPT = """You are NutriMind, a nutrition assistant. You have access to specific tools to retrieve data.
When you need to use a tool, use XML tags: <think>thought process</think><tool_call>{"name": "...", "arguments": {...}}</tool_call>
If you answer directly, do not use tool tags. Only use JSON in tool calls. Do not use parallel tool calls."""

    def __init__(
        self,
        tool_registry: Dict[str, Callable],
        max_tool_rounds: int = 6,
        tool_cache: Optional[DeterministicToolCache] = None,
        user_state_snapshot: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize environment.

        Args:
            tool_registry: Map of tool names to callable functions
            max_tool_rounds: Maximum number of tool call rounds
            tool_cache: Optional shared cache for deterministic rollouts
            user_state_snapshot: Optional user state to restore at reset
        """
        self.tool_registry = tool_registry
        self.max_tool_rounds = max_tool_rounds
        self.tool_cache = tool_cache
        self.initial_snapshot = user_state_snapshot

        # Internal state
        self._parser = ToolParser(validate_tool_name=True)
        self._messages: List[Dict[str, str]] = []
        self._current_round = 0
        self._trajectory: Optional[RolloutTrajectory] = None

    def reset(self, prompt: str) -> List[Dict[str, str]]:
        """
        Reset environment for a new rollout.

        Args:
            prompt: User query to start the conversation

        Returns:
            Initial message list for model generation
        """
        self._messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        self._current_round = 0
        self._trajectory = RolloutTrajectory(prompt=prompt)

        # Restore user state if snapshot provided
        if self.initial_snapshot is not None:
            self._restore_user_state(self.initial_snapshot)

        return deepcopy(self._messages)

    def step(self, model_output: str) -> Tuple[List[Dict[str, str]], bool, Dict[str, Any]]:
        """
        Process one step of model generation.

        Args:
            model_output: Raw text generated by the model

        Returns:
            Tuple of:
            - Updated message list for next generation (or empty if done)
            - done: Whether rollout is complete
            - info: Additional information about the step
        """
        if self._trajectory is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Parse model output
        parsed: ParseResult = self._parser.parse(model_output)

        step = RolloutStep(
            step_idx=self._current_round,
            model_output=model_output,
            think_content=parsed.think,
            action_type=parsed.type,
        )

        info = {
            "step_idx": self._current_round,
            "action_type": parsed.type,
            "think": parsed.think,
        }

        # Handle based on parse result type
        if parsed.type == "final_answer":
            # Rollout complete - model gave final answer
            self._trajectory.final_answer = parsed.content
            self._trajectory.terminated = True
            self._trajectory.termination_reason = "final_answer"
            self._trajectory.steps.append(step)

            info["final_answer"] = parsed.content
            return [], True, info

        elif parsed.type == "tool_call":
            # Execute tool and inject response
            tool_name = parsed.tool_call.name
            tool_args = parsed.tool_call.arguments

            execution_result = self._execute_tool(tool_name, tool_args)
            step.tool_execution = execution_result

            # Format and inject response
            if execution_result.success:
                response_str = format_tool_response(execution_result.result)
            else:
                response_str = format_error_response(
                    "tool_error", execution_result.error_message or "Unknown error"
                )

            step.injected_response = response_str
            self._trajectory.steps.append(step)
            self._trajectory.total_tool_calls += 1

            # Update conversation
            self._messages.append({"role": "assistant", "content": model_output})
            self._messages.append({"role": "user", "content": response_str})

            self._current_round += 1

            # Check max rounds
            if self._current_round >= self.max_tool_rounds:
                self._trajectory.terminated = True
                self._trajectory.termination_reason = "max_rounds"
                info["truncated"] = True
                return [], True, info

            info["tool_name"] = tool_name
            info["tool_success"] = execution_result.success
            return deepcopy(self._messages), False, info

        else:  # parse_error
            # Inject error message and let model retry
            error_msg = parsed.error_message or "Invalid format"
            retry_prompt = f"Your previous response had invalid format: {error_msg}. Please try again."

            step.injected_response = retry_prompt
            self._trajectory.steps.append(step)

            self._messages.append({"role": "assistant", "content": model_output})
            self._messages.append({"role": "user", "content": retry_prompt})

            self._current_round += 1

            # Check max rounds
            if self._current_round >= self.max_tool_rounds:
                self._trajectory.terminated = True
                self._trajectory.termination_reason = "max_rounds"
                info["truncated"] = True
                return [], True, info

            info["parse_error"] = error_msg
            return deepcopy(self._messages), False, info

    def get_trajectory(self) -> RolloutTrajectory:
        """Get the current rollout trajectory."""
        if self._trajectory is None:
            raise RuntimeError("No trajectory available. Call reset() first.")
        return self._trajectory

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> ToolExecutionResult:
        """Execute a tool, using cache if available."""
        # Check cache first (for deterministic GiGPO rollouts)
        if self.tool_cache is not None:
            cached = self.tool_cache.get(name, args)
            if cached is not None:
                return ToolExecutionResult(
                    tool_name=name,
                    tool_args=args,
                    result=cached,
                    success=cached.get("status") != "error",
                )

        # Validate tool exists
        if name not in self.tool_registry:
            return ToolExecutionResult(
                tool_name=name,
                tool_args=args,
                result={"status": "error", "message": f"Unknown tool: {name}"},
                success=False,
                error_message=f"Unknown tool: {name}",
            )

        # Execute tool
        try:
            tool_fn = self.tool_registry[name]
            # Handle no-arg tools
            if name in ("get_today_summary",):
                result = tool_fn()
            else:
                result = tool_fn(**args)

            # Cache result
            if self.tool_cache is not None:
                self.tool_cache.set(name, args, result)

            return ToolExecutionResult(
                tool_name=name,
                tool_args=args,
                result=result,
                success=result.get("status") != "error",
            )

        except Exception as e:
            error_result = {
                "status": "error",
                "error_type": "execution_error",
                "message": str(e),
            }
            return ToolExecutionResult(
                tool_name=name,
                tool_args=args,
                result=error_result,
                success=False,
                error_message=str(e),
            )

    def _restore_user_state(self, snapshot: Dict[str, Any]) -> None:
        """Restore user state from snapshot (for deterministic rollouts)."""
        # This will be implemented based on actual user state management
        # For now, it's a placeholder for future integration
        pass


def compute_state_key(messages: List[Dict[str, str]]) -> str:
    """
    Compute deterministic state key for GiGPO anchor state detection.

    Two rollouts share an anchor state if they have identical conversation
    context up to (and including) the last tool_response.

    Args:
        messages: Conversation history up to current point

    Returns:
        Hash string representing the state
    """
    # Serialize messages deterministically
    serialized = json.dumps(
        [(m["role"], m["content"]) for m in messages],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class RolloutGroup:
    """
    Manages a group of rollouts for the same prompt (for GRPO/GiGPO).

    Ensures deterministic tool execution across rollouts in the group
    by sharing a tool cache.
    """

    def __init__(
        self,
        prompt: str,
        task_metadata: TaskMetadata,
        tool_registry: Dict[str, Callable],
        num_rollouts: int = 8,
        max_tool_rounds: int = 6,
        user_state_snapshot: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize rollout group.

        Args:
            prompt: User query for this group
            task_metadata: Metadata about the task
            tool_registry: Tool name to function mapping
            num_rollouts: Number of rollouts (G) in this group
            max_tool_rounds: Max tool rounds per rollout
            user_state_snapshot: User state to restore for each rollout
        """
        self.prompt = prompt
        self.task_metadata = task_metadata
        self.num_rollouts = num_rollouts

        # Shared cache for deterministic tool execution
        self._shared_cache = DeterministicToolCache()

        # Create environments for each rollout
        self.envs = [
            NutriMindEnv(
                tool_registry=tool_registry,
                max_tool_rounds=max_tool_rounds,
                tool_cache=self._shared_cache,
                user_state_snapshot=user_state_snapshot,
            )
            for _ in range(num_rollouts)
        ]

        self.trajectories: List[Optional[RolloutTrajectory]] = [None] * num_rollouts

    def reset_all(self) -> List[List[Dict[str, str]]]:
        """Reset all environments and return initial message lists."""
        return [env.reset(self.prompt) for env in self.envs]

    def step_all(
        self, model_outputs: List[str]
    ) -> List[Tuple[List[Dict[str, str]], bool, Dict[str, Any]]]:
        """
        Step all environments with corresponding model outputs.

        Args:
            model_outputs: List of model outputs, one per rollout

        Returns:
            List of (messages, done, info) tuples
        """
        results = []
        for i, (env, output) in enumerate(zip(self.envs, model_outputs)):
            if self.trajectories[i] is not None and self.trajectories[i].terminated:
                # Already done, return empty
                results.append(([], True, {"already_done": True}))
            else:
                result = env.step(output)
                if result[1]:  # done
                    self.trajectories[i] = env.get_trajectory()
                results.append(result)
        return results

    def get_all_trajectories(self) -> List[RolloutTrajectory]:
        """Get all completed trajectories."""
        return [
            env.get_trajectory()
            for env in self.envs
        ]

    def find_anchor_states(self) -> Dict[str, List[int]]:
        """
        Find anchor states where rollouts diverge (for GiGPO).

        Returns:
            Dict mapping state_key to list of rollout indices at that state
        """
        anchor_states: Dict[str, List[int]] = {}

        for rollout_idx, trajectory in enumerate(self.get_all_trajectories()):
            # Build conversation progressively
            messages = [
                {"role": "system", "content": NutriMindEnv.SYSTEM_PROMPT},
                {"role": "user", "content": self.prompt},
            ]

            for step in trajectory.steps:
                # State BEFORE this step's model output
                state_key = compute_state_key(messages)

                if state_key not in anchor_states:
                    anchor_states[state_key] = []
                anchor_states[state_key].append(rollout_idx)

                # Update messages with this step
                messages.append({"role": "assistant", "content": step.model_output})
                if step.injected_response:
                    messages.append({"role": "user", "content": step.injected_response})

        # Filter to states with multiple rollouts (actual anchor points)
        return {k: v for k, v in anchor_states.items() if len(v) > 1}
