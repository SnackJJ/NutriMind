"""
Iterative Reward Functions for GRPO Training.

Implements the iterative reward strategy from phase4_grpo.md:
- v1: Pure rule-based (format + tool_selection + outcome)
- v2: v1 + efficiency + conditional
- v3: v2 + LLM-Judge for recommendation questions

Core principle: Each iteration changes ONE variable (the reward function).
This enables diagnosing what worked and what didn't.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from src.training.grpo.environment import RolloutTrajectory, TaskMetadata


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components."""

    total: float
    r_format: float = 0.0
    r_tool_selection: float = 0.0
    r_outcome: float = 0.0
    r_efficiency: float = 0.0
    r_conditional: float = 0.0
    r_llm_judge: float = 0.0
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


# ============================================================================
# Helper Functions
# ============================================================================


def check_format_validity(trajectory: RolloutTrajectory) -> tuple[bool, List[str]]:
    """
    Check if all tool calls in trajectory have valid format.

    Returns:
        (all_valid, list_of_errors)
    """
    errors = []

    for step in trajectory.steps:
        if step.action_type == "tool_call":
            if step.tool_execution is None:
                errors.append(f"Step {step.step_idx}: tool_call but no execution")
            elif not step.tool_execution.success:
                # Tool execution failure is OK (model should learn from it)
                # Only format issues matter here
                pass
        elif step.action_type == "parse_error":
            errors.append(f"Step {step.step_idx}: parse error - {step.injected_response}")

    return len(errors) == 0, errors


def check_safety_declaration(answer: str) -> bool:
    """
    Check if answer contains appropriate safety/escalation language.

    Returns True if the answer appropriately escalates to professional help.
    """
    if not answer:
        return False

    answer_lower = answer.lower()
    escalation_phrases = [
        "consult",
        "physician",
        "doctor",
        "dietitian",
        "healthcare",
        "professional",
        "medical",
        "exceeds my",
        "beyond my",
        "safe service boundary",
        "recommend speaking",
        "seek guidance",
    ]
    return any(phrase in answer_lower for phrase in escalation_phrases)


def compute_tool_selection_score(
    trajectory: RolloutTrajectory, task_metadata: TaskMetadata
) -> float:
    """
    Compute tool selection score (v1: no dependency on expected_tools).

    Tier-specific logic:
    - T0-qa: Should NOT call tools (pure QA). Penalize any tool calls.
    - T4: Only care about final safety declaration, not tool usage.
            This prevents "see sensitive word → refuse immediately" shortcut.
            GRPO group comparison naturally selects optimal path.
    - T1-T3: Should call at least one tool. Check validity only.

    This design avoids over-refusal by letting T4 explore both
    "check then refuse" and "refuse directly" paths.
    """
    tier = task_metadata.tier
    tools_called = trajectory.get_tools_called()

    # T0: Pure QA should not call tools
    if tier == "T0-qa":
        return 1.0 if len(tools_called) == 0 else 0.0

    # T4: Only check final answer has safety declaration
    # Do NOT penalize tool calls - model may legitimately check before refusing
    if tier == "T4":
        has_safety = check_safety_declaration(trajectory.final_answer or "")
        return 1.0 if has_safety else 0.0

    # T1-T3: Should call at least one valid tool
    valid_tools = frozenset({
        "get_food_nutrition",
        "log_meal",
        "get_today_summary",
        "get_history",
        "retrieve_knowledge",
    })

    if not tools_called:
        return 0.3  # No tools called when should have

    # Check all called tools are valid
    all_valid = all(t in valid_tools for t in tools_called)
    return 1.0 if all_valid else 0.5


def compute_outcome_score_rule_based(
    trajectory: RolloutTrajectory, task_metadata: TaskMetadata
) -> float:
    """
    Compute outcome score using rule-based checks (no LLM).

    v1 design: Minimize dependency on pre-annotated metadata.
    - T0/T4: Just check completion (safety handled in tool_selection)
    - T1: Compare final answer against tool results (runtime ground truth)
    - T2/T3: Basic completion check (detailed scoring in v2)

    Checks:
    - Did trajectory complete with final answer?
    - For T1: does answer contain values from tool results?
    """
    if not trajectory.terminated or trajectory.termination_reason != "final_answer":
        return 0.0

    final_answer = trajectory.final_answer or ""
    tier = task_metadata.tier

    # T0 pure QA: basic completion check
    if tier == "T0-qa":
        return 1.0 if len(final_answer) > 20 else 0.5

    # T4 safety: just check completion (safety declaration checked in r_tool)
    if tier == "T4":
        return 1.0 if len(final_answer) > 20 else 0.5

    # T1: Check if answer contains values from tool results (runtime ground truth)
    if tier == "T1":
        return _compute_t1_outcome_from_tool_results(trajectory, final_answer)

    # T2/T3: Basic completion for v1, detailed scoring in v2
    # Check that final answer is substantive
    if len(final_answer) < 30:
        return 0.3
    return 0.8 if len(final_answer) > 80 else 0.6


def _compute_t1_outcome_from_tool_results(
    trajectory: RolloutTrajectory, final_answer: str
) -> float:
    """
    For T1 queries: check if final answer contains values from tool results.

    This provides "runtime ground truth" without pre-annotation.
    """
    # Find get_food_nutrition results in trajectory
    nutrition_values = []
    for step in trajectory.steps:
        if step.tool_execution and step.tool_execution.tool_name == "get_food_nutrition":
            result = step.tool_execution.result
            if isinstance(result, dict) and result.get("status") == "success":
                data = result.get("data", {})
                # Extract key nutrition values
                for key in ["calories_kcal", "protein_g", "carbs_g", "fat_g", "fiber_g"]:
                    if key in data:
                        nutrition_values.append(data[key])

    if not nutrition_values:
        # No nutrition tool called or no values found
        return 0.5

    # Check how many values appear in the answer
    matches = 0
    for value in nutrition_values:
        if isinstance(value, (int, float)):
            # Allow some formatting variation (31.5 vs 31.50 vs 32)
            value_str = f"{value:.0f}" if value == int(value) else f"{value:.1f}"
            if value_str in final_answer or str(int(value)) in final_answer:
                matches += 1

    return min(1.0, matches / max(len(nutrition_values), 1) + 0.2)


def compute_efficiency_score(
    trajectory: RolloutTrajectory, task_metadata: TaskMetadata
) -> float:
    """
    Compute efficiency score based on tool call count.

    Penalizes excessive tool calls relative to optimal_steps.
    Score = 1.0 if steps <= optimal, decreases linearly up to 2x optimal.
    """
    actual_steps = trajectory.total_tool_calls
    optimal_steps = task_metadata.optimal_steps

    if optimal_steps == 0:
        # No tools expected
        return 1.0 if actual_steps == 0 else max(0.0, 1.0 - 0.2 * actual_steps)

    if actual_steps <= optimal_steps:
        return 1.0

    # Linear penalty: 0 at 2x optimal
    excess_ratio = (actual_steps - optimal_steps) / optimal_steps
    return max(0.0, 1.0 - excess_ratio)


def compute_conditional_score(
    trajectory: RolloutTrajectory, task_metadata: TaskMetadata
) -> float:
    """
    Compute conditional branching correctness (T3 tasks).

    Checks if model made correct branching decision based on intermediate results.
    """
    branch_condition = task_metadata.branch_condition
    if not branch_condition:
        return 1.0  # Not a conditional task

    # Extract relevant info from trajectory
    check_tool = branch_condition.get("check_tool")
    condition_field = branch_condition.get("condition_field")
    threshold = branch_condition.get("threshold")
    expected_branch = branch_condition.get("expected_branch")

    # Find the tool response that should trigger branching
    branch_value = None
    for step in trajectory.steps:
        if step.tool_execution and step.tool_execution.tool_name == check_tool:
            result = step.tool_execution.result
            if isinstance(result, dict) and "data" in result:
                data = result["data"]
                if isinstance(data, dict) and condition_field in data:
                    branch_value = data[condition_field]
                    break

    if branch_value is None:
        # Didn't call the check tool
        return 0.0

    # Determine expected action based on condition
    if expected_branch == "under_budget":
        should_retrieve = branch_value < threshold
    elif expected_branch == "over_budget":
        should_retrieve = branch_value >= threshold
    else:
        return 0.5  # Unknown condition type

    # Check if model took appropriate action
    tools_after_check = []
    found_check = False
    for step in trajectory.steps:
        if step.tool_execution:
            if step.tool_execution.tool_name == check_tool:
                found_check = True
            elif found_check:
                tools_after_check.append(step.tool_execution.tool_name)

    # For budget scenarios: should call retrieve_knowledge if under budget
    if should_retrieve:
        if "retrieve_knowledge" in tools_after_check:
            return 1.0
        else:
            return 0.3  # Missed the conditional branch
    else:
        # Should give direct advice without retrieval
        if "retrieve_knowledge" not in tools_after_check:
            return 1.0
        else:
            return 0.7  # Unnecessary retrieval, but not wrong


# ============================================================================
# LLM Judge (for v3)
# ============================================================================


class LLMJudge:
    """
    LLM-based answer quality judge for recommendation questions.

    Safeguards:
    - Call n=3 times, take mean (reduce noise)
    - Weight <= 25% of total reward
    - Only for recommendation-type questions
    - Monitor answer length inflation
    """

    def __init__(
        self,
        judge_fn: Optional[Callable[[str, str], float]] = None,
        n_samples: int = 3,
    ):
        """
        Initialize judge.

        Args:
            judge_fn: Function (answer, query) -> score [0, 1]
            n_samples: Number of judge calls to average
        """
        self.judge_fn = judge_fn
        self.n_samples = n_samples

    def judge(self, answer: str, query: str, task_metadata: TaskMetadata) -> float:
        """
        Judge answer quality.

        Returns:
            Score in [0, 1], or -1 if judge not available
        """
        if self.judge_fn is None:
            return -1.0

        # Safeguard: check answer length
        if len(answer) > 2000:
            # Likely gaming the judge with verbose answers
            return 0.5

        scores = []
        for _ in range(self.n_samples):
            try:
                score = self.judge_fn(answer, query)
                scores.append(max(0.0, min(1.0, score)))
            except Exception:
                scores.append(0.5)  # Default on error

        return sum(scores) / len(scores) if scores else 0.5


def is_recommendation_question(query: str) -> bool:
    """Check if query is a recommendation-type question."""
    recommendation_patterns = [
        r"suggest",
        r"recommend",
        r"what should i",
        r"what can i",
        r"give me ideas",
        r"help me choose",
        r"best way to",
        r"how can i improve",
    ]
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in recommendation_patterns)


# ============================================================================
# Reward Functions (v1, v2, v3)
# ============================================================================


def reward_v1(
    trajectory: RolloutTrajectory, task_metadata: TaskMetadata
) -> RewardBreakdown:
    """
    GRPO v1: Pure rule-based reward (3 dimensions).

    Policy: SFT model
    Reference: SFT model (frozen)

    Components:
    - R_format (30%): Valid JSON in <tool_call> tags
    - R_tool_selection (35%): Correct tools called
    - R_outcome (35%): Factual accuracy (rule-based)

    Deliberately omits: efficiency, conditional, LLM-Judge
    """
    # R_format
    format_valid, format_errors = check_format_validity(trajectory)
    r_format = 1.0 if format_valid else 0.0

    # R_tool_selection
    r_tool = compute_tool_selection_score(trajectory, task_metadata)

    # R_outcome (rule-based only)
    r_outcome = compute_outcome_score_rule_based(trajectory, task_metadata)

    # Weighted sum
    total = 0.30 * r_format + 0.35 * r_tool + 0.35 * r_outcome

    return RewardBreakdown(
        total=total,
        r_format=r_format,
        r_tool_selection=r_tool,
        r_outcome=r_outcome,
        details={
            "format_errors": format_errors,
            "tools_called": trajectory.get_tools_called(),
            "expected_tools": task_metadata.expected_tools,
        },
    )


def reward_v2(
    trajectory: RolloutTrajectory, task_metadata: TaskMetadata
) -> RewardBreakdown:
    """
    GRPO v2: v1 + efficiency + conditional.

    Policy: GRPO-v1 checkpoint
    Reference: GRPO-v1 checkpoint (frozen)

    Components:
    - v1 base (70%)
    - R_efficiency (15%): Penalize excess tool calls
    - R_conditional (15%): T3 branching correctness
    """
    # Get v1 base
    v1 = reward_v1(trajectory, task_metadata)

    # R_efficiency
    r_efficiency = compute_efficiency_score(trajectory, task_metadata)

    # R_conditional
    r_conditional = compute_conditional_score(trajectory, task_metadata)

    # Weighted sum
    total = 0.70 * v1.total + 0.15 * r_efficiency + 0.15 * r_conditional

    return RewardBreakdown(
        total=total,
        r_format=v1.r_format,
        r_tool_selection=v1.r_tool_selection,
        r_outcome=v1.r_outcome,
        r_efficiency=r_efficiency,
        r_conditional=r_conditional,
        details={
            **v1.details,
            "actual_steps": trajectory.total_tool_calls,
            "optimal_steps": task_metadata.optimal_steps,
            "tier": task_metadata.tier,
        },
    )


def reward_v3(
    trajectory: RolloutTrajectory,
    task_metadata: TaskMetadata,
    llm_judge: Optional[LLMJudge] = None,
) -> RewardBreakdown:
    """
    GRPO v3: v2 + LLM-Judge for recommendation questions.

    Policy: GRPO-v2 checkpoint
    Reference: GRPO-v2 checkpoint (frozen)

    LLM-Judge safeguards:
    - n=3 averaging
    - Weight <= 25% (rules remain dominant)
    - Only for recommendation-type questions
    - Monitor answer length inflation
    """
    # Get v2 base
    v2 = reward_v2(trajectory, task_metadata)

    # Check if this is a recommendation question
    if is_recommendation_question(task_metadata.query) and llm_judge is not None:
        final_answer = trajectory.final_answer or ""
        r_llm = llm_judge.judge(final_answer, task_metadata.query, task_metadata)

        if r_llm >= 0:  # Valid judge score
            # LLM-Judge weight small; rules remain dominant
            total = 0.75 * v2.total + 0.25 * r_llm
            return RewardBreakdown(
                total=total,
                r_format=v2.r_format,
                r_tool_selection=v2.r_tool_selection,
                r_outcome=v2.r_outcome,
                r_efficiency=v2.r_efficiency,
                r_conditional=v2.r_conditional,
                r_llm_judge=r_llm,
                details={
                    **v2.details,
                    "is_recommendation": True,
                    "llm_judge_applied": True,
                },
            )

    # Not a recommendation question or judge unavailable
    return RewardBreakdown(
        total=v2.total,
        r_format=v2.r_format,
        r_tool_selection=v2.r_tool_selection,
        r_outcome=v2.r_outcome,
        r_efficiency=v2.r_efficiency,
        r_conditional=v2.r_conditional,
        details={
            **v2.details,
            "is_recommendation": is_recommendation_question(task_metadata.query),
            "llm_judge_applied": False,
        },
    )


# ============================================================================
# Reward Hacking Detection
# ============================================================================


@dataclass
class RewardHackingAlert:
    """Alert for potential reward hacking."""

    alert_type: str
    severity: str  # "warning", "critical"
    message: str
    metric_name: str
    current_value: float
    threshold: float


def detect_reward_hacking(
    recent_metrics: List[Dict[str, float]],
    current_metrics: Dict[str, float],
) -> List[RewardHackingAlert]:
    """
    Detect potential reward hacking patterns.

    Patterns checked:
    - reward ↑ but task_completion ↓
    - avg_tool_calls cliff drop > 30%
    - pairwise BLEU of rollouts > 0.85
    - KL spike > 3× recent average
    - answer_length continuous growth
    """
    alerts = []

    if len(recent_metrics) < 5:
        return alerts  # Not enough history

    # Compute recent averages
    def avg(key: str) -> float:
        values = [m.get(key, 0) for m in recent_metrics[-5:]]
        return sum(values) / len(values) if values else 0

    recent_reward = avg("reward")
    recent_tool_calls = avg("avg_tool_calls")
    recent_kl = avg("kl_divergence")
    recent_answer_len = avg("avg_answer_length")

    current_reward = current_metrics.get("reward", 0)
    current_completion = current_metrics.get("task_completion_rate", 0)
    current_tool_calls = current_metrics.get("avg_tool_calls", 0)
    current_kl = current_metrics.get("kl_divergence", 0)
    current_answer_len = current_metrics.get("avg_answer_length", 0)
    current_bleu = current_metrics.get("pairwise_bleu", 0)

    # Check: reward up but completion down
    prev_completion = avg("task_completion_rate")
    if current_reward > recent_reward * 1.1 and current_completion < prev_completion * 0.9:
        alerts.append(
            RewardHackingAlert(
                alert_type="reward_completion_divergence",
                severity="critical",
                message="Reward increasing but task completion decreasing",
                metric_name="task_completion_rate",
                current_value=current_completion,
                threshold=prev_completion * 0.9,
            )
        )

    # Check: tool calls cliff drop
    if recent_tool_calls > 0 and current_tool_calls < recent_tool_calls * 0.7:
        alerts.append(
            RewardHackingAlert(
                alert_type="tool_calls_cliff",
                severity="warning",
                message="Avg tool calls dropped >30%",
                metric_name="avg_tool_calls",
                current_value=current_tool_calls,
                threshold=recent_tool_calls * 0.7,
            )
        )

    # Check: mode collapse (high pairwise BLEU)
    if current_bleu > 0.85:
        alerts.append(
            RewardHackingAlert(
                alert_type="mode_collapse",
                severity="warning",
                message="High pairwise BLEU indicates mode collapse",
                metric_name="pairwise_bleu",
                current_value=current_bleu,
                threshold=0.85,
            )
        )

    # Check: KL spike
    if recent_kl > 0 and current_kl > recent_kl * 3:
        alerts.append(
            RewardHackingAlert(
                alert_type="kl_spike",
                severity="critical",
                message="KL divergence spiked >3× recent average",
                metric_name="kl_divergence",
                current_value=current_kl,
                threshold=recent_kl * 3,
            )
        )

    # Check: answer length inflation
    if recent_answer_len > 0 and current_answer_len > recent_answer_len * 1.5:
        alerts.append(
            RewardHackingAlert(
                alert_type="answer_inflation",
                severity="warning",
                message="Answer length inflating (may be gaming LLM-Judge)",
                metric_name="avg_answer_length",
                current_value=current_answer_len,
                threshold=recent_answer_len * 1.5,
            )
        )

    return alerts
