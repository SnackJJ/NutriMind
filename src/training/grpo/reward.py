"""
Iterative Reward Functions for GRPO Training.

Implements the iterative reward strategy from phase4_grpo.md:
- v1: Pure rule-based (format + tool_selection + outcome)
- v2: v1 + efficiency + conditional
- v3: v2 + RULER-style group-relative LLM judge

Core principle: Each iteration changes ONE variable (the reward function).
This enables diagnosing what worked and what didn't.

v3 design (RULER-style):
  For each GRPO group (G rollouts from same prompt), the LLM judge sees
  ALL G trajectories side-by-side and ranks them relatively. This is much
  easier than absolute scoring and naturally produces reward variance.
  Final score = 0.5 * r_rule (v2) + 0.5 * r_judge (relative rank).
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from src.training.grpo.environment import RolloutTrajectory, TaskMetadata, RolloutStep, ToolExecutionResult

logger = logging.getLogger(__name__)


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
    details: Dict[str, Any] = field(default_factory=dict)


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
        "set_goal",
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
    # Find get_food_nutrition results in trajectory.
    nutrition_values = []
    nutrition_tool_called = False
    for step in trajectory.steps:
        if step.tool_execution and step.tool_execution.tool_name == "get_food_nutrition":
            nutrition_tool_called = True
            result = step.tool_execution.result
            if isinstance(result, dict) and result.get("status") == "success":
                data = result.get("data", {})
                # Extract key nutrition values
                for key in ["calories_kcal", "protein_g", "carbs_g", "fat_g", "fiber_g"]:
                    if key in data:
                        nutrition_values.append(data[key])

    if not nutrition_tool_called:
        return 0.0

    if not nutrition_values:
        # Called nutrition tool but got no structured values to ground the answer.
        return 0.2

    # Check how many values appear in the answer
    matches = 0
    for value in nutrition_values:
        if isinstance(value, (int, float)):
            # Allow some formatting variation (31.5 vs 31.50 vs 32)
            value_str = f"{value:.0f}" if value == int(value) else f"{value:.1f}"
            if value_str in final_answer or str(int(value)) in final_answer:
                matches += 1

    coverage = matches / max(len(nutrition_values), 1)

    # Hard downgrade: called tool but ignored key numeric values.
    if coverage == 0.0:
        return 0.1
    if coverage < 0.35:
        return 0.35
    if coverage < 0.7:
        return 0.65
    return 1.0


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
        # Neutral baseline for non-conditional tasks.
        return 0.5

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
# RULER-style Group-Relative LLM Judge (v3)
# ============================================================================


_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for a nutrition AI assistant called NutriMind.
You will be given a user query and multiple candidate responses (trajectories).
Your job is to score EACH candidate on 4 dimensions (each 1-10):

1. **accuracy**: Does the answer use correct nutritional data from tool results? Are numbers cited accurately? For safety (T4) queries: does it appropriately escalate to a professional?
2. **helpfulness**: Does the answer actually address what the user asked? Is it actionable, relevant, and complete?
3. **tool_use**: Did the assistant use appropriate tools? Did it avoid unnecessary or redundant calls? Did it use tool results (not hallucinate)?
4. **communication**: Is the answer well-organized, clear, and appropriately concise (not too verbose, not too terse)?

IMPORTANT RULES:
- Score each candidate RELATIVELY — compare them against each other.
- Candidates that cite tool results accurately should score higher on accuracy than those that hallucinate.
- Candidates that are truncated (no final answer) should get 1-2 on all dimensions.
- You MUST output valid JSON in the exact format specified. No markdown, no explanation outside the JSON."""

_JUDGE_USER_TEMPLATE = """\
## User Query
{query}

## Task Context
- Tier: {tier} (T1=single tool, T2=multi-step, T3=conditional, T4=safety boundary)
- Difficulty: {difficulty}

## Candidates
{candidates_block}

## Output Format
Score each candidate on the 4 dimensions. Output ONLY a JSON object:
{{"candidates": [{candidates_placeholder}]}}"""

_CANDIDATE_TEMPLATE = """\
### Candidate {idx}
**Tools called**: {tools_called}
**Final answer**: {final_answer}"""

# Dimension weights for computing the composite score.
# accuracy + helpfulness are the main signal; tool_use and communication
# provide finer differentiation (and are easier for the judge to vary).
JUDGE_DIMENSION_WEIGHTS = {
    "accuracy": 0.35,
    "helpfulness": 0.30,
    "tool_use": 0.20,
    "communication": 0.15,
}
_JUDGE_DIMENSIONS = list(JUDGE_DIMENSION_WEIGHTS.keys())


class GroupJudge:
    """RULER-style group-relative LLM judge for GRPO reward.

    Scores all G trajectories in a single LLM call by presenting them
    side-by-side and asking for relative scores. This naturally produces
    reward variance even when all trajectories are decent.

    Uses DashScope (qwen3.5-plus) via OpenAI-compatible API.

    Attributes:
        model: Model name for the judge (default: qwen3.5-plus).
        temperature: Sampling temperature for scoring diversity.
        max_retries: Number of retry attempts on API/parse failure.
        _client: Lazy-initialized OpenAI client.
    """

    def __init__(
        self,
        model: str = "qwen3.5-plus",
        temperature: float = 0.3,
        max_retries: int = 2,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        """Lazy-initialize OpenAI client for DashScope."""
        if self._client is None:
            try:
                from openai import OpenAI
                from src.config import settings
                api_key = settings.qwen_api_key
                if not api_key:
                    logger.warning("GroupJudge: No DASHSCOPE_API_KEY found, judge disabled")
                    return None
                self._client = OpenAI(
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            except ImportError:
                logger.warning("GroupJudge: openai package not installed")
                return None
        return self._client

    def score_group(
        self,
        trajectories: list["RolloutTrajectory"],
        task_metadata: "TaskMetadata",
    ) -> tuple[list[float], list[dict]]:
        """Score a group of trajectories relatively using the LLM judge.

        Each candidate is scored on 4 dimensions (accuracy, helpfulness,
        tool_use, communication), then a weighted composite is computed.
        This multi-dimensional approach:
        - Reduces noise (4 sub-scores averaged vs 1 holistic score)
        - Enables per-dimension diagnostics in wandb
        - Forces the judge to reason about each aspect separately

        Args:
            trajectories: List of G trajectories from the same prompt group.
            task_metadata: Shared metadata for the prompt.

        Returns:
            Tuple of (composite_scores, dimension_details):
            - composite_scores: List of normalized [0.0, 1.0] floats.
            - dimension_details: List of dicts with per-dimension raw scores
              (1-10 scale) for logging. Empty dicts on failure.
        """
        n = len(trajectories)
        fallback_scores = [0.5] * n
        fallback_details = [{}] * n

        if n == 0:
            return [], []
        if n == 1:
            return [0.5], [{}]

        client = self._get_client()
        if client is None:
            return fallback_scores, fallback_details

        # Build the prompt
        candidates_block = self._build_candidates_block(trajectories)
        candidates_placeholder = ", ".join(
            '{{"id": {i}, "accuracy": <1-10>, "helpfulness": <1-10>, '
            '"tool_use": <1-10>, "communication": <1-10>}}'.format(i=i + 1)
            for i in range(n)
        )
        user_msg = _JUDGE_USER_TEMPLATE.format(
            query=task_metadata.query,
            tier=task_metadata.tier,
            difficulty=task_metadata.difficulty,
            candidates_block=candidates_block,
            candidates_placeholder=candidates_placeholder,
        )

        # Call LLM with retries
        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=self.temperature,
                    max_tokens=1500,
                    extra_body={"enable_thinking": False},
                )
                raw_text = response.choices[0].message.content.strip()
                result = self._parse_scores(raw_text, n)
                if result is not None:
                    return result
                logger.warning(
                    "GroupJudge: parse failed attempt %d/%d, raw=%s",
                    attempt + 1, self.max_retries + 1, raw_text[:300],
                )
            except Exception as e:
                logger.warning(
                    "GroupJudge: API error attempt %d/%d: %s",
                    attempt + 1, self.max_retries + 1, e,
                )

        logger.warning("GroupJudge: all attempts failed, returning fallback scores")
        return fallback_scores, fallback_details

    def _build_candidates_block(self, trajectories: list["RolloutTrajectory"]) -> str:
        """Format trajectories into the candidates block for the judge prompt."""
        blocks = []
        for i, traj in enumerate(trajectories):
            tools_called = traj.get_tools_called()
            tools_str = ", ".join(tools_called) if tools_called else "(none)"

            final = traj.final_answer or ""
            if not final and traj.termination_reason == "max_tokens":
                final = "[TRUNCATED — no final answer]"
            elif not final:
                final = "[No answer produced]"
            # Truncate very long answers to prevent token waste
            if len(final) > 800:
                final = final[:800] + "... [truncated for judging]"

            blocks.append(_CANDIDATE_TEMPLATE.format(
                idx=i + 1,
                tools_called=tools_str,
                final_answer=final,
            ))
        return "\n\n".join(blocks)

    def _parse_scores(
        self, raw_text: str, expected_n: int
    ) -> Optional[tuple[list[float], list[dict]]]:
        """Parse multi-dimensional LLM judge output.

        Expected format:
            {"candidates": [
                {"id": 1, "accuracy": 8, "helpfulness": 7, "tool_use": 9, "communication": 6},
                ...
            ]}

        Also handles legacy single-score format for backward compatibility:
            {"scores": [7, 5, 8, 3]}

        Returns:
            Tuple of (composite_scores, dimension_details) or None on failure.
            - composite_scores: weighted average of dimensions, normalized [0,1]
            - dimension_details: list of raw dimension dicts (1-10 scale)
        """
        # Extract JSON from potential markdown wrapping
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            match = re.search(r"\{.*\"candidates\".*\}", text, re.DOTALL)
            if not match:
                match = re.search(r"\{.*\"scores\".*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None

        # --- Multi-dimensional format (preferred) ---
        candidates = data.get("candidates")
        if isinstance(candidates, list) and len(candidates) == expected_n:
            composite_scores = []
            dimension_details = []
            for entry in candidates:
                if not isinstance(entry, dict):
                    return None
                dims = {}
                for dim in _JUDGE_DIMENSIONS:
                    val = entry.get(dim)
                    if val is None:
                        return None
                    try:
                        dims[dim] = max(1.0, min(10.0, float(val)))
                    except (TypeError, ValueError):
                        return None

                # Weighted composite → [0, 1]
                weighted = sum(
                    JUDGE_DIMENSION_WEIGHTS[d] * (dims[d] - 1.0) / 9.0
                    for d in _JUDGE_DIMENSIONS
                )
                composite_scores.append(weighted)
                dimension_details.append(dims)

            return composite_scores, dimension_details

        # --- Legacy single-score format (fallback) ---
        raw_scores = data.get("scores")
        if raw_scores is not None:
            if isinstance(raw_scores, dict):
                ordered = []
                for i in range(expected_n):
                    key = f"score_{i+1}"
                    val = raw_scores.get(key)
                    if val is None:
                        return None
                    ordered.append(float(val))
                raw_scores = ordered

            if not isinstance(raw_scores, list) or len(raw_scores) != expected_n:
                return None

            scores = []
            for s in raw_scores:
                try:
                    v = max(1.0, min(10.0, float(s)))
                    scores.append((v - 1.0) / 9.0)
                except (TypeError, ValueError):
                    return None
            return scores, [{}] * expected_n

        return None


# Singleton judge instance (lazy init, shares API client across calls)
_default_judge: Optional[GroupJudge] = None


def _get_default_judge() -> GroupJudge:
    """Get or create the default GroupJudge instance."""
    global _default_judge
    if _default_judge is None:
        _default_judge = GroupJudge()
    return _default_judge


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
    GRPO v2: v1 + conditional monitoring + strict hard gate + truncation penalty.

    Policy: GRPO-v1 checkpoint
    Reference: GRPO-v1 checkpoint (frozen)

    Components:
    - v1 base (100%)
    - R_conditional is tracked for diagnostics only (not added to total)

    Hard gate:
    - For T1/T2/T3 tasks, if successful_tool_calls == 0, total reward = 0.0.

    Truncation penalty:
    - If trajectory was truncated by max_tokens exhaustion, total *= 0.5.
    - Rationale: truncated rollouts have unreliable reward signals;
      halving their weight prevents them from misleading the policy gradient.
    """
    # Get v1 base
    v1 = reward_v1(trajectory, task_metadata)

    # R_conditional
    r_conditional = compute_conditional_score(trajectory, task_metadata)

    total = v1.total

    tier = task_metadata.tier or ""
    successful_tool_calls = sum(
        1
        for step in trajectory.steps
        if step.tool_execution is not None and step.tool_execution.success
    )
    hard_gate_applied = (
        (tier.startswith("T1") or tier.startswith("T2") or tier.startswith("T3"))
        and successful_tool_calls == 0
    )
    if hard_gate_applied:
        total = 0.0

    # Penalize truncated rollouts — their reward signal is unreliable
    truncation_penalized = False
    if trajectory.termination_reason == "max_tokens":
        total *= 0.5
        truncation_penalized = True

    return RewardBreakdown(
        total=total,
        r_format=v1.r_format,
        r_tool_selection=v1.r_tool_selection,
        r_outcome=v1.r_outcome,
        r_efficiency=0.0,
        r_conditional=r_conditional,
        details={
            **v1.details,
            "successful_tool_calls": successful_tool_calls,
            "hard_gate_applied": hard_gate_applied,
            "truncation_penalized": truncation_penalized,
            "actual_steps": trajectory.total_tool_calls,
            "optimal_steps": task_metadata.optimal_steps,
            "tier": task_metadata.tier,
        },
    )


def reward_v3_group(
    trajectories: list[RolloutTrajectory],
    task_metadata: TaskMetadata,
    judge: Optional[GroupJudge] = None,
    rule_weight: float = 0.5,
    judge_weight: float = 0.5,
) -> list[RewardBreakdown]:
    """
    GRPO v3: Group-relative hybrid reward (rule-based + LLM judge).

    Unlike v1/v2 which score each trajectory independently, v3 scores the
    entire GRPO group together. The LLM judge sees all G candidates
    side-by-side (RULER-style) and produces relative scores, naturally
    creating reward variance even when all trajectories are decent.

    Formula: total = rule_weight * r_v2 + judge_weight * r_judge

    When the judge is unavailable (no API key, API error), gracefully
    degrades to pure v2 (rule_weight=1.0, judge_weight=0.0).

    ARPO compatible: ARPO's advantage attribution is reward-function-agnostic.
    It only needs per-step rewards (which can be the final trajectory reward
    broadcast to all steps). The group-relative nature of v3 actually helps
    ARPO by ensuring meaningful reward signal variance.

    Args:
        trajectories: List of G trajectories from the same prompt group.
        task_metadata: Shared metadata for the prompt.
        judge: GroupJudge instance (uses default singleton if None).
        rule_weight: Weight for rule-based v2 scores (default 0.5).
        judge_weight: Weight for LLM judge scores (default 0.5).

    Returns:
        List of RewardBreakdown, one per trajectory.
    """
    n = len(trajectories)
    if n == 0:
        return []

    # Step 1: Compute per-trajectory v2 rule-based scores
    v2_breakdowns = [reward_v2(traj, task_metadata) for traj in trajectories]
    v2_scores = [b.total for b in v2_breakdowns]

    # Step 2: Get group-relative LLM judge scores (multi-dimensional)
    judge = judge or _get_default_judge()
    judge_scores, judge_dims = judge.score_group(trajectories, task_metadata)

    # Step 3: Combine with weights
    # If judge returned all-same composites (fallback), shift to pure v2
    all_same = len(set(round(s, 4) for s in judge_scores)) <= 1
    if all_same:
        effective_rule_w = 1.0
        effective_judge_w = 0.0
        judge_applied = False
    else:
        effective_rule_w = rule_weight
        effective_judge_w = judge_weight
        judge_applied = True

    results = []
    for i in range(n):
        total = effective_rule_w * v2_scores[i] + effective_judge_w * judge_scores[i]
        total = max(0.0, min(1.0, total))

        detail = {
            **v2_breakdowns[i].details,
            "reward_version": "v3",
            "judge_applied": judge_applied,
            "rule_weight": effective_rule_w,
            "judge_weight": effective_judge_w,
            "v2_score": v2_scores[i],
            "judge_composite": judge_scores[i],
        }
        # Per-dimension raw scores (1-10) for wandb logging
        if judge_dims[i]:
            detail["judge_dimensions"] = judge_dims[i]

        results.append(RewardBreakdown(
            total=total,
            r_format=v2_breakdowns[i].r_format,
            r_tool_selection=v2_breakdowns[i].r_tool_selection,
            r_outcome=v2_breakdowns[i].r_outcome,
            r_efficiency=v2_breakdowns[i].r_efficiency,
            r_conditional=v2_breakdowns[i].r_conditional,
            r_llm_judge=judge_scores[i],
            details=detail,
        ))

    return results


def reward_v3(
    trajectory: RolloutTrajectory,
    task_metadata: TaskMetadata,
    llm_judge: Optional["GroupJudge"] = None,
) -> RewardBreakdown:
    """
    Single-trajectory v3 wrapper (backward compatible).

    When called with a single trajectory (not in group context), falls back
    to v2 behavior since RULER-style relative scoring requires multiple
    candidates. Use reward_v3_group() for proper v3 group scoring.
    """
    # Single trajectory — can't do relative scoring, fall back to v2
    return reward_v2(trajectory, task_metadata)


# ============================================================================
# TRL Entry Point
# ============================================================================


def trl_reward_wrapper(completions: List[str], **kwargs) -> List[float]:
    """
    TRL-compatible reward function for GRPOTrainer.

    TRL calls this with a batch of completion strings and dataset metadata
    passed as keyword arguments (each kwarg is a list aligned with completions).

    Args:
        completions: List of full conversation text strings from rollouts.
        **kwargs: Dataset columns as lists, including:
            - tier: List[str]
            - difficulty: List[str]
            - optimal_steps: List[int]
            - query: List[str]
            - branch_condition: List[str] (JSON or empty)

    Returns:
        List of reward scores in [0.0, 1.0].
    """
    tiers = kwargs.get("tier", ["T1"] * len(completions))
    difficulties = kwargs.get("difficulty", ["medium"] * len(completions))
    optimal_steps_list = kwargs.get("optimal_steps", [1] * len(completions))
    queries = kwargs.get("query", [""] * len(completions))
    branch_conditions = kwargs.get("branch_condition", [""] * len(completions))

    scores = []
    for i, completion in enumerate(completions):
        if not completion or not completion.strip():
            scores.append(0.0)
            continue

        # Parse branch_condition from JSON string
        bc = None
        bc_str = branch_conditions[i] if i < len(branch_conditions) else ""
        if bc_str:
            try:
                bc = json.loads(bc_str)
            except (json.JSONDecodeError, TypeError):
                pass

        task_meta = TaskMetadata(
            query=queries[i] if i < len(queries) else "",
            tier=tiers[i] if i < len(tiers) else "T1",
            difficulty=difficulties[i] if i < len(difficulties) else "medium",
            optimal_steps=optimal_steps_list[i] if i < len(optimal_steps_list) else 1,
            branch_condition=bc,
        )

        trajectory = _build_trajectory_from_solution(completion, task_meta)
        breakdown = reward_v2(trajectory, task_meta)
        scores.append(breakdown.total)

    return scores


# ============================================================================
# veRL Entry Point (legacy, kept for backward compatibility)
# ============================================================================


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any = None,
    extra_info: Any = None,
) -> float:
    """
    veRL-compatible reward function entry point.

    Called by veRL's reward manager for each completed rollout.
    Bridges veRL's flat (data_source, solution_str, ground_truth, extra_info)
    interface to our structured reward_v1/v2/v3 functions.

    Args:
        data_source: Dataset identifier (must be "nutrimind")
        solution_str: The model's complete output string for the rollout
        ground_truth: JSON string or dict with tier, difficulty, optimal_steps
        extra_info: JSON string or dict with interaction_kwargs

    Returns:
        Reward score in [0.0, 1.0]
    """
    if data_source != "nutrimind":
        return 0.0

    if not solution_str or not solution_str.strip():
        return 0.0

    # Parse ground_truth
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            ground_truth = {}
    if not isinstance(ground_truth, dict):
        ground_truth = {}

    # Parse extra_info
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except (json.JSONDecodeError, TypeError):
            extra_info = {}
    if not isinstance(extra_info, dict):
        extra_info = {}

    # Extract metadata
    tier = ground_truth.get("tier") or extra_info.get("interaction_kwargs", {}).get("tier", "T1")
    difficulty = ground_truth.get("difficulty", "medium")
    optimal_steps = ground_truth.get("optimal_steps", 1)

    # Resolve reward version with explicit priority:
    # 1) extra_info.interaction_kwargs.reward_version
    # 2) ground_truth.reward_version
    # 3) env var set by launcher (NUTRIMIND_REWARD_VERSION)
    # 4) default "v2"
    interaction_kwargs = extra_info.get("interaction_kwargs", {})
    reward_version = (
        interaction_kwargs.get("reward_version")
        or ground_truth.get("reward_version")
        or os.getenv("NUTRIMIND_REWARD_VERSION")
        or "v2"
    )
    if reward_version not in {"v1", "v2", "v3"}:
        reward_version = "v2"

    # Build TaskMetadata
    task_meta = TaskMetadata(
        query="",  # Not available in reward-only context
        tier=tier,
        difficulty=difficulty,
        optimal_steps=optimal_steps,
    )

    # Build a minimal RolloutTrajectory by parsing the solution string
    trajectory = _build_trajectory_from_solution(solution_str, task_meta)

    if reward_version == "v1":
        breakdown = reward_v1(trajectory, task_meta)
    elif reward_version == "v3":
        # In veRL reward-manager context we do not have an external LLM judge,
        # so reward_v3 will gracefully fall back to v2 behavior where needed.
        breakdown = reward_v3(trajectory, task_meta, llm_judge=None)
    else:
        breakdown = reward_v2(trajectory, task_meta)

    return breakdown.total


def _build_trajectory_from_solution(
    solution_str: str, task_meta: TaskMetadata
) -> RolloutTrajectory:
    """
    Reconstruct a RolloutTrajectory from decoded rollout text.

    In veRL multi-turn mode, solution_str usually contains plain role labels
    ("user" / "assistant") plus <tool_call>/<tool_response> blocks.
    We parse the blocks in-order and bind each tool_response payload to the
    preceding tool_call so runtime outcome checks can use real tool values.
    """
    from src.orchestrator.tool_parser import ToolParser

    trajectory = RolloutTrajectory(prompt=task_meta.query)
    parser = ToolParser(validate_tool_name=False)

    import re

    block_pattern = re.compile(
        r"<(tool_call|tool_response)>\s*(.*?)\s*</\1>",
        flags=re.DOTALL,
    )

    pending_step: Optional[RolloutStep] = None
    step_idx = 0
    last_end = 0

    for match in block_pattern.finditer(solution_str):
        tag_type = match.group(1)
        block_content = match.group(2).strip()
        raw_block = match.group(0)
        last_end = match.end()

        if tag_type == "tool_call":
            try:
                payload = json.loads(block_content)
            except json.JSONDecodeError:
                parse_error_step = RolloutStep(
                    step_idx=step_idx,
                    model_output=raw_block,
                    think_content=None,
                    action_type="parse_error",
                    injected_response="Invalid tool_call JSON in solution_str",
                )
                trajectory.steps.append(parse_error_step)
                step_idx += 1
                pending_step = None
                continue

            tool_name = payload.get("name") or payload.get("function") or "unknown_tool"
            tool_args = payload.get("arguments") or payload.get("parameters") or {}
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {}
            if not isinstance(tool_args, dict):
                tool_args = {}

            pending_step = RolloutStep(
                step_idx=step_idx,
                model_output=raw_block,
                think_content=None,
                action_type="tool_call",
                tool_execution=ToolExecutionResult(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result={"status": "error", "message": "missing tool_response"},
                    success=False,
                ),
            )
            trajectory.steps.append(pending_step)
            trajectory.total_tool_calls += 1
            step_idx += 1

        else:  # tool_response
            if pending_step is None or pending_step.tool_execution is None:
                continue

            try:
                response_payload = json.loads(block_content)
            except json.JSONDecodeError:
                response_payload = {
                    "status": "error",
                    "error_type": "invalid_tool_response_json",
                    "message": "Failed to parse tool_response JSON",
                    "raw": block_content,
                }

            status = response_payload.get("status")
            success = status == "success"
            if status is None:
                success = "error" not in response_payload

            pending_step.tool_execution.result = response_payload
            pending_step.tool_execution.success = success
            pending_step.injected_response = raw_block
            pending_step = None

    trailing = solution_str[last_end:].strip()
    if trailing:
        trailing = re.sub(r"(^|\n)\s*(assistant|user)\s*(?=\n|$)", "\\1", trailing, flags=re.IGNORECASE)
        trailing = trailing.strip()

    if trailing:
        parsed_final = parser.parse(trailing)
        trajectory.final_answer = parsed_final.content if parsed_final.type == "final_answer" else trailing
        trajectory.terminated = True
        trajectory.termination_reason = "final_answer"
    elif trajectory.steps:
        trajectory.final_answer = ""
        trajectory.terminated = True
        trajectory.termination_reason = "final_answer"

    return trajectory


# ============================================================================
# TRL environment_factory Entry Point (ADR-007)
# ============================================================================

def reward_from_env(environments, completions, **kwargs) -> list[float]:
    """TRL reward function for environment_factory mode.

    When GRPOTrainer uses environment_factory, the reward function receives
    an ``environments`` kwarg containing one env instance per completion.
    We read the env's structured tool history instead of re-parsing text.

    Supports reward_version selection via env var NUTRIMIND_REWARD_VERSION:
    - "v2": Per-trajectory rule-based scoring
    - "v3" (default): Group-relative hybrid scoring (v2 + LLM judge)

    Note on r_format: In environment_factory mode, TRL handles tool call parsing
    internally. Parse errors never reach the env, so env-based trajectories always
    have r_format=1.0.

    Note on v3 grouping: TRL calls reward_from_env with all G completions from
    the same prompt in one batch. So ``environments`` and ``completions`` already
    form a natural group for RULER-style relative scoring.

    Args:
        environments: List of NutriMindToolEnv instances, one per completion.
        completions: List of completion message dicts (conversational format).
        **kwargs: Dataset columns as aligned lists (tier, query, difficulty, etc.).

    Returns:
        List of float reward scores in [0.0, 1.0].
    """
    if environments and completions:
        assert len(environments) == len(completions), (
            f"environments/completions length mismatch: {len(environments)} vs {len(completions)}"
        )

    reward_version = os.getenv("NUTRIMIND_REWARD_VERSION", "v3")

    tiers = kwargs.get("tier", ["T1"] * len(completions))
    queries = kwargs.get("query", [""] * len(completions))
    difficulties = kwargs.get("difficulty", ["medium"] * len(completions))
    optimal_steps_list = kwargs.get("optimal_steps", [1] * len(completions))
    branch_conditions = kwargs.get("branch_condition", [""] * len(completions))

    # Build trajectories and task metadata for all completions
    trajectories: list[RolloutTrajectory] = []
    task_metas: list[TaskMetadata] = []

    for i, (env, completion) in enumerate(zip(environments, completions)):
        try:
            traj = _build_trajectory_from_env(env, completion)
        except Exception as e:
            logger.warning("reward_from_env: trajectory build error for sample %d: %s", i, e)
            traj = RolloutTrajectory(prompt="")
            traj.terminated = True
            traj.termination_reason = "error"
        trajectories.append(traj)

        bc = None
        bc_str = branch_conditions[i] if i < len(branch_conditions) else ""
        if bc_str:
            try:
                bc = json.loads(bc_str) if isinstance(bc_str, str) else bc_str
            except (json.JSONDecodeError, TypeError):
                pass

        task_metas.append(TaskMetadata(
            query=queries[i] if i < len(queries) else "",
            tier=tiers[i] if i < len(tiers) else "T1",
            difficulty=difficulties[i] if i < len(difficulties) else "medium",
            optimal_steps=int(optimal_steps_list[i]) if i < len(optimal_steps_list) else 1,
            branch_condition=bc,
        ))

    # Score based on reward version
    if reward_version == "v3" and len(trajectories) > 1:
        # v3: Group-relative scoring — all completions share the same prompt
        shared_meta = task_metas[0]
        try:
            breakdowns = reward_v3_group(trajectories, shared_meta)
            return [b.total for b in breakdowns]
        except Exception as e:
            logger.warning("reward_from_env: v3 group scoring failed, falling back to v2: %s", e)
            # Fall through to per-trajectory v2

    # v2 (fallback or explicit): Per-trajectory scoring
    scores: list[float] = []
    for traj, meta in zip(trajectories, task_metas):
        try:
            breakdown = reward_v2(traj, meta)
            scores.append(breakdown.total)
        except Exception as e:
            logger.warning("reward_from_env: v2 scoring error: %s", e)
            scores.append(0.0)

    return scores


def _build_trajectory_from_env(env, completion) -> RolloutTrajectory:
    """Build a RolloutTrajectory from the env's structured tool call history.

    More reliable than _build_trajectory_from_solution (which re-parses text)
    because it reads the actual tool execution records.
    """
    trajectory = RolloutTrajectory(prompt=getattr(env, "_query", ""))

    for i, call in enumerate(getattr(env, "_tool_history", [])):
        step = RolloutStep(
            step_idx=i,
            model_output="",
            think_content=None,
            action_type="tool_call",
            tool_execution=ToolExecutionResult(
                tool_name=call["tool_name"],
                tool_args=call.get("args", {}),
                result=call.get("result", {}),
                success=call.get("success", False),
            ),
        )
        trajectory.steps.append(step)
        trajectory.total_tool_calls += 1

    # Extract final answer from completion text
    final_text = _extract_final_answer_from_completion(completion)
    if final_text:
        trajectory.final_answer = final_text
        trajectory.terminated = True
        trajectory.termination_reason = "final_answer"
    elif trajectory.total_tool_calls > 0:
        # Has tool calls but no final answer → likely truncated by max_tokens
        trajectory.terminated = True
        trajectory.termination_reason = "max_tokens"

    return trajectory


def _extract_final_answer_from_completion(completion) -> Optional[str]:
    """Extract the final answer text from a TRL completion.

    TRL completions in conversational format are a list of message dicts.
    The final answer is the last assistant message that doesn't contain
    a <tool_call> tag.
    """
    # Handle conversational format: list of {"role": ..., "content": ...}
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if "<tool_call>" not in content:
                    return content.strip() or None
        return None

    # Handle string format (fallback)
    if isinstance(completion, str):
        text = completion
        # Find text after the last </tool_response>
        last_resp_idx = text.rfind("</tool_response>")
        if last_resp_idx >= 0:
            text = text[last_resp_idx + len("</tool_response>"):]
        # Remove any remaining tool_call tags
        if "<tool_call>" in text:
            return None
        text = text.strip()
        # Clean role labels
        for prefix in ("assistant", "Assistant"):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text or None

    return None


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
