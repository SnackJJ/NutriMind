#!/usr/bin/env python3
"""
Evaluate Teacher vs Student Model Gap.

Compares the RL-trained student model against the teacher model (e.g., Qwen3.5-Plus)
to measure how well the student has learned from the teacher's trajectories.

Evaluation dimensions:
1. Overall reward gap
2. Per-tier reward breakdown
3. Tool calling accuracy
4. Format correctness rate
5. Qualitative failure case analysis

Usage:
    # Compare student checkpoint against teacher
    python scripts/evaluate_teacher_gap.py \
        --student_model models/verl_grpo/final \
        --teacher_model qwen3.5-plus \
        --eval_prompts data/grpo/eval_prompts.jsonl \
        --output_dir eval_results/teacher_gap

    # With specific vLLM server URLs
    python scripts/evaluate_teacher_gap.py \
        --student_url http://localhost:8000/v1 \
        --teacher_url http://localhost:8001/v1 \
        --eval_prompts data/grpo/eval_prompts.jsonl

Requirements:
    - Student model served via vLLM or loaded locally
    - Teacher model accessible via API (or local vLLM)
    - Eval prompt set with metadata
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.grpo.environment import (
    NutriMindEnv,
    RolloutTrajectory,
    TaskMetadata,
)
from src.training.grpo.reward import (
    reward_v1,
    reward_v2,
    RewardBreakdown,
    check_format_validity,
)
from src.orchestrator.orchestrator import TOOL_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for a single model's evaluation."""

    model_name: str
    total_prompts: int = 0
    avg_reward: float = 0.0

    # Component scores
    avg_format_score: float = 0.0
    avg_tool_selection_score: float = 0.0
    avg_outcome_score: float = 0.0

    # Per-tier metrics
    tier_rewards: Dict[str, List[float]] = field(default_factory=dict)

    # Tool calling stats
    tool_accuracy: float = 0.0
    format_correct_rate: float = 0.0
    completion_rate: float = 0.0

    # Detailed breakdown
    per_prompt_rewards: List[float] = field(default_factory=list)


@dataclass
class GapAnalysis:
    """Analysis of the gap between student and teacher."""

    student_metrics: ModelMetrics
    teacher_metrics: ModelMetrics

    # Gap metrics
    overall_reward_gap: float = 0.0
    tier_gaps: Dict[str, float] = field(default_factory=dict)
    tool_accuracy_gap: float = 0.0
    format_gap: float = 0.0

    # Failure cases
    student_failures: List[Dict[str, Any]] = field(default_factory=list)


def generate_with_vllm(
    messages: List[Dict[str, str]],
    server_url: str,
    model_name: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """Generate response using vLLM server."""
    endpoint = f"{server_url}/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["<|im_end|>"]
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=120.0)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return ""


def run_rollout(
    query: str,
    generate_fn,
    tool_registry: Dict,
    max_rounds: int = 6,
) -> RolloutTrajectory:
    """Run a single rollout and return trajectory."""
    env = NutriMindEnv(
        tool_registry=tool_registry,
        max_tool_rounds=max_rounds,
    )

    messages = env.reset(query)
    done = False

    while not done:
        output = generate_fn(messages)
        if not output:
            break
        messages, done, info = env.step(output)

    return env.get_trajectory()


def evaluate_model(
    model_name: str,
    generate_fn,
    prompts: List[Dict[str, Any]],
    tool_registry: Dict,
    reward_version: str = "v2",
) -> ModelMetrics:
    """Evaluate a model on all prompts."""
    metrics = ModelMetrics(model_name=model_name)
    metrics.total_prompts = len(prompts)

    all_rewards = []
    all_format_scores = []
    all_tool_scores = []
    all_outcome_scores = []
    format_correct = 0
    completed = 0

    for i, prompt_data in enumerate(prompts):
        query = prompt_data["query"]
        tier = prompt_data.get("tier", "T1")

        logger.info(f"[{i+1}/{len(prompts)}] Evaluating: {query[:50]}...")

        # Run rollout
        trajectory = run_rollout(query, generate_fn, tool_registry)

        # Create task metadata
        task_meta = TaskMetadata(
            query=query,
            tier=tier,
            difficulty=prompt_data.get("difficulty", "medium"),
        )

        # Compute reward
        if reward_version == "v1":
            breakdown = reward_v1(trajectory, task_meta)
        else:
            breakdown = reward_v2(trajectory, task_meta)

        all_rewards.append(breakdown.total)
        all_format_scores.append(breakdown.r_format)
        all_tool_scores.append(breakdown.r_tool_selection)
        all_outcome_scores.append(breakdown.r_outcome)

        # Track per-tier
        if tier not in metrics.tier_rewards:
            metrics.tier_rewards[tier] = []
        metrics.tier_rewards[tier].append(breakdown.total)

        # Check format
        format_valid, _ = check_format_validity(trajectory)
        if format_valid:
            format_correct += 1

        # Check completion
        if trajectory.terminated and trajectory.termination_reason == "final_answer":
            completed += 1

        metrics.per_prompt_rewards.append(breakdown.total)

    # Aggregate metrics
    metrics.avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
    metrics.avg_format_score = sum(all_format_scores) / len(all_format_scores) if all_format_scores else 0
    metrics.avg_tool_selection_score = sum(all_tool_scores) / len(all_tool_scores) if all_tool_scores else 0
    metrics.avg_outcome_score = sum(all_outcome_scores) / len(all_outcome_scores) if all_outcome_scores else 0
    metrics.format_correct_rate = format_correct / len(prompts) if prompts else 0
    metrics.completion_rate = completed / len(prompts) if prompts else 0

    return metrics


def analyze_gap(
    student_metrics: ModelMetrics,
    teacher_metrics: ModelMetrics,
    prompts: List[Dict[str, Any]],
) -> GapAnalysis:
    """Analyze the gap between student and teacher."""
    analysis = GapAnalysis(
        student_metrics=student_metrics,
        teacher_metrics=teacher_metrics,
    )

    # Overall gap
    analysis.overall_reward_gap = teacher_metrics.avg_reward - student_metrics.avg_reward

    # Per-tier gaps
    all_tiers = set(student_metrics.tier_rewards.keys()) | set(teacher_metrics.tier_rewards.keys())
    for tier in all_tiers:
        student_avg = (
            sum(student_metrics.tier_rewards.get(tier, [0])) /
            len(student_metrics.tier_rewards.get(tier, [1]))
        )
        teacher_avg = (
            sum(teacher_metrics.tier_rewards.get(tier, [0])) /
            len(teacher_metrics.tier_rewards.get(tier, [1]))
        )
        analysis.tier_gaps[tier] = teacher_avg - student_avg

    # Format gap
    analysis.format_gap = teacher_metrics.format_correct_rate - student_metrics.format_correct_rate

    # Find failure cases (student significantly worse)
    for i, prompt in enumerate(prompts):
        student_reward = student_metrics.per_prompt_rewards[i]
        teacher_reward = teacher_metrics.per_prompt_rewards[i]

        if teacher_reward - student_reward > 0.3:  # Significant gap
            analysis.student_failures.append({
                "query": prompt["query"],
                "tier": prompt.get("tier", "T1"),
                "student_reward": student_reward,
                "teacher_reward": teacher_reward,
                "gap": teacher_reward - student_reward,
            })

    # Sort failures by gap
    analysis.student_failures.sort(key=lambda x: x["gap"], reverse=True)

    return analysis


def generate_report(analysis: GapAnalysis, output_path: str) -> None:
    """Generate evaluation report."""
    report = []
    report.append("=" * 70)
    report.append("NutriMind Teacher vs Student Evaluation Report")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("=" * 70)

    report.append("\n## Overall Metrics\n")
    report.append(f"| Metric | Student | Teacher | Gap |")
    report.append(f"|--------|---------|---------|-----|")
    report.append(f"| Avg Reward | {analysis.student_metrics.avg_reward:.3f} | {analysis.teacher_metrics.avg_reward:.3f} | {analysis.overall_reward_gap:+.3f} |")
    report.append(f"| Format Score | {analysis.student_metrics.avg_format_score:.3f} | {analysis.teacher_metrics.avg_format_score:.3f} | {analysis.teacher_metrics.avg_format_score - analysis.student_metrics.avg_format_score:+.3f} |")
    report.append(f"| Tool Selection | {analysis.student_metrics.avg_tool_selection_score:.3f} | {analysis.teacher_metrics.avg_tool_selection_score:.3f} | {analysis.teacher_metrics.avg_tool_selection_score - analysis.student_metrics.avg_tool_selection_score:+.3f} |")
    report.append(f"| Outcome Score | {analysis.student_metrics.avg_outcome_score:.3f} | {analysis.teacher_metrics.avg_outcome_score:.3f} | {analysis.teacher_metrics.avg_outcome_score - analysis.student_metrics.avg_outcome_score:+.3f} |")
    report.append(f"| Format Correct % | {analysis.student_metrics.format_correct_rate:.1%} | {analysis.teacher_metrics.format_correct_rate:.1%} | {analysis.format_gap:+.1%} |")
    report.append(f"| Completion % | {analysis.student_metrics.completion_rate:.1%} | {analysis.teacher_metrics.completion_rate:.1%} | {analysis.teacher_metrics.completion_rate - analysis.student_metrics.completion_rate:+.1%} |")

    report.append("\n## Per-Tier Analysis\n")
    report.append(f"| Tier | Student Avg | Teacher Avg | Gap |")
    report.append(f"|------|-------------|-------------|-----|")
    for tier in sorted(analysis.tier_gaps.keys()):
        student_avg = (
            sum(analysis.student_metrics.tier_rewards.get(tier, [0])) /
            max(1, len(analysis.student_metrics.tier_rewards.get(tier, [1])))
        )
        teacher_avg = (
            sum(analysis.teacher_metrics.tier_rewards.get(tier, [0])) /
            max(1, len(analysis.teacher_metrics.tier_rewards.get(tier, [1])))
        )
        report.append(f"| {tier} | {student_avg:.3f} | {teacher_avg:.3f} | {analysis.tier_gaps[tier]:+.3f} |")

    report.append("\n## Largest Failure Cases (Top 10)\n")
    for i, failure in enumerate(analysis.student_failures[:10]):
        report.append(f"\n### Case {i+1}: Gap = {failure['gap']:.3f}")
        report.append(f"- **Tier**: {failure['tier']}")
        report.append(f"- **Query**: {failure['query']}")
        report.append(f"- **Student Reward**: {failure['student_reward']:.3f}")
        report.append(f"- **Teacher Reward**: {failure['teacher_reward']:.3f}")

    report.append("\n## Summary\n")
    if analysis.overall_reward_gap < 0.05:
        report.append("**Excellent**: Student matches teacher performance (gap < 0.05)")
    elif analysis.overall_reward_gap < 0.1:
        report.append("**Good**: Student is close to teacher (gap < 0.1)")
    elif analysis.overall_reward_gap < 0.2:
        report.append("**Moderate**: Student has room for improvement (gap < 0.2)")
    else:
        report.append("**Needs Work**: Significant gap between student and teacher")

    report.append("\n" + "=" * 70)

    # Save report
    report_text = "\n".join(report)
    with open(output_path, "w") as f:
        f.write(report_text)

    print(report_text)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate teacher vs student model gap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--student_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Student model vLLM server URL",
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="student",
        help="Student model name (for vLLM)",
    )
    parser.add_argument(
        "--teacher_url",
        type=str,
        default="http://localhost:8001/v1",
        help="Teacher model vLLM server URL",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="teacher",
        help="Teacher model name (for vLLM)",
    )
    parser.add_argument(
        "--eval_prompts",
        type=str,
        default="data/grpo/eval_prompts.jsonl",
        help="Evaluation prompts file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results/teacher_gap",
        help="Output directory for results",
    )
    parser.add_argument(
        "--reward_version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Reward function version",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to evaluate (for testing)",
    )
    args = parser.parse_args()

    # Load prompts
    prompts = []
    with open(args.eval_prompts, "r") as f:
        for line in f:
            prompts.append(json.loads(line))

    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    logger.info(f"Loaded {len(prompts)} evaluation prompts")

    # Create generation functions
    def student_generate(messages):
        return generate_with_vllm(
            messages, args.student_url, args.student_model
        )

    def teacher_generate(messages):
        return generate_with_vllm(
            messages, args.teacher_url, args.teacher_model
        )

    # Evaluate student
    logger.info("\n=== Evaluating Student Model ===")
    student_metrics = evaluate_model(
        "student",
        student_generate,
        prompts,
        TOOL_REGISTRY,
        args.reward_version,
    )

    # Evaluate teacher
    logger.info("\n=== Evaluating Teacher Model ===")
    teacher_metrics = evaluate_model(
        "teacher",
        teacher_generate,
        prompts,
        TOOL_REGISTRY,
        args.reward_version,
    )

    # Analyze gap
    logger.info("\n=== Analyzing Gap ===")
    analysis = analyze_gap(student_metrics, teacher_metrics, prompts)

    # Generate report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "report.md"
    generate_report(analysis, str(report_path))

    # Save raw data
    raw_data = {
        "student_metrics": {
            "model_name": student_metrics.model_name,
            "avg_reward": student_metrics.avg_reward,
            "avg_format_score": student_metrics.avg_format_score,
            "avg_tool_selection_score": student_metrics.avg_tool_selection_score,
            "avg_outcome_score": student_metrics.avg_outcome_score,
            "format_correct_rate": student_metrics.format_correct_rate,
            "completion_rate": student_metrics.completion_rate,
            "tier_rewards": {k: list(v) for k, v in student_metrics.tier_rewards.items()},
        },
        "teacher_metrics": {
            "model_name": teacher_metrics.model_name,
            "avg_reward": teacher_metrics.avg_reward,
            "avg_format_score": teacher_metrics.avg_format_score,
            "avg_tool_selection_score": teacher_metrics.avg_tool_selection_score,
            "avg_outcome_score": teacher_metrics.avg_outcome_score,
            "format_correct_rate": teacher_metrics.format_correct_rate,
            "completion_rate": teacher_metrics.completion_rate,
            "tier_rewards": {k: list(v) for k, v in teacher_metrics.tier_rewards.items()},
        },
        "gap_analysis": {
            "overall_reward_gap": analysis.overall_reward_gap,
            "tier_gaps": analysis.tier_gaps,
            "format_gap": analysis.format_gap,
            "num_failures": len(analysis.student_failures),
        },
    }

    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(raw_data, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Overall reward gap: {analysis.overall_reward_gap:+.3f}")


if __name__ == "__main__":
    main()
