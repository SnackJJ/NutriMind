#!/usr/bin/env python3
"""
Difficulty Labeling for GRPO Prompts.

Runs the SFT model on all prompts (N=8 rollouts each) and computes
success rate to label difficulty:
- easy:   success_rate >= 0.7
- medium: 0.2 <= rate < 0.7
- hard:   success_rate < 0.2

Usage:
    python label_difficulty.py --input data/grpo/prompts.jsonl \
                               --output data/grpo/prompts_labeled.jsonl \
                               --model models/sft/final

See phase4_grpo.md Task 4.2.3 for difficulty labeling requirements.
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from src.orchestrator.orchestrator import TOOL_REGISTRY
from src.training.grpo.environment import (
    NutriMindEnv,
    TaskMetadata,
)
from src.training.grpo.reward import reward_v1, RewardBreakdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Success threshold for reward
SUCCESS_THRESHOLD = 0.7

# Difficulty thresholds
EASY_THRESHOLD = 0.7  # >= 70% success rate
HARD_THRESHOLD = 0.2  # < 20% success rate


def load_prompts(path: str) -> List[Dict[str, Any]]:
    """Load prompts from JSONL."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def save_prompts(prompts: List[Dict[str, Any]], path: str) -> None:
    """Save prompts to JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt, ensure_ascii=False) + "\n")


def run_rollouts(
    prompt_data: Dict[str, Any],
    model_generate_fn: Callable,
    num_rollouts: int = 8,
    max_tool_rounds: int = 6,
) -> List[float]:
    """
    Run multiple rollouts for a prompt and return rewards.

    Args:
        prompt_data: Prompt with metadata
        model_generate_fn: Model generation function
        num_rollouts: Number of rollouts (N)
        max_tool_rounds: Max tool rounds per rollout

    Returns:
        List of reward values
    """
    task_meta = TaskMetadata(
        query=prompt_data["query"],
        tier=prompt_data.get("tier", "T1"),
        expected_tools=prompt_data.get("expected_tools", []),
        optimal_steps=prompt_data.get("optimal_steps", 1),
        ground_truth=prompt_data.get("ground_truth"),
        branch_condition=prompt_data.get("branch_condition"),
    )

    rewards = []

    for _ in range(num_rollouts):
        env = NutriMindEnv(
            tool_registry=TOOL_REGISTRY,
            max_tool_rounds=max_tool_rounds,
        )

        messages = env.reset(task_meta.query)
        done = False

        while not done:
            output = model_generate_fn(messages)
            messages, done, info = env.step(output)

        trajectory = env.get_trajectory()
        reward_breakdown = reward_v1(trajectory, task_meta)
        rewards.append(reward_breakdown.total)

    return rewards


def compute_difficulty(rewards: List[float]) -> str:
    """Compute difficulty label from rollout rewards."""
    success_rate = sum(1 for r in rewards if r >= SUCCESS_THRESHOLD) / len(rewards)

    if success_rate >= EASY_THRESHOLD:
        return "easy"
    elif success_rate < HARD_THRESHOLD:
        return "hard"
    else:
        return "medium"


def mock_generate_fn(messages: List[Dict[str, str]]) -> str:
    """Mock generation for testing."""
    import random

    if random.random() < 0.6:
        tools = ["get_food_nutrition", "log_meal", "get_today_summary"]
        tool = random.choice(tools)
        return f'<think>\nProcessing query.\n</think>\n<tool_call>\n{{"name": "{tool}", "arguments": {{}}}}\n</tool_call>'
    else:
        return "Here's the nutrition information you requested."


def main():
    parser = argparse.ArgumentParser(description="Label GRPO prompt difficulty")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input prompts JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output prompts JSONL with difficulty labels",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="SFT model path (uses mock if not provided)",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=8,
        help="Number of rollouts per prompt",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N prompts for quick test",
    )
    args = parser.parse_args()

    # Load prompts
    prompts = load_prompts(args.input)
    logger.info(f"Loaded {len(prompts)} prompts")

    if args.sample:
        import random
        prompts = random.sample(prompts, min(args.sample, len(prompts)))
        logger.info(f"Sampled {len(prompts)} prompts for testing")

    # Setup model
    if args.model:
        logger.info(f"Loading model from {args.model}")
        # In production, load actual model here
        # For now, use mock
        logger.warning("Using mock generation - replace with actual model")
        generate_fn = mock_generate_fn
    else:
        logger.warning("No model specified, using mock generation")
        generate_fn = mock_generate_fn

    # Run labeling
    difficulty_counts = Counter()
    labeled_prompts = []

    for prompt_data in tqdm(prompts, desc="Labeling difficulty"):
        rewards = run_rollouts(
            prompt_data,
            generate_fn,
            num_rollouts=args.num_rollouts,
        )

        difficulty = compute_difficulty(rewards)
        success_rate = sum(1 for r in rewards if r >= SUCCESS_THRESHOLD) / len(rewards)

        prompt_data["difficulty"] = difficulty
        prompt_data["success_rate"] = success_rate
        prompt_data["avg_reward"] = sum(rewards) / len(rewards)

        labeled_prompts.append(prompt_data)
        difficulty_counts[difficulty] += 1

    # Report distribution
    logger.info("\nDifficulty distribution:")
    for diff, count in sorted(difficulty_counts.items()):
        pct = 100 * count / len(labeled_prompts)
        logger.info(f"  {diff}: {count} ({pct:.1f}%)")

    # Check for red flag
    hard_pct = 100 * difficulty_counts.get("hard", 0) / len(labeled_prompts)
    if hard_pct > 60:
        logger.warning(
            f"\n⚠️ RED FLAG: {hard_pct:.1f}% of prompts are 'hard' (>60%).\n"
            "This suggests SFT was insufficient. Consider improving SFT before starting RL."
        )

    # Save
    save_prompts(labeled_prompts, args.output)
    logger.info(f"\nSaved labeled prompts to {args.output}")


if __name__ == "__main__":
    main()
