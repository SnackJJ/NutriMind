#!/usr/bin/env python3
"""
Prepare GRPO prompts for TRL GRPOTrainer.

Converts NutriMind's JSONL format to HuggingFace Dataset format.

Usage:
    python scripts/prepare_trl_data.py
    python scripts/prepare_trl_data.py --dry_run
    python scripts/prepare_trl_data.py --input data/grpo/prompts.jsonl
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.training.grpo.environment import NutriMindEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = NutriMindEnv.SYSTEM_PROMPT


def get_optimal_steps(tier: str) -> int:
    """Map tier to expected optimal tool call count."""
    exact = {"T0-qa": 0, "T4": 0, "error_recovery": 2}
    if tier in exact:
        return exact[tier]
    prefix_map = {"T0": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 0}
    for prefix, steps in prefix_map.items():
        if tier.startswith(prefix):
            return steps
    return 1


def convert_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single JSONL entry to TRL-compatible format."""
    query = entry["query"]
    tier = entry.get("tier", "T1")
    difficulty = entry.get("difficulty", "medium")
    env_state = entry.get("env_state", {})
    branch_condition = entry.get("branch_condition")

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    return {
        "prompt": prompt,
        "tier": tier,
        "difficulty": difficulty,
        "optimal_steps": get_optimal_steps(tier),
        "env_state": json.dumps(env_state, ensure_ascii=False),
        "branch_condition": json.dumps(branch_condition) if branch_condition else "",
        "query": query,
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Prepare GRPO data for TRL")
    parser.add_argument(
        "--input", type=str, default="data/grpo/grpo_prompts.jsonl",
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/grpo",
        help="Output directory for HF datasets",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    entries = load_jsonl(input_path)
    logger.info(f"Loaded {len(entries)} entries from {input_path}")

    # Shuffle and split
    random.seed(args.seed)
    indices = list(range(len(entries)))
    random.shuffle(indices)

    split_idx = int(len(indices) * args.train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_data = [convert_entry(entries[i]) for i in train_indices]
    val_data = [convert_entry(entries[i]) for i in val_indices]

    # Tier distribution
    tier_counts: Dict[str, int] = {}
    for entry in train_data + val_data:
        tier_counts[entry["tier"]] = tier_counts.get(entry["tier"], 0) + 1
    logger.info("Tier distribution:")
    for tier, count in sorted(tier_counts.items()):
        logger.info(f"  {tier}: {count}")

    if args.dry_run:
        logger.info("\n=== Sample train entry ===")
        sample = train_data[0]
        print(json.dumps(sample, indent=2, ensure_ascii=False))
        logger.info("\nDry run complete. No files written.")
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    train_path = output_dir / "trl_train"
    val_path = output_dir / "trl_val"

    train_ds.save_to_disk(str(train_path))
    val_ds.save_to_disk(str(val_path))

    logger.info(f"Saved {len(train_data)} train samples to {train_path}")
    logger.info(f"Saved {len(val_data)} val samples to {val_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
