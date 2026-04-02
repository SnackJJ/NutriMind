#!/usr/bin/env python3
"""
Prepare GRPO prompts for veRL training.

Converts NutriMind's JSONL format to veRL-compatible parquet format.

Usage:
    python scripts/prepare_verl_data.py
    python scripts/prepare_verl_data.py --dry_run
    python scripts/prepare_verl_data.py --input data/grpo/prompts.jsonl --output data/grpo/
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# System prompt from src/orchestrator/orchestrator.py
SYSTEM_PROMPT = """You are NutriMind, a specialized AI nutrition assistant.

## CORE PRINCIPLES
1. **Tool-driven accuracy**: When nutritional data (calories, history, progress) can be obtained via tools, ALWAYS use the appropriate tool.
2. **Step-by-step reasoning**: Always use <think> tags to analyze the user's request and calculate any required values (e.g., body weight x target protein ratio) before taking action.
3. **Format consistency**: You MUST use the following XML tags for reasoning and tool calls:
   <think> Your internal analysis and calculations </think>
   <tool_call> {"name": "tool_name", "arguments": {"arg": "val"}} </tool_call>

## LANGUAGE RULES
- All responses must be in English.
- If the user uses Chinese, acknowledge it but provide the final answer in English.

## BEHAVIOR GUIDELINES
- **Analyze before acting**: Determine if tools are needed. Respond directly for casual conversation.
- **Synthesize results**: When using tools, summarize the data in your own words. Do not copy-paste raw tool output.
- **Safety**: If the user mentions chronic diseases (cancer, organ transplant, etc.), respond: "Your situation involves complex medical nutrition management that exceeds my safe service boundary. Please consult your physician or a registered dietitian."

## TOOL USAGE NOTES

### get_food_nutrition
Look up nutrition data for one or more foods from the USDA database.
Arguments: `foods` (list of objects with `food_name` and `amount_grams`)
Example: `{"foods": [{"food_name": "chicken breast", "amount_grams": 100}]}`

### log_meal
Record a food entry to the user's history. Only use when explicitly asked.
Arguments: `meal_type` (str: "breakfast"/"lunch"/"dinner"/"snack"), `foods` (list of objects with `food_name` and `amount_grams`)

### get_today_summary
Check today's totals and progress against current goals. No arguments needed.

### get_history
Analyze multi-day trends and goal adherence.
Arguments: `days` (int), `compare_to_goal` (bool, use true for progress reports)

### retrieve_knowledge
Search nutrition knowledge base for dietary guidelines and principles.
Arguments: `query` (str), `mode` (str: "hybrid"/"keyword"/"semantic"), `top_k` (int)
Do NOT use for specific food calories — use `get_food_nutrition` instead.

### set_goal
Set or update a daily target (calories, protein, carbs, fat).
Arguments: `nutrient` (str), `value` (float), `goal_type` (str: "lose"/"maintain"/"gain")
When user provides weight and ratio (e.g., 80kg, 2g/kg), calculate in <think> FIRST."""


def get_optimal_steps(tier: str) -> int:
    """Map tier to expected optimal tool call count."""
    # Exact tier matches
    exact_mapping = {
        "T0-qa": 0,
        "T4": 0,
        "error_recovery": 2,
    }

    if tier in exact_mapping:
        return exact_mapping[tier]

    # Prefix-based mapping
    if tier.startswith("T0"):
        return 0
    elif tier.startswith("T1"):
        return 1
    elif tier.startswith("T2"):
        return 2
    elif tier.startswith("T3"):
        return 3
    elif tier.startswith("T4"):
        return 0
    else:
        return 1  # Default


def convert_entry(entry: Dict[str, Any], index: int, split: str) -> Dict[str, Any]:
    """
    Convert a single JSONL entry to veRL format.

    Input format (NutriMind):
        {
            "query": "...",
            "tier": "T2-...",
            "difficulty": "medium",
            "env_state": {...}
        }

    Output format (veRL):
        {
            "data_source": "nutrimind",
            "prompt": [{"role": "system", ...}, {"role": "user", ...}],
            "ability": "T2",
            "reward_model": {"style": "rule", "ground_truth": {...}},
            "extra_info": {...}
        }
    """
    query = entry["query"]
    tier = entry.get("tier", "T1")
    difficulty = entry.get("difficulty", "medium")
    env_state = entry.get("env_state", {})

    # Build chat-style prompt
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    # Build ground truth for rule-based reward
    optimal_steps = get_optimal_steps(tier)
    ground_truth = {
        "tier": tier,
        "difficulty": difficulty,
        "optimal_steps": optimal_steps,
    }

    # Build extra_info with interaction kwargs
    extra_info = {
        "split": split,
        "index": index,
        "interaction_kwargs": {
            "name": "nutrimind",
            "env_state": env_state,
            "tier": tier,
            "difficulty": difficulty,
        },
    }

    return {
        "data_source": "nutrimind",
        "prompt": prompt,
        "ability": tier.split("-")[0] if "-" in tier else tier,  # T2-ambiguous -> T2
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": extra_info,
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


def save_parquet(data: List[Dict[str, Any]], path: Path) -> None:
    """
    Save data to parquet format for veRL.

    veRL expects these columns:
    - data_source: string
    - prompt: list of dicts (JSON serialized)
    - ability: string
    - reward_model: dict (JSON serialized)
    - extra_info: dict (JSON serialized)
    """
    # Serialize complex fields to JSON strings
    rows = []
    for entry in data:
        rows.append({
            "data_source": entry["data_source"],
            # New version of veRL expects prompt as a list of dicts in the parquet
            # instead of a JSON serialized string to work with jinja2 chat templates.
            "prompt": entry["prompt"],
            "ability": entry["ability"],
            "reward_model": json.dumps(entry["reward_model"], ensure_ascii=False),
            "extra_info": json.dumps(entry["extra_info"], ensure_ascii=False),
        })

    # Create PyArrow table
    table = pa.Table.from_pylist(rows)

    # Write to parquet
    pq.write_table(table, path)
    logger.info(f"Saved {len(rows)} entries to {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare GRPO data for veRL")
    parser.add_argument(
        "--input",
        type=str,
        default="data/grpo/grpo_prompts.jsonl",
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/grpo",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (rest is validation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print sample output without writing files",
    )
    args = parser.parse_args()

    # Load input data
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

    logger.info(f"Split: {len(train_indices)} train, {len(val_indices)} val")

    # Convert to veRL format
    train_data = [
        convert_entry(entries[i], idx, "train")
        for idx, i in enumerate(train_indices)
    ]
    val_data = [
        convert_entry(entries[i], idx, "val")
        for idx, i in enumerate(val_indices)
    ]

    # Tier distribution
    tier_counts = {}
    for entry in train_data + val_data:
        tier = entry["ability"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    logger.info("Tier distribution:")
    for tier, count in sorted(tier_counts.items()):
        logger.info(f"  {tier}: {count}")

    if args.dry_run:
        # Print sample output
        logger.info("\n=== Sample train entry ===")
        sample = train_data[0]
        print(json.dumps(sample, indent=2, ensure_ascii=False))

        logger.info("\n=== Prompt format ===")
        prompt = json.loads(json.dumps(sample["prompt"]))
        for msg in prompt:
            print(f"[{msg['role']}] {msg['content'][:100]}...")

        logger.info("\nDry run complete. No files written.")
        return 0

    # Save parquet files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "verl_train.parquet"
    val_path = output_dir / "verl_val.parquet"

    save_parquet(train_data, train_path)
    save_parquet(val_data, val_path)

    logger.info(f"\nCreated:")
    logger.info(f"  {train_path} ({len(train_data)} samples)")
    logger.info(f"  {val_path} ({len(val_data)} samples)")

    return 0


if __name__ == "__main__":
    exit(main())
