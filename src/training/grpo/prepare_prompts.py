#!/usr/bin/env python3
"""
Prepare GRPO Prompt Pool with Metadata.

Creates the prompt pool for GRPO training from the query pool.

v1 strategy: Minimal metadata to reduce annotation noise.
- Only outputs: query, tier, difficulty
- No expected_tools, ground_truth, or branch_condition
- Tier classification uses improved rules (not pure keyword matching)
- Human spot-check required: >90% accuracy before training

Usage:
    python prepare_prompts.py --input data/query_pool/grpo_queries.jsonl \
                              --output data/grpo/prompts.jsonl

See phase4_grpo.md Task 4.2.2 for v1/v2 metadata requirements.
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# T4 safety keywords (high precision - these are unambiguous medical terms)
T4_KEYWORDS = frozenset({
    "dialysis", "diabetes medication", "insulin dose", "insulin dosage",
    "chemotherapy", "eating disorder", "anorexia", "bulimia",
    "medication interaction", "drug interaction", "blood thinner",
    "kidney disease", "liver disease", "heart condition",
})

# T0 pure QA patterns (conceptual questions, no specific food/user data)
T0_PATTERNS = [
    r"^what is (a |an |the )?(definition|meaning|concept)",
    r"^explain (what|how|why)",
    r"^tell me about (the )?(concept|science|theory)",
    r"^define ",
    r"what('s| is) the difference between .* and ",
]

# T2 multi-step patterns (logging + query combo)
T2_PATTERNS = [
    r"log .*(and|then|,).*(how|what|check)",  # log X and check Y
    r"(track|record) .*(and|then|,)",  # track X and do Y
    r"(add|log) .* (breakfast|lunch|dinner|snack) .* (total|summary|how)",
]

# T1 single tool patterns (just query OR just log, not both)
T1_PATTERNS = [
    r"^how (much|many) (protein|calories|carbs|fat|fiber)",
    r"^(what('s| is|'re| are) the )?(nutrition|calories|protein|macros)",
    r"calories in ",
    r"protein in ",
]


def infer_tier(query: str) -> str:
    """
    Infer tier from query text using improved rules.

    Priority order: T4 > T2 > T1 > T0 > default T1

    Note: T3 (conditional) is difficult to detect with rules alone.
    Queries like "if I eat X" may be T1 (simple nutrition query) or T3 (conditional).
    For v1, we conservatively classify ambiguous "if" queries as T1.
    T3 classification should be human-verified for v2.
    """
    query_lower = query.lower().strip()

    # T4: Safety boundary (high precision keywords)
    for keyword in T4_KEYWORDS:
        if keyword in query_lower:
            return "T4"

    # T2: Multi-step (logging + query combo)
    for pattern in T2_PATTERNS:
        if re.search(pattern, query_lower):
            return "T2"

    # T1: Single tool query
    for pattern in T1_PATTERNS:
        if re.search(pattern, query_lower):
            return "T1"

    # T0: Pure conceptual QA
    for pattern in T0_PATTERNS:
        if re.search(pattern, query_lower):
            return "T0-qa"

    # Default to T1 (most common case)
    # Note: "if" keyword alone does NOT imply T3 - "if I eat chicken" is usually T1
    return "T1"


def generate_env_state() -> Dict[str, Any]:
    import uuid
    import random
    
    # Generate varied user profiles
    goals = [("maintain", 2000), ("lose", 1500), ("gain", 2800)]
    idx = random.randint(0, 2)
    goal, tdee = goals[idx]
    
    return {
        "user_id": f"grpo_user_{uuid.uuid4().hex[:8]}",
        "user_profile": {"tdee_kcal": tdee, "goal": goal},
        "user_goals": {
            "calories": tdee,
            "protein": int(tdee * 0.3 / 4),
            "fat": int(tdee * 0.25 / 9),
            "carbs": int(tdee * 0.45 / 4),
            "fiber": 25 # Default RDI
        },
        "meals_today": [], # Start empty, allow model to log or query
        "meal_history": [
            {
                "date": "yesterday", 
                "calories": tdee - random.randint(-200, 200),
                "protein_g": int(tdee * 0.3 / 4),
                "fat_g": int(tdee * 0.25 / 9),
                "carbs_g": int(tdee * 0.45 / 4),
                "fiber_g": 25
            }
        ]
    }

def process_query_v1(query: str, existing_metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Process a single query for v1 (minimal metadata + env_state).

    v1 needs: query, tier, difficulty, env_state
    No expected_tools, ground_truth, or branch_condition.
    """
    # Use existing tier if provided, otherwise infer
    tier = None
    if existing_metadata:
        tier = existing_metadata.get("tier")
    if not tier:
        tier = infer_tier(query)

    # Difficulty will be set later by SFT model rollout success rate
    difficulty = "medium"
    if existing_metadata:
        difficulty = existing_metadata.get("difficulty", "medium")

    return {
        "query": query,
        "tier": tier,
        "difficulty": difficulty,
        "env_state": generate_env_state()
    }


# =============================================================================
# v2 functions (kept for future use, not used in v1)
# =============================================================================

# Tool mappings for v2 metadata inference
TIER_TOOL_MAPPING_V2 = {
    "T0-qa": [],
    "T1": ["get_food_nutrition"],
    "T2": ["get_food_nutrition", "log_meal"],
    "T3": ["get_today_summary", "retrieve_knowledge"],
    "T4": [],
    "error-recovery": ["get_food_nutrition"],
}

QUERY_TOOL_PATTERNS_V2 = [
    (r"how much (protein|calories|carbs|fat)", ["get_food_nutrition"]),
    (r"nutrition (in|of|for)", ["get_food_nutrition"]),
    (r"log (my )?(breakfast|lunch|dinner|snack|meal)", ["get_food_nutrition", "log_meal"]),
    (r"what('s| is) my .*(today|daily)", ["get_today_summary"]),
    (r"am i (over|under) (my )?(budget|limit|goal)", ["get_today_summary"]),
    (r"suggest|recommend", ["get_today_summary", "retrieve_knowledge"]),
    (r"history|past (week|month|days)", ["get_history"]),
]


def process_query_v2(query: str, existing_metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Process a single query for v2 (full metadata).

    WARNING: expected_tools and branch_condition should be human-verified.
    Auto-generated values have high noise and should not be trusted for training.
    """
    # Start with v1 base
    result = process_query_v1(query, existing_metadata)
    tier = result["tier"]
    query_lower = query.lower()

    # Infer expected_tools (WARNING: noisy, needs human verification)
    expected_tools = None
    if existing_metadata:
        expected_tools = existing_metadata.get("expected_tools")
    if not expected_tools:
        for pattern, tools in QUERY_TOOL_PATTERNS_V2:
            if re.search(pattern, query_lower):
                expected_tools = tools
                break
        if not expected_tools:
            expected_tools = TIER_TOOL_MAPPING_V2.get(tier, [])

    result["expected_tools"] = expected_tools
    result["optimal_steps"] = max(1, len(expected_tools))
    result["ground_truth"] = existing_metadata.get("ground_truth") if existing_metadata else None
    result["branch_condition"] = existing_metadata.get("branch_condition") if existing_metadata else None

    return result


# Alias for backward compatibility
def process_query(query: str, existing_metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Process query - defaults to v1 (minimal metadata)."""
    return process_query_v1(query, existing_metadata)


def validate_prompt_v1(prompt: Dict[str, Any]) -> List[str]:
    """Validate v1 prompt metadata (minimal requirements)."""
    errors = []

    if not prompt.get("query"):
        errors.append("Missing query")

    tier = prompt.get("tier", "")
    valid_tiers = {"T0-qa", "T1", "T2", "T3", "T4", "error-recovery"}
    if tier not in valid_tiers:
        errors.append(f"Invalid tier: {tier}")

    # v1 does NOT require expected_tools or branch_condition
    # Those are v2 requirements

    return errors


def validate_prompt_v2(prompt: Dict[str, Any]) -> List[str]:
    """Validate v2 prompt metadata (full requirements)."""
    errors = validate_prompt_v1(prompt)

    tier = prompt.get("tier", "")

    if tier in ["T1", "T2", "T3"] and not prompt.get("expected_tools"):
        errors.append(f"Tier {tier} requires expected_tools (v2)")

    if tier == "T3" and not prompt.get("branch_condition"):
        errors.append("T3 requires branch_condition (v2)")

    return errors


# Alias for backward compatibility
def validate_prompt(prompt: Dict[str, Any]) -> List[str]:
    """Validate prompt - defaults to v1 (minimal requirements)."""
    return validate_prompt_v1(prompt)


def load_queries(path: str) -> List[Dict[str, Any]]:
    """Load queries from JSONL or text file."""
    queries = []
    path = Path(path)

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if isinstance(data, str):
                    queries.append({"query": data})
                else:
                    queries.append(data)
    elif path.suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append({"query": line})
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return queries


def save_prompts(prompts: List[Dict[str, Any]], path: str) -> None:
    """Save prompts to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt, ensure_ascii=False) + "\n")


def split_train_eval(
    prompts: List[Dict[str, Any]],
    eval_ratio: float = 0.04,
    seed: int = 42,
) -> tuple[List[Dict], List[Dict]]:
    """Split prompts into train and eval sets."""
    random.seed(seed)
    prompts_copy = prompts.copy()
    random.shuffle(prompts_copy)

    eval_size = max(100, int(len(prompts_copy) * eval_ratio))
    eval_prompts = prompts_copy[:eval_size]
    train_prompts = prompts_copy[eval_size:]

    return train_prompts, eval_prompts


def main():
    parser = argparse.ArgumentParser(description="Prepare GRPO prompt pool")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input query file (JSONL or TXT)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/grpo/prompts.jsonl",
        help="Output prompt file",
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default="data/grpo/eval_prompts.jsonl",
        help="Evaluation set output file",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.04,
        help="Ratio of prompts for evaluation (default 4%)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate prompts and report errors",
    )
    args = parser.parse_args()

    # Load queries
    logger.info(f"Loading queries from {args.input}")
    queries = load_queries(args.input)
    logger.info(f"Loaded {len(queries)} queries")

    # Process each query
    prompts = []
    tier_counts = {}
    errors = []

    for query_data in queries:
        query = query_data.get("query", query_data) if isinstance(query_data, dict) else query_data
        existing_meta = query_data if isinstance(query_data, dict) else None

        prompt = process_query(query, existing_meta)
        prompts.append(prompt)

        # Count tiers
        tier = prompt["tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        # Validate
        if args.validate:
            prompt_errors = validate_prompt(prompt)
            if prompt_errors:
                errors.append((query[:50], prompt_errors))

    # Report tier distribution
    logger.info("\nTier distribution:")
    for tier, count in sorted(tier_counts.items()):
        pct = 100 * count / len(prompts)
        logger.info(f"  {tier}: {count} ({pct:.1f}%)")

    # Report validation errors
    if args.validate and errors:
        logger.warning(f"\n{len(errors)} prompts have validation errors:")
        for query, errs in errors[:10]:
            logger.warning(f"  '{query}...': {errs}")
        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors) - 10} more")

    # Split train/eval
    train_prompts, eval_prompts = split_train_eval(prompts, args.eval_ratio)

    # Save
    save_prompts(train_prompts, args.output)
    save_prompts(eval_prompts, args.eval_output)

    logger.info(f"\nSaved {len(train_prompts)} training prompts to {args.output}")
    logger.info(f"Saved {len(eval_prompts)} evaluation prompts to {args.eval_output}")


if __name__ == "__main__":
    main()
