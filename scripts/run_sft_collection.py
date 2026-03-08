"""
SFT Trajectory Collection Pipeline
====================================
Steps:
  1. collect  — teacher model generates trajectories from sft_candidate_pool
  2. normalize — convert to canonical <think>/<tool_call>/<tool_response> format
  3. rule     — filter by structural rules (tool names, language, tier constraints)
  4. semantic — LLM-as-judge quality filter (qwen-max)

Usage:
  # Full pipeline
  python scripts/run_sft_collection.py

  # Collect only (e.g. resume large run)
  python scripts/run_sft_collection.py --steps collect

  # Post-process already-collected trajectories
  python scripts/run_sft_collection.py --steps normalize,rule,semantic

  # Limit collection for testing
  python scripts/run_sft_collection.py --steps collect --limit 20 --workers 3
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Resolve project root so imports work when called from any directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.sft.collect_trajectories import batch_collect
from src.training.sft.normalize import normalize_file
from src.training.sft.validate_rules import validate_file as rule_validate_file
from src.training.sft.validate_semantic import validate_semantic_file

# ── File paths ────────────────────────────────────────────────────────────────
POOL_PATH  = PROJECT_ROOT / "data/queries/sft_candidate_pool.jsonl"
RAW_PATH   = PROJECT_ROOT / "data/trajectories/real_trajectories.jsonl"
NORM_PATH  = PROJECT_ROOT / "data/trajectories/normalized_trajectories.jsonl"
RULE_PATH  = PROJECT_ROOT / "data/trajectories/rule_validated_trajectories.jsonl"
FINAL_PATH = PROJECT_ROOT / "data/trajectories/validated_trajectories.jsonl"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def tier_stats(path: Path) -> str:
    if not path.exists():
        return ""
    tiers = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    tiers.append(json.loads(line).get("tier", "?"))
                except Exception:
                    pass
    c = Counter(tiers)
    return "  " + "  ".join(f"{t}:{n}" for t, n in sorted(c.items()))


def print_status():
    print("\n── Current pipeline state ─────────────────────────────────────")
    for label, path in [
        ("Pool    ", POOL_PATH),
        ("Raw     ", RAW_PATH),
        ("Norm    ", NORM_PATH),
        ("Rule ✓  ", RULE_PATH),
        ("Final ✓ ", FINAL_PATH),
    ]:
        n = count_lines(path)
        extra = tier_stats(path) if n > 0 and path != POOL_PATH else ""
        status = f"{n:5d} lines" if path.exists() else "  [not found]"
        print(f"  {label}  {path.name:<45s}  {status}{extra}")
    print()


# ── Steps ─────────────────────────────────────────────────────────────────────

def step_collect(limit: int, workers: int):
    print("━━ Step 1/4: Collect ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if not POOL_PATH.exists():
        print(f"[ERROR] Query pool not found: {POOL_PATH}")
        sys.exit(1)

    with open(POOL_PATH, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    total_pool = len(queries)
    if limit > 0:
        queries = queries[:limit]
        print(f"  Pool: {total_pool} queries  (limited to {limit})")
    else:
        print(f"  Pool: {total_pool} queries")

    already = count_lines(RAW_PATH)
    if already:
        print(f"  Resume: {already} already collected, will skip duplicates")

    batch_collect(queries, str(RAW_PATH), workers=workers)
    print(f"  Raw total: {count_lines(RAW_PATH)}{tier_stats(RAW_PATH)}\n")


def step_normalize():
    print("━━ Step 2/4: Normalize ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if not RAW_PATH.exists():
        print(f"[ERROR] {RAW_PATH} not found, run collect first")
        sys.exit(1)
    normalize_file(str(RAW_PATH), str(NORM_PATH))
    print(f"  Normalized total: {count_lines(NORM_PATH)}\n")


def step_rule():
    print("━━ Step 3/4: Rule Validation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if not NORM_PATH.exists():
        print(f"[ERROR] {NORM_PATH} not found, run normalize first")
        sys.exit(1)
    rule_validate_file(str(NORM_PATH), str(RULE_PATH))
    print(f"  Rule-validated total: {count_lines(RULE_PATH)}{tier_stats(RULE_PATH)}\n")


def step_semantic():
    print("━━ Step 4/4: Semantic Validation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if not RULE_PATH.exists():
        print(f"[ERROR] {RULE_PATH} not found, run rule validation first")
        sys.exit(1)
    validate_semantic_file(str(RULE_PATH), str(FINAL_PATH))
    print(f"  Final validated total: {count_lines(FINAL_PATH)}{tier_stats(FINAL_PATH)}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SFT trajectory collection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="collect,normalize,rule,semantic",
        help="Comma-separated steps to run: collect,normalize,rule,semantic (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max queries to collect (0 = all, default: 0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Parallel collection threads (default: 5)",
    )
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",")]
    valid_steps = {"collect", "normalize", "rule", "semantic", "status"}
    unknown = set(steps) - valid_steps
    if unknown:
        print(f"[ERROR] Unknown steps: {unknown}. Valid: {valid_steps - {'status'}}")
        sys.exit(1)

    print_status()

    if steps == ["status"]:
        return

    if "collect" in steps:
        step_collect(args.limit, args.workers)
    if "normalize" in steps:
        step_normalize()
    if "rule" in steps:
        step_rule()
    if "semantic" in steps:
        step_semantic()

    print("━━ Done ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print_status()


if __name__ == "__main__":
    main()
