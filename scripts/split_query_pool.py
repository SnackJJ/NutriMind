"""
Split the expanded query pool into SFT Candidate Pool and GRPO Prompt Pool.

Performs a stratified random split by tier (preserving sub-tier proportions)
so that both pools have proportional tier distribution.

Usage:
    python scripts/split_query_pool.py                                  # defaults
    python scripts/split_query_pool.py --sft-ratio 0.45 --seed 2026    # custom
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def major_tier(tier: str) -> str:
    """Map sub-tier labels to their major tier for reporting.
    e.g. 'T1-basic' -> 'T1', 'T2-fuzzy' -> 'T2', 'T0-qa' -> 'T0'
    """
    if tier.startswith("T0"):
        return "T0"
    if tier.startswith("T1"):
        return "T1"
    if tier.startswith("T2"):
        return "T2"
    if tier.startswith("T3"):
        return "T3"
    if tier.startswith("T4"):
        return "T4"
    if tier.startswith("error"):
        return "error_recovery"
    return tier


def stratified_split(
    records: list[dict],
    sft_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split records into SFT and GRPO pools, stratified by the 'tier' field."""
    rng = random.Random(seed)

    # Group by the full sub-tier label so that each sub-tier is split independently
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["tier"]].append(r)

    sft_pool: list[dict] = []
    grpo_pool: list[dict] = []

    for tier_key in sorted(groups.keys()):
        items = groups[tier_key]
        rng.shuffle(items)
        split_idx = max(1, round(len(items) * sft_ratio))  # at least 1 per tier
        sft_pool.extend(items[:split_idx])
        grpo_pool.extend(items[split_idx:])

    # Final shuffle within each pool
    rng.shuffle(sft_pool)
    rng.shuffle(grpo_pool)

    return sft_pool, grpo_pool


def print_report(
    sft_pool: list[dict],
    grpo_pool: list[dict],
    total: int,
):
    """Print a tier-distribution report for both pools."""
    sft_major: dict[str, int] = defaultdict(int)
    grpo_major: dict[str, int] = defaultdict(int)
    for r in sft_pool:
        sft_major[major_tier(r["tier"])] += 1
    for r in grpo_pool:
        grpo_major[major_tier(r["tier"])] += 1

    all_tiers = sorted(set(list(sft_major.keys()) + list(grpo_major.keys())))

    print("\n" + "=" * 60)
    print(f"{'Tier':<18} {'SFT':>8} {'GRPO':>8} {'Total':>8}")
    print("-" * 60)
    for t in all_tiers:
        s = sft_major.get(t, 0)
        g = grpo_major.get(t, 0)
        print(f"{t:<18} {s:>8} {g:>8} {s + g:>8}")
    print("-" * 60)
    print(f"{'TOTAL':<18} {len(sft_pool):>8} {len(grpo_pool):>8} {total:>8}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Stratified split of query pool.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/queries/expanded_query_pool.jsonl",
        help="Path to the expanded query pool (.jsonl)",
    )
    parser.add_argument(
        "--sft-output",
        type=str,
        default="data/queries/sft_candidate_pool.jsonl",
        help="Output path for SFT Candidate Pool",
    )
    parser.add_argument(
        "--grpo-output",
        type=str,
        default="data/queries/grpo_prompt_pool.jsonl",
        help="Output path for GRPO Prompt Pool",
    )
    parser.add_argument(
        "--sft-ratio",
        type=float,
        default=0.43,
        help="Fraction of each tier allocated to SFT (default: 0.43 ≈ 2500/5800)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # ── Load ────────────────────────────────────────────────────
    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} queries from {args.input}")

    # ── Split ───────────────────────────────────────────────────
    sft_pool, grpo_pool = stratified_split(records, args.sft_ratio, args.seed)

    # ── Save ────────────────────────────────────────────────────
    save_jsonl(sft_pool, args.sft_output)
    save_jsonl(grpo_pool, args.grpo_output)
    print(f"\n✅ SFT Candidate Pool : {len(sft_pool):>5} queries -> {args.sft_output}")
    print(f"✅ GRPO Prompt Pool   : {len(grpo_pool):>5} queries -> {args.grpo_output}")

    # ── Report ──────────────────────────────────────────────────
    print_report(sft_pool, grpo_pool, len(records))


if __name__ == "__main__":
    main()
