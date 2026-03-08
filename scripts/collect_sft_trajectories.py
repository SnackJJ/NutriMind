#!/usr/bin/env python3
"""Collect SFT trajectories from sft_candidate_pool.jsonl with stratified sampling.

Usage:
    # Dry-run: show sampling plan only
    python scripts/collect_sft_trajectories.py --dry-run

    # Full collection (1800 trajectories, 3 workers)
    python scripts/collect_sft_trajectories.py --total 1800 --workers 3

    # Resume interrupted collection
    python scripts/collect_sft_trajectories.py --total 1800 --workers 3
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def map_tier(fine_tier: str) -> str:
    """Map fine-grained sub-tier to coarse tier_hint for collect_trajectories."""
    if fine_tier in ("T0-qa", "error_recovery"):
        return fine_tier
    # T1-basic -> T1, T2-fuzzy -> T2, T3-allergy -> T3, T4-reject -> T4
    return fine_tier.split("-")[0]


def stratified_sample(entries: list[dict], total: int, seed: int = 42) -> list[dict]:
    """Proportional stratified sampling across sub-tiers."""
    rng = random.Random(seed)

    # Group by sub-tier
    by_tier = defaultdict(list)
    for e in entries:
        by_tier[e["tier"]].append(e)

    pool_size = len(entries)
    ratio = total / pool_size

    sampled = []
    for tier in sorted(by_tier):
        items = by_tier[tier]
        n = max(1, round(len(items) * ratio))
        n = min(n, len(items))  # Can't sample more than available
        rng.shuffle(items)
        sampled.extend(items[:n])

    # Fine-tune to hit exact total (rounding may over/undershoot)
    if len(sampled) > total:
        rng.shuffle(sampled)
        sampled = sampled[:total]
    elif len(sampled) < total:
        # Add more from unsampled entries
        sampled_queries = {e["query"] for e in sampled}
        remaining = [e for e in entries if e["query"] not in sampled_queries]
        rng.shuffle(remaining)
        sampled.extend(remaining[: total - len(sampled)])

    return sampled


def main():
    parser = argparse.ArgumentParser(description="Collect SFT trajectories from query pool")
    parser.add_argument(
        "--pool",
        type=str,
        default="data/queries/sft_candidate_pool.jsonl",
        help="Path to SFT candidate pool (JSONL)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/trajectories/sft_trajectories.jsonl",
        help="Output trajectory file (JSONL)",
    )
    parser.add_argument("--total", type=int, default=1800, help="Total trajectories to collect")
    parser.add_argument("--workers", type=int, default=3, help="Parallel collection workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--dry-run", action="store_true", help="Show sampling plan without collecting")
    args = parser.parse_args()

    # Load pool
    pool_path = Path(args.pool)
    if not pool_path.exists():
        print(f"[ERROR] Pool file not found: {pool_path}")
        return

    with open(pool_path) as f:
        pool = [json.loads(line) for line in f if line.strip()]

    print(f"[INFO] Loaded {len(pool)} queries from {pool_path}")

    # Sample
    sampled = stratified_sample(pool, args.total, seed=args.seed)

    # Map to coarse tier_hint
    queries = []
    for entry in sampled:
        queries.append(
            {
                "query": entry["query"],
                "tier_hint": map_tier(entry["tier"]),
                "sub_tier": entry["tier"],
                "source": entry.get("source", ""),
            }
        )

    # Print sampling summary
    from collections import Counter

    coarse_counts = Counter(q["tier_hint"] for q in queries)
    sub_counts = Counter(q["sub_tier"] for q in queries)

    print(f"\n=== Sampling Plan: {len(queries)} queries ===")
    print(f"\nCoarse tier distribution:")
    for tier in ["T0-qa", "T1", "T2", "T3", "T4", "error_recovery"]:
        cnt = coarse_counts.get(tier, 0)
        print(f"  {tier:20s}: {cnt:4d}")

    print(f"\nSub-tier distribution:")
    for tier, cnt in sorted(sub_counts.items()):
        print(f"  {tier:25s}: {cnt:4d}")

    if args.dry_run:
        print("\n[DRY-RUN] No collection performed.")
        return

    # Check for existing output (resume support)
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for line in f if line.strip())
        print(f"\n[INFO] Output file exists with {existing} trajectories (resume mode)")

    # Import and run collection
    from src.training.sft.collect_trajectories import batch_collect

    print(f"\n[INFO] Starting collection: {len(queries)} queries, {args.workers} workers")
    print(f"[INFO] Output: {args.output}")
    batch_collect(queries, args.output, workers=args.workers)


if __name__ == "__main__":
    main()
