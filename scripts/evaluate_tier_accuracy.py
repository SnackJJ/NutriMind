"""
Evaluate Tier Label Accuracy via Gemini

For queries where the teacher model skipped tool-use (result = T0-qa)
but the original pool label expected tool-use (T1/T2/T3/etc.),
use Gemini to judge: was the teacher correct to skip tools, or was
the original label correct?

Usage:
    # Evaluate all mismatches
    python scripts/evaluate_tier_accuracy.py

    # Evaluate a random sample of N
    python scripts/evaluate_tier_accuracy.py --sample 100

    # Dry run: just print queries without calling Gemini
    python scripts/evaluate_tier_accuracy.py --sample 10 --dry-run
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from google import genai

# ── Gemini Client ──────────────────────────────────────────────────────────
client = genai.Client(api_key=settings.gemini_api_key)
GEMINI_MODEL = "gemini-2.5-flash"

# ── Paths ──────────────────────────────────────────────────────────────────
POOL_PATH = PROJECT_ROOT / "data/queries/sft_candidate_pool.jsonl"
BATCH3_PATH = PROJECT_ROOT / "data/trajectories/batch3_trajectories_qwen35.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data/trajectories/tier_eval_results.jsonl"

# ── Available tools (for context in the prompt) ────────────────────────────
TOOL_DESCRIPTIONS = """
Available tools for the NutriMind nutrition assistant:
1. get_food_nutrition: Look up nutrition data (calories, protein, fat, etc.) for one or more foods from a USDA database.
2. log_meal: Record a meal to user history. Can only CREATE new entries, CANNOT edit or delete.
3. get_today_summary: Get today's total intake and remaining calorie/macro budget.
4. get_history: Query multi-day nutritional trends (up to 90 days back).
5. retrieve_knowledge: RAG search over a nutrition knowledge base (dietary guidelines, supplements, medical nutrition).
"""

# ── Evaluation Prompt ──────────────────────────────────────────────────────
EVAL_PROMPT_TEMPLATE = """You are an expert evaluator for an AI nutrition assistant called NutriMind.

NutriMind has the following tools available:
{tools}

IMPORTANT SYSTEM CONSTRAINTS:
- NutriMind CANNOT edit or delete meal log entries (log_meal only creates new records)
- NutriMind CANNOT diagnose medical conditions
- NutriMind should refuse queries involving dialysis, post-surgery, cancer treatment, or drug-food interactions

Given a user query, determine whether the AI assistant SHOULD use tools or can correctly answer without tools.

## Query
"{query}"

## Original label from query pool
{original_tier}

## What the teacher model actually did
The teacher model answered this query directly WITHOUT calling any tools.

## Your task
Judge whether the teacher's decision to skip tools was CORRECT or INCORRECT.

Consider:
1. If the query asks about specific food nutrition data → tools NEEDED (get_food_nutrition)
2. If the query asks about today's intake/remaining budget → tools NEEDED (get_today_summary)  
3. If the query asks about historical trends → tools NEEDED (get_history)
4. If the query asks to log/record a meal → tools NEEDED (log_meal), but if info is too vague, asking for clarification first without tools is OK
5. If the query asks to EDIT/DELETE a log entry → tools NOT needed (capability doesn't exist, should inform user directly)
6. If the query is general nutrition advice, meal suggestions, social etiquette → tools NOT strictly needed (can answer from knowledge)
7. If the query could benefit from retrieve_knowledge but the topic is common/general nutrition → borderline (either is acceptable)
8. If the query involves safety boundaries (dialysis, surgery, cancer, etc.) → tools NOT needed (should refuse directly)

Respond with EXACTLY this JSON format, nothing else:
{{
    "verdict": "teacher_correct" | "teacher_wrong" | "borderline",
    "should_use_tool": true | false | "optional",
    "recommended_tool": "<tool_name or null>",
    "reason": "<brief 1-2 sentence explanation>"
}}
"""


def evaluate_single(query: str, original_tier: str) -> dict:
    """Call Gemini to evaluate a single query."""
    prompt = EVAL_PROMPT_TEMPLATE.format(
        tools=TOOL_DESCRIPTIONS,
        query=query,
        original_tier=original_tier,
    )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            text = response.text.strip()
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            result = json.loads(text)
            return {
                "query": query,
                "original_tier": original_tier,
                "verdict": result.get("verdict", "unknown"),
                "should_use_tool": result.get("should_use_tool"),
                "recommended_tool": result.get("recommended_tool"),
                "reason": result.get("reason", ""),
            }
        except (json.JSONDecodeError, Exception) as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return {
                "query": query,
                "original_tier": original_tier,
                "verdict": "error",
                "reason": str(e),
            }


def main():
    parser = argparse.ArgumentParser(description="Evaluate tier label accuracy via Gemini")
    parser.add_argument("--sample", type=int, default=0, help="Random sample size (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Print queries without calling Gemini")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    # Load original tier labels
    pool_tiers = {}
    with open(POOL_PATH) as f:
        for line in f:
            d = json.loads(line)
            pool_tiers[d["query"]] = d["tier"]

    # Find mismatches: result=T0-qa but original != T0
    mismatches = []
    with open(BATCH3_PATH) as f:
        for line in f:
            d = json.loads(line)
            if d["tier"] == "T0-qa":
                orig = pool_tiers.get(d["query"], "")
                if not orig.startswith("T0"):
                    mismatches.append({
                        "query": d["query"],
                        "original_tier": orig,
                    })

    print(f"Found {len(mismatches)} tier mismatches (result=T0-qa, original!=T0)")

    if args.sample > 0:
        random.seed(42)
        mismatches = random.sample(mismatches, min(args.sample, len(mismatches)))
        print(f"Sampled {len(mismatches)} for evaluation")

    if args.dry_run:
        print("\n[DRY RUN] Queries to evaluate:")
        for i, m in enumerate(mismatches):
            print(f"  {i+1}. [{m['original_tier']:>20s}] {m['query']}")
        print(f"\nTotal: {len(mismatches)} queries. No API calls made.")
        return

    # Run evaluation
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = []
    write_lock = threading.Lock()
    completed = 0

    def eval_one(item):
        return evaluate_single(item["query"], item["original_tier"])

    print(f"\nEvaluating {len(mismatches)} queries with {args.workers} workers...")
    print("-" * 60)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(eval_one, m): m for m in mismatches}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                with write_lock:
                    completed += 1
                    verdict = result.get("verdict", "?")
                    symbol = {"teacher_correct": "✓", "teacher_wrong": "✗", "borderline": "~"}.get(verdict, "?")
                    if completed % 10 == 0 or completed == len(mismatches):
                        print(f"  [{completed}/{len(mismatches)}] Last: {symbol} [{result['original_tier']}] {result['query'][:60]}...")
            except Exception as e:
                completed += 1
                print(f"  [ERROR] {e}")

    # Write results
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    verdicts = Counter(r["verdict"] for r in results)
    total = len(results)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total evaluated: {total}")
    print()

    for v in ["teacher_correct", "borderline", "teacher_wrong", "error"]:
        cnt = verdicts.get(v, 0)
        pct = cnt / total * 100 if total > 0 else 0
        symbol = {"teacher_correct": "✓", "teacher_wrong": "✗", "borderline": "~", "error": "!"}.get(v, "?")
        print(f"  {symbol} {v:<20s}: {cnt:4d} ({pct:5.1f}%)")

    print()
    really_wrong = verdicts.get("teacher_wrong", 0)
    correct_plus_border = verdicts.get("teacher_correct", 0) + verdicts.get("borderline", 0)
    print(f"Teacher was right/borderline: {correct_plus_border} ({correct_plus_border/total*100:.1f}%)")
    print(f"Teacher was actually wrong:   {really_wrong} ({really_wrong/total*100:.1f}%)")

    # Breakdown by original tier
    print("\n── Breakdown by original tier ──")
    tier_verdicts = {}
    for r in results:
        t = r["original_tier"]
        if t not in tier_verdicts:
            tier_verdicts[t] = Counter()
        tier_verdicts[t][r["verdict"]] += 1

    print(f"{'Original Tier':<25s} | {'✓ correct':>10s} | {'~ border':>10s} | {'✗ wrong':>10s} | {'total':>6s}")
    print("-" * 75)
    for t in sorted(tier_verdicts.keys()):
        vc = tier_verdicts[t]
        total_t = sum(vc.values())
        print(f"{t:<25s} | {vc.get('teacher_correct',0):>10d} | {vc.get('borderline',0):>10d} | {vc.get('teacher_wrong',0):>10d} | {total_t:>6d}")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
