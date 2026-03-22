"""
A/B test for v1 vs v2 system prompt.

Compares T0-qa rate between original prompt and "Teaching Demonstration" prompt.
Uses GRPO pool to avoid wasting SFT candidate data.

Usage:
    python -m src.training.sft.test_prompt_ab --sample 100 --workers 3
"""
import json
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import settings

# Import both prompt versions
from src.training.sft.collect_trajectories import (
    SYSTEM_PROMPT as PROMPT_V1,
    TOOLS,
    infer_tier,
    api_response_to_sft_text,
    format_tool_response,
    execute_tool,
    truncate_tool_result,
    MAX_TURNS,
)
from src.training.sft.collect_trajectories_v2 import SYSTEM_PROMPT as PROMPT_V2

TEACHER_MODEL = "qwen3.5-397b-a17b"

client = OpenAI(
    api_key=settings.qwen_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def call_api_with_retry(**kwargs):
    return client.chat.completions.create(**kwargs)


def simulate_one(query: str, system_prompt: str) -> dict:
    """Run one trajectory with given system prompt, return inferred tier."""
    api_messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": query}
    ]
    sft_messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": query}
    ]

    for turn in range(MAX_TURNS):
        response = call_api_with_retry(
            model=TEACHER_MODEL,
            messages=api_messages,
            tools=TOOLS,
            extra_body={"enable_thinking": True},
        )

        message = response.choices[0].message
        reasoning_content = getattr(message, "reasoning_content", None)
        tool_calls = message.tool_calls
        text_content = message.content

        sft_content = api_response_to_sft_text(reasoning_content, tool_calls, text_content)

        if not sft_content.strip():
            api_messages.append({"role": "user", "content": "Please provide your answer."})
            continue

        sft_messages.append({"role": "assistant", "content": sft_content})

        if not tool_calls:
            api_messages.append({"role": "assistant", "content": text_content or ""})
            break

        # Process tool call
        tc = tool_calls[0]
        func = tc.function
        name = func.name
        try:
            args = json.loads(func.arguments) if isinstance(func.arguments, str) else func.arguments
        except json.JSONDecodeError:
            args = {}

        api_messages.append({
            "role": "assistant",
            "content": text_content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": name, "arguments": func.arguments}
                }
            ]
        })

        real_result = execute_tool(name, args)
        truncated_result = truncate_tool_result(real_result)
        result_json = json.dumps(truncated_result, ensure_ascii=False, default=str)

        api_messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result_json
        })

        sft_messages.append({
            "role": "user",
            "content": format_tool_response(truncated_result, indent=None)
        })

    tier = infer_tier(sft_messages)
    return {"query": query, "tier": tier}


def stratified_sample(pool_path: str, n: int, seed: int = 42) -> list:
    """Sample n queries with stratified sampling by major tier category."""
    random.seed(seed)

    with open(pool_path) as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Group by major tier (T0, T1, T2, T3, T4, error_recovery)
    groups = {}
    for item in data:
        tier = item.get("tier", "unknown")
        major = tier.split("-")[0] if "-" in tier else tier
        groups.setdefault(major, []).append(item)

    # Sample proportionally
    result = []
    total = len(data)
    for major, items in groups.items():
        k = max(1, round(n * len(items) / total))
        sampled = random.sample(items, min(k, len(items)))
        result.extend(sampled)

    # Adjust to exact n
    random.shuffle(result)
    return result[:n]


def run_ab_test(queries: list, workers: int = 3, rpm: int = 30):
    """Run A/B test: v1 vs v2 prompt on same queries."""
    results_v1 = []
    results_v2 = []
    write_lock = threading.Lock()
    min_interval = 60.0 / rpm * workers

    def run_one(item: dict, prompt: str, version: str):
        start = time.time()
        try:
            result = simulate_one(item["query"], prompt)
            result["source_tier"] = item.get("tier", "unknown")
            result["version"] = version
            elapsed = time.time() - start
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            return result
        except Exception as e:
            return {"query": item["query"], "tier": "error", "error": str(e), "version": version}

    print(f"=== Running V1 (original prompt) on {len(queries)} queries ===")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, q, PROMPT_V1, "v1"): q for q in queries}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results_v1.append(result)
            print(f"[V1 {i+1}/{len(queries)}] {result['query'][:40]}... -> {result['tier']}")

    print(f"\n=== Running V2 (teaching prompt) on {len(queries)} queries ===")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, q, PROMPT_V2, "v2"): q for q in queries}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results_v2.append(result)
            print(f"[V2 {i+1}/{len(queries)}] {result['query'][:40]}... -> {result['tier']}")

    return results_v1, results_v2


def analyze_results(results_v1: list, results_v2: list):
    """Compare tier distributions."""
    print("\n" + "=" * 60)
    print("A/B TEST RESULTS")
    print("=" * 60)

    tiers_v1 = Counter(r["tier"] for r in results_v1)
    tiers_v2 = Counter(r["tier"] for r in results_v2)

    all_tiers = sorted(set(tiers_v1.keys()) | set(tiers_v2.keys()))

    print(f"\n{'Tier':<20} {'V1 (original)':<15} {'V2 (teaching)':<15} {'Δ':<10}")
    print("-" * 60)

    for tier in all_tiers:
        c1 = tiers_v1.get(tier, 0)
        c2 = tiers_v2.get(tier, 0)
        delta = c2 - c1
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        print(f"{tier:<20} {c1:<15} {c2:<15} {delta_str:<10}")

    t0_v1 = tiers_v1.get("T0-qa", 0)
    t0_v2 = tiers_v2.get("T0-qa", 0)
    total = len(results_v1)

    print("-" * 60)
    print(f"{'T0-qa rate':<20} {t0_v1/total*100:.1f}%{'':<10} {t0_v2/total*100:.1f}%{'':<10} {(t0_v2-t0_v1)/total*100:+.1f}%")
    print("=" * 60)

    # Queries that changed from T0 in v1 to non-T0 in v2
    v1_map = {r["query"]: r["tier"] for r in results_v1}
    v2_map = {r["query"]: r["tier"] for r in results_v2}

    improved = [(q, v1_map[q], v2_map[q]) for q in v1_map if v1_map[q] == "T0-qa" and v2_map[q] != "T0-qa"]
    regressed = [(q, v1_map[q], v2_map[q]) for q in v1_map if v1_map[q] != "T0-qa" and v2_map[q] == "T0-qa"]

    if improved:
        print(f"\n✅ Improved (T0 -> tool-use): {len(improved)} queries")
        for q, t1, t2 in improved[:5]:
            print(f"   {q[:50]}... | {t1} -> {t2}")

    if regressed:
        print(f"\n⚠️ Regressed (tool-use -> T0): {len(regressed)} queries")
        for q, t1, t2 in regressed[:5]:
            print(f"   {q[:50]}... | {t1} -> {t2}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="data/queries/grpo_prompt_pool.jsonl")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rpm", type=int, default=30, help="Requests per minute limit")
    parser.add_argument("--output", default="data/trajectories/ab_test_results.jsonl")
    args = parser.parse_args()

    queries = stratified_sample(args.pool, args.sample)
    print(f"Sampled {len(queries)} queries from {args.pool}")

    # Show sample distribution
    sample_tiers = Counter(q.get("tier", "?").split("-")[0] for q in queries)
    print(f"Sample tier distribution: {dict(sample_tiers)}")

    results_v1, results_v2 = run_ab_test(queries, workers=args.workers, rpm=args.rpm)

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results_v1 + results_v2:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {args.output}")

    analyze_results(results_v1, results_v2)
