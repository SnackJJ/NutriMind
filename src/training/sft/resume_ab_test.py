"""
Resume A/B test: only run V2 for failed queries, then merge results.

Usage:
    python -m src.training.sft.resume_ab_test --input data/trajectories/ab_test_100.jsonl
"""
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import settings
from src.training.sft.collect_trajectories import (
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


def run_v2_only(queries: list, workers: int = 3, rpm: int = 30) -> list:
    """Run V2 prompt on given queries."""
    results = []
    min_interval = 60.0 / rpm * workers

    def run_one(item: dict):
        query = item["query"]
        source_tier = item.get("source_tier", "unknown")
        start = time.time()
        try:
            result = simulate_one(query, PROMPT_V2)
            result["source_tier"] = source_tier
            result["version"] = "v2"
            elapsed = time.time() - start
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            return result
        except Exception as e:
            return {"query": query, "tier": "error", "error": str(e), "version": "v2", "source_tier": source_tier}

    print(f"=== Running V2 (teaching prompt) on {len(queries)} failed queries ===")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, q): q for q in queries}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "✓" if result["tier"] != "error" else "✗"
            print(f"[V2 {i+1}/{len(queries)}] {status} {result['query'][:40]}... -> {result['tier']}")

    return results


def analyze_results(results_v1: list, results_v2: list):
    """Compare tier distributions."""
    print("\n" + "=" * 60)
    print("A/B TEST RESULTS (MERGED)")
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

    improved = [(q, v1_map[q], v2_map[q]) for q in v1_map if v1_map[q] == "T0-qa" and v2_map.get(q, "error") not in ("T0-qa", "error")]
    regressed = [(q, v1_map[q], v2_map[q]) for q in v1_map if v1_map[q] != "T0-qa" and v2_map.get(q) == "T0-qa"]

    if improved:
        print(f"\n✅ Improved (T0 -> tool-use): {len(improved)} queries")
        for q, t1, t2 in improved[:5]:
            print(f"   {q[:50]}... | {t1} -> {t2}")
        if len(improved) > 5:
            print(f"   ... and {len(improved) - 5} more")

    if regressed:
        print(f"\n⚠️ Regressed (tool-use -> T0): {len(regressed)} queries")
        for q, t1, t2 in regressed[:5]:
            print(f"   {q[:50]}... | {t1} -> {t2}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/trajectories/ab_test_100.jsonl")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rpm", type=int, default=30)
    args = parser.parse_args()

    input_path = Path(args.input)

    # Load existing results
    with open(input_path) as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    # Separate V1 and V2 records
    v1_records = [r for r in all_records if r.get("version") == "v1"]
    v2_records = [r for r in all_records if r.get("version") == "v2"]

    # Find V2 failures (tier == "error")
    v2_failed = [r for r in v2_records if r.get("tier") == "error"]
    v2_success = [r for r in v2_records if r.get("tier") != "error"]

    print(f"Loaded {len(all_records)} records from {input_path}")
    print(f"  V1: {len(v1_records)} records")
    print(f"  V2: {len(v2_records)} records ({len(v2_success)} success, {len(v2_failed)} failed)")

    if not v2_failed:
        print("No failed V2 records to retry. Exiting.")
        exit(0)

    # Build query list for retry (use V1 info for source_tier)
    v1_by_query = {r["query"]: r for r in v1_records}
    retry_queries = []
    for r in v2_failed:
        query = r["query"]
        v1_info = v1_by_query.get(query, {})
        retry_queries.append({
            "query": query,
            "source_tier": v1_info.get("source_tier", r.get("source_tier", "unknown"))
        })

    # Run V2 on failed queries
    new_v2_results = run_v2_only(retry_queries, workers=args.workers, rpm=args.rpm)

    # Count successes
    new_success = sum(1 for r in new_v2_results if r.get("tier") != "error")
    new_failed = len(new_v2_results) - new_success
    print(f"\nRetry results: {new_success} success, {new_failed} still failed")

    # Merge: keep V1, replace V2 failures with new results
    merged_v2 = v2_success + new_v2_results

    # Write merged results
    with open(input_path, "w") as f:
        for r in v1_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in merged_v2:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nMerged results written to {input_path}")

    # Analyze
    analyze_results(v1_records, merged_v2)
