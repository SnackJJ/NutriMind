#!/usr/bin/env python3
"""
Prepare GRPO Prompt Pool with Gemini-powered `env_state`.

Samples queries from data/queries/grpo_prompt_pool.jsonl according to the
proportions defined in phase4_grpo.md:
T0: 5%
T1: 15%
T2: 30%
T3: 30%
T4: 15%
Error recovery: 5%

Uses gemini-2.5-pro to generate a realistic `env_state` corresponding to each query.
Supports parallel processing with multiple workers.
"""

import json
import random
import uuid
import sys
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("/home/jzq/Projects/NutriMind")
sys.path.append(str(PROJECT_ROOT))

from google import genai
from pydantic import BaseModel, Field

# We assume a settings module provides the API key, similar to other scripts.
try:
    from src.config import settings
except ImportError:
    print("Could not import src.config. Ensure you run this from the project root.")
    sys.exit(1)

client = genai.Client(api_key=settings.gemini_api_key)

# The Pydantic schema for structured output from Gemini
class FoodItem(BaseModel):
    name: str = Field(description="Name of the food item")
    calories: int = Field(description="Calories")
    protein_g: float = Field(description="Protein in grams")
    fat_g: float = Field(description="Fat in grams")
    carbs_g: float = Field(description="Carbs in grams")
    fiber_g: float = Field(description="Fiber in grams")

class MealLog(BaseModel):
    meal_type: str = Field(description="e.g. breakfast, lunch, snack, dinner")
    foods: List[FoodItem] = Field(description="Foods eaten in this meal")
    calories: int = Field(description="Total calories of the meal")
    protein_g: float = Field(description="Total protein in grams")
    fat_g: float = Field(description="Total fat in grams")
    carbs_g: float = Field(description="Total carbs in grams")
    fiber_g: float = Field(description="Total fiber in grams")

class MealHistoryDay(BaseModel):
    date: str = Field(description="e.g. yesterday, 2023-10-25")
    calories: int = Field(description="Total calories logged that day")
    protein_g: float = Field(description="Total protein logged that day")
    fat_g: float = Field(description="Total fat logged that day")
    carbs_g: float = Field(description="Total carbs logged that day")
    fiber_g: float = Field(description="Total fiber logged that day")

class UserProfile(BaseModel):
    tdee_kcal: int = Field(description="Total Daily Energy Expenditure in kcal")
    goal: str = Field(description="e.g. lose_weight, maintain, gain_muscle")

class UserGoals(BaseModel):
    calories: int = Field(description="Daily calorie goal")
    protein: int = Field(description="Daily protein goal in grams")
    fat: int = Field(description="Daily fat goal in grams")
    carbs: int = Field(description="Daily carb goal in grams")
    fiber: int = Field(description="Daily fiber goal in grams")

class EnvState(BaseModel):
    user_id: str = Field(description="A unique user ID like grpo_user_123")
    user_profile: UserProfile
    user_goals: UserGoals
    meals_today: List[MealLog]
    meal_history: List[MealHistoryDay]


def get_base_tier(tier_str: str) -> str:
    """Map granular tiers (T3-multi-constraint) to base tiers (T3)."""
    tier_str = tier_str.lower()
    if tier_str.startswith("t0"): return "t0"
    if tier_str.startswith("t1"): return "t1"
    if tier_str.startswith("t2"): return "t2"
    if tier_str.startswith("t3"): return "t3"
    if tier_str.startswith("t4"): return "t4"
    if "error" in tier_str: return "error_recovery"
    return "unknown"


def generate_env_state_for_query(query: str, tier: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """Use Gemini to generate a contextually relevant env_state."""
    prompt = f"""
    You are generating realistic mock user profiles (`env_state`) for a nutrition assistant testing environment.

    The user's query is: "{query}" (Tier: {tier})

    Create a realistic user snapshot (goals, profile, and what they've eaten today).
    Output purely the JSON matching the EnvState schema.
    Ensure ALL nutritional fields (fat, carbs, fiber) are provided.
    If the query implies they already ate lunch, include a lunch in `meals_today` with realistic macros and calories.
    For `meal_history`, provide at least the last 2-3 days of data so `get_history` results look realistic.
    """

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': EnvState,
                    'temperature': 0.7
                }
            )

            data = json.loads(response.text)
            if "user_id" not in data:
                data["user_id"] = f"grpo_user_{uuid.uuid4().hex[:8]}"
            return data
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "503" in err_msg or "ResourceExtausted" in err_msg or attempt < max_retries - 1:
                # Aggressive exponential backoff for rate limits/unavailable 
                sleep_time = 5 * (2 ** attempt) + random.uniform(0, 3)
                time.sleep(sleep_time)
            else:
                print(f"Failed after {max_retries} attempts: {err_msg}")
                return None
    return None


def process_single_item(item: Dict[str, Any], index: int, total: int) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Process a single item and return (index, result)."""
    try:
        env_state = generate_env_state_for_query(item["query"], item["tier"])
        if env_state is None:
            return (index, None)

        final_item = {
            "query": item["query"],
            "tier": item["tier"],
            "difficulty": "medium",
            "env_state": env_state
        }
        return (index, final_item)
    except Exception as e:
        print(f"[{index+1}/{total}] Error: {e}")
        return (index, None)

def main():
    parser = argparse.ArgumentParser(description="Prepare GRPO prompts with Gemini env_state")
    parser.add_argument("--total", type=int, default=2000, help="Total number of prompts to generate")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    input_file = PROJECT_ROOT / "data/queries/grpo_prompt_pool.jsonl"
    output_file = Path(args.output) if args.output else PROJECT_ROOT / "data/grpo/grpo_prompts.jsonl"

    if not input_file.exists():
        print(f"File not found: {input_file}")
        sys.exit(1)

    existing_queries = set()
    existing_tier_counts = {"t0": 0, "t1": 0, "t2": 0, "t3": 0, "t4": 0, "error_recovery": 0}
    
    if output_file.exists():
        print(f"Found existing output file {output_file}. Collecting existing queries to resume...")
        with open(output_file, "r") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                existing_queries.add(d["query"])
                b = get_base_tier(d.get("tier", ""))
                if b in existing_tier_counts:
                    existing_tier_counts[b] += 1
        print(f"Already generated: {len(existing_queries)} queries.")

    print("Loading queries...")
    pool = {"t0": [], "t1": [], "t2": [], "t3": [], "t4": [], "error_recovery": []}

    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            # Skip if already generated
            if data.get("query") in existing_queries:
                continue
            base = get_base_tier(data.get("tier", ""))
            if base in pool:
                pool[base].append(data)

    for k, v in pool.items():
        print(f"Base {k}: {len(v)} NEW queries available")

    total_available = sum(len(v) for v in pool.values())
    print(f"Total available: {total_available}")

    # Specific sampling logic per user request
    # T2, T3: All available
    # T0, T1, T4, Error: Proportions from 1500 baseline (5%, 15%, 15%, 5%)
    target_sizes = {
        "t0": 75,
        "t1": 225,
        "t2": len(pool["t2"]), # All T2 (~648)
        "t3": len(pool["t3"]), # All T3 (~749)
        "t4": 225,
        "error_recovery": 75
    }

    print(f"\nTarget distribution (Target Total={sum(target_sizes.values())}):")
    for tier, count in target_sizes.items():
        available = len(pool[tier])
        actual = min(count, available)
        print(f"  {tier}: {actual}/{count} (available: {available})")

    random.seed(42)
    selected_prompts = []

    for tier, count in target_sizes.items():
        k_count = min(count, len(pool[tier]))
        selected_prompts.extend(random.sample(pool[tier], k_count))

    # Shuffle to mix tiers during processing
    random.shuffle(selected_prompts)

    print(f"\nSelected {len(selected_prompts)} prompts. Starting parallel generation with {args.workers} workers...")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Thread-safe counter and file writing
    success_count = 0
    failed_count = 0
    write_lock = threading.Lock()
    counter_lock = threading.Lock()

    total = len(selected_prompts)
    if total == 0:
        print("Nothing to process. Dataset is full!")
        return

    with open(output_file, "a") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_single_item, item, i, total): i
                for i, item in enumerate(selected_prompts)
            }

            # Process as completed
            for future in as_completed(futures):
                idx, result = future.result()

                with counter_lock:
                    if result is not None:
                        success_count += 1
                        with write_lock:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f.flush()
                        status = "✓"
                    else:
                        failed_count += 1
                        status = "✗"

                    # Progress update every 10 items
                    processed = success_count + failed_count
                    if processed % 10 == 0 or processed == total:
                        print(f"Progress: {processed}/{total} ({success_count} success, {failed_count} failed)")

    print(f"\n{'='*50}")
    print(f"Generation complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {failed_count}")
    print(f"  Total:   {success_count + failed_count}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
