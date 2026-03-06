import asyncio
import json
import os
import random
from pathlib import Path
from pydantic import BaseModel, Field
import typing_extensions as typing

from google import genai
from google.genai import types

from src.config import settings

# --- API Config ---
# For stability, we use gemini-2.5-pro exclusively
MODEL_ID = "gemini-2.5-pro"

gemini_client = genai.Client(api_key=settings.gemini_api_key)

# --- Define Schema for structured output ---
class GeneratedQuery(BaseModel):
    tier: str = Field(description="The tier subtype of the query (e.g., T1-basic, T2-ambiguous, T3-limit, T0-qa, error_recovery).")
    query: str = Field(description="The generated realistic user query.")

class QueryBatch(BaseModel):
    queries: list[GeneratedQuery]

# --- Target Quantities (approx 5800 total) ---
TARGETS = {
    "T0": 250,       
    "T1": 1200,      
    "T2": 1500,      
    "T3": 1750,      
    "T4": 800,       
    "Error": 280     
}
BATCH_SIZE = 50 

MAX_CONCURRENT_REQUESTS = 3 # Reduced concurrency to prevent 503s
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
file_lock = asyncio.Lock()

# --- Helper to load USDA examples ---
def load_usda_examples(n=30):
    try:
        with open("data/usda_foods.json", "r", encoding="utf-8") as f:
            foods = json.load(f)
            return random.sample(foods, min(n, len(foods)))
    except Exception as e:
        print(f"Failed to load USDA foods: {e}")
        return ["apple", "salmon", "quinoa", "chicken breast", "broccoli"]

# --- Prompts by Tier ---

def get_t0_prompt(seeds):
    return f"""You are generating dietary queries for an AI Assistant to practice on.
Generate {BATCH_SIZE} unique "T0-qa" queries. 
T0 queries are general nutrition knowledge questions where the model learns to answer *without* invoking tracking tools.

Examples of T0 queries from our seed pool:
{chr(10).join(f"- {s['query']}" for s in seeds)}

Guidelines:
1. Cover physiology, absorption, myth-busting, and general biochemistry.
2. Vary phrasing (some casual, some direct, some slightly conversational).
3. Do NOT ask about logging or calculating specific user meals (that's T2).
"""

def get_t1_prompt(seeds, food_examples):
    return f"""You are generating dietary queries for an AI Assistant to practice on.
Generate {BATCH_SIZE} unique "T1" queries. 
T1 queries are simple factual lookups with varied language, foods, and metrics.

Examples of T1 queries from our seed pool:
{chr(10).join(f"- {s['query']} (Tier: {s['tier']})" for s in seeds)}

Here are some real food examples from our database to sprinkle in:
{food_examples}
Also intentionally include 2-3 queries with completely fictional or highly complex branded foods (e.g., "dragon meat", "Starbucks ultra venti matcha caramel latte") to naturally test fallback behavior.

Subtypes to distribute across the {BATCH_SIZE} queries:
- T1-basic (Single nutrient lookup)
- T1-compare (Compare two foods)
- T1-ranking (Top N foods for a nutrient)
- T1-yesno (Yes/no verification)
- T1-brand (Brand/packaged food query)
- T1-cooking (Cooking method impact)
- T1-unit (Unit conversions, e.g. ml to tbsp)
- T1-international (International/cultural dishes)
"""

def get_t2_prompt(seeds):
    return f"""You are generating dietary queries for an AI Assistant to practice on.
Generate {BATCH_SIZE} unique "T2" queries. 
T2 queries demand multi-step tool execution but with a STRICT DEPTH OF 1-2 STEPS (e.g., search then calculate, or calculate then log). 
DO NOT add complex conditional constraints (that's T3).

Examples of T2 queries from our seed pool:
{chr(10).join(f"- {s['query']} (Tier: {s['tier']})" for s in seeds)}

Subtypes to distribute across the {BATCH_SIZE} queries:
- T2-calculate-log (Calculate macros and log a meal)
- T2-ambiguous (Ambiguous food query requiring the agent to clarify)
- T2-error-handling (Query testing missing profile info)
- T2-history (Query past records/logs)
- T2-fuzzy (Vague input requiring estimation)
- T2-weekly (Weekly/period statistics)
- T2-update-log (Correcting a previous log)

Guidelines:
1. Vary sentence structures and implicit tool chains:
   - lookup + calculate
   - calculate + log
   - check budget + log
   - get history + lookup knowledge
2. Include realistic specifics (e.g. "I had a sandwich with 2 slices whole wheat bread, 3oz turkey...").
"""

def get_t3_prompt(seeds):
    return f"""You are generating dietary queries for an AI Assistant to practice on.
Generate {BATCH_SIZE} unique "T3" queries. 
T3 queries focus on state-dependent and conditional reasoning based on user personas, limits, budgets, or complex constraints.

Examples of T3 queries from our seed pool:
{chr(10).join(f"- {s['query']} (Tier: {s['tier']})" for s in seeds)}

Subtypes to distribute across the {BATCH_SIZE} queries:
- T3-limit (Check meal against daily sodium/sugar/etc limits)
- T3-allergy (Nut/dairy/gluten allergy alternatives)
- T3-multi-constraint (e.g., High protein, low FODMAP, under 25 min)
- T3-diet-style (Vegan, keto, halal, pescatarian)
- T3-budget (Cost-conscious suggestions)
- T3-time (Time/prep-constrained cooking)
- T3-exercise (Pre/post workout nutrition timing)
- T3-social (Navigating parties, restaurants, holidays)
- T3-pregnancy/T3-elderly (Life stage specific restrictions)

Guidelines:
Ensure the condition is implicitly present in the query (e.g., "I've already hit 2000mg sodium today, can I eat soup?").
"""

def get_t4_prompt(seeds):
    return f"""You are generating dietary queries for an AI Assistant to practice on.
Generate {BATCH_SIZE} unique "T4" queries. 
T4 queries define our strict medical integration boundaries. These test the agent's ability to recognize when to refer to a human doctor.

Examples of T4 queries from our seed pool:
{chr(10).join(f"- {s['query']} (Tier: {s['tier']})" for s in seeds)}

Subtypes to distribute across the {BATCH_SIZE} queries:
- T4-info (General info on medical diet, expecting a cautious answer+disclaimer)
- T4-guidance (Personal medical situation asking for guidance, expecting boundaries)
- T4-reject (Requests requiring a doctor, which the agent MUST politely decline)

Topics to aggressively cover:
1. Drug Interactions (Warfarin, MAOIs, Statins, Metformin)
2. Extreme Dieting (Eating disorders, dangerous fasting, severe deficits)
3. Disease Self-Diagnosis (Diagnosing symptoms based on diet)
4. Post-surgery / Active Treatment (Dialysis meal plans, chemotherapy appetite loss, transplant recovery)
"""

def get_error_prompt(seeds):
    return f"""You are generating dietary queries for an AI Assistant to practice on.
Generate {BATCH_SIZE} unique "error_recovery" queries. 
These are INTENTIONALLY tricky, ambiguous, flawed, or impossible queries designed to trigger the agent's fallback/recovery/clarification tools.

Examples of error queries from our seed pool:
{chr(10).join(f"- {s['query']}" for s in seeds)}

Types of errors to simulate:
1. Missing Quantities ("Log some trail mix")
2. Out-of-Vocabulary / Gibberish foods ("Macros for unicorn meat", "How much protein in flumblygrump")
3. Ambiguous Metric ("What is the vitamin split?")
4. Conflicting Commands ("Search apple. Wait no, calculate banana.")
5. Out of Bounds Goals ("Set calorie goal to 50", "Set protein to -20g")
6. Extreme Quantities ("Log 5000kg of rice")
7. Type Errors ("Get history for 'blue' days")
"""

# --- Execution ---

async def call_gemini_and_save(prompt_text: str, batch_id: str, output_file: str) -> int:
    async with semaphore:
        for attempt in range(5):  # Try up to 5 times
            try:
                print(f"[{batch_id}] Calling {MODEL_ID} (Attempt {attempt+1}/5)...")
                
                response = await asyncio.to_thread(
                    gemini_client.models.generate_content,
                    model=MODEL_ID,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=QueryBatch,
                        temperature=0.8
                    )
                )
                
                if response.text:
                    parsed = QueryBatch.model_validate_json(response.text)
                    print(f"[{batch_id}] Success: returned {len(parsed.queries)} queries.")
                    
                    # Save immediately upon success
                    async with file_lock:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            for q in parsed.queries:
                                record = {
                                    "tier": q.tier,
                                    "query": q.query,
                                    "source": "gemini-sft-expansion"
                                }
                                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    print(f"[{batch_id}] Batch saved to file.")
                    return len(parsed.queries)
                    
            except Exception as e:
                print(f"[{batch_id}] Failed: {e}")
                # Exponential backoff: 3s, 6s, 12s, 24s
                backoff = 3 * (2 ** attempt)
                if attempt < 4:
                    print(f"[{batch_id}] Waiting {backoff} seconds before retry...")
                    await asyncio.sleep(backoff)
                else:
                    print(f"[{batch_id}] Max retries reached. Giving up on this batch.")
        return 0


async def main():
    # 1. Read existing collection seeds
    seed_pool_path = "data/queries/candidate_seeds.json"
    print(f"Loading seeds from {seed_pool_path}...")
    with open(seed_pool_path, "r", encoding="utf-8") as f:
        all_seeds = json.load(f)
        
    # Group by base tier
    seed_groups = {"T0": [], "T1": [], "T2": [], "T3": [], "T4": [], "Error": []}
    for s in all_seeds:
        t = str(s.get("tier", ""))
        if t.startswith("T0"): seed_groups["T0"].append(s)
        elif t.startswith("T1"): seed_groups["T1"].append(s)
        elif t.startswith("T2"): seed_groups["T2"].append(s)
        elif t.startswith("T3"): seed_groups["T3"].append(s)
        elif t.startswith("T4"): seed_groups["T4"].append(s)
        elif "error" in t.lower() or "not-found" in t.lower() or "ambiguous" in t.lower():
            seed_groups["Error"].append(s)
            
    # 2. Check existing generated data for resume capability
    output_path = "data/queries/expanded_query_pool.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    existing_counts = {"T0": 0, "T1": 0, "T2": 0, "T3": 0, "T4": 0, "Error": 0}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    q = json.loads(line)
                    t = q.get("tier", "")
                    if t.startswith("T0"): existing_counts["T0"] += 1
                    elif t.startswith("T1"): existing_counts["T1"] += 1
                    elif t.startswith("T2"): existing_counts["T2"] += 1
                    elif t.startswith("T3"): existing_counts["T3"] += 1
                    elif t.startswith("T4"): existing_counts["T4"] += 1
                    elif "error" in t.lower() or "not-found" in t.lower() or "ambiguous" in t.lower():
                        existing_counts["Error"] += 1
                except:
                    pass
        print("Existing counts based on file:", existing_counts)
    else:
        # Create empty file
        with open(output_path, 'w', encoding='utf-8') as f: pass

    # 3. Process each tier iteratively
    tier_mapping = [
        ("T0", get_t0_prompt),
        ("T1", get_t1_prompt),
        ("T2", get_t2_prompt),
        ("T3", get_t3_prompt),
        ("T4", get_t4_prompt),
        ("Error", get_error_prompt),
    ]
    
    total_generated_this_run = 0
    
    # We will dispatch ALL batch tasks across tiers conceptually, 
    # but the semaphore restricts actual concurrency
    tasks = []
    
    # Pre-load food examples for T1
    food_examples = ", ".join(load_usda_examples(40))

    for tier_name, prompt_func in tier_mapping:
        target_count = TARGETS[tier_name]
        existing = existing_counts[tier_name]
        
        if existing >= target_count:
            print(f"👍 {tier_name} already has {existing} queries (Target: {target_count}). Skipping.")
            continue
            
        remaining = target_count - existing
        batches_needed = (remaining + BATCH_SIZE - 1) // BATCH_SIZE # ceiling division
        
        print(f"--- Queueing {tier_name} expansion ({batches_needed} batches needed to fill {remaining}) ---")
        seeds_for_tier = seed_groups[tier_name]
        if not seeds_for_tier:
            seeds_for_tier = [{"tier": tier_name, "query": f"Example {tier_name} query."}]

        for i in range(batches_needed):
            batch_id = f"{tier_name}-batch-{i+1}/{batches_needed}"
            sampled_seeds = random.sample(seeds_for_tier, min(10, len(seeds_for_tier)))
            
            if tier_name == "T1":
                prompt = prompt_func(sampled_seeds, food_examples)
            else:
                prompt = prompt_func(sampled_seeds)
                
            tasks.append(asyncio.create_task(call_gemini_and_save(prompt, batch_id, output_path)))
            
    if tasks:
        print(f"\n🚀 Launching {len(tasks)} batch tasks across all tiers...")
        results = await asyncio.gather(*tasks)
        total_generated_this_run = sum(results)
        
    print(f"\n✅ All done! Generated {total_generated_this_run} new queries this run.")
    print(f"Results appended to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
