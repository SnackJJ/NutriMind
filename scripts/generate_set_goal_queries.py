"""
Generate set_goal tool queries using Gemini.

Covers three tiers:
- T1: Single set_goal call (e.g., "Set my protein target to 130g")
- T2: set_goal + status check (e.g., "I want to lose weight, set calories to 1500 and show me today's progress")
- T3: Conditional set_goal (e.g., "Check if I've been over my calorie goal this week, if so lower it by 200")

Usage:
    python scripts/generate_set_goal_queries.py
    python scripts/generate_set_goal_queries.py --count 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from google import genai

client = genai.Client(api_key=settings.gemini_api_key)
GEMINI_MODEL = "gemini-2.5-pro"

OUTPUT_PATH = PROJECT_ROOT / "data/queries/set_goal_queries.jsonl"

GENERATION_PROMPT = """You are helping create training data for a nutrition AI assistant called NutriMind.

The assistant has these tools:
1. get_food_nutrition(foods: list of food_name + amount_grams) → Returns nutrition data
2. log_meal(meal_type, foods: list of food_name + amount_grams) → Creates meal log entry
3. get_today_summary() → Returns today's intake and remaining calorie budget
4. get_history(days, metric, compare_to_goal) → Returns multi-day trends. When compare_to_goal=true, includes adherence analysis
5. retrieve_knowledge(query, mode, top_k) → RAG search over nutrition knowledge base
6. set_goal(metric, target_value, goal_type) → Set or update a nutrition target
   - metric: "calories" | "protein" | "fat" | "carbs"
   - target_value: number (kcal for calories, grams for macros)
   - goal_type (optional): "lose" | "maintain" | "gain"

## Task
Generate {count} diverse user queries for the set_goal tool, covering three tiers:

### Tier 1 (T1): Simple Goal Setting (~40 queries)
User directly wants to set or update ONE nutrition target. Single tool call.

Examples:
- "Set my protein target to 130g"
- "Change my daily calorie goal to 1800"
- "I want to aim for 2200 calories per day"
- "Update my carbs limit to 200 grams"
- "Set fat intake goal to 70g daily"
- "My trainer recommended 150g protein, please set that"
- "Can you change my calorie budget to 1600?"
- "I'd like to lower my carb target to 150g"

Variations to include:
- Different phrasings (set, change, update, adjust, make it, I want)
- All four metrics (calories, protein, fat, carbs)
- Different target values (realistic ranges)
- Mentions of goal_type (lose weight → lose, bulk up → gain, maintain weight → maintain)
- Casual vs formal tone
- With/without units mentioned

### Tier 2 (T2): Set Goal + Check Status (~35 queries)
User sets a goal AND wants to see current progress/status. Two sequential tool calls.

Examples:
- "I'm trying to lose weight. Set my calories to 1500 and show me how much I have left today"
- "Update my protein goal to 120g and tell me how close I am today"
- "Set my carb limit to 180g, then check today's intake for me"
- "Change my calorie target to 2000 for weight maintenance, and show today's summary"
- "I want to bulk up - set protein to 160g and check if I've hit it today"

Expected tool flow: set_goal → get_today_summary

### Tier 3 (T3): Conditional Goal Setting (~25 queries)
User asks to check history/trends FIRST, then conditionally set goal based on results. Requires branching logic.

Examples:
- "Check if I've been exceeding my calorie goal this week. If I'm consistently over, lower the target by 200"
- "Look at my protein intake for the past week. If I'm averaging under 80g, set my goal to 100g"
- "Review my calorie adherence this month. If adherence is below 60%, maybe I should adjust my target"
- "See how I did with carbs last week. If I kept going over, I need to set a more realistic limit"
- "Check my weekly trend - if my fat intake has been too low, bump up the goal by 10g"

Expected tool flow: get_history(compare_to_goal=true) → [based on result] → set_goal

## Requirements
1. Each query must be a natural, realistic user message (1-3 sentences)
2. ALL queries must be in English
3. Include diverse food/nutrition contexts (weight loss, muscle building, health conditions, athletic training, general health)
4. Use varied phrasings and tones (casual, direct, questioning, requesting)
5. Cover all four metrics approximately equally
6. Include both explicit numbers and relative adjustments ("lower by 200", "increase to 150")
7. Some queries should mention goal_type (lose/maintain/gain), others should not
8. Make queries realistic - use sensible target values

## Output Format
Return a JSON array where each object has:
- "query": the user message
- "tier": "T1", "T2", or "T3"
- "expected_tools": list of tool names in expected call order

Return ONLY the JSON array, no other text.
"""


def generate_queries(count: int = 100) -> list:
    """Generate set_goal queries using Gemini."""
    prompt = GENERATION_PROMPT.format(count=count)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "temperature": 0.9,
                    "max_output_tokens": 32000,
                }
            )
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            queries = json.loads(text)
            return queries
        except Exception as e:
            print(f"[Attempt {attempt+1}/3] Error: {e}")
            if attempt < 2:
                time.sleep(3)
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100, help="Number of queries to generate")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    print(f"Generating {args.count} set_goal queries via Gemini {GEMINI_MODEL}...")
    queries = generate_queries(args.count)

    if not queries:
        print("ERROR: No queries generated")
        sys.exit(1)

    # Deduplicate
    seen = set()
    unique = []
    for q in queries:
        if q["query"] not in seen:
            seen.add(q["query"])
            unique.append(q)

    # Add source tag
    for q in unique:
        q["source"] = "gemini-set-goal-generation"

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for q in unique:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    from collections import Counter
    tier_c = Counter(q["tier"] for q in unique)

    print(f"\nGenerated {len(unique)} unique queries → {output_file}")
    print("\nTier distribution:")
    for t, c in sorted(tier_c.items()):
        print(f"  {t}: {c}")

    # Print sample queries per tier
    print("\n── Sample queries ──")
    for tier in ["T1", "T2", "T3"]:
        tier_qs = [q for q in unique if q["tier"] == tier][:3]
        print(f"\n{tier}:")
        for q in tier_qs:
            print(f"  - {q['query']}")
            print(f"    Tools: {' → '.join(q.get('expected_tools', []))}")


if __name__ == "__main__":
    main()
