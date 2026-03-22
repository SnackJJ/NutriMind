"""
Generate multi-step tool-use queries using Gemini.

These are queries that inherently require 2+ sequential tool calls,
designed to supplement training data for teaching the student model
proper multi-step tool orchestration.

Usage:
    python scripts/generate_multistep_queries.py
    python scripts/generate_multistep_queries.py --count 50
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
GEMINI_MODEL = "gemini-2.5-flash"

OUTPUT_PATH = PROJECT_ROOT / "data/queries/multistep_generated.jsonl"

TOOL_DESCRIPTIONS = """
Available tools for the NutriMind nutrition assistant:
1. get_food_nutrition(foods: [{food_name, amount_grams}]) → Returns nutrition data (calories, protein, fat, carbs, fiber, sodium, etc.) with match_confidence (high/medium/low). When confidence is "low", the returned food may not match the query.
2. log_meal(meal_type, foods: [{food_name, amount_grams}]) → Creates a new meal log entry. CANNOT edit or delete.
3. get_today_summary() → Returns today's total intake (calories, protein, fat, carbs) and remaining calorie budget.
4. get_history(days, metric, compare_to_goal) → Returns multi-day nutritional trends (up to 90 days back). Can compare against goals.
5. retrieve_knowledge(query, mode: hybrid|semantic|keyword, top_k) → RAG search over a nutrition knowledge base. Returns relevance_score and passages. May need retry with different mode if score < 0.4.
"""

GENERATION_PROMPT = """You are helping create training data for a nutrition AI assistant called NutriMind.

The assistant has these tools:
{tools}

## Task
Generate {count} user queries that INHERENTLY require **multiple sequential tool calls** to answer properly. Each query must lead to 2-5 tool calls in sequence.

## Multi-step patterns to cover (generate a balanced mix):

### Pattern A: Lookup → Log (2 steps)
User wants to log a meal but needs nutrition info first.
Example: "I had 200g of grilled chicken breast with half a cup of brown rice for lunch. Log it and tell me the total calories."
Tools: get_food_nutrition → log_meal

### Pattern B: Check Budget → Suggest (2 steps)
User asks what they can still eat today based on remaining budget.
Example: "I want to hit my protein goal today. How much do I have left and what should I eat?"
Tools: get_today_summary → get_food_nutrition (to verify suggestion)

### Pattern C: History Analysis → Comparison (2 steps)
User wants to analyze trends or compare periods.
Example: "How does my average calorie intake this week compare to last week? Am I improving?"
Tools: get_history(days=14) → analyze and present

### Pattern D: Food Lookup with Retry (2 steps)
User asks about a food that initially returns low confidence match.
Example: "What's the nutrition in a serving of 'dragon fruit'?" (might need retry with "Pitaya, raw")
Tools: get_food_nutrition → get_food_nutrition (retry with USDA naming)

### Pattern E: RAG Search with Mode Switch (2-3 steps)
User asks about dietary guidelines where first search doesn't find good results.
Example: "What does the latest research say about intermittent fasting and muscle retention?"
Tools: retrieve_knowledge(hybrid) → retrieve_knowledge(keyword) → answer

### Pattern F: Multi-food Comparison → Decision (2 steps)
User wants to compare multiple foods to make a choice.
Example: "I'm choosing between quinoa and couscous for dinner. Compare their protein and fiber content."
Tools: get_food_nutrition(both foods) → provide comparison

### Pattern G: Full Workflow - Check → Lookup → Log (3 steps)
Complete nutrition tracking workflow in one request.
Example: "Check how much protein I've had today, then log a snack of 150g Greek yogurt with honey, and update me on my remaining budget."
Tools: get_today_summary → log_meal → get_today_summary (or get_food_nutrition → log_meal → get_today_summary)

### Pattern H: History + Knowledge (2-3 steps)
User asks about their trends AND wants guidance based on established guidelines.
Example: "I've been eating a lot of sodium lately. Show me my sodium trend for the past week and tell me what the recommended daily limit is."
Tools: get_history → retrieve_knowledge

## Requirements
1. Each query must be a natural, realistic user message (1-3 sentences)
2. Queries must be in English
3. Include a mix of all patterns (A through H)
4. Make queries diverse in food types, dietary situations, and user backgrounds
5. Do NOT include queries that can be answered without tools
6. Do NOT include queries about editing/deleting logs (system can't do that)
7. Vary complexity: some 2-step, some 3-step, a few 4-5 step

## Output format
Return a JSON array of objects, each with:
- "query": the user message
- "tier": one of "T1-multi", "T2-multi", "T3-multi" based on complexity (T1=2 steps, T2=3 steps, T3=4+ steps)
- "pattern": which pattern (A-H) this query follows
- "expected_tools": list of tool names in expected call order

Return ONLY the JSON array, no other text.
"""


def generate_queries(count: int = 50) -> list:
    """Generate multi-step queries using Gemini."""
    prompt = GENERATION_PROMPT.format(tools=TOOL_DESCRIPTIONS, count=count)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "temperature": 0.9,
                    "max_output_tokens": 16000,
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
    parser.add_argument("--count", type=int, default=80, help="Number of queries to generate")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    print(f"Generating {args.count} multi-step tool-use queries via Gemini...")
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
        q["source"] = "gemini-multistep-generation"

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for q in unique:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    from collections import Counter
    tier_c = Counter(q["tier"] for q in unique)
    pattern_c = Counter(q["pattern"] for q in unique)

    print(f"\nGenerated {len(unique)} unique queries → {output_file}")
    print("\nTier distribution:")
    for t, c in sorted(tier_c.items()):
        print(f"  {t}: {c}")
    print("\nPattern distribution:")
    for p, c in sorted(pattern_c.items()):
        print(f"  {p}: {c}")

    # Print a few examples
    print("\n── Sample queries ──")
    for q in unique[:5]:
        print(f"  [{q['tier']}|{q['pattern']}] {q['query']}")
        print(f"    Tools: {' → '.join(q.get('expected_tools', []))}")


if __name__ == "__main__":
    main()
