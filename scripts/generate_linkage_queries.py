"""
Generate complex multi-step "Linkage" queries via Gemini.
Uses google-genai SDK for high-quality reasoning.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from google import genai
from google.genai import types
from src.config import settings

client = genai.Client(api_key=settings.gemini_api_key)

# Use standard prompt for complex linkage
LINKAGE_PROMPT = """You are a senior nutrition data architect. 
Your task is to generate complex, multi-step queries for a nutrition AI assistant.

These queries MUST involve cross-tool interaction (Linkage). 
A good linkage query cannot be answered by one tool alone.

### SCENARIOS:
1. (History + KB): User reviews their past intake trends and asks for food/diet adjustments based on specific medical/fitness guidelines in the knowledge base.
2. (Summary + Nutrition + KB): User checks today's remaining budget, lookups a specific food's nutrition, and asks for a recommendation that fits both the budget and fitness goals.
3. (Error Recovery + KB): User tries to log/lookup a non-standard or problematic food, fails, and then asks for a healthy alternative/substitute from the knowledge base.
4. (Goal Analysis + Nutrition): User checks history for goal adherence, identifies a missing macro, and asks to find a food high in that macro but low in fat.

### GENERATION RULES:
- Language: English.
- Tone: Natural, like a real user tracking their health.
- Output: Return a JSON array of objects. Each object has: "query", "tier" (always "T2" or "T3"), "scenario" (A, B, or C).

Generate 50 diverse queries. Return ONLY the JSON array."""

def generate_batch():
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=LINKAGE_PROMPT,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.8,
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error generating with Gemini: {e}")
        return []

def main():
    all_queries = []
    target_count = 250
    print(f"Generating {target_count} linkage queries via Gemini 2.5 Pro...")
    
    while len(all_queries) < target_count:
        batch = generate_batch()
        if not batch:
            continue
        for q in batch:
            q["source"] = "gemini_linkage_generation"
            all_queries.append(q)
            if len(all_queries) >= target_count:
                break
        print(f"Progress: {len(all_queries)}/{target_count}")

    output_path = "data/queries/batch4_linkage_new.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for q in all_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    
    print(f"Done! Generated {len(all_queries)} queries -> {output_path}")

if __name__ == "__main__":
    main()
