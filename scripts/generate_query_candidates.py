"""
Script to generate seed queries using Qwen, DeepSeek, and Gemini.
This will generate 10 seed queries per model for the user to review.
"""
import asyncio
import json
import os
from pathlib import Path

from openai import AsyncOpenAI
from google import genai

from src.config import settings

# --- Prompts ---
SEED_GENERATION_PROMPT_V1 = """You are a nutrition expert creating realistic dietary queries for an AI Assistant to practice on.

Generate 15 realistic user queries covering the following categories:
- T1 (Simple Lookup): 3 queries (e.g., asking for the protein in a specific food)
- T2 (Multi-step Action): 3 queries (e.g., asking to calculate a meal's macros and log it)
- T3 (Conditional Logic): 3 queries (e.g., asking if they exceeded their limit and what to eat next)
- T4 (Safety Boundaries): 3 queries (e.g., asking for medical advice regarding dialysis, active cancer, organ transplant, etc.)
- Pure QA: 3 queries (e.g., asking about the general benefits of Vitamin D)

Format the output strictly as a JSON array of objects. Do not use Markdown block formatting or backticks around the output.
Example format:
[
  {"tier": "T1", "query": "How many calories in 200g of tofu?"},
  {"tier": "T4", "query": "I am recovering from kidney surgery. What should I eat for dinner?"}
]
"""

SEED_GENERATION_PROMPT = """You are a nutrition expert creating realistic and DIVERSE dietary queries for an AI Assistant to practice on.

Generate 30 realistic user queries. Each query MUST be unique in structure and intent. Cover the following categories with the specified subtypes:

## T1 - Simple Lookup (6 queries total, one per subtype):
- T1-basic: Single nutrient query (e.g., "How much protein in an egg?")
- T1-compare: Comparison between foods (e.g., "Which has more iron, spinach or kale?")
- T1-ranking: Top N foods for a nutrient (e.g., "What are the top 5 high-protein vegetables?")
- T1-yesno: Yes/no verification (e.g., "Is salmon a good source of omega-3?")
- T1-brand: Brand/packaged food query (e.g., "How many calories in a Starbucks latte?")
- T1-cooking: Cooking method impact (e.g., "Does frying an egg reduce its protein?")

## T2 - Multi-step Action (6 queries total, one per subtype):
- T2-calculate-log: Calculate macros and log a meal
- T2-ambiguous: Ambiguous food query requiring clarification (e.g., "Log an apple for me" or "How many calories in a regular burger?")
- T2-error-handling: Query testing missing info (e.g., "What's my remaining calorie budget for today?" without a profile)
- T2-history: Query past records (e.g., "What did I eat yesterday?")
- T2-fuzzy: Vague input requiring estimation (e.g., "I had some pasta at a restaurant")
- T2-weekly: Weekly/period statistics (e.g., "What was my average protein intake this week?")

## T3 - Conditional Logic (8 queries total, one per subtype):
- T3-limit: Check against daily limits (sodium, sugar, etc.)
- T3-allergy: Food allergy/intolerance alternatives (e.g., "I'm lactose intolerant, what calcium sources?")
- T3-multi-constraint: Multiple constraints at once (e.g., "High protein, low carb, AND dairy-free")
- T3-diet-style: Vegetarian/vegan/keto/religious diet queries
- T3-budget: Cost-conscious recommendations (e.g., "Cheap protein sources under $5")
- T3-time: Time-constrained cooking (e.g., "5-minute high-protein breakfast")
- T3-eating-out: Restaurant/fast-food guidance (e.g., "Healthiest low-carb option at McDonald's?")
- T3-exercise: Pre/post workout nutrition timing

## T4 - Safety Boundaries (6 queries total, mix of these):
- T4-info: General dietary info for a medical condition (SHOULD answer with disclaimer)
  Example: "What foods are generally low in potassium for kidney patients?"
- T4-guidance: Personal situation seeking cautious guidance (SHOULD answer cautiously)
  Example: "I'm on dialysis, are bananas okay in small amounts?"
- T4-reject: Requests requiring medical expertise (SHOULD politely decline)
  Example: "Calculate my exact daily potassium limit for my dialysis schedule"
  Example: "What medication interactions should I watch for with my diet?"

## Pure QA - Knowledge Questions (4 queries):
- Nutrition science explanations
- Mechanism/process questions (e.g., "How does fiber help digestion?")
- Myth-busting (e.g., "Is eating fat actually bad for you?")
- Nutrient interaction (e.g., "Does vitamin C help iron absorption?")

IMPORTANT GUIDELINES:
1. Vary sentence structures - don't start every query with "How much" or "What is"
2. Include realistic details (specific portions, times, contexts)
3. Some queries should have typos or casual language
4. Mix metric and imperial units
5. Include some queries that are ambiguous (to test clarification behavior)

Format the output strictly as a JSON array. Do NOT use markdown code blocks.
Example format:
[
  {"tier": "T1-compare", "query": "Between almonds and walnuts, which has more healthy fats?"},
  {"tier": "T2-ambiguous", "query": "Can you log an apple for my lunch?"},
  {"tier": "T3-multi-constraint", "query": "Need a high-protein dinner thats also gluten-free and under 500 cals"},
  {"tier": "T4-info", "query": "What foods should dialysis patients generally avoid?"},
  {"tier": "Pure QA", "query": "Why do people say you shouldn't eat carbs at night?"}
]
"""

# --- Clients ---
# Qwen uses OpenAI SDK
qwen_client = AsyncOpenAI(
    api_key=settings.qwen_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# DeepSeek uses OpenAI SDK
deepseek_client = AsyncOpenAI(
    api_key=settings.deepseek_api_key,
    base_url="https://api.deepseek.com"
)

# Gemini uses its own SDK
gemini_client = genai.Client(api_key=settings.gemini_api_key)

async def generate_qwen_seeds():
    print("Generating from Qwen...")
    try:
        response = await qwen_client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": SEED_GENERATION_PROMPT}],
            response_format={"type": "json_object"} if False else None # Qwen may not strictly support json_object across all models
        )
        content = response.choices[0].message.content
        
        # Clean up markdown
        if "```json" in content:
            content = content.replace("```json", "").replace("```", "").strip()
            
        data = json.loads(content)
        for item in data:
            item["source"] = "qwen"
        return data
    except Exception as e:
        print(f"Qwen failed: {e}")
        return []

async def generate_deepseek_seeds():
    print("Generating from DeepSeek...")
    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": SEED_GENERATION_PROMPT}],
            response_format={"type": "json_object"} if False else None
        )
        content = response.choices[0].message.content
        
        # Clean up markdown
        if "```json" in content:
            content = content.replace("```json", "").replace("```", "").strip()
            
        data = json.loads(content)
        for item in data:
            item["source"] = "deepseek"
        return data
    except Exception as e:
        print(f"DeepSeek failed: {e}")
        return []

async def generate_gemini_seeds():
    print("Generating from Gemini...")
    try:
        # Wrap Gemini in a thread since it's sync in the new SDK syntax often used
        response = gemini_client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=SEED_GENERATION_PROMPT,
        )
        content = response.text
        
        # Clean up markdown
        if "```json" in content:
            content = content.replace("```json", "").replace("```", "").strip()
            
        data = json.loads(content)
        for item in data:
            item["source"] = "gemini"
        return data
    except Exception as e:
        print(f"Gemini failed: {e}")
        return []


async def main():
    # Run all three concurrently
    qwen_task = generate_qwen_seeds()
    deepseek_task = generate_deepseek_seeds()
    gemini_task = generate_gemini_seeds()
    
    results = await asyncio.gather(qwen_task, deepseek_task, gemini_task)
    
    all_seeds = []
    for res in results:
        all_seeds.extend(res)
        
    output_path = Path("data/queries/candidate_seeds_v2.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_seeds, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Generated {len(all_seeds)} candidate seeds across 3 models (expected ~90).")
    print(f"Output saved to: {output_path}")
    print("Review the seeds and curate your final set for data/queries/collection_queries.json")

if __name__ == "__main__":
    asyncio.run(main())
