"""Query-aware mock data generation for user-state tools.

Used ONLY during trajectory collection to generate semantically consistent
mock data for get_today_summary and get_history. Production tools are unchanged.

See phase2.6_trajectory_collection.md Section 2.2 for design rationale.
"""

import json
import random
from datetime import date, timedelta
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

# Use a cheap, fast model for mock generation
MOCK_GENERATOR_MODEL = "qwen-plus"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazy init to avoid import-time API key issues."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.qwen_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    return _client


# =============================================================================
# Mock State Generation
# =============================================================================

_MOCK_STATE_PROMPT = """Based on this user query, generate realistic nutrition tracking data.

Query: "{query}"

Generate a JSON object with these fields:
- calorie_budget: daily calorie goal (1200-3500, typical 2000)
- protein_target: daily protein goal in grams (60-200)
- fat_target: daily fat goal in grams (30-150)
- carbs_target: daily carbs goal in grams (20-400)
- today_eaten: calories consumed today (0 to calorie_budget * 1.3)
- protein_g: protein eaten today (0 to protein_target * 1.5)
- fat_g: fat eaten today (0 to fat_target * 1.5)
- carbs_g: carbs eaten today (0 to carbs_target * 1.5)
- fiber_g: fiber eaten today (5-50)
- meal_count: number of meals logged today (0-5)
- history_trend: "increasing" | "decreasing" | "stable"
- history_7d_avg_calories: 7-day average (similar to today_eaten ±20%)
- goal_adherence_pct: percentage of days within target (40-95)
- context_foods: A list of 8-10 food names that are SEMANTICALLY CONSISTENT with the query context (e.g., if query is about keto, include steak, eggs, avocado; if vegan, include beans, tofu, quinoa).

IMPORTANT: Make all values and foods SEMANTICALLY CONSISTENT with the query.
Examples:
- "keto" → carbs_target should be low (<50g), context_foods should be keto-friendly.
- "I've been eating too much" → today_eaten > calorie_budget.
- "bulk up" → calorie_budget and protein_target should be on the higher side.

Return ONLY valid JSON, no explanation."""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5), reraise=True)
def _call_mock_generator(query: str) -> dict[str, Any]:
    """Call LLM to generate query-aware mock state."""
    client = _get_client()
    response = client.chat.completions.create(
        model=MOCK_GENERATOR_MODEL,
        messages=[
            {"role": "user", "content": _MOCK_STATE_PROMPT.format(query=query)}
        ],
        temperature=0.7,
    )
    content = response.choices[0].message.content.strip()

    # Handle markdown code blocks
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    return json.loads(content)


def _generate_fallback_state() -> dict[str, Any]:
    """Generate random but realistic fallback state when LLM fails."""
    budget = random.choice([1500, 1800, 2000, 2200, 2500])
    p_target = random.choice([80, 100, 120, 150])
    f_target = random.choice([50, 65, 80])
    c_target = random.choice([150, 200, 250, 300])
    
    eaten_ratio = random.uniform(0.3, 1.1)
    today_eaten = round(budget * eaten_ratio)

    return {
        "calorie_budget": budget,
        "protein_target": p_target,
        "fat_target": f_target,
        "carbs_target": c_target,
        "today_eaten": today_eaten,
        "protein_g": round(p_target * random.uniform(0.5, 1.2), 1),
        "fat_g": round(f_target * random.uniform(0.5, 1.2), 1),
        "carbs_g": round(c_target * random.uniform(0.5, 1.2), 1),
        "fiber_g": round(random.uniform(10, 35), 1),
        "meal_count": random.randint(1, 4),
        "history_trend": random.choice(["stable", "increasing", "decreasing"]),
        "history_7d_avg_calories": round(budget * random.uniform(0.85, 1.05)),
        "goal_adherence_pct": round(random.uniform(50, 85), 1),
        "context_foods": ["Oatmeal", "Chicken Breast", "Steamed Rice", "Apple", "Greek Yogurt", "Salad", "Coffee", "Egg", "Protein Shake"],
    }


def generate_mock_state(query: str) -> dict[str, Any]:
    """Generate mock user state that is semantically consistent with the query.

    Args:
        query: The user's original query

    Returns:
        Dict with fields needed by mock_today_summary and mock_history
    """
    try:
        state = _call_mock_generator(query)
        # Validate required fields exist
        required = ["calorie_budget", "today_eaten", "protein_g", "fat_g", "carbs_g", "context_foods"]
        if not all(k in state for k in required):
            raise ValueError(f"Missing required fields in LLM response: {state}")
        return state
    except Exception as e:
        print(f"[MOCK] LLM generation failed, using fallback: {e}")
        return _generate_fallback_state()


# =============================================================================
# Mock Tool Responses
# =============================================================================

def mock_today_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Generate mock response matching get_today_summary format.

    Args:
        state: Mock state from generate_mock_state()

    Returns:
        Dict in the same format as get_today_summary()
    """
    today = date.today().isoformat()
    budget = state.get("calorie_budget", 2000)
    eaten = state.get("today_eaten", 1200)
    meal_count = state.get("meal_count", 2)
    common_foods = state.get("context_foods", ["Oatmeal", "Chicken Breast", "Steamed Rice", "Apple", "Greek Yogurt", "Salad", "Coffee", "Egg", "Protein Shake"])
    
    # Generate realistic meal breakdown if meals were logged
    meals_logged = []
    
    if meal_count > 0:
        meal_types = ["breakfast", "lunch", "dinner", "snack"][:meal_count]
        cal_per_meal = eaten / meal_count

        for i, meal_type in enumerate(meal_types):
            hour = 8 + i * 4  # breakfast at 8, lunch at 12, etc.
            f1, f2 = random.sample(common_foods, 2)
            meals_logged.append({
                "meal_id": f"mock_{i+1}",
                "meal_type": meal_type,
                "logged_at": f"{today}T{hour:02d}:00:00",
                "food_names": f"{f1}, {f2}",
                "calories_kcal": round(cal_per_meal * random.uniform(0.8, 1.2), 1),
                "protein_g": round(state.get("protein_g", 60) / meal_count * random.uniform(0.7, 1.3), 1),
                "fat_g": round(state.get("fat_g", 50) / meal_count * random.uniform(0.7, 1.3), 1),
                "carbs_g": round(state.get("carbs_g", 150) / meal_count * random.uniform(0.7, 1.3), 1),
            })

    return {
        "status": "success",
        "data": {
            "date": today,
            "total_calories": float(eaten),
            "calorie_budget": float(budget),
            "remaining_calories": round(budget - eaten, 1),
            "protein_g": float(state.get("protein_g", 60)),
            "fat_g": float(state.get("fat_g", 50)),
            "carbs_g": float(state.get("carbs_g", 150)),
            "fiber_g": float(state.get("fiber_g", 20)),
            "meal_count": meal_count,
            "food_summary": ", ".join([m["food_names"] for m in meals_logged]),
            "meals_logged": meals_logged,
        }
    }


def mock_history(
    state: dict[str, Any],
    days: int = 7,
    metric: str = "all",
    compare_to_goal: bool = False
) -> dict[str, Any]:
    """Generate mock response matching get_history format.

    Args:
        state: Mock state from generate_mock_state()
        days: Number of days to include
        metric: Which metric(s) to return
        compare_to_goal: Whether to include goal adherence analysis

    Returns:
        Dict in the same format as get_history()
    """
    days = min(max(days, 1), 90)
    today = date.today()
    start_date = today - timedelta(days=days - 1)

    base_calories = state.get("history_7d_avg_calories", state.get("today_eaten", 1800))
    trend = state.get("history_trend", "stable")
    common_foods = state.get("context_foods", ["Chicken salad", "Greek yogurt", "Apple", "Omelet", "Pasta", "Banana", "Protein shake"])

    # Generate daily breakdown with trend
    daily_breakdown = []
    for i in range(days):
        day_date = start_date + timedelta(days=i)

        # Apply trend factor
        if trend == "increasing":
            trend_factor = 0.9 + (i / days) * 0.2  # 0.9 → 1.1
        elif trend == "decreasing":
            trend_factor = 1.1 - (i / days) * 0.2  # 1.1 → 0.9
        else:
            trend_factor = 1.0

        day_calories = round(base_calories * trend_factor * random.uniform(0.85, 1.15), 1)

        f1, f2 = random.sample(common_foods, 2)
        day_data = {
            "date": day_date.isoformat(),
            "calories_kcal": day_calories,
            "protein_g": round(state.get("protein_g", 80) * random.uniform(0.8, 1.2), 1),
            "fat_g": round(state.get("fat_g", 65) * random.uniform(0.8, 1.2), 1),
            "carbs_g": round(state.get("carbs_g", 200) * random.uniform(0.8, 1.2), 1),
            "fiber_g": round(state.get("fiber_g", 25) * random.uniform(0.8, 1.2), 1),
            "meal_count": random.randint(2, 4),
            "food_summary": f"{f1}, {f2}",
        }
        daily_breakdown.append(day_data)

    # Calculate averages
    n = len(daily_breakdown)
    daily_averages = {
        "calories_kcal": round(sum(d["calories_kcal"] for d in daily_breakdown) / n, 1),
        "protein_g": round(sum(d["protein_g"] for d in daily_breakdown) / n, 1),
        "fat_g": round(sum(d["fat_g"] for d in daily_breakdown) / n, 1),
        "carbs_g": round(sum(d["carbs_g"] for d in daily_breakdown) / n, 1),
    }

    # Filter columns if specific metric requested
    if metric != "all":
        display_key = f"{metric}_kcal" if metric == "calories" else f"{metric}_g"
        daily_breakdown = [{"date": d["date"], display_key: d[display_key]} for d in daily_breakdown]

    result_data = {
        "period": f"Last {days} days ({start_date.isoformat()} to {today.isoformat()})",
        "days_with_data": n,
        "daily_averages": daily_averages,
        "trend": trend,
        "daily_breakdown": daily_breakdown,
    }

    # Add goal adherence if requested
    if compare_to_goal:
        adherence_pct = state.get("goal_adherence_pct", 70)
        budget = state.get("calorie_budget", 2000)

        days_within = round(n * adherence_pct / 100)
        days_over = random.randint(0, n - days_within)
        days_under = n - days_within - days_over

        metrics_to_check = ["calories", "protein", "fat", "carbs"] if metric == "all" else [metric]
        goal_adherence = {}

        for m in metrics_to_check:
            if m == "calories":
                target = budget
                avg = daily_averages["calories_kcal"]
            else:
                # Use targets from state to ensure consistency
                target = state.get(f"{m}_target", 100.0)
                avg = daily_averages.get(f"{m}_g", target)

            goal_adherence[m] = {
                "target_value": target,
                "daily_average": avg,
                "days_within_target": days_within,
                "days_over_target": days_over,
                "days_under_target": days_under,
                "adherence_pct": adherence_pct,
                "avg_deviation": round(avg - target, 1),
            }

        result_data["goal_adherence"] = goal_adherence

    return {"status": "success", "data": result_data}
