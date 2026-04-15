"""
TRL environment_factory for NutriMind GRPO agent training.

Implements the environment_factory protocol required by TRL's GRPOTrainer:
- reset(**kwargs) is called at the start of each rollout
- Public methods (no _ prefix) are auto-discovered as tools via Qwen3 chat template
- Tool methods must have type-hinted args, Google docstrings, and return str

TRL handles the full multi-turn loop automatically:
    generate → detect <tool_call> → call env method → inject <tool_response> → continue

Tool categories:
    Real DB/RAG:     get_food_nutrition, retrieve_knowledge
    State rendering: get_today_summary, get_history (read from env_state metadata)
    Write mocking:   log_meal, set_goal (return success without DB writes)

All calls go through ToolCache for deterministic results within a GRPO group.
"""

import hashlib
import json
import logging
from datetime import date, timedelta
from typing import Any

from src.training.grpo.tool_cache import ToolCache

logger = logging.getLogger(__name__)


def _render_today_summary(env_state: dict) -> dict:
    """Render get_today_summary output from env_state metadata.

    Output format matches src/tools/get_today_summary.get_today_summary().
    Uses a fixed date string to ensure determinism across rollouts.
    """
    meals = env_state.get("meals_today", [])
    goals = env_state.get("user_goals", {})
    profile = env_state.get("user_profile", {})
    budget = float(goals.get("calories", profile.get("tdee_kcal", 2000)))

    total_cal = sum(m.get("calories", 0) for m in meals)
    total_pro = sum(m.get("protein_g", 0) for m in meals)
    total_fat = sum(m.get("fat_g", 0) for m in meals)
    total_carbs = sum(m.get("carbs_g", 0) for m in meals)
    total_fiber = sum(m.get("fiber_g", 0) for m in meals)

    # Fixed date for determinism (actual date doesn't affect reward)
    today_str = "2026-04-15"
    meals_logged = []
    for i, m in enumerate(meals):
        food_names = ", ".join(f.get("name", "unknown") for f in m.get("foods", []))
        hour = min(8 + i * 4, 23)  # Cap at 23 to avoid invalid timestamps
        meals_logged.append({
            "meal_id": str(i + 1),
            "meal_type": m.get("meal_type", "snack"),
            "logged_at": f"{today_str}T{hour:02d}:00:00",
            "food_names": food_names,
            "calories_kcal": float(m.get("calories", 0)),
            "protein_g": float(m.get("protein_g", 0)),
            "fat_g": float(m.get("fat_g", 0)),
            "carbs_g": float(m.get("carbs_g", 0)),
        })

    return {
        "status": "success",
        "data": {
            "date": today_str,
            "total_calories": float(total_cal),
            "calorie_budget": budget,
            "remaining_calories": round(budget - total_cal, 1),
            "protein_g": float(total_pro),
            "fat_g": float(total_fat),
            "carbs_g": float(total_carbs),
            "fiber_g": float(total_fiber),
            "meal_count": len(meals),
            "food_summary": "; ".join(ml["food_names"] for ml in meals_logged),
            "meals_logged": meals_logged,
        },
    }


def _render_history(env_state: dict, days: int, metric: str, compare_to_goal: bool) -> dict:
    """Render get_history output from env_state metadata.

    Output format matches src/tools/get_history.get_history().
    """
    history = env_state.get("meal_history", [])
    goals = env_state.get("user_goals", {})

    # Clamp days
    days = max(1, min(days, 90))
    history = history[:days]

    if not history:
        return {
            "status": "error",
            "error_type": "no_data_in_range",
            "message": f"No logged meals in the past {days} days.",
        }

    today = date(2026, 4, 15)  # Fixed date for determinism
    start_date = today - timedelta(days=days - 1)

    daily_breakdown = []
    for i, h in enumerate(history):
        day_date = h.get("date", (start_date + timedelta(days=i)).isoformat())
        daily_breakdown.append({
            "date": day_date if isinstance(day_date, str) and "-" in str(day_date) else (start_date + timedelta(days=i)).isoformat(),
            "calories_kcal": float(h.get("calories", 0)),
            "protein_g": float(h.get("protein_g", 0)),
            "fat_g": float(h.get("fat_g", 0)),
            "carbs_g": float(h.get("carbs_g", 0)),
            "fiber_g": float(h.get("fiber_g", 0)),
            "meal_count": h.get("meal_count", 3),
            "food_summary": h.get("food_summary", ""),
        })

    n = len(daily_breakdown)
    daily_averages = {
        "calories_kcal": round(sum(d["calories_kcal"] for d in daily_breakdown) / n, 1),
        "protein_g": round(sum(d["protein_g"] for d in daily_breakdown) / n, 1),
        "fat_g": round(sum(d["fat_g"] for d in daily_breakdown) / n, 1),
        "carbs_g": round(sum(d["carbs_g"] for d in daily_breakdown) / n, 1),
        "fiber_g": round(sum(d["fiber_g"] for d in daily_breakdown) / n, 1),
    }

    # Trend: compare first vs second half
    trend = "stable"
    if n >= 4:
        trend_key = "calories_kcal" if metric == "all" else (
            f"{metric}_kcal" if metric == "calories" else f"{metric}_g"
        )
        mid = n // 2
        first_avg = sum(d.get(trend_key, 0) for d in daily_breakdown[:mid]) / mid
        second_avg = sum(d.get(trend_key, 0) for d in daily_breakdown[mid:]) / (n - mid)
        if first_avg > 0:
            if second_avg > first_avg * 1.05:
                trend = "increasing"
            elif second_avg < first_avg * 0.95:
                trend = "decreasing"

    # Filter columns for specific metric
    if metric != "all":
        display_key = f"{metric}_kcal" if metric == "calories" else f"{metric}_g"
        daily_breakdown = [{"date": d["date"], display_key: d.get(display_key, 0)} for d in daily_breakdown]

    result_data: dict[str, Any] = {
        "period": f"Last {days} days ({start_date.isoformat()} to {today.isoformat()})",
        "days_with_data": n,
        "daily_averages": daily_averages,
        "trend": trend,
        "daily_breakdown": daily_breakdown,
    }

    if compare_to_goal:
        _DEFAULTS = {"calories": 2000.0, "protein": 90.0, "fat": 65.0, "carbs": 250.0}
        metrics_to_check = ["calories", "protein", "fat", "carbs"] if metric == "all" else [metric]
        tolerance = 0.10

        # Rebuild full breakdown for adherence calc
        full_bd = []
        for h in env_state.get("meal_history", [])[:days]:
            full_bd.append({
                "calories_kcal": float(h.get("calories", 0)),
                "protein_g": float(h.get("protein_g", 0)),
                "fat_g": float(h.get("fat_g", 0)),
                "carbs_g": float(h.get("carbs_g", 0)),
            })

        col_map = {"calories": "calories_kcal", "protein": "protein_g", "fat": "fat_g", "carbs": "carbs_g"}
        goal_adherence = {}
        for m in metrics_to_check:
            target = float(goals.get(m, _DEFAULTS.get(m, 100)))
            values = [d.get(col_map.get(m, m), 0) for d in full_bd]
            if not values:
                continue
            avg_val = sum(values) / len(values)
            within = sum(1 for v in values if abs(v - target) / max(target, 1) <= tolerance)
            over = sum(1 for v in values if v > target * (1 + tolerance))
            under = len(values) - within - over
            goal_adherence[m] = {
                "target_value": target,
                "daily_average": round(avg_val, 1),
                "days_within_target": within,
                "days_over_target": over,
                "days_under_target": under,
                "adherence_pct": round(within / len(values) * 100, 1),
                "avg_deviation": round(avg_val - target, 1),
            }

        result_data["goal_adherence"] = goal_adherence

    return {"status": "success", "data": result_data}


class NutriMindToolEnv:
    """TRL-compatible training environment for NutriMind.

    Each GRPO rollout creates one instance. Public methods are automatically
    registered as tools by TRL via Qwen3's chat template.

    Attributes (private, not exposed as tools):
        _env_state: User profile/meals/history from dataset metadata.
        _tool_history: Ordered record of tool calls for reward computation.
        _tool_calls_count: Number of tools invoked this rollout.
    """

    # Class-level cache shared across all instances in the same prompt group.
    _cache: ToolCache = ToolCache()

    def reset(self, **kwargs) -> str | None:
        """Initialize the environment for a new rollout.

        Called by TRL at the start of each generation. Receives all dataset
        row fields as keyword arguments.

        Args:
            **kwargs: Dataset fields including prompt, env_state, tier, query, etc.

        Returns:
            None (no extra text appended to user message).
        """
        # Parse env_state
        raw = kwargs.get("env_state", "{}")
        self._env_state: dict = json.loads(raw) if isinstance(raw, str) else (raw or {})

        # Metadata for reward function
        self._tier: str = kwargs.get("tier", "T1")
        self._query: str = kwargs.get("query", "")
        self._difficulty: str = kwargs.get("difficulty", "medium")
        self._optimal_steps: int = int(kwargs.get("optimal_steps", 1))
        self._branch_condition: str = kwargs.get("branch_condition", "")

        # Trajectory tracking
        self._tool_history: list[dict] = []
        self._tool_calls_count: int = 0

        # Cache group management — same prompt = same group
        prompt = kwargs.get("prompt", "")
        group_id = hashlib.md5(str(prompt).encode()).hexdigest()[:16]
        NutriMindToolEnv._cache.new_group(group_id)

        return None

    # ------------------------------------------------------------------
    # Real tools (call actual DB / RAG)
    # ------------------------------------------------------------------

    def get_food_nutrition(self, foods: list[dict]) -> str:
        """Look up nutrition data for one or more foods from the USDA database.

        Args:
            foods: List of foods to query. Each item should have 'food_name' (str)
                and 'amount_grams' (float). Example:
                [{"food_name": "chicken breast", "amount_grams": 150}]

        Returns:
            JSON with total macros, per-food breakdown, and macro ratios.
        """
        from src.tools.get_food_nutrition import get_food_nutrition as _real

        result = self._cached_call("get_food_nutrition", {"foods": foods}, lambda: _real(foods))
        return result

    def retrieve_knowledge(self, query: str, mode: str = "hybrid", top_k: int = 3) -> str:
        """Search the nutrition knowledge base for dietary guidelines and research.

        Args:
            query: Natural-language question or topic to search for.
            mode: Retrieval strategy — "hybrid" (default), "semantic", or "keyword".
            top_k: Maximum number of passages to return (default 3).

        Returns:
            JSON with ranked passages including content, source, and relevance score.
        """
        from src.tools.retrieve_knowledge import retrieve_knowledge as _real

        args = {"query": query, "mode": mode, "top_k": top_k}
        result = self._cached_call("retrieve_knowledge", args, lambda: _real(query, mode, top_k))
        return result

    # ------------------------------------------------------------------
    # State-rendering tools (read from env_state metadata)
    # ------------------------------------------------------------------

    def get_today_summary(self) -> str:
        """Get today's nutritional intake summary and remaining calorie budget.

        Returns:
            JSON with total calories, macronutrients, remaining budget,
            and a list of logged meals.
        """
        result = self._cached_call(
            "get_today_summary", {},
            lambda: _render_today_summary(self._env_state),
        )
        return result

    def get_history(self, days: int = 7, metric: str = "all", compare_to_goal: bool = False) -> str:
        """Query multi-day nutritional history, trends, and goal adherence.

        Args:
            days: Number of past days to include (1–90, default 7).
            metric: Which nutrient to focus on — "calories", "protein",
                "fat", "carbs", or "all" (default).
            compare_to_goal: If true, include goal-adherence analysis
                comparing daily intake against targets.

        Returns:
            JSON with daily averages, trend direction, daily breakdown,
            and optional goal adherence stats.
        """
        args = {"days": days, "metric": metric, "compare_to_goal": compare_to_goal}
        result = self._cached_call(
            "get_history", args,
            lambda: _render_history(self._env_state, days, metric, compare_to_goal),
        )
        return result

    # ------------------------------------------------------------------
    # Write-mocking tools (return success without touching DB)
    # ------------------------------------------------------------------

    def log_meal(self, meal_type: str, foods: list[dict]) -> str:
        """Record a meal entry to the user's food diary.

        Args:
            meal_type: One of "breakfast", "lunch", "dinner", or "snack".
            foods: List of foods consumed, each with 'food_name' (str)
                and 'amount_grams' (float).

        Returns:
            JSON confirmation with meal_id and total calories.
        """
        args = {"meal_type": meal_type, "foods": foods}

        def _mock() -> dict:
            # Validate meal_type
            if meal_type not in ("breakfast", "lunch", "dinner", "snack"):
                return {"status": "error", "error_type": "invalid_meal_type",
                        "message": f"meal_type must be one of breakfast/lunch/dinner/snack"}
            if not foods:
                return {"status": "error", "error_type": "missing_required_field",
                        "message": "foods list cannot be empty"}

            # Use real nutrition lookup for realistic calorie values
            from src.tools.get_food_nutrition import get_food_nutrition
            nutrition = get_food_nutrition(foods)
            total_cal = 0.0
            if isinstance(nutrition, dict) and nutrition.get("status") in ("success", "partial_success"):
                total_cal = nutrition.get("data", {}).get("total", {}).get("calories_kcal", 0.0)

            meal_id = hashlib.md5(
                json.dumps({"meal_type": meal_type, "foods": foods}, sort_keys=True).encode()
            ).hexdigest()[:8]

            return {
                "status": "success",
                "meal_id": f"train_{meal_id}",
                "total_calories": round(total_cal, 1),
            }

        result = self._cached_call("log_meal", args, _mock)
        return result

    def set_goal(self, metric: str, target_value: float, goal_type: str | None = None) -> str:
        """Set or update a daily nutritional target.

        Args:
            metric: Nutrient to set — "calories", "protein", "fat", "carbs", or "fiber".
            target_value: Daily target in kcal (calories) or grams (macros).
            goal_type: Optional overall direction — "lose", "maintain", or "gain".

        Returns:
            JSON confirmation with previous and new target values.
        """
        args = {"metric": metric, "target_value": target_value, "goal_type": goal_type}

        def _mock() -> dict:
            valid_metrics = ("calories", "protein", "fat", "carbs", "fiber")
            if metric not in valid_metrics:
                return {"status": "error", "error_type": "invalid_metric",
                        "message": f"metric must be one of {valid_metrics}"}

            prev = self._env_state.get("user_goals", {}).get(metric)
            return {
                "status": "success",
                "data": {
                    "metric": metric,
                    "previous_value": prev,
                    "new_value": target_value,
                    "goal_type": goal_type or "maintain",
                },
            }

        result = self._cached_call("set_goal", args, _mock)
        return result

    # ------------------------------------------------------------------
    # Internal helpers (prefixed with _ so TRL ignores them)
    # ------------------------------------------------------------------

    def _cached_call(self, tool_name: str, args: dict, fn: callable) -> str:
        """Execute a tool call through the cache, record it, and return JSON str."""
        result_str = NutriMindToolEnv._cache.get_or_call(tool_name, args, fn)

        # Ensure result is a string
        if not isinstance(result_str, str):
            result_str = json.dumps(result_str, ensure_ascii=False, default=str)

        # Record for reward function
        self._tool_calls_count += 1
        try:
            parsed = json.loads(result_str)
        except (json.JSONDecodeError, TypeError):
            parsed = {"raw": result_str}

        self._tool_history.append({
            "tool_name": tool_name,
            "args": args,
            "result": parsed,
            "success": parsed.get("status") != "error" if isinstance(parsed, dict) else True,
        })

        return result_str
