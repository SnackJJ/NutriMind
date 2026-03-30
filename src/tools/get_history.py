"""get_history tool — multi-day nutritional history and trends."""

from datetime import date, timedelta

from src.utils.db import get_connection
from src.utils.logger import logger

USER_ID = "default"

METRIC_COLUMNS = {
    "calories": "total_calories",
    "protein":  "total_protein_g",
    "fat":      "total_fat_g",
    "carbs":    "total_carbs_g",
}

_GOAL_DEFAULTS = {"calories": 2000.0, "protein": 90.0, "fat": 65.0, "carbs": 250.0}
_TOLERANCE = 0.10  # ±10% counts as within target


def _adherence_stats(daily_values: list[float], target: float) -> dict:
    n = len(daily_values)
    days_within = sum(1 for v in daily_values if abs(v - target) / target <= _TOLERANCE)
    days_over   = sum(1 for v in daily_values if v > target * (1 + _TOLERANCE))
    days_under  = sum(1 for v in daily_values if v < target * (1 - _TOLERANCE))
    daily_avg   = round(sum(daily_values) / n, 1)
    return {
        "target_value":       target,
        "daily_average":      daily_avg,
        "days_within_target": days_within,
        "days_over_target":   days_over,
        "days_under_target":  days_under,
        "adherence_pct":      round(days_within / n * 100, 1),
        "avg_deviation":      round(daily_avg - target, 1),
    }


def get_history(days: int = 7, metric: str = "all", compare_to_goal: bool = False) -> dict:
    """Query multi-day nutritional history, trends, and optionally goal adherence.

    Args:
        days:            Number of past days to include (1–90).
        metric:          calories / protein / fat / carbs / all.
        compare_to_goal: When True, include goal adherence analysis in the response.

    Returns:
        {status, data: {period, daily_averages, trend, daily_breakdown,
                        goal_adherence (only when compare_to_goal=True)}}
    """
    valid_metrics = ["calories", "protein", "fat", "carbs", "all"]
    if metric not in valid_metrics:
        return {
            "status": "error",
            "error_type": "invalid_metric",
            "message": f"metric must be one of {valid_metrics}",
        }
    if days < 1 or days > 90:
        return {
            "status": "error",
            "error_type": "invalid_date_range",
            "message": "days must be between 1 and 90",
        }

    conn = get_connection()
    if not conn:
        return {"status": "error", "error_type": "db_error", "message": "Database not available"}

    start_date = (date.today() - timedelta(days=days - 1)).isoformat()
    today = date.today().isoformat()

    try:
        rows = conn.execute(
            """SELECT log_date, total_calories, total_protein_g, total_fat_g,
                      total_carbs_g, total_fiber_g, meal_count, food_summary
               FROM daily_summary
               WHERE user_id = ? AND log_date >= ? AND log_date <= ?
               ORDER BY log_date""",
            (USER_ID, start_date, today),
        ).fetchall()

        if not rows:
            return {
                "status": "error",
                "error_type": "no_data_in_range",
                "message": f"No logged meals in the past {days} days.",
            }

        daily_breakdown = [
            {
                "date": r["log_date"],
                "calories_kcal": float(r["total_calories"] or 0),
                "protein_g":     float(r["total_protein_g"] or 0),
                "fat_g":         float(r["total_fat_g"] or 0),
                "carbs_g":       float(r["total_carbs_g"] or 0),
                "fiber_g":       float(r["total_fiber_g"] or 0),
                "meal_count":    r["meal_count"],
                "food_summary":  r["food_summary"],
            }
            for r in rows
        ]

        n = len(daily_breakdown)
        daily_averages = {
            "calories_kcal": round(sum(d["calories_kcal"] for d in daily_breakdown) / n, 1),
            "protein_g":     round(sum(d["protein_g"]     for d in daily_breakdown) / n, 1),
            "fat_g":         round(sum(d["fat_g"]         for d in daily_breakdown) / n, 1),
            "carbs_g":       round(sum(d["carbs_g"]       for d in daily_breakdown) / n, 1),
            "fiber_g":       round(sum(d["fiber_g"]       for d in daily_breakdown) / n, 1),
        }

        # Simple trend: compare first half vs second half of the requested metric
        trend = "stable"
        if n >= 4:
            # Determine which key to compute trend on
            if metric == "all":
                trend_key = "calories_kcal"
            else:
                trend_key = f"{metric}_kcal" if metric == "calories" else f"{metric}_g"

            mid = n // 2
            first_half_avg = sum(d[trend_key] for d in daily_breakdown[:mid]) / mid
            second_half_avg = sum(d[trend_key] for d in daily_breakdown[mid:]) / (n - mid)
            if second_half_avg > first_half_avg * 1.05:
                trend = "increasing"
            elif second_half_avg < first_half_avg * 0.95:
                trend = "decreasing"

        # Filter breakdown columns when specific metric requested (after trend calculation)
        if metric != "all":
            display_key = f"{metric}_kcal" if metric == "calories" else f"{metric}_g"
            daily_breakdown = [{"date": d["date"], display_key: d[display_key]} for d in daily_breakdown]

        result_data: dict = {
            "period": f"Last {days} days ({start_date} to {today})",
            "days_with_data": n,
            "daily_averages": daily_averages,
            "trend": trend,
            "daily_breakdown": daily_breakdown,
        }

        if compare_to_goal:
            metrics_to_check = ["calories", "protein", "fat", "carbs"] if metric == "all" else [metric]
            goal_rows = conn.execute(
                "SELECT metric, target_value FROM user_goals WHERE user_id = ? AND metric IN ({})".format(
                    ",".join("?" * len(metrics_to_check))
                ),
                (USER_ID, *metrics_to_check),
            ).fetchall()
            targets = {r["metric"]: float(r["target_value"]) for r in goal_rows}
            for m in metrics_to_check:
                if m not in targets:
                    targets[m] = _GOAL_DEFAULTS[m]

            col_map = {"calories": "calories_kcal", "protein": "protein_g", "fat": "fat_g", "carbs": "carbs_g"}
            full_breakdown = [
                {
                    "calories_kcal": float(r["total_calories"] or 0),
                    "protein_g":     float(r["total_protein_g"] or 0),
                    "fat_g":         float(r["total_fat_g"] or 0),
                    "carbs_g":       float(r["total_carbs_g"] or 0),
                }
                for r in rows
            ]
            result_data["goal_adherence"] = {
                m: _adherence_stats([d[col_map[m]] for d in full_breakdown], targets[m])
                for m in metrics_to_check
            }

        logger.info(f"get_history: {days} days, {n} data points, metric={metric}, compare_to_goal={compare_to_goal}")

        return {"status": "success", "data": result_data}

    except Exception as e:
        logger.error(f"get_history error: {e}")
        return {"status": "error", "error_type": "internal_error", "message": str(e)}
    finally:
        conn.close()
