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


def get_history(days: int = 7, metric: str = "all") -> dict:
    """Query multi-day nutritional history and trends.

    Args:
        days: Number of past days to include (1–90).
        metric: calories / protein / fat / carbs / all.

    Returns:
        {status, data: {period, daily_averages, trend, daily_breakdown}}
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
                      total_carbs_g, total_fiber_g, meal_count
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
            }
            for r in rows
        ]

        n = len(daily_breakdown)
        daily_averages = {
            "calories_kcal": round(sum(d["calories_kcal"] for d in daily_breakdown) / n, 1),
            "protein_g":     round(sum(d["protein_g"]     for d in daily_breakdown) / n, 1),
            "fat_g":         round(sum(d["fat_g"]         for d in daily_breakdown) / n, 1),
            "carbs_g":       round(sum(d["carbs_g"]       for d in daily_breakdown) / n, 1),
        }

        # Filter breakdown columns when specific metric requested
        if metric != "all":
            col = METRIC_COLUMNS[metric]
            key = list(METRIC_COLUMNS.keys())[list(METRIC_COLUMNS.values()).index(col)]
            display_key = f"{key}_kcal" if key == "calories" else f"{key}_g"
            daily_breakdown = [{"date": d["date"], display_key: d[display_key if display_key in d else f"{key}_g"]} for d in daily_breakdown]

        # Simple trend: compare first half vs second half average calories
        trend = "stable"
        if n >= 4:
            mid = n // 2
            first_half_avg = sum(d["calories_kcal"] for d in daily_breakdown[:mid] if "calories_kcal" in d) / mid if "calories_kcal" in daily_breakdown[0] else 0
            second_half_avg = sum(d["calories_kcal"] for d in daily_breakdown[mid:] if "calories_kcal" in d) / (n - mid) if "calories_kcal" in daily_breakdown[0] else 0
            if second_half_avg > first_half_avg * 1.05:
                trend = "increasing"
            elif second_half_avg < first_half_avg * 0.95:
                trend = "decreasing"

        logger.info(f"get_history: {days} days, {n} data points, metric={metric}")

        return {
            "status": "success",
            "data": {
                "period": f"Last {days} days ({start_date} to {today})",
                "days_with_data": n,
                "daily_averages": daily_averages,
                "trend": trend,
                "daily_breakdown": daily_breakdown,
            },
        }

    except Exception as e:
        logger.error(f"get_history error: {e}")
        return {"status": "error", "error_type": "internal_error", "message": str(e)}
    finally:
        conn.close()
