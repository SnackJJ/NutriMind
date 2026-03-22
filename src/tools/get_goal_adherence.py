"""get_goal_adherence tool — analyse adherence to nutrition goals.

DEPRECATED [2026-03-13]: Merged into get_history(compare_to_goal=True).
This file is retained for reference only. Do not use in new trajectories.
"""

from datetime import date, timedelta

from src.utils.db import get_connection
from src.utils.logger import logger

USER_ID = "default"

# Map metric names to daily_summary column names
METRIC_COLUMNS = {
    "calories": "total_calories",
    "protein":  "total_protein_g",
    "fat":      "total_fat_g",
    "carbs":    "total_carbs_g",
}

# ±10 % window counts as "within target"
_TOLERANCE = 0.10


def _analyse_metric(daily_values: list[float], target: float) -> dict:
    """Compute adherence stats for a single metric from a list of daily actuals."""
    n = len(daily_values)
    if n == 0:
        return {
            "target_value": target,
            "daily_average": None,
            "days_within_target": 0,
            "days_over_target": 0,
            "days_under_target": 0,
            "adherence_pct": 0.0,
            "avg_deviation": None,
            "trend": "stable",
        }

    days_within = sum(1 for v in daily_values if abs(v - target) / target <= _TOLERANCE)
    days_over   = sum(1 for v in daily_values if v > target * (1 + _TOLERANCE))
    days_under  = sum(1 for v in daily_values if v < target * (1 - _TOLERANCE))

    daily_average = round(sum(daily_values) / n, 1)
    adherence_pct = round(days_within / n * 100, 1)
    avg_deviation = round(daily_average - target, 1)

    # Trend: compare first-half vs second-half average
    trend = "stable"
    if n >= 4:
        mid = n // 2
        first_avg  = sum(daily_values[:mid]) / mid
        second_avg = sum(daily_values[mid:]) / (n - mid)
        if second_avg > first_avg * 1.05:
            trend = "increasing"
        elif second_avg < first_avg * 0.95:
            trend = "decreasing"

    return {
        "target_value": target,
        "daily_average": daily_average,
        "days_within_target": days_within,
        "days_over_target": days_over,
        "days_under_target": days_under,
        "adherence_pct": adherence_pct,
        "avg_deviation": avg_deviation,
        "trend": trend,
    }


def get_goal_adherence(days: int = 7, metric: str = "all") -> dict:
    """Analyse adherence to nutrition goals over a specified period.

    Args:
        days:   Number of past days to analyse (1–90).
        metric: calories / protein / fat / carbs / all.

    Returns:
        {status, data: {period, metric(s): {target_value, daily_average, ...}}}
    """
    valid_metrics = ["calories", "protein", "fat", "carbs", "all"]
    if metric not in valid_metrics:
        return {
            "status": "error",
            "error_type": "invalid_metric",
            "message": f"Invalid metric '{metric}'. Valid metrics: {valid_metrics}",
        }

    if days < 1 or days > 90:
        return {
            "status": "error",
            "error_type": "invalid_date_range",
            "message": f"Days must be between 1 and 90. Got {days}.",
        }

    conn = get_connection()
    if not conn:
        return {"status": "error", "error_type": "db_error", "message": "Database not available"}

    metrics_to_report = ["calories", "protein", "fat", "carbs"] if metric == "all" else [metric]
    start_date = (date.today() - timedelta(days=days - 1)).isoformat()
    today = date.today().isoformat()

    try:
        # Fetch targets from user_goals (fall back to sensible defaults)
        goal_rows = conn.execute(
            "SELECT metric, target_value FROM user_goals WHERE user_id = ? AND metric IN ({})".format(
                ",".join("?" * len(metrics_to_report))
            ),
            (USER_ID, *metrics_to_report),
        ).fetchall()
        targets = {r["metric"]: float(r["target_value"]) for r in goal_rows}

        # Apply defaults for any missing metrics
        _defaults = {"calories": 2000.0, "protein": 90.0, "fat": 65.0, "carbs": 250.0}
        for m in metrics_to_report:
            if m not in targets:
                targets[m] = _defaults[m]

        # Fetch daily summary rows for the period
        rows = conn.execute(
            """SELECT total_calories, total_protein_g, total_fat_g, total_carbs_g
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

        # Build per-metric value lists
        col_map = {
            "calories": "total_calories",
            "protein":  "total_protein_g",
            "fat":      "total_fat_g",
            "carbs":    "total_carbs_g",
        }
        results = {}
        for m in metrics_to_report:
            col = col_map[m]
            values = [float(r[col] or 0) for r in rows]
            results[m] = _analyse_metric(values, targets[m])

        period = f"Last {days} days ({start_date} to {today})"
        logger.info(f"get_goal_adherence: days={days}, metric={metric}, rows={len(rows)}")

        if metric == "all":
            return {
                "status": "success",
                "data": {
                    "period": period,
                    "metrics": results,
                },
            }
        else:
            return {
                "status": "success",
                "data": {
                    "period": period,
                    "metric": metric,
                    **results[metric],
                },
            }

    except Exception as e:
        logger.error(f"get_goal_adherence error: {e}")
        return {"status": "error", "error_type": "internal_error", "message": str(e)}
    finally:
        conn.close()
