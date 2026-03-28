"""get_today_summary tool — today's nutritional snapshot."""

from datetime import date

from src.utils.db import get_connection
from src.utils.logger import logger

USER_ID = "default"


def get_today_summary() -> dict:
    """Return a nutritional snapshot for the current day.

    Returns:
        {status, data: {date, total_calories, calorie_budget, remaining_calories,
                        protein_g, fat_g, carbs_g, fiber_g, meal_count, food_summary,
                        meals_logged: [{meal_id, meal_type, logged_at, food_names,
                                        calories_kcal, protein_g, fat_g, carbs_g}]}}
    """
    conn = get_connection()
    if not conn:
        return {"status": "error", "error_type": "db_error", "message": "Database not available"}

    today = date.today().isoformat()

    try:
        # Daily totals from the summary view
        row = conn.execute(
            """SELECT total_calories, total_protein_g, total_fat_g,
                      total_carbs_g, total_fiber_g, meal_count, food_summary
               FROM daily_summary
               WHERE user_id = ? AND log_date = ?""",
            (USER_ID, today),
        ).fetchone()

        # Calorie budget: prefer user_goals, fall back to tdee_kcal, then 2000
        goal_row = conn.execute(
            "SELECT target_value FROM user_goals WHERE user_id = ? AND metric = 'calories'",
            (USER_ID,),
        ).fetchone()

        if goal_row:
            calorie_budget = float(goal_row["target_value"])
        else:
            profile = conn.execute(
                "SELECT tdee_kcal FROM user_profiles WHERE user_id = ?", (USER_ID,)
            ).fetchone()
            calorie_budget = float(profile["tdee_kcal"]) if profile and profile["tdee_kcal"] else 2000.0

        if not row:
            logger.info(f"get_today_summary: {today}, no meals logged — returning default zeros")
            return {
                "status": "success",
                "data": {
                    "date":               today,
                    "total_calories":     0.0,
                    "calorie_budget":     calorie_budget,
                    "remaining_calories": calorie_budget,
                    "protein_g":  0.0,
                    "fat_g":      0.0,
                    "carbs_g":    0.0,
                    "fiber_g":    0.0,
                    "meal_count": 0,
                    "food_summary": "",
                    "meals_logged": [],
                },
            }

        total_calories = float(row["total_calories"] or 0)

        # Per-meal breakdown
        meal_rows = conn.execute(
            """SELECT ml.log_id, ml.meal_type, ml.logged_at,
                      ROUND(SUM(mli.calories_kcal), 1) AS kcal,
                      ROUND(SUM(mli.protein_g), 1)     AS protein_g,
                      ROUND(SUM(mli.fat_g), 1)         AS fat_g,
                      ROUND(SUM(mli.carbs_g), 1)       AS carbs_g,
                      GROUP_CONCAT(mli.food_name, ', ') AS food_names
               FROM meal_logs ml
               JOIN meal_log_items mli ON ml.log_id = mli.log_id
               WHERE ml.user_id = ? AND DATE(ml.logged_at) = ?
               GROUP BY ml.log_id
               ORDER BY ml.logged_at""",
            (USER_ID, today),
        ).fetchall()

        meals_logged = [
            {
                "meal_id":   str(m["log_id"]),
                "meal_type": m["meal_type"],
                "logged_at": m["logged_at"],
                "food_names":    m["food_names"] or "",
                "calories_kcal": float(m["kcal"] or 0),
                "protein_g":     float(m["protein_g"] or 0),
                "fat_g":         float(m["fat_g"] or 0),
                "carbs_g":       float(m["carbs_g"] or 0),
            }
            for m in meal_rows
        ]

        logger.info(f"get_today_summary: {today}, {total_calories} kcal, {len(meals_logged)} meals")

        return {
            "status": "success",
            "data": {
                "date":               today,
                "total_calories":     total_calories,
                "calorie_budget":     calorie_budget,
                "remaining_calories": round(calorie_budget - total_calories, 1),
                "protein_g":  float(row["total_protein_g"] or 0),
                "fat_g":      float(row["total_fat_g"] or 0),
                "carbs_g":    float(row["total_carbs_g"] or 0),
                "fiber_g":    float(row["total_fiber_g"] or 0),
                "meal_count": row["meal_count"],
                "food_summary": row["food_summary"] or "",
                "meals_logged": meals_logged,
            },
        }

    except Exception as e:
        logger.error(f"get_today_summary error: {e}")
        return {"status": "error", "error_type": "internal_error", "message": str(e)}
    finally:
        conn.close()
