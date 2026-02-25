"""log_meal tool — persist a meal to the SQLite database."""

from datetime import datetime

from src.tools.get_food_nutrition import get_food_nutrition
from src.utils.db import get_connection
from src.utils.logger import logger

USER_ID = "default"


def log_meal(meal_type: str, foods: list, timestamp: str = None) -> dict:
    """Persist a single meal record to user history.

    Args:
        meal_type: One of breakfast / lunch / dinner / snack.
        foods: List of {food_name: str, amount_grams: float}.
        timestamp: ISO-8601 string; defaults to now.

    Returns:
        {status, meal_id, total_calories} on success.
    """
    valid_types = ["breakfast", "lunch", "dinner", "snack"]
    if meal_type not in valid_types:
        return {
            "status": "error",
            "error_type": "invalid_meal_type",
            "message": f"meal_type must be one of {valid_types}",
        }

    if not foods:
        return {
            "status": "error",
            "error_type": "missing_required_field",
            "message": "foods list cannot be empty",
        }

    # Compute nutrition per food item
    calc = get_food_nutrition(foods)
    if calc.get("status") == "error":
        return calc

    breakdown = calc["data"]["breakdown"]
    total_calories = calc["data"]["total"]["calories_kcal"]

    logged_at = timestamp or datetime.now().isoformat(timespec="seconds")

    conn = get_connection()
    if not conn:
        return {"status": "error", "error_type": "db_error", "message": "Database not available"}

    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO meal_logs (user_id, meal_type, logged_at) VALUES (?, ?, ?)",
            (USER_ID, meal_type, logged_at),
        )
        log_id = cur.lastrowid

        # Insert one row per food in the breakdown
        food_inputs = {f["food_name"].lower(): f for f in foods}
        for item in breakdown:
            matched_key = next(
                (k for k in food_inputs if k in item["food_name"].lower() or item["food_name"].lower() in k),
                None,
            )
            amount_g = food_inputs[matched_key]["amount_grams"] if matched_key else 100.0
            cur.execute(
                """INSERT INTO meal_log_items
                   (log_id, food_name, amount_grams, calories_kcal, protein_g, fat_g, carbs_g, fiber_g)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    log_id,
                    item["food_name"],
                    amount_g,
                    item.get("calories_kcal", 0),
                    item.get("protein_g", 0),
                    item.get("fat_g", 0),
                    item.get("carbs_g", 0),
                    item.get("fiber_g", 0),
                ),
            )

        conn.commit()
        logger.info(f"log_meal: log_id={log_id}, type={meal_type}, kcal={total_calories}")

        return {
            "status": "success",
            "meal_id": str(log_id),
            "total_calories": total_calories,
        }

    except Exception as e:
        logger.error(f"log_meal error: {e}")
        return {"status": "error", "error_type": "internal_error", "message": str(e)}
    finally:
        conn.close()
