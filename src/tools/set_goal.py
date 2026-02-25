"""set_goal tool — persist a nutrition target to the database."""

from datetime import datetime

from src.utils.db import get_connection
from src.utils.logger import logger

USER_ID = "default"


def set_goal(metric: str, target_value: float, goal_type: str = None) -> dict:
    """Set or update a specific nutrition target for the user.

    Args:
        metric: One of "calories", "protein", "fat", "carbs"
        target_value: Daily target value in kcal or grams
        goal_type: Optional overall goal direction - "lose", "maintain", or "gain"

    Returns:
        {status, data: {metric, previous_value, new_value, goal_type}}
    """
    valid_metrics = ["calories", "protein", "fat", "carbs"]
    if metric not in valid_metrics:
        return {
            "status": "error",
            "error_type": "invalid_metric",
            "message": f"Invalid metric '{metric}'. Valid metrics: {valid_metrics}",
        }

    if metric == "calories" and (target_value < 1000 or target_value > 5000):
        return {
            "status": "error",
            "error_type": "value_out_of_range",
            "message": f"Calorie target must be between 1000-5000 kcal. Got {target_value}.",
        }

    if metric in ["protein", "fat", "carbs"] and target_value <= 0:
        return {
            "status": "error",
            "error_type": "value_out_of_range",
            "message": f"{metric.capitalize()} target must be > 0. Got {target_value}.",
        }

    valid_goal_types = ["lose", "maintain", "gain"]
    if goal_type and goal_type not in valid_goal_types:
        return {
            "status": "error",
            "error_type": "invalid_goal_type",
            "message": f"Invalid goal_type '{goal_type}'. Valid types: {valid_goal_types}",
        }

    conn = get_connection()
    if not conn:
        return {"status": "error", "error_type": "db_error", "message": "Database not available"}

    now = datetime.now().isoformat(timespec="seconds")

    try:
        # Get previous value
        row = conn.execute(
            "SELECT target_value FROM user_goals WHERE user_id = ? AND metric = ?",
            (USER_ID, metric),
        ).fetchone()
        previous_value = float(row["target_value"]) if row else None

        # Upsert goal
        conn.execute(
            """INSERT INTO user_goals (user_id, metric, target_value, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id, metric) DO UPDATE SET
                   target_value = excluded.target_value,
                   updated_at   = excluded.updated_at""",
            (USER_ID, metric, target_value, now),
        )

        # Persist goal_type to user profile when provided
        resolved_goal_type = goal_type
        if goal_type:
            conn.execute(
                "UPDATE user_profiles SET goal = ?, updated_at = ? WHERE user_id = ?",
                (goal_type, now, USER_ID),
            )
        else:
            # Read existing goal_type from profile
            profile = conn.execute(
                "SELECT goal FROM user_profiles WHERE user_id = ?", (USER_ID,)
            ).fetchone()
            resolved_goal_type = profile["goal"] if profile and profile["goal"] else "maintain"

        conn.commit()
        logger.info(f"set_goal: {metric} = {target_value} (was {previous_value}, type={resolved_goal_type})")

        return {
            "status": "success",
            "data": {
                "metric": metric,
                "previous_value": previous_value,
                "new_value": target_value,
                "goal_type": resolved_goal_type,
            },
        }

    except Exception as e:
        logger.error(f"set_goal error: {e}")
        return {"status": "error", "error_type": "internal_error", "message": str(e)}
    finally:
        conn.close()
