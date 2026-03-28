"""Create user-state tables in the existing usda.db.

Idempotent — safe to run multiple times (CREATE TABLE IF NOT EXISTS).
Run once before using log_meal / get_today_summary / get_history / set_goal tools.
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.db import get_db_path
from src.utils.logger import logger

DDL = """
-- ============================================================
-- USER PROFILE
-- ============================================================
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id              TEXT PRIMARY KEY DEFAULT 'default',
    weight_kg            REAL CHECK(weight_kg IS NULL OR (weight_kg > 0 AND weight_kg < 500)),
    height_cm            REAL CHECK(height_cm IS NULL OR (height_cm > 0 AND height_cm < 300)),
    age                  INTEGER CHECK(age IS NULL OR (age > 0 AND age < 150)),
    gender               TEXT CHECK(gender IS NULL OR gender IN ('male', 'female', 'other')),
    activity_level       TEXT CHECK(activity_level IS NULL OR activity_level IN
                           ('sedentary', 'light', 'moderate', 'active', 'very_active')),
    goal                 TEXT CHECK(goal IS NULL OR goal IN ('lose', 'maintain', 'gain')),
    tdee_kcal            REAL CHECK(tdee_kcal IS NULL OR tdee_kcal > 0),
    allergies            TEXT NOT NULL DEFAULT '[]',
    conditions           TEXT NOT NULL DEFAULT '[]',
    dietary_preferences  TEXT NOT NULL DEFAULT '[]',
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ============================================================
-- MEAL LOGGING
-- ============================================================
CREATE TABLE IF NOT EXISTS meal_logs (
    log_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   TEXT    NOT NULL DEFAULT 'default',
    meal_type TEXT    NOT NULL CHECK(meal_type IN ('breakfast', 'lunch', 'dinner', 'snack')),
    logged_at TEXT    NOT NULL DEFAULT (datetime('now')),
    note      TEXT,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS meal_log_items (
    item_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id        INTEGER NOT NULL,
    food_name     TEXT    NOT NULL,
    amount_grams  REAL    NOT NULL DEFAULT 100.0,
    calories_kcal REAL    NOT NULL DEFAULT 0 CHECK(calories_kcal >= 0),
    protein_g     REAL    NOT NULL DEFAULT 0 CHECK(protein_g >= 0),
    fat_g         REAL    NOT NULL DEFAULT 0 CHECK(fat_g >= 0),
    carbs_g       REAL    NOT NULL DEFAULT 0 CHECK(carbs_g >= 0),
    fiber_g       REAL    NOT NULL DEFAULT 0 CHECK(fiber_g >= 0),
    FOREIGN KEY (log_id) REFERENCES meal_logs(log_id) ON DELETE CASCADE
);

-- ============================================================
-- GOAL MANAGEMENT
-- ============================================================
CREATE TABLE IF NOT EXISTS user_goals (
    user_id       TEXT    NOT NULL DEFAULT 'default',
    metric        TEXT    NOT NULL CHECK(metric IN ('calories', 'protein', 'fat', 'carbs')),
    target_value  REAL    NOT NULL CHECK(target_value > 0),
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, metric),
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
);

-- ============================================================
-- INDEXES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_meal_logs_user_date ON meal_logs(user_id, logged_at);
CREATE INDEX IF NOT EXISTS idx_meal_log_items_log  ON meal_log_items(log_id);

-- ============================================================
-- DAILY SUMMARY VIEW
-- ============================================================
CREATE VIEW IF NOT EXISTS daily_summary AS
SELECT
    ml.user_id,
    DATE(ml.logged_at)                   AS log_date,
    COUNT(DISTINCT ml.log_id)            AS meal_count,
    ROUND(SUM(mli.calories_kcal), 1)     AS total_calories,
    ROUND(SUM(mli.protein_g), 1)         AS total_protein_g,
    ROUND(SUM(mli.fat_g), 1)             AS total_fat_g,
    ROUND(SUM(mli.carbs_g), 1)           AS total_carbs_g,
    ROUND(SUM(mli.fiber_g), 1)           AS total_fiber_g,
    GROUP_CONCAT(COALESCE(mli.food_name, ''), ', ')  AS food_summary
FROM meal_logs ml
JOIN meal_log_items mli ON ml.log_id = mli.log_id
GROUP BY ml.user_id, DATE(ml.logged_at);
"""

DEFAULT_USER = """
INSERT OR IGNORE INTO user_profiles (user_id, tdee_kcal, goal)
VALUES ('default', 2000.0, 'maintain');
"""

DEFAULT_GOALS = """
INSERT OR IGNORE INTO user_goals (user_id, metric, target_value) VALUES
    ('default', 'calories', 2000.0),
    ('default', 'protein',   90.0),
    ('default', 'fat',       65.0),
    ('default', 'carbs',    250.0);
"""


def init_user_tables():
    db_path = get_db_path()
    logger.info(f"Initializing user tables in: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(DDL)
        conn.execute(DEFAULT_USER)
        conn.executescript(DEFAULT_GOALS)
        conn.commit()

        # Verify
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"
        ).fetchall()]
        logger.info(f"Tables/views present: {tables}")

        user = conn.execute("SELECT * FROM user_profiles WHERE user_id='default'").fetchone()
        logger.info(f"Default user: tdee_kcal={user[7]}, goal={user[6]}")

        goals = conn.execute("SELECT metric, target_value FROM user_goals WHERE user_id='default'").fetchall()
        logger.info(f"Default goals: {dict(goals)}")

        print("✓ User tables initialized successfully.")
    finally:
        conn.close()


if __name__ == "__main__":
    init_user_tables()
