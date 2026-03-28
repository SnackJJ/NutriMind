"""Unit tests for NutriAgent tool functions.

Write tests use an isolated temp SQLite DB (via `user_db` fixture) so the
real data/usda.db is never modified.  Read-only tools (search_food,
calculate_meal) are also redirected to the same temp DB so tests remain
self-contained.
"""

import sqlite3
from contextlib import ExitStack, contextmanager
from datetime import datetime
from unittest.mock import patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Shared DDL (mirrors scripts/init_user_tables.py + actual foods schema)
# ─────────────────────────────────────────────────────────────────────────────

_USER_DDL = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id              TEXT PRIMARY KEY DEFAULT 'default',
    weight_kg            REAL,
    height_cm            REAL,
    age                  INTEGER,
    gender               TEXT,
    activity_level       TEXT,
    goal                 TEXT,
    tdee_kcal            REAL,
    allergies            TEXT NOT NULL DEFAULT '[]',
    conditions           TEXT NOT NULL DEFAULT '[]',
    dietary_preferences  TEXT NOT NULL DEFAULT '[]',
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS meal_logs (
    log_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   TEXT    NOT NULL DEFAULT 'default',
    meal_type TEXT    NOT NULL CHECK(meal_type IN ('breakfast','lunch','dinner','snack')),
    logged_at TEXT    NOT NULL DEFAULT (datetime('now')),
    note      TEXT
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

CREATE TABLE IF NOT EXISTS user_goals (
    user_id       TEXT NOT NULL DEFAULT 'default',
    metric        TEXT NOT NULL CHECK(metric IN ('calories','protein','fat','carbs')),
    target_value  REAL NOT NULL CHECK(target_value > 0),
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, metric)
);

CREATE TABLE IF NOT EXISTS foods (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fdc_id          INTEGER UNIQUE,
    description     TEXT NOT NULL,
    category        TEXT,
    energy_kcal     REAL DEFAULT 0,
    protein_g       REAL DEFAULT 0,
    total_fat_g     REAL DEFAULT 0,
    carbohydrate_g  REAL DEFAULT 0,
    fiber_g         REAL DEFAULT 0,
    sugars_g        REAL DEFAULT 0,
    sodium_mg       REAL DEFAULT 0,
    iron_mg         REAL DEFAULT 0,
    calcium_mg      REAL DEFAULT 0,
    potassium_mg    REAL DEFAULT 0,
    cholesterol_mg  REAL DEFAULT 0,
    saturated_fat_g REAL DEFAULT 0,
    vitamin_a_mcg   REAL DEFAULT 0,
    vitamin_c_mg    REAL DEFAULT 0,
    vitamin_d_mcg   REAL DEFAULT 0,
    water_g         REAL DEFAULT 0
);

CREATE VIEW IF NOT EXISTS daily_summary AS
SELECT
    ml.user_id,
    DATE(ml.logged_at)               AS log_date,
    COUNT(DISTINCT ml.log_id)        AS meal_count,
    ROUND(SUM(mli.calories_kcal), 1) AS total_calories,
    ROUND(SUM(mli.protein_g),     1) AS total_protein_g,
    ROUND(SUM(mli.fat_g),         1) AS total_fat_g,
    ROUND(SUM(mli.carbs_g),       1) AS total_carbs_g,
    ROUND(SUM(mli.fiber_g),       1) AS total_fiber_g,
    GROUP_CONCAT(COALESCE(mli.food_name, ''), ', ') AS food_summary
FROM meal_logs ml
JOIN meal_log_items mli ON ml.log_id = mli.log_id
GROUP BY ml.user_id, DATE(ml.logged_at);
"""

# columns: fdc_id, description, category, energy_kcal, protein_g, total_fat_g,
#          carbohydrate_g, fiber_g, sugars_g, sodium_mg, iron_mg, calcium_mg,
#          potassium_mg, cholesterol_mg, saturated_fat_g, vitamin_a_mcg,
#          vitamin_c_mg, vitamin_d_mcg, water_g
_TEST_FOODS = [
    (1, "Chicken, broiler, breast, raw", "Poultry",
     165.0, 31.0, 3.6, 0.0, 0.0, 0.0, 74.0, 0.4, 11.0, 256.0, 73.0, 1.0, 6.0, 0.0, 0.1, 65.0),
    (2, "Rice, white, long-grain, raw", "Grains",
     365.0, 7.1, 0.7, 80.0, 1.3, 0.0, 5.0, 0.8, 10.0, 115.0, 0.0, 0.2, 0.0, 0.0, 0.0, 12.0),
    (3, "Egg, whole, raw, fresh", "Eggs",
     143.0, 12.6, 9.5, 0.7, 0.0, 0.0, 142.0, 1.7, 50.0, 138.0, 372.0, 3.1, 149.0, 0.0, 2.0, 76.0),
]

_FOOD_INSERT = """
    INSERT OR IGNORE INTO foods
        (fdc_id, description, category, energy_kcal, protein_g, total_fat_g,
         carbohydrate_g, fiber_g, sugars_g, sodium_mg, iron_mg, calcium_mg,
         potassium_mg, cholesterol_mg, saturated_fat_g, vitamin_a_mcg,
         vitamin_c_mg, vitamin_d_mcg, water_g)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


@pytest.fixture
def user_db(tmp_path):
    """Return path to a fresh isolated SQLite DB with user tables + test foods."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_USER_DDL)
    conn.execute(
        "INSERT OR IGNORE INTO user_profiles (user_id, tdee_kcal, goal) VALUES ('default', 2000.0, 'maintain')"
    )
    conn.executemany(
        "INSERT OR IGNORE INTO user_goals (user_id, metric, target_value) VALUES ('default', ?, ?)",
        [("calories", 2000.0), ("protein", 90.0), ("fat", 65.0), ("carbs", 250.0)],
    )
    conn.executemany(_FOOD_INSERT, _TEST_FOODS)
    conn.commit()
    conn.close()
    return db_path


def _conn_factory(db_path):
    """Return a callable that opens a Row-enabled connection to `db_path`."""
    def _open():
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        return c
    return _open


@contextmanager
def patch_connections(db_path):
    """Patch get_connection in all tool modules to use the temp DB."""
    factory = _conn_factory(db_path)
    targets = [
        "src.tools.get_food_nutrition.get_connection",
        "src.tools.log_meal.get_connection",
        "src.tools.get_today_summary.get_connection",
        "src.tools.get_history.get_connection",
        "src.tools.set_goal.get_connection",
    ]
    with ExitStack() as stack:
        for t in targets:
            stack.enter_context(patch(t, factory))
        yield


# ─────────────────────────────────────────────────────────────────────────────
# Mock food retriever for isolated tests
# ─────────────────────────────────────────────────────────────────────────────

# Map of test food queries to their fdc_ids in _TEST_FOODS
_TEST_FOOD_MAP = {
    "chicken": 1,
    "chicken breast": 1,
    "rice": 2,
    "rice white": 2,
    "egg": 3,
}


class MockFoodRetriever:
    """Mock retriever that returns fdc_ids matching _TEST_FOODS."""

    def retrieve(self, query: str, allow_fallback: bool = True):
        query_lower = query.lower()
        for key, fdc_id in _TEST_FOOD_MAP.items():
            if key in query_lower:
                return [{
                    "content": f"Test food for {query}",
                    "metadata": {"fdc_id": fdc_id},
                    "rerank_score": 0.95,
                }]
        return []  # Not found


@contextmanager
def patch_food_retriever():
    """Patch the food retriever to use mock data matching _TEST_FOODS."""
    mock_retriever = MockFoodRetriever()
    with patch("src.tools.get_food_nutrition._get_food_retriever", return_value=mock_retriever):
        yield


# ─────────────────────────────────────────────────────────────────────────────
# get_food_nutrition
# ─────────────────────────────────────────────────────────────────────────────

from src.tools.get_food_nutrition import get_food_nutrition


class TestGetFoodNutrition:
    def test_single_food_success(self, user_db):
        with patch_connections(user_db), patch_food_retriever():
            result = get_food_nutrition([{"food_name": "chicken breast", "amount_grams": 100.0}])
        assert result["status"] == "success"
        assert result["data"]["total"]["calories_kcal"] == pytest.approx(165.0, abs=0.5)
        assert "protein_g" in result["data"]["total"]
        assert result["data"]["breakdown"][0]["amount_grams"] == 100.0

    def test_unknown_food_returns_error(self, user_db):
        with patch_connections(user_db), patch_food_retriever():
            result = get_food_nutrition([{"food_name": "xyzzy_nonexistent_food_12345", "amount_grams": 100.0}])
        assert result["status"] == "error"
        assert result["error_type"] == "all_foods_not_found"

    def test_amount_scaling(self, user_db):
        with patch_connections(user_db), patch_food_retriever():
            r100 = get_food_nutrition([{"food_name": "chicken breast", "amount_grams": 100.0}])
            r200 = get_food_nutrition([{"food_name": "chicken breast", "amount_grams": 200.0}])
        assert r100["status"] == "success"
        assert r200["status"] == "success"
        assert r200["data"]["total"]["calories_kcal"] == pytest.approx(
            r100["data"]["total"]["calories_kcal"] * 2, abs=0.5
        )

    def test_two_foods_total_correct(self, user_db):
        foods = [
            {"food_name": "chicken breast", "amount_grams": 100.0},
            {"food_name": "rice white", "amount_grams": 100.0},
        ]
        with patch_connections(user_db), patch_food_retriever():
            r_combined = get_food_nutrition(foods)
            r_chicken  = get_food_nutrition([foods[0]])
            r_rice     = get_food_nutrition([foods[1]])

        assert r_combined["status"] in ("success", "partial_success")
        expected = (
            r_chicken["data"]["total"]["calories_kcal"]
            + r_rice["data"]["total"]["calories_kcal"]
        )
        assert r_combined["data"]["total"]["calories_kcal"] == pytest.approx(expected, abs=1.0)

    def test_unknown_food_partial_failure(self, user_db):
        with patch_connections(user_db), patch_food_retriever():
            result = get_food_nutrition([{"food_name": "xyzzy_nonexistent_12345", "amount_grams": 100.0}])
        assert result["status"] in ("partial_failure", "error")


# ─────────────────────────────────────────────────────────────────────────────
# log_meal
# ─────────────────────────────────────────────────────────────────────────────

from src.tools.log_meal import log_meal


class TestLogMeal:
    def test_valid_meal_logged_successfully(self, user_db):
        with patch_connections(user_db), patch_food_retriever():
            result = log_meal(
                meal_type="lunch",
                foods=[{"food_name": "chicken breast", "amount_grams": 150.0}],
            )
        assert result["status"] == "success"
        assert "meal_id" in result
        assert isinstance(result["meal_id"], str)
        assert result["total_calories"] > 0

    def test_invalid_meal_type_returns_error(self, user_db):
        with patch_connections(user_db), patch_food_retriever():
            result = log_meal(meal_type="brunch", foods=[{"food_name": "egg", "amount_grams": 100.0}])
        assert result["status"] == "error"
        assert result["error_type"] == "invalid_meal_type"

    def test_empty_foods_returns_error(self, user_db):
        with patch_connections(user_db), patch_food_retriever():
            result = log_meal(meal_type="breakfast", foods=[])
        assert result["status"] == "error"
        assert result["error_type"] == "missing_required_field"

    def test_meal_persisted_to_db(self, user_db):
        with patch_connections(user_db), patch_food_retriever():
            result = log_meal(
                meal_type="dinner",
                foods=[{"food_name": "egg", "amount_grams": 100.0}],
            )
        assert result["status"] == "success"
        # Verify the row exists in the DB
        conn = sqlite3.connect(user_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT meal_type FROM meal_logs WHERE log_id = ?", (int(result["meal_id"]),)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row["meal_type"] == "dinner"


# ─────────────────────────────────────────────────────────────────────────────
# get_today_summary
# ─────────────────────────────────────────────────────────────────────────────

from src.tools.get_today_summary import get_today_summary


class TestGetTodaySummary:
    def test_no_meals_today_returns_default_zeros(self, user_db):
        with patch_connections(user_db):
            result = get_today_summary()
        assert result["status"] == "success"
        data = result["data"]
        assert data["total_calories"] == 0.0
        assert data["meal_count"] == 0
        assert data["meals_logged"] == []
        assert data["calorie_budget"] > 0
        assert data["remaining_calories"] == data["calorie_budget"]

    def test_after_logging_returns_summary(self, user_db):
        with patch_connections(user_db), patch_food_retriever():
            log_meal("breakfast", [{"food_name": "egg", "amount_grams": 100.0}])
            result = get_today_summary()
        assert result["status"] == "success"
        data = result["data"]
        assert data["total_calories"] > 0
        assert "protein_g" in data
        assert "remaining_calories" in data

    def test_calorie_budget_from_goals(self, user_db):
        # Calorie budget should match the default goal of 2000 kcal
        with patch_connections(user_db), patch_food_retriever():
            log_meal("snack", [{"food_name": "chicken breast", "amount_grams": 100.0}])
            result = get_today_summary()
        assert result["status"] == "success"
        assert result["data"]["calorie_budget"] == pytest.approx(2000.0, abs=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# get_history
# ─────────────────────────────────────────────────────────────────────────────

from src.tools.get_history import get_history


class TestGetHistory:
    def test_invalid_days_below_range(self, user_db):
        with patch_connections(user_db):
            result = get_history(days=0)
        assert result["status"] == "error"
        assert result["error_type"] == "invalid_date_range"

    def test_invalid_days_above_range(self, user_db):
        with patch_connections(user_db):
            result = get_history(days=91)
        assert result["status"] == "error"
        assert result["error_type"] == "invalid_date_range"

    def test_invalid_metric_returns_error(self, user_db):
        with patch_connections(user_db):
            result = get_history(days=7, metric="sodium")
        assert result["status"] == "error"
        assert result["error_type"] == "invalid_metric"

    def test_no_data_returns_error(self, user_db):
        with patch_connections(user_db):
            result = get_history(days=7)
        assert result["status"] == "error"
        assert result["error_type"] == "no_data_in_range"

    def test_with_data_returns_breakdown(self, user_db):
        with patch_connections(user_db), patch_food_retriever():
            log_meal("lunch", [{"food_name": "egg", "amount_grams": 100.0}])
            result = get_history(days=7)
        assert result["status"] == "success"
        data = result["data"]
        assert "daily_averages" in data
        assert "daily_breakdown" in data
        assert data["days_with_data"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# set_goal
# ─────────────────────────────────────────────────────────────────────────────

from src.tools.set_goal import set_goal


class TestSetGoal:
    def test_valid_goal_update(self, user_db):
        with patch_connections(user_db):
            result = set_goal(metric="protein", target_value=120.0)
        assert result["status"] == "success"
        data = result["data"]
        assert data["metric"] == "protein"
        assert data["new_value"] == 120.0
        assert data["previous_value"] == pytest.approx(90.0)  # default from fixture

    def test_invalid_metric_returns_error(self, user_db):
        with patch_connections(user_db):
            result = set_goal(metric="fiber", target_value=25.0)
        assert result["status"] == "error"
        assert result["error_type"] == "invalid_metric"

    def test_calories_out_of_range_returns_error(self, user_db):
        with patch_connections(user_db):
            result = set_goal(metric="calories", target_value=500.0)
        assert result["status"] == "error"
        assert result["error_type"] == "value_out_of_range"

    def test_goal_type_persisted(self, user_db):
        with patch_connections(user_db):
            result = set_goal(metric="calories", target_value=1800.0, goal_type="lose")
        assert result["status"] == "success"
        assert result["data"]["goal_type"] == "lose"
        # Verify it was written to user_profiles
        conn = sqlite3.connect(user_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT goal FROM user_profiles WHERE user_id='default'").fetchone()
        conn.close()
        assert row["goal"] == "lose"
