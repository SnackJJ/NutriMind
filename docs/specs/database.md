# Database Specification

## Data Sources

| Source | Description | Records |
|--------|-------------|---------|
| **USDA SR Legacy** | Standard Reference Legacy, common foods | ~8,000 |
| **USDA Foundation Foods** | High-precision analytical data | ~2,000 |

Combined total: ~10,000 food items covering 95%+ of common dietary queries.

## Download Instructions

Data is available from USDA FoodData Central:
- https://fdc.nal.usda.gov/download-datasets.html

Download:
- `FoodData_Central_sr_legacy_food_csv_2018-04.zip`
- `FoodData_Central_foundation_food_csv_YYYY-MM.zip`

## SQLite Schema

### Main Tables

```sql
-- ============================================================
-- FOOD DATA TABLES (denormalized — all nutrients as columns)
-- ============================================================

CREATE TABLE foods (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fdc_id          INTEGER UNIQUE,
    description     TEXT    NOT NULL,
    category        TEXT,
    -- Core macronutrients
    energy_kcal     REAL DEFAULT 0,
    protein_g       REAL DEFAULT 0,
    total_fat_g     REAL DEFAULT 0,
    carbohydrate_g  REAL DEFAULT 0,
    fiber_g         REAL DEFAULT 0,
    sugars_g        REAL DEFAULT 0,
    -- Minerals
    sodium_mg       REAL DEFAULT 0,
    iron_mg         REAL DEFAULT 0,
    calcium_mg      REAL DEFAULT 0,
    potassium_mg    REAL DEFAULT 0,
    -- Other
    cholesterol_mg  REAL DEFAULT 0,
    saturated_fat_g REAL DEFAULT 0,
    vitamin_a_mcg   REAL DEFAULT 0,
    vitamin_c_mg    REAL DEFAULT 0,
    vitamin_d_mcg   REAL DEFAULT 0,
    water_g         REAL DEFAULT 0
);

-- Note: nutrients and nutrient_types tables are NOT used.
-- All nutrient values are embedded as columns in the foods table above.

-- Indexes (built during download_usda.py)
CREATE INDEX idx_foods_description ON foods(description);
CREATE INDEX idx_foods_category ON foods(category);

-- food_aliases — PLANNED (Task 1.2.2, not yet created)
-- CREATE TABLE food_aliases (
--     alias  TEXT    PRIMARY KEY COLLATE NOCASE,
--     fdc_id INTEGER NOT NULL REFERENCES foods(fdc_id) ON DELETE CASCADE
-- );

-- foods_fts FTS5 table — PLANNED (after food_aliases is populated)
-- CREATE VIRTUAL TABLE foods_fts USING fts5(
--     description, category,
--     content='foods', content_rowid='id'
-- );

-- ============================================================
-- USER DATA TABLES (not yet created — Task 1.2.3/1.2.4)
-- ============================================================

CREATE TABLE user_profiles (
    user_id              TEXT PRIMARY KEY DEFAULT 'default',
    weight_kg            REAL CHECK(weight_kg IS NULL OR (weight_kg > 0 AND weight_kg < 500)),
    height_cm            REAL CHECK(height_cm IS NULL OR (height_cm > 0 AND height_cm < 300)),
    age                  INTEGER CHECK(age IS NULL OR (age > 0 AND age < 150)),
    gender               TEXT CHECK(gender IS NULL OR gender IN ('male', 'female', 'other')),
    activity_level       TEXT CHECK(activity_level IS NULL OR activity_level IN
                           ('sedentary', 'light', 'moderate', 'active', 'very_active')),
    goal                 TEXT CHECK(goal IS NULL OR goal IN ('lose', 'maintain', 'gain')),
    tdee_kcal            REAL CHECK(tdee_kcal IS NULL OR tdee_kcal > 0),
    -- ^ Populated from user_profile.md at session init; used by get_today_summary for budget
    allergies            TEXT NOT NULL DEFAULT '[]',
    -- ^ JSON array, e.g. '["shellfish","peanuts"]'
    conditions           TEXT NOT NULL DEFAULT '[]',
    -- ^ JSON array, e.g. '["diabetes","hypertension","ckd"]'; used by get_user_profile
    dietary_preferences  TEXT NOT NULL DEFAULT '[]',
    -- ^ JSON array, e.g. '["vegetarian","low_sodium","gluten_free"]'
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE meal_logs (
    log_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   TEXT    NOT NULL DEFAULT 'default',
    meal_type TEXT    NOT NULL CHECK(meal_type IN ('breakfast', 'lunch', 'dinner', 'snack')),
    logged_at TEXT    NOT NULL DEFAULT (datetime('now')),
    note      TEXT,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
);

CREATE TABLE meal_log_items (
    item_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id        INTEGER NOT NULL,
    fdc_id        INTEGER,                              -- nullable if food not in DB
    food_name     TEXT    NOT NULL,
    amount        TEXT    NOT NULL DEFAULT '100g',
    calories_kcal REAL    NOT NULL DEFAULT 0 CHECK(calories_kcal >= 0),
    protein_g     REAL    NOT NULL DEFAULT 0 CHECK(protein_g >= 0),
    fat_g         REAL    NOT NULL DEFAULT 0 CHECK(fat_g >= 0),
    carbs_g       REAL    NOT NULL DEFAULT 0 CHECK(carbs_g >= 0),
    fiber_g       REAL    NOT NULL DEFAULT 0 CHECK(fiber_g >= 0),
    FOREIGN KEY (log_id) REFERENCES meal_logs(log_id) ON DELETE CASCADE,
    FOREIGN KEY (fdc_id) REFERENCES foods(fdc_id)
);

-- ============================================================
-- GOAL MANAGEMENT TABLE
-- ============================================================

CREATE TABLE user_goals (
    user_id       TEXT    NOT NULL DEFAULT 'default',
    metric        TEXT    NOT NULL CHECK(metric IN ('calories', 'protein', 'fat', 'carbs')),
    target_value  REAL    NOT NULL CHECK(target_value > 0),
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, metric),
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_meal_logs_user_date ON meal_logs(user_id, logged_at);
CREATE INDEX idx_meal_log_items_log  ON meal_log_items(log_id);

-- Daily intake summary (computed view)
CREATE VIEW daily_summary AS
SELECT
    ml.user_id,
    DATE(ml.logged_at)                   AS log_date,
    COUNT(DISTINCT ml.log_id)            AS meal_count,
    ROUND(SUM(mli.calories_kcal), 1)     AS total_calories,
    ROUND(SUM(mli.protein_g), 1)         AS total_protein_g,
    ROUND(SUM(mli.fat_g), 1)             AS total_fat_g,
    ROUND(SUM(mli.carbs_g), 1)           AS total_carbs_g,
    ROUND(SUM(mli.fiber_g), 1)           AS total_fiber_g
FROM meal_logs ml
JOIN meal_log_items mli ON ml.log_id = mli.log_id
GROUP BY ml.user_id, DATE(ml.logged_at);
```

### Nutrient Columns

The foods table embeds 16 nutrient columns directly. No separate `nutrients` table is used.

| Column | Unit | Notes |
|--------|------|-------|
| energy_kcal | kcal | Maps to USDA nutrient_id 1008 |
| protein_g | g | 1003 |
| total_fat_g | g | 1004 |
| carbohydrate_g | g | 1005 |
| fiber_g | g | 1079 |
| sugars_g | g | 2000 |
| sodium_mg | mg | 1093 |
| iron_mg | mg | 1089 |
| calcium_mg | mg | 1087 |
| potassium_mg | mg | 1092 |
| cholesterol_mg | mg | 1253 |
| saturated_fat_g | g | 1258 |
| vitamin_a_mcg | mcg | 1106 |
| vitamin_c_mg | mg | 1162 |
| vitamin_d_mcg | mcg | 1114 |
| water_g | g | 1051 |

## Data Processing Pipeline

```python
# scripts/download_usda.py

import sqlite3
import pandas as pd
from pathlib import Path

def process_usda_data(sr_legacy_path: Path, foundation_path: Path, output_db: Path):
    conn = sqlite3.connect(output_db)

    # 1. Load SR Legacy
    sr_foods = pd.read_csv(sr_legacy_path / "food.csv")
    sr_nutrients = pd.read_csv(sr_legacy_path / "food_nutrient.csv")

    # 2. Load Foundation Foods
    ff_foods = pd.read_csv(foundation_path / "food.csv")
    ff_nutrients = pd.read_csv(foundation_path / "food_nutrient.csv")

    # 3. Combine and deduplicate
    all_foods = pd.concat([
        sr_foods.assign(data_source='sr_legacy'),
        ff_foods.assign(data_source='foundation')
    ])

    # 4. Clean descriptions
    all_foods['description'] = all_foods['description'].str.lower().str.strip()

    # 5. Insert into SQLite
    all_foods.to_sql('foods', conn, if_exists='replace', index=False)

    # 6. Process nutrients (pivot to per-food rows)
    # ... nutrient processing ...

    # 7. Build FTS index
    conn.execute("INSERT INTO foods_fts(foods_fts) VALUES('rebuild');")

    conn.commit()
    conn.close()
```

## Query Interface

### search_food Implementation

```python
def search_food(food_name: str, amount_grams: float = 100.0) -> dict:
    conn = get_db_connection()

    # 1. Try exact match first
    result = conn.execute("""
        SELECT id, description, category,
               energy_kcal, protein_g, total_fat_g, carbohydrate_g, fiber_g,
               sugars_g, sodium_mg, cholesterol_mg, saturated_fat_g,
               iron_mg, calcium_mg, potassium_mg
        FROM foods
        WHERE LOWER(description) = LOWER(?)
    """, (food_name,)).fetchone()

    # 2. If no exact match, use LIKE fallback
    if not result:
        results = conn.execute("""
            SELECT id, description, category,
                   energy_kcal, protein_g, total_fat_g, carbohydrate_g, fiber_g,
                   sugars_g, sodium_mg, cholesterol_mg, saturated_fat_g,
                   iron_mg, calcium_mg, potassium_mg
            FROM foods
            WHERE description LIKE ?
            LIMIT 5
        """, (f"%{food_name}%",)).fetchall()

        if not results:
            return {"status": "error", "error_type": "food_not_found"}

        result = results[0]

    # 3. Apply amount scaling (relative to 100g base)
    scale = amount_grams / 100.0

    return {
        "status": "success",
        "data": {
            "food_name": result[1],
            "amount_grams": amount_grams,
            "calories_kcal": round(result[3] * scale, 1),
            "protein_g":     round(result[4] * scale, 1),
            "fat_g":         round(result[5] * scale, 1),
            "carbs_g":       round(result[6] * scale, 1),
            "fiber_g":       round(result[7] * scale, 1),
            "sugars_g":      round(result[8] * scale, 1),
            "sodium_mg":     round(result[9] * scale, 1),
            "cholesterol_mg":round(result[10] * scale, 1),
            "saturated_fat_g":round(result[11] * scale, 1),
            "iron_mg":       round(result[12] * scale, 1),
            "calcium_mg":    round(result[13] * scale, 1),
            "potassium_mg":  round(result[14] * scale, 1),
        }
    }
```

### Amount Scaling

All tools accept `amount_grams: float` (grams as a number). Scaling is a direct division by 100:

```python
scale = amount_grams / 100.0
# e.g. 150g chicken breast: scale = 1.5, calories = energy_kcal * 1.5
```

### calculate_meal Implementation

`calculate_meal` calls `search_food` internally for each item, then aggregates results.
It does **not** expose a shared `_query_food_nutrients` function.

```python
def calculate_meal(foods: list[dict]) -> dict:
    """Aggregate nutrition across multiple food items."""
    if not foods:
        return {"status": "error", "error_type": "empty_food_list"}

    breakdown = []
    failed_items = []

    for item in foods:
        result = search_food(item["food_name"], item.get("amount_grams", 100.0))
        if result["status"] == "success":
            breakdown.append(result["data"])
        else:
            failed_items.append({"food_name": item["food_name"], "error": result["error_type"]})

    if not breakdown:
        return {"status": "error", "error_type": "all_foods_not_found"}

    # Aggregate macros
    keys = ("calories_kcal", "protein_g", "fat_g", "carbs_g")
    total = {k: round(sum(b[k] for b in breakdown), 1) for k in keys}

    # Status
    if failed_items and breakdown:
        status = "partial_success"
    else:
        status = "success"

    return {
        "status": status,
        "data": {"total": total, "breakdown": breakdown, "macro_ratio": ...},
    }
```

## Data Quality Notes

1. **SR Legacy** is the primary source for common foods
2. **Foundation Foods** provides higher precision but limited coverage
3. When both sources have the same food, prefer Foundation Foods
4. Some portion sizes are approximate (e.g., "1 medium apple" = ~180g)
5. `food_aliases` — **planned** (Task 1.2.2, not yet created); will map common names
   (e.g., "chicken breast", "egg") to canonical USDA descriptions

## Maintenance

```bash
# Check database integrity
sqlite3 data/usda.db "PRAGMA integrity_check;"

# Export statistics
sqlite3 data/usda.db "SELECT COUNT(*) FROM foods;"

# Sample food lookup
sqlite3 data/usda.db "SELECT description, energy_kcal, protein_g FROM foods WHERE description LIKE '%chicken%' LIMIT 5;"
```
