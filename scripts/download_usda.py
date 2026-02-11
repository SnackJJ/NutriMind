"""
Build the USDA SQLite database from FoodData Central SR Legacy JSON.

Extracts ~7800 foods with their core nutrients (per 100g) into a compact
SQLite database that search_food and calculate_meal query at runtime.

Usage:
    python scripts/download_usda.py                        # defaults
    python scripts/download_usda.py --db-path data/usda.db # explicit
"""

import argparse
import json
import sqlite3
from pathlib import Path

# Nutrient IDs we care about (FoodData Central numbering, per 100g basis)
NUTRIENT_MAP = {
    1008: "energy_kcal",       # Energy (kcal)
    1003: "protein_g",         # Protein (g)
    1004: "total_fat_g",       # Total lipid / fat (g)
    1005: "carbohydrate_g",    # Carbohydrate, by difference (g)
    1079: "fiber_g",           # Fiber, total dietary (g)
    2000: "sugars_g",          # Sugars, total (g)
    1093: "sodium_mg",         # Sodium (mg)
    1253: "cholesterol_mg",    # Cholesterol (mg)
    1258: "saturated_fat_g",   # Fatty acids, total saturated (g)
    1089: "iron_mg",           # Iron (mg)
    1087: "calcium_mg",        # Calcium (mg)
    1092: "potassium_mg",      # Potassium (mg)
    1106: "vitamin_a_mcg",     # Vitamin A, RAE (mcg)
    1162: "vitamin_c_mg",      # Vitamin C (mg)
    1114: "vitamin_d_mcg",     # Vitamin D (D2 + D3) (mcg)
    1051: "water_g",           # Water (g)
}


def create_schema(conn: sqlite3.Connection):
    """Create the foods table with embedded nutrient columns."""
    nutrient_cols = ",\n        ".join(f"{col} REAL DEFAULT 0" for col in NUTRIENT_MAP.values())
    conn.executescript(f"""
        DROP TABLE IF EXISTS foods;
        DROP TABLE IF EXISTS food_nutrients;

        CREATE TABLE foods (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fdc_id INTEGER UNIQUE,
            description TEXT NOT NULL,
            category TEXT,
            {nutrient_cols}
        );

        CREATE INDEX idx_foods_description ON foods(description);
        CREATE INDEX idx_foods_category ON foods(category);
    """)


def extract_nutrients(food: dict) -> dict:
    """Extract nutrient values from a food item's foodNutrients array."""
    result = {col: 0.0 for col in NUTRIENT_MAP.values()}
    for fn in food.get("foodNutrients", []):
        nid = fn.get("nutrient", {}).get("id")
        if nid in NUTRIENT_MAP:
            result[NUTRIENT_MAP[nid]] = fn.get("amount", 0.0) or 0.0
    return result


def import_foods(conn: sqlite3.Connection, json_path: str, data_key: str):
    """Import foods from a FoodData Central JSON file."""
    print(f"Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    foods = data.get(data_key, [])
    print(f"  Found {len(foods)} foods under key '{data_key}'")

    cols = list(NUTRIENT_MAP.values())
    placeholders = ", ".join(["?"] * (3 + len(cols)))
    col_names = ", ".join(["fdc_id", "description", "category"] + cols)
    sql = f"INSERT OR IGNORE INTO foods ({col_names}) VALUES ({placeholders})"

    rows = []
    for food in foods:
        fdc_id = food.get("fdcId")
        desc = food.get("description", "").strip()
        category = (food.get("foodCategory", {}) or {}).get("description", "")
        nutrients = extract_nutrients(food)
        row = (fdc_id, desc, category) + tuple(nutrients[c] for c in cols)
        rows.append(row)

    conn.executemany(sql, rows)
    conn.commit()
    print(f"  Inserted {len(rows)} rows")


def print_stats(conn: sqlite3.Connection):
    """Print summary statistics."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM foods")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT category) FROM foods WHERE category != ''")
    cats = cur.fetchone()[0]

    cur.execute("SELECT AVG(energy_kcal), AVG(protein_g), AVG(total_fat_g), AVG(carbohydrate_g) FROM foods")
    avg = cur.fetchone()

    print(f"\n{'='*50}")
    print(f"Total foods:       {total}")
    print(f"Categories:        {cats}")
    print(f"Avg per 100g:      {avg[0]:.0f} kcal | {avg[1]:.1f}g protein | {avg[2]:.1f}g fat | {avg[3]:.1f}g carbs")
    print(f"{'='*50}")

    # Spot check some common foods
    print("\nSpot check:")
    for name in ["chicken breast", "egg", "rice", "banana", "spinach"]:
        cur.execute("SELECT description, energy_kcal, protein_g, total_fat_g, carbohydrate_g FROM foods WHERE description LIKE ? LIMIT 1", (f"%{name}%",))
        row = cur.fetchone()
        if row:
            print(f"  {row[0][:50]:50s} | {row[1]:>6.0f} kcal | {row[2]:>5.1f}g P | {row[3]:>5.1f}g F | {row[4]:>5.1f}g C")
        else:
            print(f"  '{name}': NOT FOUND")


def main():
    parser = argparse.ArgumentParser(description="Build USDA SQLite database from FoodData Central JSON.")
    parser.add_argument("--db-path", default="data/usda.db", help="Path to output SQLite database")
    parser.add_argument("--sr-legacy", default="data/raw/sr_legacy/FoodData_Central_sr_legacy_food_json_2021-10-28.json")
    parser.add_argument("--foundation", default="data/raw/foundation/foundationDownload.json")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        create_schema(conn)

        # Import SR Legacy (7793 foods — comprehensive everyday foods)
        if Path(args.sr_legacy).exists():
            import_foods(conn, args.sr_legacy, "SRLegacyFoods")
        else:
            print(f"⚠ SR Legacy not found at {args.sr_legacy}, skipping")

        # Import Foundation (316 foods — high-quality analytical data)
        if Path(args.foundation).exists():
            import_foods(conn, args.foundation, "FoundationFoods")
        else:
            print(f"⚠ Foundation not found at {args.foundation}, skipping")

        print_stats(conn)
    finally:
        conn.close()

    print(f"\n✅ Database saved to {db_path} ({db_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
