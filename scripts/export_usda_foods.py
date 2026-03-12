"""
Export a list of food names from the USDA SQLite database to data/usda_foods.json.
This file is used by expand_query_pool.py to inject real food names into T1 prompts.

Usage:
    python scripts/export_usda_foods.py
    python scripts/export_usda_foods.py --db-path data/usda.db --output data/usda_foods.json --limit 500
"""

import argparse
import json
import sqlite3
from pathlib import Path


def export_food_names(db_path: str, output_path: str, limit: int) -> int:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Pull distinct food descriptions, ordered by data_type so Foundation Foods
    # (higher quality entries) surface first, then SR Legacy fills the rest.
    cursor.execute(
        """
        SELECT DISTINCT description
        FROM foods
        ORDER BY description
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()

    names = [row[0] for row in rows]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2, ensure_ascii=False)

    return len(names)


def main():
    parser = argparse.ArgumentParser(description="Export USDA food names to JSON.")
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/usda.db",
        help="Path to the USDA SQLite database (default: data/usda.db)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/usda_foods.json",
        help="Output JSON file path (default: data/usda_foods.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max number of food names to export (default: 500)",
    )
    args = parser.parse_args()

    if not Path(args.db_path).exists():
        print(f"ERROR: Database not found at {args.db_path}")
        print("Run scripts/download_usda.py and scripts/init_user_tables.py first.")
        raise SystemExit(1)

    count = export_food_names(args.db_path, args.output, args.limit)
    print(f"Exported {count} food names -> {args.output}")


if __name__ == "__main__":
    main()
