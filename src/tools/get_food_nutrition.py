"""get_food_nutrition tool — single/multi-food nutrition lookup from USDA database.

Merges the former search_food (single-food query) and calculate_meal
(multi-food aggregation) into one unified tool that the model calls for
ANY food nutrition query.
"""

import threading
import yaml
from src.utils.db import get_connection
from src.utils.logger import logger
from src.retrieval.hybrid_retriever import HybridRetriever

_food_retriever: HybridRetriever | None = None
_food_init_lock = threading.Lock()


def _get_food_retriever() -> HybridRetriever:
    """Lazy-load the hybrid retriever for food search."""
    global _food_retriever
    if _food_retriever is None:
        with _food_init_lock:
            if _food_retriever is None:
                with open("configs/tools.yaml", "r") as f:
                    full_config = yaml.safe_load(f)
                    rag_config = full_config.get("rag", {})
                    food_config = full_config.get("food_search", {})

                # Use model config from rag section
                food_config.update({
                    "embedding_model": rag_config.get("embedding_model"),
                    "reranker_model": rag_config.get("reranker_model")
                })

                _food_retriever = HybridRetriever(
                    food_config,
                    collection_name="usda_foods",
                    instruction="Represent this food item for retrieval: "
                )
                logger.info("Food HybridRetriever initialized")
    return _food_retriever




def _lookup_single(food_name: str, amount_grams: float) -> dict:
    """Look up a single food using hybrid search and fetch details from USDA DB."""
    try:
        retriever = _get_food_retriever()
        results = retriever.retrieve(food_name, allow_fallback=False)

        if not results:
            return {
                "status": "error",
                "error_type": "food_not_found",
                "message": f"Food '{food_name}' not found in USDA database.",
            }

        best_match = results[0]
        score = best_match.get("rerank_score", best_match.get("rrf_score", 0.0))
        low_confidence = best_match.get("low_confidence", False)

        # Confidence tiers based on hybrid score
        if score > 0.85 and not low_confidence:
            confidence = "high"
        elif score > 0.5 and not low_confidence:
            confidence = "medium"
        else:
            confidence = "low"

        fdc_id = best_match["metadata"]["fdc_id"]

        conn = get_connection()
        if not conn:
            return {"status": "error", "error_type": "db_error", "message": "Database not available"}

        multiplier = amount_grams / 100.0
        cursor = conn.cursor()

        sql = """SELECT description, category,
                        energy_kcal, protein_g, total_fat_g, carbohydrate_g,
                        fiber_g, sugars_g, sodium_mg, cholesterol_mg,
                        saturated_fat_g, iron_mg, calcium_mg, potassium_mg
                 FROM foods
                 WHERE fdc_id = ?"""
        cursor.execute(sql, (fdc_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return {
                "status": "error",
                "error_type": "food_id_broken",
                "message": f"FDC ID {fdc_id} found in index but missing in database.",
            }

        return {
            "status": "success",
            "data": {
                "food_name": row[0],
                "category": row[1] or "",
                "amount_grams": amount_grams,
                "match_confidence": confidence,
                "match_score": round(score, 4),
                "calories_kcal": round(row[2] * multiplier, 1),
                "protein_g": round(row[3] * multiplier, 1),
                "fat_g": round(row[4] * multiplier, 1),
                "carbs_g": round(row[5] * multiplier, 1),
                "fiber_g": round(row[6] * multiplier, 1),
                "sugars_g": round(row[7] * multiplier, 1),
                "sodium_mg": round(row[8] * multiplier, 1),
                "cholesterol_mg": round(row[9] * multiplier, 1),
                "saturated_fat_g": round(row[10] * multiplier, 1),
                "iron_mg": round(row[11] * multiplier, 2),
                "calcium_mg": round(row[12] * multiplier, 1),
                "potassium_mg": round(row[13] * multiplier, 1),
            },
        }
    except Exception as e:
        logger.error(f"get_food_nutrition hybrid lookup error: {e}")
        return {"status": "error", "error_type": "internal_error", "message": str(e)}


def get_food_nutrition(foods: list) -> dict:
    """Look up nutrition for one or more foods from the USDA database.

    Args:
        foods: List of {food_name: str, amount_grams: float}.
               For a single food, provide a list with one item.

    Returns:
        {status, data: {total, breakdown, macro_ratio}} on success.
        For a single food, 'total' equals that food's nutrients.
    """
    if not foods:
        return {"status": "error", "error_type": "empty_food_list", "message": "No foods provided."}

    total = {"calories_kcal": 0.0, "protein_g": 0.0, "fat_g": 0.0, "carbs_g": 0.0}
    breakdown = []
    failed_items = []

    for item in foods:
        name = item.get("food_name")
        amount_grams = float(item.get("amount_grams", 100.0))
        res = _lookup_single(name, amount_grams)
        if res.get("status") == "success":
            data = res["data"]
            total["calories_kcal"] += data.get("calories_kcal", 0)
            total["protein_g"] += data.get("protein_g", 0)
            total["fat_g"] += data.get("fat_g", 0)
            total["carbs_g"] += data.get("carbs_g", 0)
            breakdown.append(data)
        else:
            logger.warning(f"Food not found in get_food_nutrition: {name}")
            failed_items.append({
                "food_name": name,
                "amount_grams": amount_grams,
                "error": res.get("error_type", "unknown"),
                "message": res.get("message", ""),
            })

    total_macros = total["protein_g"] * 4 + total["fat_g"] * 9 + total["carbs_g"] * 4
    if total_macros > 0:
        macro_ratio = {
            "protein_pct": round((total["protein_g"] * 4) / total_macros * 100, 1),
            "fat_pct": round((total["fat_g"] * 9) / total_macros * 100, 1),
            "carbs_pct": round((total["carbs_g"] * 4) / total_macros * 100, 1),
        }
    else:
        macro_ratio = {"protein_pct": 0, "fat_pct": 0, "carbs_pct": 0}

    result = {
        "status": "success" if breakdown else "partial_failure",
        "data": {
            "total": total,
            "breakdown": breakdown,
            "macro_ratio": macro_ratio,
        },
    }
    if failed_items:
        result["data"]["failed_items"] = failed_items
        if breakdown:
            result["status"] = "partial_success"
        else:
            result["status"] = "error"
            result["error_type"] = "all_foods_not_found"
    return result
