"""
USDA FoodData Central API 使用示例
完全免费，最权威的营养数据源
"""

import requests
import json
from typing import Dict, List, Optional

# 获取API Key: https://fdc.nal.usda.gov/api-key-signup.html
API_KEY = "DEMO_KEY"  # 替换为你的API Key
BASE_URL = "https://api.nal.usda.gov/fdc/v1"


def search_foods(query: str, page_size: int = 10) -> Dict:
    """
    搜索食品

    Args:
        query: 搜索关键词（如 "McDonald's Big Mac"）
        page_size: 返回结果数量

    Returns:
        包含搜索结果的字典
    """
    url = f"{BASE_URL}/foods/search"
    params = {
        "api_key": API_KEY,
        "query": query,
        "pageSize": page_size,
        "dataType": ["Branded"],  # 只搜索品牌食品
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_food_details(fdc_id: int) -> Dict:
    """
    获取单个食品的详细营养信息

    Args:
        fdc_id: FoodData Central ID

    Returns:
        完整的营养信息字典
    """
    url = f"{BASE_URL}/food/{fdc_id}"
    params = {"api_key": API_KEY}

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def extract_nutrients(food_data: Dict) -> Dict[str, float]:
    """
    从食品数据中提取主要营养素

    Returns:
        营养素字典 {nutrient_name: amount}
    """
    nutrients = {}

    nutrient_mapping = {
        "Energy": "calories",
        "Protein": "protein_g",
        "Total lipid (fat)": "fat_g",
        "Carbohydrate, by difference": "carbs_g",
        "Fiber, total dietary": "fiber_g",
        "Sugars, total including NLEA": "sugar_g",
        "Sodium, Na": "sodium_mg",
        "Cholesterol": "cholesterol_mg",
        "Fatty acids, total saturated": "saturated_fat_g",
    }

    for nutrient in food_data.get("foodNutrients", []):
        nutrient_name = nutrient.get("nutrient", {}).get("name")
        if nutrient_name in nutrient_mapping:
            amount = nutrient.get("amount", 0)
            nutrients[nutrient_mapping[nutrient_name]] = amount

    return nutrients


# === 示例1: 搜索麦当劳的产品 ===
def example_search_mcdonalds():
    """搜索麦当劳菜单项"""
    print("=== 搜索麦当劳产品 ===\n")

    results = search_foods("McDonald's Big Mac", page_size=5)

    print(f"共找到 {results['totalHits']} 个结果\n")

    for food in results.get("foods", [])[:3]:
        print(f"📦 {food['description']}")
        print(f"   FDC ID: {food['fdcId']}")
        print(f"   品牌: {food.get('brandOwner', 'N/A')}")
        print(f"   数据类型: {food['dataType']}")
        print()


# === 示例2: 获取详细营养信息 ===
def example_get_nutrition(fdc_id: int = 2346495):  # Big Mac的示例ID
    """获取特定食品的详细营养信息"""
    print(f"=== 获取 FDC ID {fdc_id} 的营养信息 ===\n")

    food_data = get_food_details(fdc_id)

    print(f"🍔 {food_data['description']}")
    print(f"品牌: {food_data.get('brandOwner', 'N/A')}")
    print(f"份量: {food_data.get('servingSize', 'N/A')} {food_data.get('servingSizeUnit', '')}")
    print()

    nutrients = extract_nutrients(food_data)

    print("营养成分:")
    for nutrient, amount in nutrients.items():
        print(f"  • {nutrient}: {amount}")


# === 示例3: 批量搜索多个连锁餐厅 ===
def example_search_multiple_chains():
    """搜索多个连锁餐厅的菜单"""
    print("=== 批量搜索连锁餐厅 ===\n")

    chains = ["McDonald's", "Starbucks", "Subway", "KFC", "Taco Bell"]

    for chain in chains:
        results = search_foods(chain, page_size=3)
        print(f"{chain}: 找到 {results['totalHits']} 个产品")


# === 示例4: 比较两个产品 ===
def example_compare_foods():
    """比较两个食品的营养成分"""
    print("=== 比较营养成分 ===\n")

    # 搜索两个产品
    bigmac_results = search_foods("McDonald's Big Mac", page_size=1)
    whopper_results = search_foods("Burger King Whopper", page_size=1)

    if bigmac_results["foods"] and whopper_results["foods"]:
        bigmac_id = bigmac_results["foods"][0]["fdcId"]
        whopper_id = whopper_results["foods"][0]["fdcId"]

        bigmac_data = get_food_details(bigmac_id)
        whopper_data = get_food_details(whopper_id)

        bigmac_nutrients = extract_nutrients(bigmac_data)
        whopper_nutrients = extract_nutrients(whopper_data)

        print(f"🍔 Big Mac vs Whopper\n")
        print(f"{'Nutrient':<20} {'Big Mac':<15} {'Whopper':<15}")
        print("-" * 50)

        for nutrient in set(bigmac_nutrients.keys()) | set(whopper_nutrients.keys()):
            bigmac_val = bigmac_nutrients.get(nutrient, 0)
            whopper_val = whopper_nutrients.get(nutrient, 0)
            print(f"{nutrient:<20} {bigmac_val:<15.1f} {whopper_val:<15.1f}")


if __name__ == "__main__":
    # 运行示例
    try:
        example_search_mcdonalds()
        print("\n" + "="*60 + "\n")

        # 注意：需要实际的FDC ID才能运行以下示例
        # example_get_nutrition()

        example_search_multiple_chains()

    except requests.exceptions.HTTPError as e:
        print(f"❌ API错误: {e}")
        print("可能是API Key问题，请在 https://fdc.nal.usda.gov/api-key-signup.html 注册")
    except Exception as e:
        print(f"❌ 错误: {e}")


"""
输出示例:

=== 搜索麦当劳产品 ===

共找到 1,234 个结果

📦 MCDONALD'S, BIG MAC
   FDC ID: 2346495
   品牌: McDonald's Corporation
   数据类型: Branded

📦 MCDONALD'S, BIG MAC (WITHOUT BUN)
   FDC ID: 2346496
   品牌: McDonald's Corporation
   数据类型: Branded

=== 批量搜索连锁餐厅 ===

McDonald's: 找到 1,234 个产品
Starbucks: 找到 856 个产品
Subway: 找到 342 个产品
KFC: 找到 189 个产品
Taco Bell: 找到 267 个产品
"""
