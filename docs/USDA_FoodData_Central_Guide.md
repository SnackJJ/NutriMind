# USDA FoodData Central 使用指南

## 🌟 为什么选择 USDA FoodData Central

### ✅ 最权威的营养数据源

- **美国农业部官方数据库**：政府权威机构维护
- **法定参考标准**：所有美国食品标签必须符合USDA标准
- **科学验证**：实验室分析 + 品牌官方提交的双重验证

### 💰 完全免费

- **无使用费用**
- **公共领域数据**（CC0 1.0协议）
- **可商业使用**
- **无版权限制**

### 📊 覆盖范围广

- **375,000+** 食品条目（2026年数据）
- **品牌食品数据库**（Branded Foods）：包含所有主流连锁餐厅
- **基础食品**：USDA实验室分析的食材数据
- **定期更新**：每季度更新一次

## 🔑 快速开始

### 1. 获取API Key（免费）

访问：https://fdc.nal.usda.gov/api-key-signup.html

- 填写基本信息（姓名、邮箱）
- 立即获得API Key
- 或使用 `DEMO_KEY` 进行测试

### 2. API限制

| 项目 | 限制 |
|------|------|
| **请求频率** | 1,000请求/小时/IP |
| **超限惩罚** | 暂时封禁1小时 |
| **提升额度** | 可联系USDA申请更高限额 |

### 3. 主要API端点

```bash
# 基础URL
https://api.nal.usda.gov/fdc/v1/

# 搜索食品
GET /foods/search?api_key=YOUR_KEY&query=McDonald's

# 获取单个食品详情
GET /food/{fdcId}?api_key=YOUR_KEY

# 批量获取多个食品
POST /foods?api_key=YOUR_KEY
Body: {"fdcIds": [123, 456, 789]}

# 获取食品列表（分页）
GET /foods/list?api_key=YOUR_KEY&pageSize=100
```

## 🍔 连锁餐厅数据覆盖

### 已验证有数据的餐厅（部分）

- ✅ McDonald's (麦当劳)
- ✅ Starbucks (星巴克)
- ✅ Subway
- ✅ KFC
- ✅ Burger King
- ✅ Taco Bell
- ✅ Pizza Hut
- ✅ Domino's Pizza
- ✅ Wendy's
- ✅ Chick-fil-A
- ✅ Chipotle
- ✅ Panera Bread
- ✅ Dunkin'
- ✅ Five Guys
- ✅ Panda Express

### 数据字段示例

每个食品包含：

```json
{
  "fdcId": 2346495,
  "description": "MCDONALD'S, BIG MAC",
  "brandOwner": "McDonald's Corporation",
  "servingSize": 219,
  "servingSizeUnit": "g",
  "foodNutrients": [
    {
      "nutrient": {
        "name": "Energy",
        "unitName": "kcal"
      },
      "amount": 257
    },
    {
      "nutrient": {
        "name": "Protein",
        "unitName": "g"
      },
      "amount": 13.24
    }
    // ... 更多营养素
  ]
}
```

## 📦 可用的营养素数据

USDA提供150+种营养素，常用的包括：

### 基础营养素
- Energy (calories)
- Protein
- Total lipid (fat)
- Carbohydrate
- Fiber
- Sugars

### 微量营养素
- Sodium
- Cholesterol
- Saturated fat
- Vitamins (A, C, D, E, K, B群)
- Minerals (钙, 铁, 钾, 镁等)

## 🚀 推荐使用策略

### 方案1：纯本地缓存（初期）

```
优点：
- 简单快速
- 无API依赖
- 响应迅速

缺点：
- 需要手动更新
- 覆盖有限

适用场景：MVP阶段
```

### 方案2：API + 缓存（推荐）

```
优点：
- 数据最新
- 自动更新
- 覆盖全面

实现：
1. 首次查询调用API
2. 缓存到本地数据库
3. 设置30天过期时间
4. 过期后重新查询

适用场景：生产环境
```

### 方案3：定期批量同步

```
优点：
- 减少API调用
- 数据完整

实现：
1. 每周批量下载常见餐厅数据
2. 存储到本地PostgreSQL/MongoDB
3. 应用直接查询本地库

适用场景：高并发场景
```

## 📝 代码示例

参考：`/examples/usda_api_example.py`

基础使用：

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.nal.usda.gov/fdc/v1"

# 搜索麦当劳大麦克
response = requests.get(
    f"{BASE_URL}/foods/search",
    params={
        "api_key": API_KEY,
        "query": "McDonald's Big Mac",
        "dataType": "Branded"
    }
)

results = response.json()
print(f"找到 {results['totalHits']} 个结果")

# 获取第一个结果的详细信息
if results['foods']:
    fdc_id = results['foods'][0]['fdcId']
    food_detail = requests.get(
        f"{BASE_URL}/food/{fdc_id}",
        params={"api_key": API_KEY}
    ).json()

    print(f"产品: {food_detail['description']}")
    for nutrient in food_detail['foodNutrients']:
        name = nutrient['nutrient']['name']
        amount = nutrient.get('amount', 0)
        unit = nutrient['nutrient']['unitName']
        print(f"  {name}: {amount} {unit}")
```

## ⚠️ 注意事项

### 1. 数据质量

- ✅ **品牌食品**（Branded Foods）：由品牌官方提交，准确度高
- ⚠️ **用户贡献**：部分数据可能由用户提交，需验证
- 💡 **建议**：优先使用 `dataType: "Branded"` 过滤

### 2. 数据时效性

- 季度更新，不是实时数据
- 餐厅菜单变化时可能有延迟
- 建议定期重新同步数据

### 3. 搜索技巧

```python
# ❌ 不推荐：过于宽泛
search_foods("burger")  # 返回几千个结果

# ✅ 推荐：精确品牌+产品名
search_foods("McDonald's Big Mac")

# ✅ 推荐：添加过滤
search_foods("Big Mac", dataType="Branded", brandOwner="McDonald's")
```

### 4. API限制应对

如果每小时1000次不够用：

1. **实施缓存**：相同查询不重复请求
2. **批量查询**：使用 `/foods` 端点一次查询多个
3. **申请提额**：联系 USDA 说明用途，通常会批准
4. **多IP轮询**：部署多个服务器分散请求（需遵守服务条款）

## 🔗 相关资源

- **官方文档**：https://fdc.nal.usda.gov/api-guide.html
- **API规范**：https://fdc.nal.usda.gov/api-spec/fdc_api.html
- **注册API Key**：https://fdc.nal.usda.gov/api-key-signup.html
- **数据字典**：https://fdc.nal.usda.gov/data-documentation.html
- **GitHub示例**：https://github.com/littlebunch/fdc-api

## 🆚 对比其他数据源

| 特性 | USDA FDC | Nutritionix | MyFitnessPal |
|------|----------|-------------|--------------|
| **价格** | 免费 | $99+/月 | 需订阅 |
| **权威性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **数据量** | 375k+ | 700k+ | 2M+ |
| **更新频率** | 季度 | 实时 | 实时 |
| **API限制** | 1k/小时 | 按套餐 | 受限 |
| **推荐度** | 🏆 首选 | 💰 预算充足 | 🚫 API限制多 |

## 🎯 NutriAgent 集成建议

### 阶段1：MVP（当前）

```
- 使用 DEMO_KEY 测试
- 手动收集10-15家常见餐厅
- 存储为 JSON 在 data/nutrition/restaurants/
```

### 阶段2：Beta（1-2个月后）

```
- 申请正式API Key
- 实现API调用 + Redis缓存
- 覆盖50+连锁餐厅
```

### 阶段3：生产（3-6个月后）

```
- 定期批量同步到PostgreSQL
- 实现多源数据融合（USDA + 自建数据库）
- 覆盖200+餐厅
```

## 💡 最佳实践

1. **始终缓存结果**：减少API调用
2. **错误处理**：API可能暂时不可用，准备降级方案
3. **数据验证**：检查返回的营养数据是否合理
4. **版本控制**：记录数据更新时间，方便溯源
5. **用户反馈**：允许用户报告数据问题

---

**结论**：USDA FoodData Central 是营养应用的最佳免费数据源，强烈推荐作为 NutriAgent 的主要数据来源。
