# 使用 SOTA Agent API 收集真实 Trajectory

> 创建时间: 2026-02-27 | 更新时间: 2026-02-28 | 状态: 当前方案

## 1. 背景与动机

### 1.1 问题

当前 SFT 数据生成流程使用 GPT-4o/Gemini **离线合成** trajectory：

```
Seed Data → LLM Expansion → Auto-Check → SFT Training
```

**核心缺陷**：`tool_response` 是 LLM 想象的，不是真实工具执行的结果。

```python
# 合成的 tool_response（LLM 凭记忆编的）
{"calories": 165, "protein": 31}

# 真实的 tool_response（USDA 数据库实际返回）
{"calories": 164.8, "protein": 30.91, "fat": 3.57, "carbs": 0.0, ...}
```

这导致：
- 训练数据与部署时分布不一致
- 模型学不到真实的数据格式、字段结构和数值精度
- 错误处理场景（`food_not_found`、`ambiguous_name` 等）覆盖不足

### 1.2 调研结论

根据对 Trajectory Distillation 的调研（FireAct, Agent-FLAN, AgentTrek 等），**在真实系统中运行 SOTA 模型收集 trajectory** 是更有效的方法：

| 方法 | 数据质量 | 分布一致性 | 错误覆盖 |
|------|---------|-----------|---------|
| 离线合成 | 中 | 低 | 需手工设计 |
| **真实系统运行（本方案）** | 高 | 高 | 自然产生 |

### 1.3 方案演进与选型

调研了多种方案后，最终选定 **API 驱动的 Simulation Loop**：

| 候选方案 | 评估结论 |
|----------|--------|
| Claude Code + MCP 拦截 | ❌ 放弃：用户 Query 和 Answer 无法通过 MCP 获取；Think 内容不暴露给 MCP；Session 边界难以切分 |
| Gemini Thinking API | ❌ 不推荐：Thinking 格式与 Qwen 不兼容；跨生态风格漂移；额外集成成本 |
| **Qwen API Simulation Loop（本方案）** | ✅ 采用：同生态对齐；完整掌控数据流；OpenAI 兼容；Think 格式天然一致 |

---

## 2. 方案概述：API 驱动的 Simulation Loop

### 2.1 核心原理

**放弃"被动拦截"，改为在自有框架内主动驱动 SOTA 教师模型，闭环执行真实工具调用。**

```
┌──────────────────┐  1. Query + Tool Schema  ┌─────────────────┐
│                  ├────────────────────────> │                 │
│   Controller     │  2. <think> + tool_call  │  Qwen3.5-Plus   │
│  (你的 Python)   │ <────────────────────── │  (教师模型 API)  │
│                  │                          │                 │
└────────┬─────────┘                          └─────────────────┘
  3. 执行 │ ▲ 4. 真实 JSON（含 USDA 实际数据）
   Tool  ▼ │
┌──────────────────┐
│  NutriMind Tools │  search_food / calculate_meal / ...
│  （本地真实实现） │  直接查 USDA SQLite，返回真实数值
└──────────────────┘
         │
         │  5. 保存完整 messages 列表
         ▼
[ system, user, assistant(<think>+tool_call), tool(真实数据), assistant(最终答案) ]
```

### 2.2 为什么是 Qwen 生态

目标微调模型是 **Qwen2.5-3B**，教师模型选 Qwen 家族的大杯有以下优势：

1. **Tokenization 完全一致**：`<think>`、`<tool_call>` 等特殊 token encoding 相同
2. **推理风格对齐**：小模型学大模型时不需要克服"风格迁移"障碍
3. **工具调用格式天然一致**：Qwen 系列 function calling schema 统一
4. **价格极具竞争力**：Qwen3-235B 仅需 \$0.26/M input，远低于 Claude/GPT

### 2.3 教师模型选型（2026年2月）

| Tier | 推荐模型 | 理由 | 价格 (Output/M) |
|------|--------|------|----------------|
| **T1 基础查询** | `qwen3.5-flash` | 量大价廉，批量生成 | \$0.29 |
| **T2 多步工具链** | `qwen3-235b-a22b` | Hybrid Thinking 稳定，格式完美 | \$0.78 |
| **T3 条件分支** | `qwen3.5-plus-2026-02-15` | 最新最强 Qwen，推理深度足够 | \$2.40 |
| **T4 专家升级** | `qwen3-max-2026-01-23`（Thinking 版）| 最强推理，escalation 决策需要 | \$6.00 |
| **错误恢复** | `qwen3-coder-next` | 专为工具执行失败+恢复优化 | \$0.30 |

> **500 条高质量轨迹预估总成本：约 \$10–20**（全程 Qwen 生态）

---

## 3. 实现：核心 Simulation Loop

在 `src/training/sft/collect_trajectories.py` 中实现，直接复用现有工具代码。

### 3.1 核心代码

```python
"""
SOTA Agent Trajectory Collector
使用 Qwen API 教师模型 + 本地真实工具，收集高质量 SFT 训练轨迹。
"""
import json
import time
from pathlib import Path
from openai import OpenAI  # Qwen 兼容 OpenAI SDK

# 导入本地真实工具（6 个原子工具）
from src.tools.search_food import search_food
from src.tools.calculate_meal import calculate_meal
from src.tools.log_meal import log_meal
from src.tools.get_today_summary import get_today_summary
from src.tools.get_history import get_history
from src.tools.retrieve_knowledge import retrieve_knowledge
from src.config import settings

# 教师模型配置
TEACHER_MODELS = {
    "T1": "qwen3.5-flash",
    "T2": "qwen3-235b-a22b",
    "T3": "qwen3.5-plus-2026-02-15",
    "T4": "qwen3-max-2026-01-23",
    "error_recovery": "qwen3-coder-next",
}

client = OpenAI(
    api_key=settings.qwen_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 工具 Schema（与 Phase 1 冻结版本一致）
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_food",
            "description": "Search USDA database for food nutrition information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Food name to search (English preferred)"},
                    "limit": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_meal",
            "description": "Calculate total nutrition for a meal given foods and amounts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "foods": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "food_id": {"type": "integer"},
                                "amount_grams": {"type": "number"}
                            },
                            "required": ["food_id", "amount_grams"]
                        }
                    }
                },
                "required": ["foods"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "log_user_data",
            "description": "Log or query user dietary data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["log_meal", "get_today", "get_history"]},
                    "data": {"type": "object"}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": "Retrieve nutrition knowledge from English-language RAG knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                    "domain": {"type": "string", "enum": ["medical", "dietary", "sports", "supplement", "general"]}
                },
                "required": ["query"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are NutriMind, a specialized AI nutrition assistant.

## LANGUAGE RULES (strictly follow)
- ALL output must be in English: <think> blocks, tool parameters, and final answers.
- User may write in Chinese or English — always respond in English.

## BEHAVIOR
Before every tool call, reason step-by-step in <think>...</think> tags.
Use tools only when you need specific data. Answer directly when general knowledge suffices.

## SAFETY BOUNDARY (T4)
If the user's situation involves: dialysis, post-surgery recovery, active cancer,
organ transplant, or complex drug-nutrient interactions — DO NOT use tools.
Instead, respond directly:
"Your situation involves complex medical nutrition management that exceeds my safe
service boundary. Please consult your physician or a registered dietitian."

## TOOL CALL FORMAT
<think>Step-by-step reasoning in English about what data is needed and why.</think>
Then call the tool via function calling with English parameters.
After receiving tool results, synthesize a clear, helpful answer in English.

## AVAILABLE TOOLS
- search_food(name): Look up single food nutrition from USDA database
- calculate_meal(foods[{name, amount_grams}]): Total nutrition for a meal
- log_meal(meal_type, foods[]): Record a meal to user history
- get_today_summary(): Today's intake and remaining calorie budget
- get_history(days, metric): Multi-day nutrition trends
- retrieve_knowledge(query, top_k, domain): Search nutrition knowledge base"""


def execute_tool(name: str, arguments: dict) -> dict:
    """执行本地真实工具，返回真实结果。"""
    try:
        if name == "search_food":
            return search_food(**arguments)
        elif name == "calculate_meal":
            return calculate_meal(**arguments)
        elif name == "log_meal":
            return log_meal(**arguments)
        elif name == "get_today_summary":
            return get_today_summary(**arguments)
        elif name == "get_history":
            return get_history(**arguments)
        elif name == "retrieve_knowledge":
            return retrieve_knowledge(**arguments)
        else:
            return {"status": "error", "message": f"Unknown tool: {name}"}
    except Exception as e:
        # 真实的错误会被记录，用于生成错误恢复轨迹
        return {"status": "error", "error_type": type(e).__name__, "message": str(e)}


def infer_tier(messages: list) -> str:
    """从 messages 中推断 Tier。"""
    tool_calls = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_calls.extend([tc["function"]["name"] for tc in msg["tool_calls"]])

    if not tool_calls:
        # Check if final answer contains safety disclaimer keywords → T4
        final_answers = [m.get("content", "") for m in messages
                         if m.get("role") == "assistant" and not m.get("tool_calls")]
        safety_keywords = ["safe service boundary", "registered dietitian", "consult your physician"]
        if any(kw in ans for ans in final_answers for kw in safety_keywords):
            return "T4"
        return "pure_qa"
    if any(m.get("role") == "tool" and "error" in m.get("content", "") for m in messages):
        return "error_recovery"
    if len(tool_calls) == 1:
        return "T1"
    if len(tool_calls) <= 3:
        return "T2"
    return "T3"


def simulate_trajectory(query: str, tier_hint: str = "T2") -> dict | None:
    """
    运行单条 Trajectory 仿真。
    返回包含完整 messages 列表的 dict，可直接转为 SFT 训练格式。
    """
    model = TEACHER_MODELS.get(tier_hint, TEACHER_MODELS["T2"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]

    max_turns = 6  # 防止死循环
    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # 将 assistant 消息加入历史
        messages.append(msg.model_dump(exclude_none=True))

        # 如果模型没有调用工具，对话结束
        if not msg.tool_calls:
            break

        # 执行每一个工具调用（核心：真实本地工具）
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            real_result = execute_tool(name, args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": json.dumps(real_result, ensure_ascii=False)
            })

    tier = infer_tier(messages)
    return {
        "query": query,
        "tier": tier,
        "messages": messages,
        "metadata": {
            "teacher_model": model,
            "real_tool_executed": True,
            "turns": len([m for m in messages if m["role"] == "tool"]),
        }
    }


def batch_collect(queries: list[dict], output_path: str):
    """
    批量收集 trajectory。
    queries 格式: [{"query": "...", "tier_hint": "T2"}, ...]
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    collected = 0
    for item in queries:
        try:
            traj = simulate_trajectory(item["query"], item.get("tier_hint", "T2"))
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            collected += 1
            print(f"[{collected}/{len(queries)}] {item['query'][:50]}... → {traj['tier']}")
            time.sleep(0.5)  # 避免请求过快
        except Exception as e:
            print(f"[ERROR] Query failed: {item['query'][:50]}... | {e}")

    print(f"\n✅ 收集完成：{collected}/{len(queries)} 条轨迹 → {output_path}")
```

### 3.2 System Prompt 设计原则

为了保证 `<think>` 内容被教师模型输出到正文（而不是进入 extended thinking 字段），Prompt 设计遵循：

1. **明确要求**：在 `SYSTEM_PROMPT` 中显式要求每次工具调用前必须输出 `<think>` 标签
2. **格式示例**：在 Prompt 中展示期望的 `<think>` + 工具调用的格式
3. **验证**：收集后检查每条轨迹中 `<think>` 是否存在，不存在则过滤或 Post-hoc 补充

---

## 4. 数据收集流程

### 4.1 Query 准备（复用 Phase 2 种子）

直接复用 Phase 2 中已有的 ~150 条种子数据，按 Tier 分配：

```python
# scripts/prepare_collection_queries.py

queries = []
for seed in load_jsonl("data/seeds/all_seeds.jsonl"):
    queries.append({
        "query": seed["messages"][1]["content"],   # user message
        "tier_hint": seed["tier"]
    })

# 也可扩展：对每条种子生成 2-3 个变体
save_json(queries, "data/queries/collection_queries.json")
```

### 4.2 批量执行

```bash
# 收集 500 条轨迹（约 1-2 小时）
python -m src.training.sft.collect_trajectories \
    --queries data/queries/collection_queries.json \
    --output data/trajectories/real_trajectories.jsonl \
    --limit 500
```

### 4.3 验证与清洗

```bash
# 复用现有 validate_rules.py，检查格式合规性
python src/training/sft/validate_rules.py \
    --input data/trajectories/real_trajectories.jsonl

# 输出示例:
# Total: 520
# Valid: 498 (95.8%)
# Filtered: 22
#   - Missing <think>: 10
#   - Tool call format error: 7
#   - Empty final answer: 5
```

---

## 5. 与现有 SFT 训练流程集成

### 5.1 格式转换

收集到的 messages 格式已经和 SFT 训练格式高度一致，只需做轻量转换：

```python
def convert_to_sft_format(traj: dict) -> dict:
    """已经是 messages 格式，只需确认 system prompt 一致即可。"""
    messages = traj["messages"]
    # 验证 system prompt 是最新版本
    assert messages[0]["role"] == "system"
    return {"messages": messages, "tier": traj["tier"]}
```

### 5.2 数据混合策略（更新版）

| Tier | 合成数据占比 | 真实 Trajectory 占比 | 理由 |
|------|------------|---------------------|------|
| T1 | 80% | 20% | 简单查询，合成足够；真实数据验证分布 |
| T2 | 50% | 50% | 均衡，两者互补 |
| T3 | 20% | 80% | 条件分支依赖真实中间结果 |
| T4 | 20% | 80% | escalation 决策需要真实推理链 |
| 错误恢复 | 0% | 100% | 合成的错误场景质量差，全用真实的 |

---

## 6. 预期效果

### 6.1 对比实验设计

| 实验组 | 数据来源 | 预期效果 |
|--------|---------|---------| 
| A. Baseline | 纯合成 trajectory | 基线 |
| B. Real | 纯 API 仿真真实 trajectory | T3/T4 提升明显 |
| C. Mixed | 合成 + 真实混合 | 最佳性价比 |
| D. Mixed + GRPO | C + GRPO 强化 | 最终最佳 |

### 6.2 预期提升（基于 FireAct 等研究估计）

| 指标 | Baseline (A) | Mixed (C) | Mixed + GRPO (D) |
|------|-------------|-----------|-----------------|
| T1 准确率 | ~95% | ~96% | ~97% |
| T2 成功率 | ~70% | ~80% | ~85% |
| T3 条件分支 | ~60% | ~75% | ~80% |
| T4 升级精准 | ~75% | ~85% | ~90% |

**关键假设**：真实 trajectory 对 T3 条件分支帮助最大，因为条件分支的正确性依赖于真实的中间工具返回值。

---

## 7. 为什么不用 Gemini / Claude

| 跨生态模型 | 问题 |
|-----------|------|
| Gemini 3.1 Thinking | Thinking 内容在独立字段，不在正文；格式与 Qwen `<think>` 标签不兼容；额外 SDK 适配成本 |
| Claude Sonnet 4.x | API 价格高（Output \$15/M）；非 Qwen 生态，需要 prompt 引导 `<think>` 格式并验证稳定性 |
| DeepSeek V3.2 | 可作为补充（OpenAI 兼容，\$0.42/M Output），但 `<think>` 格式与 Qwen 有细微差异 |

---

## 8. 下一步行动

- [ ] 实现 `src/training/sft/collect_trajectories.py`
- [ ] 准备 `data/queries/collection_queries.json`（从 Phase 2 种子提取）
- [ ] Pilot：收集 50 条验证格式和 `<think>` 内容稳定性
- [ ] 批量收集 500+ 条（按 Tier 分配）
- [ ] 运行 `validate_rules.py` 验证
- [ ] 与现有合成数据混合，生成 `mixed_trajectories.jsonl`
- [ ] 对比实验：合成 vs 真实 vs 混合

---

## 参考

- [FireAct: Toward Language Agent Fine-tuning](https://arxiv.org/abs/2310.05915)
- [Agent-FLAN: Designing Data and Methods of Effective Agent Tuning](https://arxiv.org/abs/2403.12881)
- [AgentTrek: Agent Trajectory Synthesis via Guiding Replay](https://agenttrek.github.io/)
- [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789)
- [Qwen3 技术报告](https://qwen.readthedocs.io/)
