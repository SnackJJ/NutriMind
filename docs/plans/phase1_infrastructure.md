# Phase 1: 基础设施搭建（Orchestrator + Tools）

> 优先级: Foundation | 预估工时: 6-8 天 | 依赖: 无

## 🎯 目标

搭建完整的 agentic 运行基础设施，使系统能够执行 observe → think → act → observe 循环。

## 📊 任务依赖图

```
Task 1.1 (环境) ─┬─→ Task 1.2 (数据库) ─┬─→ Task 1.3 (工具) ─→ Task 1.4 (Orchestrator) ─→ Task 1.6 (Smoke Test)
                 │                       │
                 └─→ Task 1.5 (RAG) ─────┘
```

## 📋 交付物

- [x] 可运行的 Orchestrator 状态机
- [x] **8 个原子工具**实现 + 单元测试（已重构 log_user_data 为三个工具，新增目标管理两个工具）
- [x] USDA SQLite 数据库 (已有数据)
- [x] RAG 向量存储初始化脚本 — **已拆分至 Phase 1.2**
- [x] 统一日志与监控框架
- [x] 端到端 smoke test (模拟完整 agentic loop)

## 📝 详细任务

### Task 1.1: 环境与依赖 (0.5 天)

- [x] **1.1.1** 确认 `pyproject.toml` 包含所有必需依赖
  - vllm, transformers, torch, trl, peft, datasets
  - chromadb, sentence-transformers
  - openai, httpx, pydantic, loguru
- [x] **1.1.2** 创建 `.env.example` 模板文件，包含所需环境变量
  ```
  QWEN_API_KEY=xxx
  OPENAI_API_KEY=xxx
  ```
- [ ] **1.1.3** 验证虚拟环境可正常安装所有依赖
  ```bash
  uv sync
  ```
- [x] **1.1.4** 完善 `configs/` 配置结构
  - `configs/model.yaml` — 模型推理配置 (server URL, timeout, max_tokens)
  - `configs/tools.yaml` — 工具配置 (数据库路径, RAG 参数)
  - `configs/orchestrator.yaml` — 状态机配置 (max_turns, safety_thresholds)
- [x] **1.1.5** 实现统一日志框架
  - 配置 loguru 输出格式 (console + file rotation)
  - 定义日志级别规范 (DEBUG/INFO/WARNING/ERROR)
  - 添加结构化日志字段 (request_id, tool_name, latency_ms)

### Task 1.2: 数据库搭建 (1 天)

- [x] **1.2.1** 实现 `scripts/download_usda.py`
  - 下载 USDA SR Legacy & Foundation Foods CSV
  - 解析并导入 SQLite (按 `specs/database.md` schema)
  - 构建 FTS5 全文索引
  - 填入 `nutrient_types` 核心营养素 ID 映射
- [ ] **1.2.2** 实现 `food_aliases` 初始填充
  - 解析 USDA description 提取常用简称
  - 手动补充常见别名 (chicken breast, egg, rice 等)
- [x] **1.2.3** 创建 `user_profiles` 默认记录
- [x] **1.2.4** 创建 `user_goals` 表（存储用户营养目标）
  - `user_id`, `metric` (calories/protein/fat/carbs), `target_value`, `created_at`, `updated_at`
  - 默认目标从 `user_profiles.tdee_kcal` 初始化 calories 目标
- [x] **1.2.5** 数据库完整性测试
  - 食物总数 ≥ 8000
  - 核心营养素覆盖率 ≥ 95%
  - FTS 搜索功能验证

### Task 1.3: 工具实现 (2.5 天)

实现路径: `src/tools/`，每个工具需遵循 `specs/tools.md` 定义。

- [x] **1.3.1** `get_food_nutrition.py` — 食物营养查询（合并原 search_food + calculate_meal）
  - 单/多食物统一接口 `get_food_nutrition(foods[{food_name, amount_grams}])`
  - 实现 exact match → fallback ANY-match 查找链
  - 处理 `food_not_found`, `all_foods_not_found` 错误
  - 编写单元测试

- [x] ~~**1.3.2** `calculate_meal.py`~~ — 已合并入 `get_food_nutrition.py`

- [x] **1.3.3a** `log_meal.py` — 记录单次餐食
  - 实现 `log_meal(meal_type, foods[])` 写入 meal_logs + meal_log_items
  - 编写单元测试

- [x] **1.3.3b** `get_today_summary.py` — 当日摄入汇总
  - 实现 `get_today_summary()` 查询当日摄入 + 剩余预算
  - 编写单元测试

- [x] **1.3.3c** `get_history.py` — 多日历史趋势
  - 实现 `get_history(days, metric)` 查询历史 daily_summary 视图
  - 编写单元测试

> 原 `log_user_data.py` 已拆分为上述 3 个原子工具，旧文件需删除。

- [x] **1.3.4** `retrieve_knowledge.py` — RAG 知识检索
  - **注意：工具实现随 Phase 1.2 RAG 重写一并完成**
  - 详见 `docs/plans/phase1.2_rag_knowledge_base.md` Phase F

> `call_expert_nutritionist` / `expert_client.py` 已由架构决策删除（T4 改为安全边界声明，无需云端 API）。

- [x] **1.3.5a** `set_goal.py` — 设定/调整营养目标
  - 实现 `set_goal(metric, target_value, goal_type?)` 写入 user_goals 表
  - 安全约束：calories 限制 1000-5000 kcal，macro 限制 > 0
  - 返回 previous_value + new_value
  - 编写单元测试
  - > ⚠️ **[2026-03-08] 从 SFT schema 移除** — query pool 中无对应查询类型（0/2495 覆盖），模型无法学习；`src/tools/set_goal.py` 保留供 GRPO 阶段使用

- [x] **1.3.5b** `get_goal_adherence.py` — 目标达成率分析
  - 实现 `get_goal_adherence(days, metric)` 聚合 daily_summary vs user_goals
  - 计算 adherence_pct, days_within_target, avg_deviation
  - 编写单元测试

### Task 1.4: Orchestrator 实现 (1.5 天)

实现路径: `src/orchestrator/`，遵循 `specs/orchestrator.md` 状态机设计。

- [x] **1.4.1** 定义数据结构
  - `OrchestratorConfig` dataclass
  - `ParsedOutput` / `ToolCall` 类型
  - 状态枚举 (START, INFERENCE, TOOL_CALL, TOOL_EXEC, CHECK_LIMIT, ANSWER, ERROR, END)

- [x] **1.4.2** 实现 `parse_model_output()` — 响应解析
  - 提取 `<think>` 块
  - 提取 `<tool_call>` 块并 JSON 解析
  - 识别 final answer (无 tool_call)
  - 处理 parse error

- [x] **1.4.3** 实现 `orchestrate()` 主循环
  - START: 构建初始 messages (system prompt + user input)
  - INFERENCE: 调用 3B 模型
  - TOOL_CALL → TOOL_EXEC: 分发执行工具
  - CHECK_LIMIT: 检查轮次上限
  - ANSWER: 返回最终答案
  - ERROR: 错误恢复 / 回退

- [x] **1.4.4** 实现安全检查与错误处理
  - `safety_check_expert_response()` — 过敏原检测 + 极端卡路里值拦截
  - 格式恢复提示 (invalid format → 引导重新生成)
  - 工具执行超时处理 (单工具 30s 上限)
  - 状态机死循环检测 (连续 N 次相同工具调用 → 强制终止)

- [x] **1.4.5** 实现 `src/orchestrator/inference.py` — 推理客户端
  - 抽象 `InferenceBackend` 接口
  - `MockBackend` — 本地开发测试用 (返回预设响应)
  - `VLLMBackend` — 服务器高性能推理 (vLLM serve)
  - 通过配置切换 backend (`configs/model.yaml`)

- [x] **1.4.6** 添加关键指标埋点
  - 工具调用耗时 (tool_name, latency_ms)
  - Token 使用量 (prompt_tokens, completion_tokens)
  - 状态机轮次统计 (turns_count, final_state)

### Task 1.5: RAG 知识库初始化

> **已拆分至 Phase 1.2**: 全部 RAG 相关任务（知识源收集、解析、分块、索引、检索）
> 已迁移到独立的 `phase1.2_rag_knowledge_base.md` 计划。
> 详见 `docs/specs/rag.md` 和 `docs/plans/phase1.2_rag_knowledge_base.md`。

### Task 1.6: 端到端 Smoke Test (0.5 天)

- [x] **1.6.1** 编写集成测试脚本 (`tests/test_smoke.py`)
  - T1: 单步食物查询 → `get_food_nutrition` → 回答
  - T1: 目标达成率查询 → `get_goal_adherence` → 回答
  - T2: 多食物计算 → `get_food_nutrition` → `log_meal`
  - T2: 设定目标并查看进度 → ~~`set_goal`~~ → `get_goal_adherence`（`set_goal` 已从 SFT schema 移除）
  - T3: 条件分支 (`get_today_summary` → 预算检查 → `retrieve_knowledge`)
  - T3: 条件分支 (`get_goal_adherence` → 达标率低 → `retrieve_knowledge`)
  - T4: 安全边界声明（无工具调用，模型直接输出免责声明）
  - Pure QA: 直接回答

- [x] **1.6.2** 使用 mock model 验证 orchestrator 完整流程
  - 模拟模型输出 (返回预设的 tool_call)
  - 验证状态机转换正确性
  - 验证工具调用链完整性

## ✅ Phase 1 完成标准

| 检查项 | 标准 |
|--------|------|
| Orchestrator 状态机 | 所有状态转换路径可执行 |
| **8 个原子工具** | 各自有 ≥ 3 个单元测试通过 |
| USDA 数据库 | ≥ 8000 食物, FTS 搜索正常 |
| RAG 索引 | **已拆分至 Phase 1.2** |
| 端到端 | Mock-model smoke test 全部通过 |
