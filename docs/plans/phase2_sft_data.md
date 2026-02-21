# Phase 2: SFT 数据生成 (v4 混合真实仿真路线)

> 优先级: Foundation | 预估工时: 8-10 天 | 依赖: Phase 1

## 🎯 目标

生成 8,000-10,000 条高质量**全英文** SFT 训练数据。覆盖 T1-T4 四个复杂度层级 + 对比式 Pure QA + 错误恢复。
全面采用 **"混合生成：离线合成 + 真实 API 仿真 (Simulation Loop)"** 的策略，彻底对齐真实系统的工具分布、数值精度与格式规范。

## 📋 交付物

- [ ] ~5,000 条统一生成的 Query 种子
- [ ] 分流提取 ~2,000 条进行 Teacher Model Trajectory 仿真
- [ ] 存留 ~1,500 条高纯度黄金数据进入 Train/Val JSONL
- [ ] ~3,000 条留存的 Query 直接输送给 GRPO 题库
- [ ] 基于 Qwen 生态大模型的 API Simulation Loop 数据采集器 (`collect_trajectories.py`)
- [ ] 格式归一化脚本 (`normalize.py`)
- [ ] 自动验证脚本 (`validate_rules.py` + `validate_semantic.py`)
- [ ] 数据分布统计报告

## 📊 目标数据分布 (按初次分流提取池 5000 条总计分布计算)

| Tier | 终态数量 (Queries) | 分布占比 | 生成策略 (仿真:GRPO) | 描述 |
|------|--------------------|----------|----------------------|------|
| **T1**: 单步工具调用 | ~1,750 | 35% | 分流抽取 | 简单查询 → `get_food_nutrition` / `get_today_summary` → answer |
| **T2**: 多步工具链 | ~1,250 | 25% | 分流抽取 | `get_food_nutrition` → `log_meal` 等顺序规划 |
| **T3**: 条件分支 | ~750 | 15% | 分流抽取 | 强依赖真实返回结果执行分支判定 |
| **T4**: 安全声明 | ~500 | 10% | 分流抽取 | **零工具调用**。识别医疗红线并输出免责声明 |
| **Pure QA** | ~500 | 10% | 分流抽取 | 无需工具。同主题配对式生成 |
| **Error Recovery**| ~250 | 5% | 分流抽取 | 完全依赖真实工具环境产生报错 (如 `food_not_found`) |

> **关键架构决策对齐 (v4)**:
> 1. 原 `log_user_data` 拆除，分为 `log_meal`, `get_today_summary`, `get_history` 3 个独立原子工具
> 2. `search_food` + `calculate_meal` 合并为 `get_food_nutrition`，单工具处理所有食物营养查询（单食物/多食物）
> 3. T4 完全不调用工具，作为免责声明（兜底）处理
> 4. 语言架构必须实现 **Full English** (User Query, Think, Parameters, Final Answer 全英文)

## 🔄 分阶段策略

采用 **"快速验证 → 数据驱动迭代"** 原则：

### Phase 2a: 基础数据生成与分流 (本阶段)

1. **统一生成种子 Query 池 (5,000 条)**:
   - 全面覆盖 T1-T4, Pure QA, Error Recovery。
   - 所有 Query 必须为全英文。
2. **分流验证 (SFT vs GRPO)**:
   - 从 5,000 条种子中提取 **~2,000 条** 用作 SFT 训练源数据。
   - 剩余的 **~3,000 条** 留作未来 Phase 4 GRPO 的 Prompt 题库，**不需要在当前步骤运行 Teacher Model 跑轨迹**。
3. **SFT 轨迹仿真与清洗**:
   - 将上述挑出的 2,000 条 SFT Query 打入 `collect_trajectories.py`，让 Teacher API (Qwen 大杯) 结合本地 Tools 跑出真实轨迹。
   - 经过 `normalize`、`validate_rules` 以及 `validate_semantic` 清洗后，预计存活 **~1,500 条** 高纯度黄金数据进入 `train.jsonl` 作为基座行为克隆使用。

### Phase 2b: 条件触发扩充

| Eval 结果 | 诊断 | 行动 |
|-----------|------|------|
| SFT 存活率 < 50% | 规则/裁判过于严格或 Prompt 偏差 | **优化 Validator 或调整 Teacher Prompt** |
| 整体存活达标 | 数据库基底稳固 | **直接进入 SFT (Phase 3) 和 GRPO 准备 (Phase 4)** |

---

## 📝 详细任务

### Task 2.0: 前置验证 (0.5 天)

- [ ] **2.0.1** 确认真实工具测试完毕，包括 Mock DB 和知识库返回行为正常。
- [ ] **2.0.2** 确认 6 个原子工具的 JSON Schema 定义已被冻结，参数均为 snake_case。
- [ ] **2.0.3** 敲定 Full English System Prompt。

### Task 2.1: 种子数据编写 (2 天)

围绕 5 个人工构造的维度矩阵：食物类型 / 用户意图 / 用户画像 / 查询风格 / 生僻边界。均为**全英文**。

#### 2.1.1 T1 种子 (30-35 条)
常规单工具：
```
assistant: <think>Simple food lookup. Single-step tool call to get macros.</think>
           <tool_call>{"name": "get_food_nutrition", "arguments": {"foods": [{"food_name": "chicken breast", "amount_grams": 100}]}}</tool_call>
```

#### 2.1.2 T2 种子 (25-30 条)
**注意**：不需要先 search，直接算。
```
assistant: <think>Multi-step T2: look up meal nutrition first, then log it.</think>
           <tool_call>{"name": "get_food_nutrition", "arguments": ...}</tool_call>
```

#### 2.1.3 T3 种子 (15-20 条)
核心要求：必须有前后信息依赖（查完 A，基于 A 的结果查 B）。

#### 2.1.4 T4 种子 (安全边界声明，12-15 条)
**T4 = 零工具调用**。
必须在 `<think>` 中描述高危原因，直接输出 Disclaimer。包括 Type A / Type B 变种。

#### 2.1.5 Pure QA 种子 (20-25 对)
包含负向判断的同主题变种（不需要用工具）。

#### 2.1.6 错误恢复 (10-15 条)
故意给予不存在的食物/打错单位，观察模型的 `<think>` 反思及新 `tool_call` 行动。

---

### Task 2.2: Simulation Loop (API 驱动收集管线) (2 天)

**教师模型选型 (Qwen 全家桶)**：
保证 `<think>` 标签输出格式和 Tokenizer 分布严格拉齐。
- T1 (合成主导): `qwen3.5-flash`
- T2 (半合成半真): `qwen3-235b-a22b` 
- T3 (重逻辑推导): `qwen3.5-plus`
- T4 (医疗安全判定): `qwen3-max`
- 错误恢复: `qwen3-coder-next`

**开发环节**：
- [ ] 编写 `src/training/sft/collect_trajectories.py`
- [ ] 设置 API 调用拦截与本地 `execute_tool` 流转环路
- [ ] 以 `qwen3.5-flash` 批量扩充纯合成数据 (基于 avoid list 与 varied context)
- [ ] 归总 `mixed_trajectories.jsonl`，保留 `teacher_model` 和 `real_tool_executed` 标记

---

### Task 2.3: 格式归一化与洗牌 (0.5 天)

- [ ] **2.3.1** 实现 `training/sft/normalize.py`。确保 JSON 解析合法、脱去 markdown fence，统一 `think` 和 `tool_call` 格式对齐 Qwen 原生。
- [ ] **2.3.2** 附带完整元数据:
  ```json
  {
    "source_model": "qwen3.5-plus",
    "is_simulated": true,
    "tier": "T3",
    "data": { ... }
  }
  ```

---

### Task 2.4: 自动验证管线 (1.5 天)

#### 2.4.1 规则验证
- 严格限定仅含 6 种合法工具
- **强制约束 100% English**：`<think>` 块和回答中禁止出现 >5% 比例中文字符
- 校验模型产生的 T4 是否恰好为 0 次 `tool_call`
- 校验单次循环以 `answer` 结束，且中间不含有孤立 `think`

#### 2.4.2 语义裁判 (LLM-as-Judge)
- 借由 `GPT-4o-mini` 或者 `Qwen-Max` 打回"工具调了但最终回答牛头不对马嘴"的冗余样本。

---

### Task 2.5: 数据后处理与划分 (0.5 天)

- 淘汰无效数据、清洗重复数据（余弦相似度排重）。
- 按 90:10 拆分 train/val、统计按层级分布指标报告。

---

### Task 2.6: 人工质量抽查 (0.5 天)

- 抽样验证数值逼真度 (是否有离谱营养比例)。
- T4 防御漏报率核查。

## ✅ Phase 2 完成标准

| 检查项 | 标准 |
|--------|------|
| 生成数据量 | ~10,000 条 |
| 清洗后数据量 | ≥ 8,000 条 |
| API 仿真比例 | T3及T4 达到核心占比 (80%) |
| 规则验证通过率 | ≥ 95% |
| 全英语合规率 | 100% 英文执行与回答 |
| 工具正确性 | get_food_nutrition 等调用遵循独立解耦逻辑 |
| 训练集/验证集 | Train 90% / Val 10% + 统计报告 |

---
## Detailed Sub-Plans
- **[Phase 2.5: 数据扩充管线 (SFT Data Scaling Pipeline)](phase2.5_sft_data_pipeline.md)**
- **[Phase 2.6: 轨迹收集与验证 (Trajectory Collection & Validation)](phase2.6_trajectory_collection.md)**