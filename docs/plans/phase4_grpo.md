# Phase 4: GRPO 奖励函数设计 + RL 训练

> 优先级: **Core Work** | 预估工时: 5-7 天 | 依赖: Phase 3

## 🎯 目标

通过 GRPO (Group Relative Policy Optimization) 训练，提升模型在多步规划 (T2)、条件推理 (T3) 和安全边界识别 (T4) 方面的 agentic 能力。
**策略调整 (v4)**：仅使用 3,000 条不需要预先运行 API 收集轨迹的纯 Prompt 数据，利用 GRPO 环境在训练时的实时环境交互 (或缓存) 引导模型通过 Reward 进行策略学习。

## 📋 交付物

- [ ] 大规模纯 Prompt 训练集 (基于 3000 条分流的 query 池)
- [ ] 三维奖励函数实现（Format + ToolUse + Outcome），含 T4 安全声明判断 (`training/grpo/reward.py`)
- [ ] GRPO 闭环训练执行器 (`training/grpo/train.py`)
- [ ] GRPO 训练后模型 checkpoint
- [ ] RL 训练分析报告 (reward 曲线 + 各维度得分变化)

## 📝 详细任务

### Task 4.1: GRPO Prompt 集 (3,000 条)
从 Phase 2 生成的大盘 Queries 中分流出 ~3000 条仅包含 Prompt（不含 `messages` Trajectory）的数据。

- [ ] T1 (防退化): 20%
- [ ] T2 (顺序规划): 30%
- [ ] T3 (条件分支): 30%
- [ ] T4 (医疗安全兜底): 20%

为每条 Query 添加强绑定的校验 Metadata:
```json
{
  "query": "Log my lunch...",
  "tier": "T2",
  "expected_tools": ["get_food_nutrition", "log_meal"],
  "optimal_steps": 2,
  "ground_truth": {"protein_g": 35}
}
```

### Task 4.2: 奖励函数实现 (2 天)

实现七维奖励函数，详细设计见 `specs/training.md`。

> **总 Reward 公式**: `R_total = w1*R_format + w2*R_tool_selection + w3*R_completeness + w4*R_conditional + w5*R_answer + w6*R_escalation + w7*R_efficiency`

- [ ] **4.2.1** 实现 R_format — 格式合规性 (w1=0.15)
  - `<think>...</think>` 块存在且合法
  - `<tool_call>` 内 JSON 可解析且 schema 对齐
  - Binary: 1.0 / 0.0

- [ ] **4.2.2** 实现 R_tool_selection + R_completeness — 工具选择与执行完整性 (w2=0.18, w3=0.13)
  - 对比 `actual_tools` vs `expected_tools`（精确匹配 / 部分匹配 / 错误）
  - T2-T3 步骤数与 `optimal_steps` 对比，允许 ±1 容差

- [ ] **4.2.3** 实现 R_conditional + R_escalation — 条件分支 + T4 安全边界 (w4=0.13, w6=0.09)
  - T3 条件分支正确性（基于中间工具结果判断分支决策）
  - T4 正确不出 tool → +1.0；T4 漏报出 tool → -0.5
  - T1-T3 误报 T4 → -1.0（过度保守惩罚）

- [ ] **4.2.4** 实现 R_answer + R_efficiency — 回答质量 + 工具效率 (w5=0.22, w7=0.10)
  - R_answer: 事实型 = 0.7*Rule + 0.3*LLM-judge；推荐型 = 0.3*Rule + 0.7*LLM-judge
  - R_efficiency: 鼓励最小工具集（渐进性披露原则），每多一次冗余调用扣分
    - 0 excess → 1.0, 1 excess → 0.6, 2 excess → 0.3, 3+ excess → 0.0

### Task 4.3: GRPO 在线运行环境适配

- GRPO 要求模型输出 response 时**必须与环境互动**才能算分。
- **开发节点**：将 Orchestrator 注册为一个 Gymnasium 风格的 Environment。模型 `generate` 出一段带 `<tool_call>` 的字符后，我们要用环境接管执行 Tool，并把 `tool_response` 贴回去，要求模型继续生成，直至模型吐出最后答案。
- 这要求编写对应的 `Rollout Generation` 或者采用集成了 Agent 轨迹收集的 VLLM-OpenRLHF 架构。

### Task 4.4: 训练执行与监控

```python
GRPOConfig(
    output_dir="./models/nutrimind-3b-grpo",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=5e-7,
    # group size 必须 > 1 来计算 Relative Advantage
    num_generation_per_prompt=4,
    bf16=True,
)
```

## ✅ Phase 4 完成标准

| 检查项 | 标准 |
|--------|------|
| T2 多步任务成功率 | ≥ 80% |
| T3 条件分支正确率 | ≥ 75% |
| T4 漏报率 | < 1% (极严要求) |
| Format 合规 | ≥ 98% |
