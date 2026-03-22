# Phase 5: 消融实验 (Ablation Experiments)

> 优先级: **Core Work** | 预估工时: 3-4 天 | 依赖: Phase 4

## 🎯 目标

通过系统性消融实验，量化 RL 对 agentic 能力各层面的贡献，为 resume narrative 提供有力数据支撑。

## 📋 交付物

- [ ] 完整消融实验结果矩阵
- [ ] SFT vs SFT+GRPO 分层对比报告
- [ ] 4B (SFT+GRPO) vs GPT-4o API 成本-性能对比
- [ ] 各奖励维度贡献分析
- [ ] LaTeX/Markdown 格式化报告

## 📝 详细任务

### Task 5.1: 实验设计 (0.5 天)

- [ ] **5.1.1** 定义消融实验矩阵

| 实验 ID | 模型配置 | 目的 |
|---------|---------|------|
| **A** | Qwen3-4B (base, no fine-tune) | Zero-shot 基线 |
| **B** | SFT-only (Phase 3 模型) | SFT 基线 |
| **C** | SFT + GRPO (Phase 4 模型) | 完整方案 |
| **D** | GPT-4o (直接 API 调用, 同样的 tools) | 上限参照 |
| **E** | SFT + GRPO (仅 R_format + R_tool) | 最小 reward 消融 |
| **F** | SFT + GRPO (无 R_conditional) | 条件 reward 消融 |

- [ ] **5.1.2** 设计统一评估测试集
  - **T1 测试集**: 100 条 (不在训练集中)
  - **T2 测试集**: 80 条 (不在训练集中)
  - **T3 测试集**: 60 条 (不在训练集中)
  - **T4 测试集**: 40 条（Type B 触发安全声明）+ 40 条 T1-T3（Type A 不应触发声明）
  - **Pure QA**: 20 条
  - **错误处理**: 20 条
  - 总计: 320 条

- [ ] **5.1.3** 定义评估指标
  - Per-tier success rate
  - Format validity rate
  - Average tool count per task (效率)
  - Redundant tool call rate
  - Error recovery rate
  - **T4 安全声明 precision / recall**（是否正确识别高危场景并输出免责声明）
  - Answer quality (LLM-judge)
  - End-to-end latency

### Task 5.2: 评估基础设施 (0.5 天)

- [ ] **5.2.1** 实现统一评估框架 `training/evaluate.py`
  ```python
  def evaluate_model(model_config, test_set, tools_env) -> EvalReport:
      """Run full agentic evaluation for one model configuration."""
      results = []
      for sample in test_set:
          trajectory = run_agentic_loop(model_config, sample["query"], tools_env)
          metrics = compute_metrics(trajectory, sample["metadata"])
          results.append(metrics)
      return aggregate_results(results)
  ```

- [ ] **5.2.2** 实现 GPT-4o 对比评估
  - 用同样的 system prompt + tool schemas 调用 GPT-4o
  - 记录: 准确率、工具使用模式、延迟、成本

- [ ] **5.2.3** 实现结果自动生成
  - JSON 结构化输出
  - Markdown 报告模板
  - 对比表格自动生成

### Task 5.3: 核心消融 — SFT vs SFT+GRPO (1 天)

- [ ] **5.3.1** 运行实验 A (Base model)
- [ ] **5.3.2** 运行实验 B (SFT-only)
- [ ] **5.3.3** 运行实验 C (SFT+GRPO)
- [ ] **5.3.4** 生成核心对比表

  ```
  ┌─────────────────────────────────────────────────────────┐
  │          SFT vs SFT+GRPO — Per-Tier Comparison          │
  ├──────────┬────────┬──────────┬───────────┬──────────────┤
  │ Metric   │  Base  │ SFT-only │ SFT+GRPO │  Δ (RL)     │
  ├──────────┼────────┼──────────┼───────────┼──────────────┤
  │ T1 Acc   │        │          │           │              │
  │ T2 Succ  │        │          │           │              │
  │ T3 Corr  │        │          │           │              │
  │ T4 Safety│        │          │           │              │
  │ Format   │        │          │           │              │
  │ Redund.  │        │          │           │              │
  │ Recovery │        │          │           │              │
  │ QA Qual  │        │          │           │              │
  └──────────┴────────┴──────────┴───────────┴──────────────┘
  ```

- [ ] **5.3.5** 分层深入分析
  - T2 失败案例分析: 哪些工具链模式 RL 改进了？
  - T3 失败案例分析: 哪类条件分支更难学？
  - 错误恢复分析: RL 是否改善了重试行为？

### Task 5.4: 成本-性能消融 — 3B vs GPT-4o (0.5 天)

- [ ] **5.4.1** 运行实验 D (GPT-4o)
- [ ] **5.4.2** 生成成本-性能对比

  ```
  ┌────────────────────────────────────────────────────────┐
  │         3B (SFT+GRPO) vs GPT-4o — Cost-Performance     │
  ├──────────┬───────────────┬──────────┬──────────────────┤
  │ Metric   │ 3B (SFT+GRPO) │  GPT-4o  │  Ratio          │
  ├──────────┼───────────────┼──────────┼──────────────────┤
  │ T1 Acc   │               │          │                  │
  │ T2 Succ  │               │          │                  │
  │ T3 Corr  │               │          │                  │
  │ Latency  │               │          │                  │
  │ Cost/req │               │          │                  │
  │ $/1K req │               │          │                  │
  └──────────┴───────────────┴──────────┴──────────────────┘
  ```

### Task 5.5: Reward 维度消融 (1 天)

- [ ] **5.5.1** 训练实验 E (仅 R_format + R_tool_selection)
  - 设置 w3=w4=w5=w6=0
  - 快速训练 (可用更少步数)

- [ ] **5.5.2** 训练实验 F (移除 R_conditional)
  - 设置 w4=0, 其余不变
  - 观察 T3 条件分支是否仍有改善

- [ ] **5.5.3** 生成 Reward 维度贡献分析
  - 各维度对最终性能的边际贡献
  - 哪些维度是必需的，哪些是锦上添花

### Task 5.6: 报告撰写 (0.5 天)

- [ ] **5.6.1** 整合所有实验数据
- [ ] **5.6.2** 输出结构化报告
  - **核心发现**: RL 对每个 Tier 的量化提升
  - **Insight 1**: T2-T3 提升的关键因素
  - **Insight 2**: 3B 模型的能力天花板在哪里
  - **Insight 3**: 成本-性能最优点分析
  - **Insight 4**: Reward 设计的权衡

## ✅ Phase 5 完成标准

| 检查项 | 标准 |
|--------|------|
| 消融矩阵 | 所有 6 个实验完成 |
| SFT vs SFT+GRPO | T2-T3 有统计显著提升 |
| GPT-4o 对比 | 成本-性能数据齐全 |
| Reward 消融 | 各维度贡献量化 |
| 报告 | 结构化 + 数据可视化 |
