# NutriMind Agent 总执行计划 (Master Plan)

> 基于 PRD v2 | 创建时间: 2026-02-23 | 状态: 初版

## 📋 项目概览

**核心目标**: 探索通过 SFT + GRPO 训练，让 Qwen3-4B 成为营养领域可靠的 agentic problem-solver，建立 4B 模型在固定 6 工具集内的能力边界，通过安全声明兜底高风险场景。

**核心研究问题**: RL 能将 4B 模型的 agentic 能力推到多远？哪里是必须触发安全边界声明的临界点？

---

## 🏗️ 阶段总览

| 阶段 | 名称 | 优先级 | 预估工时 | 依赖 | 关键交付物 |
|------|------|--------|---------|------|-----------|
| **Phase 1** | 基础设施搭建 (Orchestrator + 8 原子工具) | Foundation | 5-7 天 | 无 | 可运行的 agentic loop + 8 个原子工具实现 |
| **Phase 2** | SFT 种子与初步轨迹收集 (T1-T4) | Foundation | 4-5 天 | Phase 1 | ~1,500 条 SFT + 3,000 GRPO Prompts |
| **Phase 3** | SFT 训练 | Foundation | 2-3 天 | Phase 2 | SFT 基线模型 (T1 ≥ 95%, Format ≥ 98%) |
| **Phase 4** | GRPO 奖励函数设计 + RL 训练 | **Core** | 5-7 天 | Phase 3 | GRPO 优化模型 (T2 ≥ 80%, T3 ≥ 75%) |
| **Phase 5** | 消融实验 (SFT vs SFT+RL, 分层分析) | **Core** | 3-4 天 | Phase 4 | 消融对比报告 |
| **Phase 6** | 离线评估 + Agentic 能力边界分析 | **Core** | 3-4 天 | Phase 5 | 完整评估报告 + 边界分析 |
| **Phase 7** | 部署 (vLLM serving) + 监控 | Wrap-up | 2-3 天 | Phase 6 | 可部署的完整系统 |

**总预估工时**: 24-33 天

**重点提示**: Phase 1-3 是前置条件，应快速完成；Phase 4-6 是项目核心，主要精力应集中于此。

---

## 📊 阶段依赖关系图

```
Phase 1 (基础设施)
  │
  ├──▶ Phase 2 (SFT 数据生成)
  │      │
  │      └──▶ Phase 3 (SFT 训练)
  │             │
  │             └──▶ Phase 4 (GRPO 训练)  ◀── 核心工作
  │                    │
  │                    └──▶ Phase 5 (消融实验)  ◀── 核心工作
  │                           │
  │                           └──▶ Phase 6 (评估 + 边界分析)  ◀── 核心工作
  │                                  │
  │                                  └──▶ Phase 7 (部署 + 监控)
  │
  └──▶ [可并行] 知识库文档收集 (供 Phase 2 使用)
```

---

## 🎯 成功指标速查

### Agentic 能力指标 (核心 — 衡量 RL 效果)

| 指标 | 目标 | 主要考查阶段 |
|------|------|-------------|
| T1 单步工具调用准确率 | ≥ 95% | SFT 基线 |
| T2 多步任务成功率 | ≥ 80% | **RL 核心** |
| T3 条件分支正确率 | ≥ 75% | **RL 核心** |
| T4 安全声明精准率 (正确识别高危场景并输出声明) | ≥ 85% | SFT + RL |
| T4 安全声明召回率 (不遗漏高危场景) | ≥ 90% | SFT + RL |
| 冗余工具调用率 | ≤ 10% | RL 效率 |
| 错误恢复率 (工具失败后重试) | ≥ 60% | RL |

### 基线指标

| 指标 | 目标 |
|------|------|
| 工具调用格式有效性 | ≥ 98% |
| 营养 QA 质量 (无工具查询) | 相比基座模型下降 ≤ 3% |

### 系统级指标 (部署后)

| 指标 | 目标 |
|------|------|
| P50 延迟 (本地, TransformersBackend) | < 2s |
| P50 延迟 (本地, VLLMBackend) | < 1s |
| 服务稳定性 (100 连续请求) | 0 crash |

---

## 🔧 技术栈概要

| 组件 | 技术选型 |
|------|---------|
| 基座模型 | Qwen3-4B |
| 教师模型 (数据采集用) | Qwen-Max / Gemini 2.5 Flash (仅用于采集 SFT 轨迹) |
| 推理服务 | vLLM (生产) / Transformers (开发) |
| 食物数据库 | SQLite (USDA SR Legacy + Foundation Foods) |
| 向量存储 | Chroma |
| Embedding | bge-small-en-v1.5 |
| 训练框架 | transformers + trl (SFT + GRPO) |
| LoRA | peft |
| 运行时 | Python 3.10+ |

---

## 📁 目录结构

```
NutriMind/
├── agent/                        # 规划与规格文档
│   ├── PRD.md
│   ├── Plan/                     # 执行计划 (本目录)
│   │   ├── master_plan.md        # 总计划
│   │   ├── phase1_infrastructure.md
│   │   ├── phase2_sft_data.md
│   │   ├── phase3_sft_training.md
│   │   ├── phase4_grpo.md
│   │   ├── phase5_ablation.md
│   │   ├── phase6_evaluation.md
│   │   └── phase7_deployment.md
│   └── specs/                    # 模块规格
├── serving/                      # 推理服务代码
│   ├── orchestrator.py
│   ├── inference.py
│   └── tools/
├── training/                     # 训练代码
│   ├── sft/
│   └── grpo/
├── data/                         # 数据
│   ├── usda.db
│   └── knowledge/
├── scripts/                      # 工具脚本
└── configs/                      # 配置
```

---

## ⚠️ 风险与应对

| 风险 | 严重性 | 可能性 | 应对策略 |
|------|--------|--------|---------|
| 3B 模型无法可靠生成 JSON 工具调用 | 高 | 中 | SFT 中加大 T1 格式训练比例；增加 format compliance 奖励权重 |
| GRPO 训练不稳定 / reward hacking | 高 | 中 | 渐进式权重调整；多维度奖励分散风险；设置 hard constraint |
| T3 条件分支能力难以学习 | 中 | 高 | 增加 T3 种子数据多样性；设计更细粒度的 conditional reward |
| USDA 数据食物名模糊匹配准确率不足 → think 退化 | 高 | 高 (已确认) | **Phase 2.6 Step 0 前置修复**: alias 表 (100-200条) + `match_confidence` 字段 + 干跑污染率估算 (目标 <10%)；tool 修好后再收集数据 |
| GPU 显存不足 (3B + GRPO) | 中 | 低 | LoRA + gradient checkpointing + bfloat16 |
| SFT 数据质量不一 (合成数据偏差) | 中 | 中 | 自动验证流程 + 10% 人工抽查 |

---

## 📝 各阶段详细计划

每个阶段的详细执行步骤见各自的独立文档：

- **[Phase 1: 基础设施搭建](phase1_infrastructure.md)**
- **[Phase 2: SFT 数据生成](phase2_sft_data.md)**
  - **[Phase 2.5: 数据扩充管线](phase2.5_sft_data_pipeline.md)**
  - **[Phase 2.6: 轨迹收集与验证](phase2.6_trajectory_collection.md)**
- **[Phase 3: SFT 训练](phase3_sft_training.md)**
- **[Phase 4: GRPO 训练](phase4_grpo.md)**
- **[Phase 5: 消融实验](phase5_ablation.md)**
- **[Phase 6: 评估与边界分析](phase6_evaluation.md)**
- **[Phase 7: 部署与监控](phase7_deployment.md)**

---

## ✅ 总体进度检查点 (Milestones)

| 里程碑 | 检查条件 | 预期完成 |
|--------|---------|---------|
| **M1: 基础设施就绪** | Orchestrator 可运行 agentic loop；8 个原子工具均有单元测试通过 | Phase 1 结束 |
| **M2: 数据就绪** | ~1500 条 SFT + 3000 条 GRPO Prompts 生成完成，格式验证通过率 > 95% | Phase 2 结束 |
| **M3: SFT 基线达标** | T1 ≥ 95%, Format ≥ 98%，无严重退化 | Phase 3 结束 |
| **M4: RL 增益验证** | T2 ≥ 80%, T3 ≥ 75%，相比 SFT-only 有可量化提升 | Phase 4 结束 |
| **M5: 消融完成** | SFT vs SFT+GRPO 对比数据齐全，分层分析完成 | Phase 5 结束 |
| **M6: 评估报告完成** | 所有 PRD Section 6 指标评估完成，边界分析文档产出 | Phase 6 结束 |
| **M7: 系统可部署** | vLLM 服务启动，端到端 demo 可运行 | Phase 7 结束 |
