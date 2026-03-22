# NutriMind Agent: Product Requirements Document

> **⚠️ 文档状态说明**
> 此 PRD 是需求基线文档，记录各阶段的设计意图。**不代表当前实现**。
> 当前实现规格请参考 `docs/specs/`；架构决策和变更理由见 `docs/ideas/architecture_alignment.md`。

## Revision History

| 版本 | 日期 | 主要变更 |
|------|------|---------|
| **v1.0** | 2026-02 | 初始 PRD，5 工具设计（含 call_expert_nutritionist），T4 = 专家升级 |
| **v2.0** | 2026-02 | 工具调整，Tier 定义细化，GRPO reward 框架确立 |
| **v3.0** | **2026-03-01** | **架构重大变更**：① T4 改为安全边界声明（删除 expert 路径）② log_user_data 拆分为 3 个原子工具 ③ 全英语训练架构 ④ 知识库改为英文权威来源 |

> **v3.0 决策详情**见 [`architecture_alignment.md`](ideas/architecture_alignment.md)

---

## 1. Vision

NutriMind Agent explores **how to make a small model a reliable agentic problem-solver in a specific domain**. Through SFT + RL (GRPO) training, a lightweight Qwen3-4B learns to answer nutrition queries, invoke tools for precise computation and data retrieval, and chain multi-step tool calls to handle complex tasks. When the task involves complex medical nutrition management, the model declares a safety boundary and recommends professional consultation.

**Core Research Question**: How far can RL push a small model's agentic capabilities in a domain-specific setting, and where is the boundary that necessitates a safety declaration?


## 2. Core Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                        Orchestrator                         │
│         (State machine managing the agentic loop)           │
└───────────────────────────────────────────────────────────────┘
                              │
              ┌─────────────┴─────────────┐
              │                             │
              ▼                             ▼
┌───────────────┐              ┌───────────────┐
│  4B Agentic   │              │  Local Tools  │
│    Model      │              │  (Python)     │
│  (Qwen3-4B)  │              │               │
└───────────────┘              └───────────────┘
                                        │
        ┌─────────┐┌─────────┐┌────────┐┌────────┐┌─────────┐┌────────┐┌──────────┐
        ▼         ▼         ▼        ▼        ▼         ▼        ▼
  search_food  calc   log_meal get_today get_hist retrieve get_goal
              _meal            _summary  ory     _knowledge _adherence
```

> **v3 变更**：删除了 Expert LLM 警层（T4 改为本地安全边界声明）。全部计算在本地完成，无云端 API 依赖。

**Components:**
- **Orchestrator**: Manages the agentic loop (observe → think → act → observe) → [specs/orchestrator.md](specs/orchestrator.md)
- **4B Agentic Model**: SFT + GRPO fine-tuned Qwen3-4B. Handles nutrition QA, tool selection, multi-step planning, and escalation decisions
- **Local Tools**: Deterministic Python functions for data retrieval, calculation, and logging → [specs/tools.md](specs/tools.md)

## 3. Agentic Complexity Taxonomy

Task complexity is defined by the **agentic reasoning depth** required, not by routing destination. This taxonomy drives both training data distribution and RL reward design.

| Tier | Agentic Complexity | Example | Training Focus |
|------|-------------------|---------|----------------|
| **T1** | Single-step tool call | "Protein in 100g chicken?" → `search_food` → answer | SFT (pattern matching) |
| **T2** | Multi-step tool chain (2-3 tools, data dependency) | "Log my lunch and calculate total calories" → `calculate_meal` → `log_meal` → `get_today_summary` → answer | **RL core target**: sequential planning |
| **T3** | Conditional branching (next step depends on intermediate result) | "Am I over my calorie budget? If so, suggest a low-cal dinner" → `get_today_summary` → *branch*: if over → `retrieve_knowledge` → answer | **RL core target**: dynamic decision-making |
| **T4 \[v3 updated\]** | Safety boundary declaration (no tool calls) | Dialysis / post-transplant / drug-nutrient interactions → model outputs disclaimer, recommends professional | SFT (learn to recognize clinical boundary) |

**Design Rationale:**
- T1 is solved by SFT — the model learns input→tool→output patterns
- T2-T3 are where RL adds value — the model must learn *sequential decision-making* and *conditional reasoning* that pure imitation cannot reliably teach
- T4 is a meta-cognitive skill — the model learns when to stop and delegate. The boundary between T3 and T4 is itself an empirical finding from training

## 4. Training Strategy

### Phase A: SFT — Establish Baseline Agentic Capability

**Objective**: Teach the model the function calling format, basic tool selection, and nutrition domain knowledge.

**Data**:
- T1 examples: ~40% of dataset (anchor tool-calling format)
- T2 examples: ~30% (expose multi-step patterns)
- T3 examples: ~15% (expose conditional patterns)
- T4 examples: ~10% (escalation patterns)
- Pure QA (no tool): ~5% (retain conversational ability)

**Data Generation Pipeline**:
1. Define tool schemas in standard function calling format (JSON)
2. Use strong model (Claude/GPT-4o) to generate diverse (query, tool_call_trajectory) pairs per tier
3. Quality filter: validate tool call format + verify tool chain logic + LLM-as-judge for answer quality
4. Augment: paraphrase queries for linguistic diversity

**SFT Output Format** (Hermes-style function calling):
```
<|im_start|>system
You are NutriMind, a nutrition assistant. You have access to the following tools:
{tool_schemas_json}
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
<think>{reasoning about which tools to use and why}</think>
<tool_call>{"name": "search_food", "arguments": {"query": "chicken breast", "amount_g": 100}}</tool_call>
<|im_end|>
<|im_start|>tool
{"calories": 165, "protein_g": 31, ...}
<|im_end|>
<|im_start|>assistant
{final answer incorporating tool results}
<|im_end|>
```

### Phase B: GRPO — Optimize Agentic Decision Quality

**Objective**: Improve the model's multi-step planning, conditional reasoning, and escalation judgment beyond what SFT imitation can achieve.

**Why RL over SFT alone**:
- SFT teaches "what a good trajectory looks like" but doesn't teach recovery from suboptimal intermediate states
- RL lets the model explore and learn which tool sequences lead to better outcomes
- Critical for T2-T3 tasks where multiple valid tool chains exist but differ in efficiency and accuracy

**Reward Function Design** (multi-dimensional):

| Reward Dimension | Signal Type | Description |
|-----------------|-------------|-------------|
| **Format Compliance** | Rule-based (process) | Valid JSON tool calls, correct schema adherence. Binary: +1 / 0 |
| **Tool Selection Accuracy** | Rule-based (process) | Did the model call the right tool(s) for this query type? Compare against ground truth tool set |
| **Execution Completeness** | Rule-based (outcome) | For T2-T3: Did the full tool chain execute without unnecessary steps or missing steps? |
| **Conditional Correctness** | Rule-based (outcome) | For T3: Did the model branch correctly based on intermediate results? |
| **Answer Quality** | LLM-as-judge (outcome) | Is the final answer nutritionally accurate and helpful given the tool outputs? |
| **Escalation Appropriateness** | Rule-based (outcome) | T4 queries escalated = reward; T1-T2 queries escalated = penalty |
| **Tool Efficiency** | Rule-based (process) | Penalizes redundant tool calls beyond optimal count. Encourages minimum viable tool usage |

**Reward Composition**:
```
R_total = w1 * R_format + w2 * R_tool_selection + w3 * R_completeness
        + w4 * R_conditional + w5 * R_answer + w6 * R_escalation
        + w7 * R_efficiency
```
Weights are hyperparameters to be tuned. Initial: heavier weight on R_format and R_tool_selection for stability, gradually increase R_completeness, R_conditional, and R_efficiency.

**Training Focus**:
- RL training data is concentrated on T2-T3 tasks (where RL has the most impact)
- T1 tasks included at low ratio to prevent regression
- T4 tasks included to maintain escalation judgment

## 5. Module Specifications

| Module | Spec Document | Description |
|--------|---------------|-------------|
| Tools | [specs/tools.md](specs/tools.md) | 8 atomic tool definitions, invocation format, error handling |
| Orchestrator | [specs/orchestrator.md](specs/orchestrator.md) | State machine, loop logic, termination conditions |
| Database | [specs/database.md](specs/database.md) | USDA schema, user data schema, data sources |
| RAG | [specs/rag.md](specs/rag.md) | Vector store, embedding, English knowledge sources |
| Training | [specs/training.md](specs/training.md) | SFT data pipeline, GRPO reward function, training configs |

> **specs/expert_prompt.md** 已删除（v3 删除 expert 路径）。

**Tech Stack**: See [tech-stack.md](tech-stack.md)

## 6. Success Metrics

### Agentic Capability Metrics (Core — measures RL impact)

| Metric | Target | Tier Focus |
|--------|--------|------------|
| T1 Single-step tool call accuracy | ≥ 95% | SFT baseline |
| T2 Multi-step task success rate | ≥ 80% | **RL core** |
| T3 Conditional branching correctness | ≥ 75% | **RL core** |
| T4 Safety boundary precision (correctly declare when needed) | ≥ 85% | SFT + RL |
| T4 Safety boundary recall (don't miss clinical cases) | ≥ 90% | SFT + RL |
| Redundant tool call rate | ≤ 10% | RL efficiency |
| Error recovery rate (retry after tool failure) | ≥ 60% | RL |

### Baseline Metrics

| Metric | Target |
|--------|--------|
| Tool-call format validity | ≥ 98% |
| Nutrition QA quality (no-tool queries) | ≤ 3% degradation vs base model |

### Ablation Comparisons (for resume narrative)

| Comparison | Purpose |
|-----------|---------|
| SFT-only vs SFT+GRPO on T2-T3 tasks | Quantify RL's contribution to multi-step agentic ability |
| 3B (SFT+GRPO) vs direct GPT-4o API | Cost-performance tradeoff validation |

### System-Level Metrics (Post-Deployment)

| Metric | Target |
|--------|--------|
| Expert API call rate | ≤ 25% |
| P50 Latency (local) | < 1s |
| P95 Latency (expert path) | < 5s |

## 7. Implementation Roadmap

| Phase | Deliverable | Priority |
|-------|-------------|----------|
| **Phase 1** | Orchestrator + 8 atomic tools | Foundation |
| **Phase 2** | SFT data generation (T1-T4 distribution, quality filtering, **full English**) | Foundation |
| **Phase 3** | SFT training on Qwen3-4B | Foundation |
| **Phase 4** | GRPO reward function design + RL training | **Core work** |
| **Phase 5** | Ablation experiments (SFT vs SFT+RL, per-tier analysis) | **Core work** |
| **Phase 6** | Offline evaluation + analysis of agentic capability boundary | **Core work** |
| **Phase 7** | Deployment (vLLM serving) + monitoring | Wrap-up |

**Key shift from v1**: Phases 4-6 are the project's core, not Phase 1-3. SFT is a prerequisite to get through quickly; RL experimentation and evaluation is where the main effort goes.

## 8. Business Value

- **Cost**: ~70% reduction in cloud API costs (80% queries handled locally)
- **Latency**: Sub-second responses for T1-T3 tasks
- **Accuracy**: Deterministic computation via tools > LLM mental arithmetic
- **Reliability**: Functional even when cloud APIs are unavailable
- **Graceful Degradation**: Cascading ensures complex tasks still get expert-quality answers