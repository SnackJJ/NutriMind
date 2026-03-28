# Phase 5: Ablation Experiments

> Priority: **Core Work** | Estimated: 3-4 days | Depends on: Phase 4

## 🎯 Goal

Systematically quantify the contribution of each training component (SFT → GRPO iterations → GiGPO), providing strong data support for resume narrative and technical blog.

## 📋 Deliverables

- [ ] Complete ablation result matrix (7 experiments × per-tier metrics)
- [ ] SFT vs GRPO vs GiGPO layered comparison report
- [ ] GRPO vs GiGPO head-to-head analysis (core finding)
- [ ] 4B (SFT+GiGPO) vs GPT-4o cost-performance comparison
- [ ] Reward iteration contribution analysis (v1 → v2 → v3)
- [ ] Reward hacking case studies with resolution details
- [ ] Formatted report (Markdown + charts)

---

## 📝 Detailed Tasks

### Task 5.1: Experiment Design (0.5 days)

#### 5.1.1 Ablation Matrix

| Exp | Model Config | Algorithm | Reward | Purpose | Priority |
|-----|-------------|-----------|--------|---------|----------|
| **A** | Qwen3-4B (base, no fine-tune) | — | — | Zero-shot baseline | Required |
| **B** | SFT-only (Phase 3) | — | — | SFT baseline | Required |
| **C** | SFT → GRPO-v1 | GRPO | v1 (3-dim rules) | Minimal RL | Required |
| **D** | SFT → GRPO-v2 | GRPO | v2 (+efficiency, +conditional) | Full GRPO | **Core** |
| **F** | SFT → GiGPO | GiGPO | v2 | **Direct SFT→GiGPO (fair comparison with D)** | **Core** |
| **G** | GPT-4o (API, same tools) | — | — | Upper bound reference | Required |
| **E** | GRPO-v2 → GRPO-v3 | GRPO | v3 (+LLM-Judge) | Open-ended quality | Optional |

**Key comparisons**:
- B vs D: SFT vs SFT+GRPO (RL contribution)
- **D vs F: GRPO vs GiGPO (credit assignment contribution) ← core finding** (same base, same reward, same steps — only difference is advantage estimation)
- C → D: Reward iteration progression (v1 → v2)
- F vs G: 4B+GiGPO vs GPT-4o (cost-performance)
- E: Optional — LLM-Judge exploration, run only if time permits

#### 5.1.2 Unified Evaluation Test Set (held out from all training)

| Tier | Count | Description |
|------|-------|-------------|
| T1 | 100 | Single-step tool calls |
| T2 | 80 | Multi-step chains |
| T3 | 60 | Conditional branching |
| T4-positive | 40 | Should trigger safety declaration |
| T4-negative | 40 | Should NOT trigger safety (false positive test) |
| Pure QA | 20 | No tool needed |
| Error recovery | 20 | Injected tool failures |
| **Total** | **360** | |

#### 5.1.3 Evaluation Metrics

**Per-tier metrics**:
- Task success rate (primary)
- Format validity rate
- Average tool calls per task
- Redundant tool call rate

**Safety metrics**:
- T4 precision (correctly identified high-risk)
- T4 recall (didn't miss high-risk)
- False positive rate (over-escalation)

**Quality metrics**:
- Answer factual accuracy (rule-based)
- Answer quality (LLM-judge, for recommendation tasks)
- Hallucination rate (claims vs tool evidence)

**Efficiency metrics**:
- End-to-end latency
- Cost per request (for GPT-4o comparison)

---

### Task 5.2: Evaluation Infrastructure (0.5 days)

- [ ] **5.2.1** Implement `training/evaluate.py` — unified eval runner
- [ ] **5.2.2** Implement GPT-4o comparison (same system prompt + tool schemas)
- [ ] **5.2.3** Implement result table auto-generation

---

### Task 5.3: Core Ablation — Training Progression (1 day)

Run all model checkpoints on the 360-prompt test set.

- [ ] **5.3.1** Run Exp A (base model)
- [ ] **5.3.2** Run Exp B (SFT-only)
- [ ] **5.3.3** Run Exp C (GRPO-v1)
- [ ] **5.3.4** Run Exp D (GRPO-v2)
- [ ] **5.3.5** Run Exp E (GiGPO)
- [ ] **5.3.6** Generate progression table:

```
┌──────────────────────────────────────────────────────────────────────┐
│            Training Progression — Per-Tier Success Rate               │
├──────────┬──────┬──────┬─────────┬─────────┬────────┬───────────────┤
│ Metric   │ Base │ SFT  │ GRPO-v1 │ GRPO-v2 │ GiGPO  │ Δ(GiGPO-SFT) │
├──────────┼──────┼──────┼─────────┼─────────┼────────┼───────────────┤
│ T1 Acc   │      │      │         │         │        │               │
│ T2 Succ  │      │      │         │         │        │               │
│ T3 Corr  │      │      │         │         │        │               │
│ T4 Prec  │      │      │         │         │        │               │
│ T4 Recall│      │      │         │         │        │               │
│ Format   │      │      │         │         │        │               │
│ Redund.  │      │      │         │         │        │               │
│ Recovery │      │      │         │         │        │               │
└──────────┴──────┴──────┴─────────┴─────────┴────────┴───────────────┘
```

- [ ] **5.3.7** Deep analysis per tier:
  - T2 failure analysis: Which tool chain patterns did RL improve? Which remain hard?
  - T3 failure analysis: Which conditional branches are hardest to learn?
  - Error recovery analysis: Does RL improve retry behavior?

---

### Task 5.4: GRPO vs GiGPO Head-to-Head (0.5 days) — Core Finding

This is the **most important comparison** for the project narrative.

#### 5.4.1 Primary Comparison: SFT → GRPO vs SFT → GiGPO (Strict Control)

**Design rationale**: To fairly compare GRPO vs GiGPO, both must start from the **same checkpoint** and train for the **same number of steps**. The only variable is the advantage estimation method.

- [ ] **5.4.1a** Train: SFT → GRPO (N steps, reward v2)
- [ ] **5.4.1b** Train: SFT → GiGPO (N steps, reward v2)

```
┌──────────────────────────────────────────────────────────────────┐
│   GRPO vs GiGPO — Strictly Controlled Comparison                   │
├──────────┬────────────────┬────────────────┬─────────────────────┤
│ Metric   │ SFT→GRPO (N)   │ SFT→GiGPO (N)  │ Δ                   │
├──────────┼────────────────┼────────────────┼─────────────────────┤
│ T1 Acc   │                │                │ (expect ~same)       │
│ T2 Succ  │                │                │ (expect GiGPO > GRPO)│
│ T3 Corr  │                │                │ (expect GiGPO >> GRPO)│
│ Avg Steps│                │                │                      │
│ Tool Div │                │                │                      │
└──────────┴────────────────┴────────────────┴─────────────────────┘
```

**Control variables**:
- Same base model: SFT checkpoint
- Same reward function: v2
- Same training steps: N
- Same hyperparameters (lr, kl_coef, etc.)
- **Only difference**: advantage estimation (trajectory-level vs step-level)

#### 5.4.2 Secondary Comparison: Progressive Training Value

- [ ] **5.4.2a** Compare SFT→GiGPO vs SFT→GRPO→GiGPO
  - Does iterative GRPO→GiGPO outperform direct GiGPO?
  - If yes: validates the progressive training strategy
  - If no: direct GiGPO is simpler and preferred

#### 5.4.3 Qualitative Analysis

- [ ] **5.4.3a** Pick 10 T3 examples where GiGPO succeeds but GRPO fails
  - What did GiGPO learn about the branching step?
  - How does the step-level advantage differ at the decision point?
  - Visualize anchor state grouping for selected examples

---

### Task 5.5: Cost-Performance — 4B vs GPT-4o (0.5 days)

- [ ] **5.5.1** Run Exp G (GPT-4o)
- [ ] **5.5.2** Generate cost-performance table:

```
┌──────────────────────────────────────────────────────────┐
│         4B (SFT+GiGPO) vs GPT-4o — Cost-Performance      │
├──────────┬───────────────┬──────────┬────────────────────┤
│ Metric   │ 4B+GiGPO      │  GPT-4o  │  4B/GPT-4o Ratio   │
├──────────┼───────────────┼──────────┼────────────────────┤
│ T1 Acc   │               │          │                     │
│ T2 Succ  │               │          │                     │
│ T3 Corr  │               │          │                     │
│ Latency  │               │          │                     │
│ Cost/req │               │          │                     │
│ $/1K req │               │          │                     │
└──────────┴───────────────┴──────────┴────────────────────┘
```

---

### Task 5.6: Reward Iteration Analysis (0.5 days)

- [ ] **5.6.1** Per-reward-version analysis: What did each iteration add?

```
v1 → v2: Expected improvement in T2/T3, reduction in redundant calls
v2 → v3: Expected improvement in recommendation answer quality
```

- [ ] **5.6.2** Reward hacking case studies
  - Document at least 1 concrete hacking instance discovered during training
  - Show: what happened → how it was detected → what was fixed → result after fix
  - This is the **most interview-compelling** content

- [ ] **5.6.3** Reward dimension contribution: If v1 (3-dim) gets 80% of v2's performance, are the extra dimensions worth the complexity?

---

### Task 5.7: Report (0.5 days)

- [ ] **5.7.1** Compile all results into structured report
- [ ] **5.7.2** Key findings:
  - **Finding 1**: RL (GRPO) improves T2/T3 by X% over SFT-only
  - **Finding 2**: GiGPO improves T2/T3 by Y% over GRPO through step-level credit assignment
  - **Finding 3**: Iterative reward design catches reward hacking that one-shot design would miss
  - **Finding 4**: 4B+GiGPO achieves Z% of GPT-4o's capability at 1/N the cost
  - **Finding 5**: [Whatever unexpected thing you discover]

---

## ✅ Phase 5 Completion Criteria

| Checkpoint | Criteria |
|------------|---------|
| Ablation matrix | All 7 experiments completed |
| GRPO vs GiGPO | Head-to-head comparison with controlled variables |
| Training progression | Clear monotonic improvement from SFT → v1 → v2 → GiGPO |
| GPT-4o comparison | Cost-performance data complete |
| Reward hacking | At least 1 documented case study |
| Report | Structured, data-backed, with visualizations |
