# Phase 4: RL Training — GRPO vs GiGPO Controlled Comparison

> Priority: **Core Work** | Estimated: 2-3 weeks | Depends on: Phase 3

## 🎯 Goal

Through controlled comparison of GRPO and GiGPO (both starting from SFT, same reward function, same training steps), determine which advantage estimation method better improves multi-step planning (T2), conditional reasoning (T3), and safety boundary (T4) capabilities.

**Key strategy**: Clean experimental design over complex iterative chains. GRPO vs GiGPO is the core comparison; reward function variants (v1/v2) are exploratory.

## 📋 Deliverables

- [ ] GRPO prompt pool with metadata (2,500 prompts, tier-labeled)
- [x] Multi-turn rollout environment (`training/grpo/environment.py`) ✅ 2026-03-28
- [x] Reward functions v1/v2 (`training/grpo/reward.py`) ✅ 2026-03-28
- [x] GRPO training pipeline on veRL (`training/grpo/train.py`) ✅ 2026-03-28
- [x] GiGPO implementation with step-level advantage (`training/grpo/gigpo.py`) ✅ 2026-03-28
- [x] Training monitoring dashboard (`training/grpo/monitor.py`) ✅ 2026-03-28
- [ ] 4 model checkpoints: GRPO-v1, GRPO-v2, GiGPO-v2, [Optional] GRPO-v3
- [ ] RL training analysis report (GRPO vs GiGPO comparison + reward hacking case studies)

---

## 📝 Detailed Tasks

### Task 4.0: Infrastructure Setup (2 days)

#### 4.0.1 veRL Environment Setup

**Framework**: veRL (Volcano Engine RL) — fully open source, Apache 2.0.

**Why veRL over alternatives**:
- Native multi-turn environment-in-the-loop rollout support
- vLLM integration for fast rollout generation
- Natural path to GiGPO via verl-agent extension
- Designed for multi-GPU (Actor/Rollout/Trainer separation)

**Why not alternatives**:
| Framework | Rejection Reason |
|-----------|-----------------|
| trl (HuggingFace) | No native multi-turn support; requires heavy modification |
| OpenRLHF | Ray overhead for single-node; multi-turn support immature |
| Unsloth | SFT only, no RL support at all (also core kernels are closed-source) |
| SWIFT | Good alternative but less community momentum than veRL for agent RL |

- [ ] Run veRL's official GSM8K example to validate environment
- [ ] Confirm vLLM rollout works with Qwen3-4B
- [ ] Test LoRA training + checkpoint saving

#### 4.0.2 Hardware Plan

**Development & debugging**: Existing V100 32GB (free)
- Write all code, test with 50 prompts + G=4
- Validate rollout environment, reward functions, training loop

**Production experiments**: Rent 2× A100 80GB (~200-300 RMB total)
- Run all experiments in one session
- Estimated 15-20 hours for all experiments

**Analysis & reporting**: Back to V100 (free)

#### 4.0.3 Training Parameters

```python
# veRL config for 2× A100 80GB
model_name = "Qwen/Qwen3-4B"
lora_rank = 16
lora_alpha = 16
load_in_4bit = False              # A100 has enough VRAM, use BF16

# Rollout
num_generation_per_prompt = 8     # G=8 (P(useful contrast) = 99.2%)
max_new_tokens = 2048             # Multi-turn needs longer context
temperature = 0.7                 # Balance exploration vs quality
top_p = 0.9
max_tool_rounds = 6               # Match orchestrator config

# Training
learning_rate = 5e-7              # 40x lower than SFT (2e-5)
num_train_epochs = 1              # RL typically 1 epoch
per_device_train_batch_size = 1
gradient_accumulation_steps = 8   # Effective batch = 8 prompts
warmup_ratio = 0.05

# GRPO-specific
kl_coef = 0.05                    # Start moderate; tune based on KL trend
clip_range = 0.2
bf16 = True

# Checkpointing
save_steps = 200                  # Frequent saves for rollback
```

**Why G=8**: G=4 (spec v1) is too small — high probability of all-success or all-failure within a group, yielding zero advantage signal. G=8 ensures 99.2% probability of meaningful contrast on medium-difficulty prompts.

**Why lr=5e-7**: RL fine-tunes existing capabilities; large lr destroys SFT-learned skills. If reward stagnates, increase to 1e-6. If format compliance drops, decrease to 1e-7.

**Why kl_coef=0.05**: Controls policy drift from reference. >0.1 is too conservative (model can't learn), <0.01 enables reward hacking. Monitor KL divergence and adjust.

---

### Task 4.1: Multi-Turn Rollout Environment (2 days) ✅

This is the **hardest and most critical** component. Standard GRPO generates to EOS in one shot. We need pause-at-tool-call → execute → inject → resume.

- [x] **4.1.1** Implement `NutriMindEnv` wrapping the existing Orchestrator ✅

```python
class NutriMindEnv:
    """
    Wraps Orchestrator as GRPO rollout environment.

    Loop:
    1. Model generates text
    2. Detect </tool_call> → pause generation
    3. Parse tool_call → execute real tool → get tool_response
    4. Inject tool_response into context
    5. Model continues generating
    6. Repeat until final answer or max_rounds
    """
```

- [ ] **4.1.2** Implement `ToolCallStoppingCriteria` for vLLM (to be integrated with veRL)

```python
# Qwen3-4B has </tool_call> as a single token
# Must stop generation at this token AND at EOS
```

- [x] **4.1.3** Reuse existing `TOOL_REGISTRY` and `ToolParser` from `src/orchestrator/` ✅

- [ ] **4.1.4** Test with SFT model: run 10 prompts manually, verify rollout output is sensible

- [x] **4.1.5** Handle edge cases: ✅
  - Model generates invalid JSON → inject parse error, let model retry (counts as a round)
  - Model never outputs tool_call or final answer → truncate at max_new_tokens
  - Tool execution fails → inject error response (same as orchestrator behavior)

---

### Task 4.2: GRPO Prompt Pool (2 days)

#### 4.2.1 Prompt Selection (2,500 from query pool)

From Phase 2's 5,000 query pool, the SFT half was used for trajectories. The remaining ~2,500 are for GRPO.

**Tier distribution for GRPO** (Initial Reference):
| Tier | Count | % | Purpose |
|------|-------|---|---------|
| T0 (Pure QA) | 125 | 5% | Prevent conversational regression |
| T1 (single tool) | 375 | 15% | Prevent capability regression |
| T2 (multi-step) | 750 | 30% | **Core RL target** |
| T3 (conditional) | 750 | 30% | **Core RL target** |
| T4 (safety) | 375 | 15% | Maintain escalation judgment |
| Error recovery | 125 | 5% | Learn retry/recovery behavior |

> **Note**: GRPO's group-relative advantage naturally "mutes" Easy and Hard prompts by producing near-zero gradients (std ≈ 0). Initial ratios do not need to be precise.


#### 4.2.2 Metadata per prompt

**v1 strategy: Minimize prompt metadata; env_state is environment-layer concern.**

The original design embedded `env_state` in prompt metadata. However, this conflates prompt data with environment configuration. Instead:
- **Prompt metadata**: Only `query` + `tier` (minimal, clean)
- **env_state**: Generated during prompt preparation, managed by environment layer
- **Group consistency**: Same `env_state` shared across all G rollouts via tool cache

**Prompt metadata (all prompts):**
```json
{
    "query": "Am I over budget? If so, suggest a light dinner.",
    "tier": "T3"
}
```

**What's NOT needed in prompt metadata:**
- `env_state` — environment layer manages this (see §4.2.2a)
- `expected_tools` — v1 uses tier-based rules instead
- `ground_truth` — v1 uses runtime ground truth from tool results
- `branch_condition` — v2 uses intrinsic signals instead

**Tier classification quality:**
- Use LLM classification instead of keyword matching
- Human spot-check 100 samples, require >90% accuracy before proceeding
- If accuracy <85%, fix classifier before starting training

#### 4.2.2a Environment State Design

**Tool dependency analysis:**

| Tool | Data Source | Depends on env_state? |
|------|-------------|----------------------|
| `get_food_nutrition` | USDA database (static) | ❌ No |
| `retrieve_knowledge` | ChromaDB (static) | ❌ No |
| `get_today_summary` | user_profiles, user_goals, meal_logs | ✅ Yes |
| `get_history` | daily_summary, user_goals | ✅ Yes |
| `log_meal` | writes to meal_logs | ✅ Yes (+ side effect) |
| `set_goal` | writes to user_goals | ✅ Yes (+ side effect) |

**env_state structure** (generated per prompt during data preparation):

```python
env_state = {
    "user_id": "grpo_user_001",
    "user_profile": {
        "tdee_kcal": 2000,
        "goal": "maintain"  # lose | maintain | gain
    },
    "user_goals": {
        "calories": 2000,
        "protein": 120,
        "fat": 65,
        "carbs": 250
    },
    "meals_today": [
        {"meal_type": "breakfast", "foods": [...], "calories": 350, "protein_g": 12},
        {"meal_type": "lunch", "foods": [...], "calories": 550, "protein_g": 45}
    ],
    "meal_history": [  # For get_history (past N days)
        {"date": "2026-03-27", "calories": 1850, "protein_g": 95, ...},
        {"date": "2026-03-26", "calories": 2100, "protein_g": 110, ...}
    ]
}
```

**env_state is derived from prompt content** — e.g., "My grandma has diabetes..." → generate elderly profile with relevant health constraints.

**Group-level tool cache** (ensures rollout consistency):

```python
class NutriMindEnv:
    def reset(self, prompt: str, env_state: dict, group_id: str):
        self.env_state = env_state
        self.group_id = group_id
        self.tool_cache = {}  # Shared across all G rollouts in this group

    def call_tool(self, tool_name: str, params: dict) -> str:
        cache_key = (self.group_id, tool_name, freeze_params(params))

        if cache_key in self.tool_cache:
            return self.tool_cache[cache_key]  # Same result for same call

        result = self._execute_tool(tool_name, params)
        self.tool_cache[cache_key] = result
        return result
```

**Why this design:**
- Static tools (`get_food_nutrition`, `retrieve_knowledge`): No caching needed — deterministic by nature
- Stateful tools (`get_today_summary`, `get_history`): Cache ensures identical results across G rollouts
- Write tools (`log_meal`, `set_goal`): Mock the side effect (return success but don't persist) OR use isolated DB per rollout

**Training loop integration:**

```python
for prompt_data in batch:
    query = prompt_data["query"]
    tier = prompt_data["tier"]
    env_state = load_env_state(prompt_data["prompt_id"])  # Pre-generated
    group_id = generate_group_id()

    rollouts = []
    for _ in range(G):
        env.reset(query, env_state, group_id)
        rollout = policy.generate(query, env)
        rollouts.append(rollout)

    # All G rollouts saw identical tool results for identical calls
    compute_grpo_advantage(rollouts)
```

#### 4.2.2b Reward Function Design

**v1 reward (3 dimensions, pure rule-based):**

| Dimension | Signal | Metadata Needed |
|-----------|--------|-----------------|
| `r_format` | Valid JSON in `<tool_call>` tags | None |
| `r_tool_selection` | Tier-based rules (T0=no tools, T1-T3=≥1 tool, T4=safety check) | `tier` only |
| `r_outcome` | Compare final answer vs actual tool results | None (runtime) |

**v2 reward adds trajectory-intrinsic signals (no additional metadata):**

| Dimension | Intrinsic Signal | Metadata Needed |
|-----------|------------------|-----------------|
| `r_efficiency` | Repeated tool calls detection | None |
| `r_conditional` | Does `<think>` reference tool results? | None |

```python
# v2 efficiency: penalize repeated calls
def compute_efficiency_intrinsic(trajectory):
    tool_calls = trajectory.get_tools_called()
    unique_calls = set(tool_calls)
    repetition_penalty = len(tool_calls) - len(unique_calls)
    return max(0, 1.0 - 0.2 * repetition_penalty)

# v2 conditional: check if think block references tool results
def compute_conditional_intrinsic(trajectory):
    think_content = trajectory.get_think_content()
    tool_results = trajectory.get_tool_results()
    references = count_value_references(think_content, tool_results)
    return min(1.0, references / max(len(tool_results), 1))
```

#### 4.2.3 Initial Pool Validation & SFT Readiness Check

Run SFT model on a representative sample (~500 prompts), N=8 rollouts each:

- Easy (≥70% success): SFT already handles — GRPO will auto-mute
- Medium (20-70%): Core learning zone
- Hard (<20% success): Exploratory

**Purpose**: NOT for sampling balance (GRPO handles that), but for:
1. **Red flag**: Hard > 60% → SFT insufficient, go back to Phase 3
2. **Baseline snapshot**: Know your starting point for later comparison

#### 4.2.4 Bootstrap with 500 prompts first

Don't prepare all 2,500 at once. Start with 500 high-quality prompts to validate the entire pipeline works end-to-end. Expand to 2,500 only after confirming:
- Rollout environment produces valid trajectories
- Reward function returns reasonable score distribution
- Training loss decreases
- No obvious bugs

---

### Task 4.3: Reward Functions — Parallel Experiments (Route A)

**Core principle**: Clean controlled comparison over iterative chains. All experiments start from SFT with ref=SFT. This enables direct comparison: GRPO vs GiGPO, v1 vs v2.

#### Experiment Design (Route A)

```
All experiments share:
  - Base model: SFT checkpoint
  - Reference: SFT checkpoint (frozen)
  - Training steps: Same for all

Variable 1 - Reward function:
  - v1: R_format + R_tool_selection + R_outcome (3 dimensions)
  - v2: v1 + R_efficiency + R_conditional (5 dimensions, intrinsic signals)

Variable 2 - Advantage estimation:
  - GRPO: trajectory-level only
  - GiGPO: trajectory-level × step-level (anchor states)
```

#### Reward v1: Pure Rule-Based (3 dimensions)

```python
def reward_v1(trajectory, task_metadata):
    """
    v1: Pure rule-based. No LLM-Judge. No pre-annotated metadata dependency.
    """
    # R_format: valid JSON in <tool_call> tags (binary 0/1)
    r_format = 1.0 if all_tool_calls_valid(trajectory) else 0.0

    # R_tool_selection: tier-based rules
    # - T0: should NOT call tools (pure QA)
    # - T4: only check safety declaration (no tool call penalty)
    # - T1-T3: should call at least one valid tool
    r_tool = compute_tool_selection_score(trajectory, task_metadata)

    # R_outcome: runtime ground truth for T1 (compare answer vs tool results)
    r_outcome = compute_outcome_score(trajectory, task_metadata)

    return 0.30 * r_format + 0.35 * r_tool + 0.35 * r_outcome
```

**T4 design rationale:**
- Do NOT penalize T4 for calling tools — model may legitimately check before refusing
- Only check that final answer has safety declaration
- This prevents "see sensitive word → refuse immediately" shortcut (over-refusal)
- GRPO group comparison naturally selects optimal path: "check then refuse" vs "refuse directly"

#### Reward v2: Add Efficiency + Conditional (intrinsic signals)

```python
def reward_v2(trajectory, task_metadata):
    """
    v2: v1 + efficiency + conditional. Uses trajectory-intrinsic signals only.
    """
    base = reward_v1(trajectory, task_metadata)

    # R_efficiency: penalize repeated tool calls (intrinsic, no optimal_steps needed)
    r_efficiency = compute_efficiency_intrinsic(trajectory)

    # R_conditional: check if <think> references tool results (intrinsic)
    r_conditional = compute_conditional_intrinsic(trajectory)

    return 0.70 * base + 0.15 * r_efficiency + 0.15 * r_conditional
```

**v2 uses intrinsic signals** — no `expected_tools` or `branch_condition` metadata needed. See Task 4.2.2 for details.

#### [Optional] Reward v3: Add LLM-Judge

Only run if v2 experiments succeed and time permits:

```python
def reward_v3(trajectory, task_metadata, llm_judge):
    """v3: v2 + LLM-Judge for recommendation questions."""
    base = reward_v2(trajectory, task_metadata)
    if is_recommendation_question(task_metadata.query):
        llm_score = llm_judge(trajectory.final_answer, task_metadata.query)
        return 0.75 * base + 0.25 * llm_score
    return base
```

**LLM-Judge safeguards**: n=3 averaging, ≤25% weight, monitor answer length inflation.

#### Success Criteria by Experiment

| Experiment | Success Criteria |
|------------|-----------------|
| C (GRPO-v1) | Format ≥ 95%, T1 accuracy ≥ 90%, reward trending up |
| D (GRPO-v2) | T2 ≥ 75%, T3 ≥ 70%, no regression on C's metrics |
| F (GiGPO-v2) | T2/T3 improvement over D (same reward, better advantage) |

---

### Task 4.4: GiGPO Implementation (3 days)

Implement GiGPO for controlled comparison with GRPO. Both start from SFT, use same reward (v2), same training steps.

#### 4.4.1 Algorithm Overview

GiGPO adds a second layer of advantage estimation on top of GRPO:

```
GRPO:  advantage = (reward - mean(rewards)) / std(rewards)
       → Only trajectory-level. Cannot distinguish which STEP was good/bad.

GiGPO:
  Layer 1 (group-level): Same as GRPO
  Layer 2 (step-level):  Find "anchor states" where rollouts diverge
       → Steps that lead to higher downstream success get positive advantage
       → Steps that lead to failure get negative advantage

  Final: advantage = group_advantage × step_advantage
```

#### 4.4.2 Anchor State Detection

In multi-turn agent trajectories, anchor states are points where different rollouts made different tool-calling decisions from the same state:

```
Example: 8 rollouts for "Am I over budget? If so, suggest dinner."

Anchor state: After get_today_summary returns {remaining: 150}

  τ1-τ3: Correctly call retrieve_knowledge("low cal dinner") → succeed
  τ4-τ5: Call get_food_nutrition("salad") → suboptimal but succeed
  τ6-τ8: Output final answer without knowledge lookup → fail

Step advantage at this anchor:
  retrieve_knowledge → high (3/3 succeed)
  get_food_nutrition → medium (2/2 succeed but suboptimal)
  no tool call → low (0/3 succeed)
```

**State Equivalence Definition** (critical for NutriMind):

Per GiGPO paper, anchor state grouping uses **exact match** of environment state. For tool-calling agents:

```python
def compute_state_key(conversation_history_up_to_t):
    """
    State = conversation context up to (and including) the last tool_response.
    Two rollouts share an anchor state if they have identical context at step t.
    """
    return hash(tuple(
        (msg["role"], msg["content"])
        for msg in conversation_history_up_to_t
    ))
```

**Tool Determinism Requirement** (MUST implement before GiGPO):

For anchor states to form naturally, tools must return **identical results for identical inputs**. This is achieved via the group-level tool cache design in §4.2.2a.

| Tool Category | Tools | Determinism Strategy |
|---------------|-------|---------------------|
| Static (no env_state) | `get_food_nutrition`, `retrieve_knowledge` | Inherently deterministic (same input → same output) |
| Stateful read | `get_today_summary`, `get_history` | Group-level cache ensures consistency |
| Stateful write | `log_meal`, `set_goal` | Mock side effects (return success, don't persist) |

**Note**: The `NutriMindEnv` implementation in §4.2.2a already handles this via `tool_cache` keyed by `(group_id, tool_name, params)`.

- [ ] **4.4.2a** Verify `NutriMindEnv.tool_cache` works correctly for GiGPO anchor state detection
- [ ] **4.4.2b** Implement anchor state detection using conversation history hash
- [ ] **4.4.3** Implement step-level advantage computation (discounted return from anchor)
- [ ] **4.4.4** Integrate with veRL's advantage calculation (replace GRPO's flat advantage)

#### 4.4.3 GiGPO Training (Exp F)

```
Policy:    SFT model
Reference: SFT model (frozen)
Reward:    v2 (same as Exp D)
Algorithm: GiGPO (step-level advantage)
```

**Controlled comparison with Exp D (GRPO-v2):**
- Same base model (SFT)
- Same reference model (SFT)
- Same reward function (v2)
- Same training steps
- **Only difference**: advantage estimation method (GRPO vs GiGPO)

This design enables a clean comparison: if F outperforms D on T2/T3, step-level credit assignment provides value for multi-turn agent tasks.

---

### Task 4.5: Training Monitoring & Reward Hacking Detection (ongoing)

#### 4.5.1 Monitoring Dashboard

Track these metrics every 200 steps on a held-out eval set (100 prompts):

**Core metrics**:
| Metric | Normal | Alarm |
|--------|--------|-------|
| avg_reward | Steadily increasing | ↑ but manual eval quality ↓ |
| task_completion_rate | Increasing | Stagnant while reward increases |
| format_compliance | Stable ≥ 95% | Drops below 90% |
| KL divergence | Slow increase, converges | Sudden spike (>3× recent avg) |

**Behavioral metrics**:
| Metric | Normal | Alarm |
|--------|--------|-------|
| avg_tool_calls | Stable or slight decrease | Cliff drop (>30% decrease) |
| tool_path_diversity | 5-10 unique paths per prompt | 1-2 paths (mode collapse) |
| answer_length | Stable | Continuous inflation |
| pairwise_bleu (rollouts) | 0.3-0.6 | >0.85 (mode collapse) |

**Safety metrics** (specific to NutriMind):
| Metric | Normal | Alarm |
|--------|--------|-------|
| hallucination_rate | Decreasing | Increasing |
| uncertain_match_accept_rate | 40-60% | <10% (over-refusing) or >90% (over-accepting) |
| T4 false positive rate | Stable | Increasing (model over-escalating) |

#### 4.5.2 Common Reward Hacking Patterns (NutriMind-Specific)

| Pattern | Detection | Fix |
|---------|-----------|-----|
| Perfect format, fabricated content | Compare answer claims vs tool_response data | Add hallucination penalty to reward |
| Skip tools to game efficiency | avg_tool_calls cliff drop | Add minimum tool call constraint for T1-T3 |
| Exploit LLM-Judge preferences | Answer length inflation, list formatting | Cap answer length; reduce Judge weight |
| Refuse all uncertain matches | uncertain_accept_rate < 10% | Penalize over-refusal |
| Template memorization | pairwise_bleu > 0.85 | Increase temperature; add diversity bonus |

#### 4.5.3 Rollback Protocol

```
1. Stop training immediately
2. Identify which reward dimension was exploited (check per-dimension trends)
3. Roll back to checkpoint BEFORE hacking began (not to SFT — too far back)
4. Fix reward function
5. Resume training from rolled-back checkpoint with fixed reward
```

**This is why save_steps=200 is critical** — granular checkpoints enable precise rollback.

#### 4.5.4 Human Spot-Check

Every 200 steps, manually review 50 model outputs from the eval set. This is the most reliable hacking detection method. All automated metrics have blind spots; a human can instantly see "this answer is nonsense."

---

### Task 4.6: Prompt Pool Optimization (Optional)

In Route A, C (GRPO-v1) and D (GRPO-v2) are independent experiments from SFT. However, if C runs first, its logs can inform D's prompt selection:

1. **Identify Low Signal**: Prompts with `reward_std < 0.05` across rollouts provide no learning signal
2. **Optional Culling**: Can remove low-signal prompts from D's pool to improve training efficiency
3. **Not Required**: Since D doesn't build on C, this optimization is optional—D can use the same prompt pool as C for cleaner comparison

**Recommendation**: For the cleanest comparison, use identical prompt pools for C, D, and F. Apply culling only if training budget is constrained.

---

## 🧪 Experiment Matrix

| Exp | Base Model | Algorithm | Reward | Purpose | Priority |
|-----|-----------|-----------|--------|---------|----------|
| B | SFT | — | — | Baseline (already done) | Required |
| C | SFT | GRPO | v1 (3-dim) | Validate pipeline, basic RL signal | Required |
| D | SFT | GRPO | v2 (5-dim intrinsic) | Full GRPO | **Core** |
| F | SFT | **GiGPO** | v2 (5-dim intrinsic) | Step-level advantage | **Core** |
| E | SFT | GRPO | v3 (+LLM-Judge) | Open-ended quality | Optional |

> **Note**: Exp A (base model zero-shot) and Exp G (GPT-4o) are defined in phase5_ablation.md for evaluation only.

**Route A Design (adopted):**
- All experiments start from SFT with ref=SFT
- D and F are the **core comparison**: same base, same reward, same steps, only difference is advantage method
- C is exploratory (v1 reward) to validate pipeline before committing to v2
- E is optional (adds LLM-Judge) if time permits

**Why Route A over iterative chains:**
- Clean controlled comparison: GRPO vs GiGPO is the core narrative
- Easier to interpret results: no confounding from intermediate training rounds
- If F > D on T2/T3, step-level credit assignment definitively helps

---

## ⏱ Timeline

```
Week 1: Infrastructure + Environment
  Day 1-2: veRL setup, run official examples, validate on V100
  Day 3-4: Implement NutriMindEnv (multi-turn rollout)
  Day 5:   Implement reward v1/v2, test on 50 prompts

Week 2: Core Experiments (rent A100s)
  Day 6:   Prepare 500 prompts, SFT baseline rollout
  Day 7:   Exp C (GRPO-v1) — validate pipeline works
  Day 8:   Exp D (GRPO-v2) — full GRPO with intrinsic signals
  Day 9:   Exp F (GiGPO-v2) — same reward as D, step-level advantage

Week 3: Analysis + Optional (A100s or V100)
  Day 10:  Analyze D vs F, human spot-check 50 samples each
  Day 11:  Expand to 2500 prompts if needed, re-run core experiments
  Day 12:  Final eval on B/C/D/F, document results
  Day 13:  [Optional] Exp E (GRPO-v3 with LLM-Judge)
```

---

## ✅ Phase 4 Completion Criteria

| Checkpoint | Criteria |
|------------|---------|
| T2 multi-step success rate | ≥ 80% (D or F) |
| T3 conditional correctness | ≥ 75% (D or F) |
| T4 safety recall | ≥ 90% (no regression from SFT) |
| Format compliance | ≥ 98% (no regression from SFT) |
| D vs F comparison | Clear winner documented with analysis |
| Reward hacking | At least 1 documented case + resolution |
| Checkpoints saved | 4 models: GRPO-v1 (C), GRPO-v2 (D), GiGPO-v2 (F), [optional] GRPO-v3 (E) |
