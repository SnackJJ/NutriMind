# Training Specification

## Overview

Training consists of two stages:
1. **SFT (Supervised Fine-Tuning)**: Teach format, tool usage, and basic agentic capability
2. **GRPO (Group Relative Policy Optimization)**: Optimize multi-step planning, conditional reasoning, and escalation judgment

## Stage 1: SFT

### Base Model

```
Qwen3-4B
```

> **Decision (2026-03-13)**: Switched from Qwen2.5-3B-Instruct to Qwen3-4B.
> - Native single-token `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>` (all in vocabulary)
> - Qwen3-4B ≈ Qwen2.5-7B performance level, with only ~33% more parameters than 3B
> - Pre-trained with agentic capabilities (Hermes-style tool calling, hybrid thinking mode)
> - 4bit LoRA on single 4090 (24GB) feasible: estimated ~15 GB VRAM

### Data Mixture (Tier-Based)

| Tier | Query Pool | % | Expected Validated | Description |
|------|-----------|---|-------------------|-------------|
| T0: Pure QA (no tool) | 100 | 4% | ~90 | Retain conversational ability |
| T1: Single-step tool call | 525 | 21% | ~450 | Simple lookup → `get_food_nutrition` / `get_today_summary` → answer |
| T2: Multi-step tool chain (2-3 tools) | 650 | 26% | ~350 | Sequential planning, data dependency between steps |
| T3: Conditional branching | 750 | 30% | ~250 | Next step depends on intermediate result |
| T4: Safety boundary declaration | 350 | 14% | ~200 | Model recognizes clinical boundary, outputs disclaimer (no tool) |
| Error Recovery | 125 | 5% | ~60 | Model handles API failures or missing data naturally |
| **Total** | **2,500** | **100%** | **~1,000–1,300** | |

> **Query Pool**: Total 5,000 queries split 50/50 between SFT and GRPO. The SFT half (2,500) goes through teacher trajectory collection → validation → final dataset. Validation pass rate ~40–52% yields ~1,000–1,300 usable trajectories.

**Design Rationale**:
- T1 anchors SFT — the model learns input → tool → output patterns
- T2-T3 exposure gives the model multi-step and conditional patterns to bootstrap RL
- T4 teaches escalation judgment (meta-cognitive skill)
- Pure QA prevents catastrophic forgetting of conversational ability

### Data Generation Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│ Query Pool  │────▶│ qwen3.5-plus │────▶│  Normalize   │────▶│  Auto-Check  │────▶│   Human     │
│  (English)  │     │ (Online Loop)│     │  (Qwen3 SFT) │     │ + Lang Check │     │   Review    │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └─────────────┘
```

Pipeline: `collect_trajectories.py` (pure text mode) → `normalize.py` (Qwen3-4B SFT format) → `validate_rules.py` → `validate_semantic.py`

> **Language policy**: All training data is in English — user queries, `<think>` blocks,
> tool parameters, and final answers. Knowledge base is English-language authoritative sources.

#### Pure Text Tool Calling (No Function Calling API)

> **Decision Reference**: [ADR-001: Pure Text Tool Calling](../decisions/001-pure-text-tool-calling.md)

SFT training data uses **pure text representation** for tool calls — not structured API fields:

| Approach | Tool Call Representation | Why We Chose Pure Text |
|----------|-------------------------|------------------------|
| Function Calling API | `tool_calls: [{id, function: {name, arguments}}]` | ❌ Loses `<think>` content; format ≠ inference |
| **Pure Text (Ours)** | `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` | ✅ Preserves reasoning; format == inference |

**Format identity principle**:
```
Teacher model output == SFT training target == Student model output
```

This eliminates the "format gap" problem:
- **Collection**: Teacher outputs `<think>...<tool_call>...</tool_call>` as plain text
- **Training**: Student learns to produce the exact same text pattern
- **Inference**: Student outputs the same format; agent parser extracts and executes

The agent-side parser (see [specs/tools.md § Tool Calling Protocol](tools.md#tool-calling-protocol)) is shared between collection and inference, ensuring behavioral consistency.

#### Step 1: Seed Data (Manual)

Create 50-100 high-quality seed trajectories covering all tiers.

**SFT Training Format (Qwen3-4B native)**:

After normalization, each trajectory is a `messages` array using Qwen3's native chat format. The `<think>`, `<tool_call>`, `<tool_response>` tags are single tokens in Qwen3-4B's vocabulary. When `apply_chat_template(enable_thinking=True)` is applied, this produces the correct token sequence for SFT.

```jsonc
// T1: Single-step tool call
{
    "tier": "T1",
    "messages": [
        {"role": "system", "content": "You are NutriMind, a specialized AI nutrition assistant. ..."},
        {"role": "user", "content": "How much protein is in 100g chicken breast?"},
        {"role": "assistant", "content": "<think>\nSimple food lookup. Single-step tool call to get_food_nutrition.\n</think>\n<tool_call>\n{\"name\": \"get_food_nutrition\", \"arguments\": {\"foods\": [{\"food_name\": \"chicken breast\", \"amount_grams\": 100}]}}\n</tool_call>"},
        {"role": "user", "content": "<tool_response>\n{\"status\": \"success\", \"data\": {\"foods\": [{\"food_name\": \"chicken breast\", \"amount_grams\": 100, \"protein_g\": 31.0, \"calories_kcal\": 165}]}}\n</tool_response>"},
        {"role": "assistant", "content": "100g of chicken breast contains approximately 31g of protein and 165 kcal."}
    ]
}

// T2: Multi-step tool chain (get_food_nutrition → log_meal)
{
    "tier": "T2",
    "messages": [
        {"role": "system", "content": "You are NutriMind, a specialized AI nutrition assistant. ..."},
        {"role": "user", "content": "Log my lunch: 200g rice and 150g salmon. What's my total protein?"},
        {"role": "assistant", "content": "<think>\nMulti-step T2: first look up nutrition for both foods, then log the meal.\n</think>\n<tool_call>\n{\"name\": \"get_food_nutrition\", \"arguments\": {\"foods\": [{\"food_name\": \"rice\", \"amount_grams\": 200}, {\"food_name\": \"salmon\", \"amount_grams\": 150}]}}\n</tool_call>"},
        {"role": "user", "content": "<tool_response>\n{\"status\": \"success\", \"data\": {\"total\": {\"protein_g\": 45.2, \"calories_kcal\": 520}}}\n</tool_response>"},
        {"role": "assistant", "content": "<think>\nGot nutrition totals. Now log this as lunch.\n</think>\n<tool_call>\n{\"name\": \"log_meal\", \"arguments\": {\"meal_type\": \"lunch\", \"foods\": [{\"food_name\": \"rice\", \"amount_grams\": 200}, {\"food_name\": \"salmon\", \"amount_grams\": 150}]}}\n</tool_call>"},
        {"role": "user", "content": "<tool_response>\n{\"status\": \"success\", \"total_calories\": 520}\n</tool_response>"},
        {"role": "assistant", "content": "Your lunch has been logged. Total protein: 45.2g, calories: 520 kcal."}
    ]
}

// T3: Conditional branching (branch based on intermediate result)
{
    "tier": "T3",
    "messages": [
        {"role": "system", "content": "You are NutriMind, a specialized AI nutrition assistant. ..."},
        {"role": "user", "content": "Am I over my calorie budget today? If so, suggest a low-cal dinner."},
        {"role": "assistant", "content": "<think>\nConditional T3: check today's budget first, then branch based on result.\n</think>\n<tool_call>\n{\"name\": \"get_today_summary\", \"arguments\": {}}\n</tool_call>"},
        {"role": "user", "content": "<tool_response>\n{\"status\": \"success\", \"data\": {\"total_calories\": 1850, \"calorie_budget\": 2000, \"remaining_calories\": 150}}\n</tool_response>"},
        {"role": "assistant", "content": "<think>\nOnly 150 kcal remaining. User needs low-calorie dinner suggestions. Retrieve relevant knowledge.\n</think>\n<tool_call>\n{\"name\": \"retrieve_knowledge\", \"arguments\": {\"query\": \"low calorie dinner under 150 calories\", \"domain\": \"meal_planning\"}}\n</tool_call>"},
        {"role": "user", "content": "<tool_response>\n{\"status\": \"success\", \"data\": {\"passages\": [\"...\"]}}\n</tool_response>"},
        {"role": "assistant", "content": "You have 150 kcal left today. Here are some light dinner options: a small mixed green salad with lemon dressing (~80 kcal), or a cup of clear vegetable broth (~30 kcal) with a side of steamed cucumbers."}
    ]
}

// T4: Safety boundary declaration (NO tool calls)
{
    "tier": "T4",
    "messages": [
        {"role": "system", "content": "You are NutriMind, a specialized AI nutrition assistant. ..."},
        {"role": "user", "content": "I'm on dialysis and want to build muscle. What should I eat?"},
        {"role": "assistant", "content": "<think>\nUser is on dialysis — active renal replacement therapy. Protein, potassium, and phosphorus management requires clinical-level intervention. This exceeds my safe service boundary. No tool calls.\n</think>\nYour situation involves complex medical nutrition management (dialysis with concurrent fitness goals) that exceeds my safe service boundary. Please consult your nephrologist or a renal dietitian for personalized guidance."}
    ]
}
```

**Key format rules**:
- `role: "assistant"` carries `<think>...</think>` + `<tool_call>...</tool_call>` in `content`
- `role: "user"` wraps tool results as `<tool_response>...</tool_response>` in `content`
- Final answer is a plain `role: "assistant"` message (no tool call tags)
- All 5 tools: `get_food_nutrition`, `log_meal`, `get_today_summary`, `get_history`, `retrieve_knowledge`
- 100% English in all fields

#### Step 2: Teacher Model Trajectory Collection (Online Python Loop)

Instead of prompting the model to hallucinate offline JSON, we run an **Online Execution Loop** via `collect_trajectories.py` using **qwen3.5-plus** as the teacher model.

**The Challenge**: Dashboard APIs (like qwen3.5-plus) natively attempt to optimize execution by emitting multiple tool calls concurrently. We need strictly sequential, single-tool-call trajectories for the 4B student model.

**The Solution (Collection + Normalization)**:
We enforce sequential execution and Chain-of-Thought via Python-level injection and history management:
1. **Schema Injection**: We dynamically inject a mandatory `"thought"` parameter into *every* JSON schema passed to the Teacher Model. This forces the model to generate its reasoning in English *before* filling in the actual tool arguments.
2. **Sequential Forcing (History Truncation)**: If the Teacher attempts parallel tool calls, the Python collection script silences the extra calls, executing only the first one.
3. **History Rewriting (Reconstruction)**: The script extracts the `"thought"` argument from the JSON, formats it as `<think>...</think>`, deletes the injected `"thought"` from the arguments, rewrites the Teacher's message history to reflect a single step, and appends the real environment response.
4. **Normalization** (`normalize.py`): Converts OpenAI messages format to Qwen3-4B native SFT format — `tool_calls` array → `<tool_call>` tags, `role: "tool"` → `role: "user"` with `<tool_response>` wrapper.

**Resulting Dataset**: The 4B student model ingests a perfect, patient, strictly sequential step-by-step thinking loop in its native chat format, while utilizing the Teacher's high-intelligence planning capabilities against a real mock environment.

#### Step 3: Auto-Validation

Validation runs on normalized trajectories (Qwen3-4B SFT format). See `validate_rules.py` for the actual implementation.

```python
VALID_TOOLS = {"get_food_nutrition", "log_meal", "get_today_summary", "get_history", "retrieve_knowledge"}

def validate_trajectory(trajectory: dict) -> tuple[bool, list[str]]:
    errors = []
    messages = trajectory.get("messages", [])
    tier = trajectory.get("tier", "")

    tool_count = 0
    has_final_answer = False

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "assistant":
            # Check for <think> and <tool_call> blocks
            has_think = "<think>" in content
            tool_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)

            if tool_match:
                tool_count += 1
                try:
                    tool_json = json.loads(tool_match.group(1))
                    if tool_json.get("name") not in VALID_TOOLS:
                        errors.append(f"Invalid tool: {tool_json['name']}")
                except json.JSONDecodeError:
                    errors.append("Invalid tool_call JSON")

                if not has_think:
                    errors.append("Missing <think> before <tool_call>")

                # Language check: <think> must be English
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                if think_match and check_chinese_chars(think_match.group(1)) > 0.05:
                    errors.append("<think> contains >5% Chinese characters")
            else:
                has_final_answer = True
                if check_chinese_chars(content) > 0.05:
                    errors.append("Final answer contains >5% Chinese characters")

    # Tier consistency checks
    if tier == "T0-qa" and tool_count > 0:
        errors.append("T0-qa should have no tool calls")
    if tier == "T4" and tool_count > 0:
        errors.append("T4 should have NO tool calls (must be safety boundary declaration)")
    if tier == "T1" and tool_count != 1:
        errors.append(f"T1 should have exactly 1 tool call, got {tool_count}")
    if not has_final_answer:
        errors.append("Trajectory does not end with a final answer")

    return len(errors) == 0, errors
```

#### Step 4: Human Review (Sampled)

- Review 10% of generated data
- Focus on: factual accuracy, natural language, edge cases
- Reject rate target: < 5%

### Measured Token Length Distribution (2026-03-22)

Data: `data/trajectories/sft_train_trajectory.jsonl` (n=1,449)

| Stat | Value |
|------|-------|
| Min | 418 |
| Max | 10,174 |
| Mean | 2,879 |
| Median | 2,633 |
| P25 | 1,696 |
| P75 | 3,700 |
| P90 | 4,787 |
| P95 | 5,342 |
| P99 | 7,798 |

**Token length buckets:**
| Range | Count | % |
|-------|-------|---|
| [0, 1024) | 93 | 6.4% |
| [1024, 2048) | 328 | 22.6% |
| [2048, 3072) | 451 | 31.1% |
| [3072, 4096) | 271 | 18.7% |
| [4096, 6144) | 225 | 15.5% |
| [6144, 8192) | 70 | 4.8% |
| > 8192 | 11 | 0.8% |

**Per-tier stats:**
| Tier | n | Min | Max | Mean | Median | P95 |
|------|---|-----|-----|------|--------|-----|
| T0-qa | 78 | 482 | 1,420 | 799 | 728 | 1,295 |
| T1 | 413 | 418 | 4,092 | 1,573 | 1,444 | 2,700 |
| T2 | 319 | 1,073 | 7,310 | 2,853 | 2,697 | 4,755 |
| T3 | 282 | 1,773 | 10,174 | 4,591 | 4,360 | 8,355 |
| T4 | 199 | 498 | 2,168 | 943 | 875 | 1,589 |
| error-recovery | 158 | 2,194 | 7,798 | 4,590 | 4,478 | 6,831 |

> **Decision**: `max_seq_length=8192` covers 99.2% of data. Samples exceeding 8192 tokens (11 total) are truncated.

### Training Configuration

For SFT, we recommend **Unsloth** over raw Transformers/TRL for the 4B model. Unsloth provides massive multi-card/single-card VRAM savings through its specialized Triton kernels and LoRA optimizations, which is critical when our `max_seq_length` expands to 8192 for full Agentic XML trajectories.

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Load Model with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-4B",
    max_seq_length=8192,
    dtype=None,          # Auto-detect (bf16 recommended)
    load_in_4bit=True,   # Critical for consumer GPUs while maintaining quality
)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,      # Unsloth is optimized with 0 dropout
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
)

training_args = TrainingArguments(
    output_dir="./models/nutrimind-4b-sft",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Kept small due to 8192 sequence length
    gradient_accumulation_steps=16, # Increased to maintain effective batch size
    learning_rate=2e-5,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    fp16=not is_bf16_supported(),
    bf16=is_bf16_supported(),
    optim="adamw_8bit",             # 8-bit optimizer for memory reduction
    group_by_length=True,           # Length bucketing to reduce padding waste
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=8192,
    dataset_num_proc=2,
    packing=False,       # Disabled packing to avoid chunking tool JSON logic
    args=training_args,
)
```

#### Loss Masking with `train_on_responses_only`

Use Unsloth's `train_on_responses_only` to mask loss on system/user/tool_response turns. Only compute loss on assistant turns (`<|im_start|>assistant`), which includes `<think>` blocks, `<tool_call>` JSON, and final answers.

```python
from unsloth import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
```

**Why this works**:
- `normalize.py` converts `role: "tool"` to `role: "user"` + `<tool_response>` wrapper
- So the masking boundary aligns: `<|im_start|>user\n` covers system prompts, user queries, and tool responses
- `<|im_start|>assistant\n` covers everything we want to train on

**Verification before training**:
```python
# MUST verify labels are correct before training
sample = trainer.train_dataset[0]
labels = sample["labels"]
tokens = sample["input_ids"]

# Check that assistant turns have valid labels (not -100)
# and user/system turns are masked (-100)
for i, (tok, lbl) in enumerate(zip(tokens, labels)):
    print(f"{i}: {tokenizer.decode([tok])!r} -> label={lbl}")
```

> ⚠️ **Known issue**: Some Unsloth versions may incorrectly mask `<|im_start|>assistant` tokens on small models. If verification fails, use a custom DataCollator instead.

---

## Stage 2: Iterative GRPO + GiGPO

### Framework: veRL

**Decision**: Use veRL (Volcano Engine RL, Apache 2.0) instead of trl.

| Requirement | trl | veRL |
|-------------|-----|------|
| Multi-turn tool calling rollout | ❌ Requires heavy modification | ✅ Native environment-in-the-loop |
| vLLM integration for fast rollout | ❌ | ✅ Native |
| GiGPO support | ❌ | ✅ Via verl-agent extension |
| Single-node multi-GPU | ⚠️ Basic | ✅ Actor/Rollout/Trainer separation |

### Architecture: SFT (Unsloth) → RL (veRL)

- **Stage 1 (SFT)**: Unsloth. No generation needed, only Forward/Backward on fixed teacher data. Unsloth's Triton kernels optimize LoRA + 8-bit AdamW for maximum throughput.
- **Stage 2 (GRPO/GiGPO)**: veRL + vLLM. RL requires generating massive amounts of text (G=8 rollouts × 2,500 prompts × multi-turn). vLLM's PagedAttention and continuous batching are essential.

### Environment-in-the-Loop GRPO Setup (Multi-Turn)

Standard GRPO generates a single completion to EOS. For multi-step tool usage, we need an **Environment-in-the-Loop** architecture:

1. **Step-wise Rollout (vLLM)**: Generation pauses at `</tool_call>` token (single token in Qwen3-4B vocabulary)
2. **Environment Simulator (Python)**: Reuses existing `TOOL_REGISTRY` and `ToolParser` from `src/orchestrator/`
3. **Context Injection**: Tool response formatted as `<tool_response>...</tool_response>` and injected into context
4. **Resume Generation**: vLLM continues generation until next `</tool_call>` or EOS
5. **Max rounds**: 6 (matching orchestrator config)

### Hardware Strategy

**Development**: Existing V100 32GB — write code, test with 50 prompts + G=4
**Production experiments**: Rent 2× A100 80GB — run all experiments (~15-20 hours, ~200-300 RMB)
**Analysis**: Back to V100

### Training Configuration

```python
# veRL config for 2× A100 80GB
model_name = "Qwen/Qwen3-4B"
lora_rank = 16
lora_alpha = 16
load_in_4bit = False              # A100 has enough VRAM

# Rollout
num_generation_per_prompt = 8     # G=8 (was 4 in spec v1 — too small)
max_new_tokens = 2048
temperature = 0.7
top_p = 0.9
max_tool_rounds = 6

# Training
learning_rate = 5e-7              # 40x lower than SFT
num_train_epochs = 1
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
warmup_ratio = 0.05

# GRPO-specific
kl_coef = 0.05                    # Moderate; adjust based on KL trend
clip_range = 0.2
bf16 = True

# Checkpointing
save_steps = 200                  # Frequent saves for rollback
```

### Training Focus (Using ~2,500 Prompts)

- **Pure Prompts only**: Unlike SFT, GRPO uses prompts without trajectory bodies
- **Core target**: T2-T3 tasks (where RL has the most impact)
- T1 tasks at 15% to prevent regression
- T4 tasks at 15% to maintain escalation judgment

### Prompt Difficulty Labeling

Before training, run SFT model on all prompts (N=8 rollouts) to compute success rate:
- Easy (≥70% success): 40% of training batch
- Medium (20-70%): 40% — **main learning signal**
- Hard (<20%): 20% — exploratory

Re-evaluate difficulty at training midpoint; shift distribution toward harder prompts.

**Red flag**: If Hard > 60%, SFT was insufficient. Go back and improve SFT first.

### Iterative Reward Strategy

**Core principle**: Each iteration changes ONE variable (the reward function). Don't design the "perfect" 7-dimensional reward upfront.

#### GRPO v1: Pure Rule Reward (3 dimensions)

```
Policy:    SFT model
Reference: SFT model (frozen)
```

```python
def reward_v1(trajectory_info, task_metadata):
    r_format = 1.0 if all_tool_calls_valid(trajectory_info) else 0.0
    r_tool = compute_tool_selection_score(trajectory_info, task_metadata)
    r_outcome = compute_rule_outcome_score(trajectory_info, task_metadata)
    return 0.30 * r_format + 0.35 * r_tool + 0.35 * r_outcome
```

**Deliberately omits**: R_efficiency, R_conditional, LLM-Judge — to be added in later rounds.

#### GRPO v2: Add Efficiency + Conditional

```
Policy:    GRPO-v1 checkpoint
Reference: GRPO-v1 checkpoint (frozen)
lr:        2.5e-7 (lower)
```

```python
def reward_v2(trajectory_info, task_metadata):
    base = reward_v1(trajectory_info, task_metadata)
    r_efficiency = compute_efficiency_score(trajectory_info, task_metadata)
    r_conditional = compute_conditional_score(trajectory_info, task_metadata)
    return 0.70 * base + 0.15 * r_efficiency + 0.15 * r_conditional
```

#### GRPO v3: Add LLM-Judge (for recommendation questions only)

```
Policy:    GRPO-v2 checkpoint
Reference: GRPO-v2 checkpoint (frozen)
lr:        1e-7 (even lower)
```

```python
def reward_v3(trajectory_info, task_metadata):
    base = reward_v2(trajectory_info, task_metadata)
    if get_question_type(trajectory_info) == "recommendation":
        llm_score = llm_judge(trajectory_info["final_answer"], task_metadata)
        return 0.75 * base + 0.25 * llm_score  # Rules remain dominant
    return base
```

**LLM-Judge safeguards**: n=3 averaging, weight ≤ 25%, only for recommendation tasks, monitor answer length inflation.

#### Reference Model Strategy

Each iteration updates reference to previous output:
- v1: ref = SFT
- v2: ref = GRPO-v1
- v3: ref = GRPO-v2

Rationale: Progressive refinement. If cumulative drift causes issues, fall back to ref=SFT.

### GiGPO: Step-Level Credit Assignment

After GRPO v2, implement GiGPO to add step-level advantage:

```
GRPO:  advantage = (reward - mean) / std  → trajectory-level only
GiGPO: advantage = group_advantage × step_advantage  → two layers
```

**Step advantage** is computed by finding "anchor states" where rollouts diverge, then comparing downstream success rates for different decisions at each anchor point.

```
Policy:    GRPO-v2 checkpoint
Reference: GRPO-v2 checkpoint (frozen)
Reward:    Same as v2 (only advantage calculation changes)
Algorithm: GiGPO (via verl-agent)
```

Using the SAME reward as GRPO-v2 enables a clean controlled comparison.

### SFT vs GRPO vs GRPO+GiGPO Computation Cost

```
SFT:  1,449 samples × 3 epochs × 2 ops = ~8,700 model operations → 3-4 hours (V100)
GRPO: 2,500 prompts × G=8 × ~3 rounds × ~400 tokens = ~24M tokens generated → 16-20 hours (V100)
      Forward: ~9,600×/prompt (autoregressive) + 16×/prompt (log_prob)
      Backward: 8×/prompt

GRPO is 50-200x more expensive than SFT per iteration. This is inherent to the algorithm.
All experiments combined: ~15-20 hours on 2× A100.
```

### Reward Hacking Detection

Monitor every 200 steps on held-out eval set (100 prompts):

| Metric | Normal | Alarm |
|--------|--------|-------|
| reward ↑ but task_completion ↓ | — | Almost certain hacking |
| avg_tool_calls cliff drop >30% | — | Skipping tools for efficiency |
| pairwise BLEU of rollouts >0.85 | — | Mode collapse |
| KL spike >3× recent average | — | Found reward exploit |
| answer_length continuous growth | — | Gaming LLM-Judge |

**Rollback protocol**: Stop → identify exploited dimension → rollback to pre-hacking checkpoint → fix reward → resume.

### Experiment Matrix

| Exp | Base | Algorithm | Reward | Purpose | Priority |
|-----|------|-----------|--------|---------|----------|
| B | SFT | — | — | Baseline | Required |
| C | SFT | GRPO | v1 | Basic RL | Required |
| D | SFT | GRPO | v2 | +efficiency, +conditional | **Core** |
| F | SFT | **GiGPO** | v2 | Direct SFT→GiGPO (fair comparison with D) | **Core** |
| E | GRPO-v2 | GRPO | v3 | +LLM-Judge (open-ended quality) | Optional |

> **Note**: Exp A (base model zero-shot) and Exp G (GPT-4o) are defined in phase5_ablation.md for evaluation only.

**Core comparison**: D vs F — same base (SFT), same reward (v2), same steps. Only difference is advantage estimation method.

**Exp E rationale**: v3 adds LLM-Judge for recommendation-type answers. It's exploratory — run only if D/F results are satisfactory and time permits. v3 output does not feed into GiGPO comparison.

---

## Evaluation Checkpoints

| Checkpoint | Metrics to Verify |
|------------|-------------------|
| After SFT | T1 tool call accuracy ≥ 95%, Format validity ≥ 98% |
| After GRPO-v1 | Format ≥ 95%, overall reward trending upward |
| After GRPO-v2 | T2 ≥ 80%, T3 ≥ 75%, redundant calls decreased |
| After GiGPO | T2/T3 measurably higher than GRPO-v2 |
| Final | All PRD Section 6 metrics, GRPO vs GiGPO comparison table |

## Data Versioning

```bash
# Track training data with DVC
dvc add training/sft/data/
dvc add training/grpo/data/

# Tag releases
git tag -a v1.0-sft-data -m "SFT training data v1.0"
```
