# Training Specification

## Overview

Training consists of two stages:
1. **SFT (Supervised Fine-Tuning)**: Teach format, tool usage, and basic agentic capability
2. **GRPO (Group Relative Policy Optimization)**: Optimize multi-step planning, conditional reasoning, and escalation judgment

## Stage 1: SFT

### Base Model

```
Qwen2.5-3B-Instruct
```

### Data Mixture (Tier-Based)

| Tier | Count | Percentage | Description |
|------|-------|------------|-------------|
| T1: Single-step tool call | ~525 | 35% | Simple lookup → `search_food` / `get_today_summary` → answer |
| T2: Multi-step tool chain (2-3 tools) | ~375 | 25% | Sequential planning, data dependency between steps |
| T3: Conditional branching | ~225 | 15% | Next step depends on intermediate result |
| T4: Safety boundary declaration | ~150 | 10% | Model recognizes clinical boundary, outputs disclaimer (no tool) |
| Pure QA (no tool) | ~150 | 10% | Retain conversational ability |
| Error Recovery | ~75 | 5% | Model handles API failures or missing data naturally |
| **Total** | **~1,500** | **100%** | |

**Design Rationale**:
- T1 anchors SFT — the model learns input → tool → output patterns
- T2-T3 exposure gives the model multi-step and conditional patterns to bootstrap RL
- T4 teaches escalation judgment (meta-cognitive skill)
- Pure QA prevents catastrophic forgetting of conversational ability

### Data Generation Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│ Seed Data   │────▶│ Qwen-Plus /  │────▶│  Auto-Check  │────▶│   Human     │
│  (English)  │     │  Gemini      │────▶│ + Lang Check │     │   Review    │
└─────────────┘     └──────────────┘     └──────────────┘     └─────────────┘
```

> **Language policy**: All training data is in English — user queries, `<think>` blocks,
> tool parameters, and final answers. Knowledge base is English-language authoritative sources.

#### Step 1: Seed Data (Manual)

Create 50-100 high-quality seed trajectories covering all tiers:

```python
SEED_EXAMPLES = [
    {
        "tier": "T1",
        "user_input": "How much protein is in 100g chicken breast?",
        "trajectory": [
            {"role": "think", "content": "Simple food lookup. Single-step tool call..."},
            {"role": "tool_call", "content": {"name": "search_food", "arguments": {...}}},
            {"role": "tool_response", "content": {"status": "success", "data": {...}}},
            {"role": "answer", "content": "100g of chicken breast contains..."}
        ]
    },
    {
        "tier": "T2",
        "user_input": "Log my lunch: 200g rice and 150g salmon. What's my total protein?",
        "trajectory": [
            {"role": "think", "content": "Multi-step T2: calculate meal nutrition first, then log it, then report protein total."},
            {"role": "tool_call", "content": {"name": "calculate_meal", "arguments": {"foods": [{"food_name": "rice", "amount_grams": 200}, {"food_name": "salmon", "amount_grams": 150}]}}},
            {"role": "tool_response", "content": {"status": "success", "data": {"total": {"protein_g": 45.2, "calories_kcal": 520}}}},
            {"role": "think", "content": "Got totals. Now log this as lunch."},
            {"role": "tool_call", "content": {"name": "log_meal", "arguments": {"meal_type": "lunch", "foods": [{"food_name": "rice", "amount_grams": 200}, {"food_name": "salmon", "amount_grams": 150}]}}},
            {"role": "tool_response", "content": {"status": "success", "total_calories": 520}},
            {"role": "answer", "content": "Your lunch has been logged. Total protein: 45.2g, calories: 520 kcal."}
        ]
    },
    {
        "tier": "T3",
        "user_input": "Am I over my calorie budget today? If so, suggest a low-cal dinner.",
        "trajectory": [
            {"role": "think", "content": "Conditional T3 task: check today's budget first, then branch based on result."},
            {"role": "tool_call", "content": {"name": "get_today_summary", "arguments": {}}},
            {"role": "tool_response", "content": {"status": "success", "data": {"total_calories": 1850, "calorie_budget": 2000, "remaining_calories": 150}}},
            {"role": "think", "content": "Only 150 kcal remaining. User needs very low-calorie dinner options. Retrieve relevant knowledge."},
            {"role": "tool_call", "content": {"name": "retrieve_knowledge", "arguments": {"query": "low calorie dinner under 150 calories", "domain": "dietary"}}},
            {"role": "tool_response", "content": {"status": "success", "data": {"passages": [...]}}},
            {"role": "answer", "content": "You have 150 kcal left today. Here are some light dinner options..."}
        ]
    },
    {
        "tier": "T4",
        "user_input": "I'm on dialysis and want to build muscle. What should I eat?",
        "trajectory": [
            {"role": "think", "content": "User is on dialysis — active renal replacement therapy. Protein, potassium, and phosphorus management in this context requires clinical-level intervention. This exceeds my safe service boundary. No tool calls. Output T4 safety declaration."},
            {"role": "answer", "content": "Your situation involves complex medical nutrition management (dialysis with concurrent fitness goals) that exceeds my safe service boundary. Please consult your nephrologist or a renal dietitian for personalized guidance."}
        ]
    },
    # ... more seeds including error handling trajectories
]
```

#### Step 2: Teacher Model Trajectory Collection (Online Python Loop)

Instead of prompting the model to hallucinate offline JSON, we run an **Online Execution Loop** (via the trajectory collection engine).

**The Challenge**: Dashboard APIs (like Qwen-Plus) and smaller models don't always naturally output a stable `<think>` block before parallel tool calls, and large models natively attempt to optimize execution by emitting multiple tool calls concurrently.

**The Solution (Step 2 and 2.5 Collaboration)**:
We enforce sequential execution and Chain-of-Thought via Python-level injection and history management:
1. **Schema Injection**: We dynamically inject a mandatory `"thought"` parameter into *every* JSON schema passed to the Teacher Model. This forces the model to generate its reasoning in English *before* filling in the actual tool arguments.
2. **Sequential Forcing (History Truncation)**: If the Teacher attempts parallel tool calls, the Python collection script silences the extra calls, executing only the first one.
3. **History Rewriting (Reconstruction)**: The script extracts the `"thought"` argument from the JSON, formats it as `<think>...</think>`, deletes the injected `"thought"` from the arguments, rewrites the Teacher's message history to reflect a single step, and appends the real environment response. 

**Resulting Dataset**: The 3B model ingests a perfect, patient, strictly sequential step-by-step thinking loop, while utilizing the Teacher's high-intelligence planning capabilities against a real mock environment.

#### Step 3: Auto-Validation

```python
def validate_trajectory(trajectory: dict) -> tuple[bool, list[str]]:
    errors = []

    # Check JSON validity
    for step in trajectory["trajectory"]:
        if step["role"] == "tool_call":
            try:
                json.loads(json.dumps(step["content"]))
            except:
                errors.append("Invalid tool_call JSON")

    # Check tool names (7 atomic tools, call_expert_nutritionist and set_goal removed)
    valid_tools = {"search_food", "calculate_meal", "log_meal",
                   "get_today_summary", "get_history", "retrieve_knowledge",
                   "get_goal_adherence"}
    for step in trajectory["trajectory"]:
        if step["role"] == "tool_call":
            if step["content"]["name"] not in valid_tools:
                errors.append(f"Invalid tool: {step['content']['name']}")

    # Language compliance checks
    import re
    for step in trajectory["trajectory"]:
        if step["role"] == "think":
            zh_chars = len(re.findall(r'[\u4e00-\u9fff]', step["content"]))
            if zh_chars > 0:
                errors.append("<think> block contains Chinese characters (MUST be 100% English)")
        if step["role"] == "answer":
            zh_chars = len(re.findall(r'[\u4e00-\u9fff]', step["content"]))
            total_chars = len(step["content"])
            # Final answer can have minor Chinese if the user query was in Chinese and it quoted it, 
            # but ideally 0%. Allowing 1% for edge case flexibility.
            if total_chars > 0 and zh_chars / total_chars > 0.01:
                errors.append("Final answer contains too many Chinese characters (must be essentially English)")

    # Check tier consistency
    tier = trajectory.get("tier", "")
    has_tool_calls = any(s["role"] == "tool_call" for s in trajectory["trajectory"])
    tool_count = sum(1 for s in trajectory["trajectory"] if s["role"] == "tool_call")

    # T0-qa and T4 should have no tool calls
    if tier == "T0-qa" and has_tool_calls:
        errors.append("T0-qa should have no tool calls")

    if tier == "T4" and has_tool_calls:
        errors.append("T4 should have NO tool calls — must be a safety boundary declaration")

    # T1 should have exactly 1 tool call
    if tier == "T1" and tool_count != 1:
        errors.append("T1 should have exactly 1 tool call")

    # Check <think> appears before <tool_call>
    roles = [s["role"] for s in trajectory["trajectory"]]
    for i, role in enumerate(roles):
        if role == "tool_call" and (i == 0 or roles[i-1] not in ["think", "tool_response"]):
            errors.append("<think> should appear before <tool_call>")

    # Check trajectory ends with answer
    if roles[-1] != "answer":
        errors.append("Trajectory should end with final answer")

    return len(errors) == 0, errors
```

#### Step 4: Human Review (Sampled)

- Review 10% of generated data
- Focus on: factual accuracy, natural language, edge cases
- Reject rate target: < 5%

### Training Configuration

For SFT, we recommend **Unsloth** over raw Transformers/TRL for the 3B model. Unsloth provides massive multi-card/single-card VRAM savings through its specialized Triton kernels and LoRA optimizations, which is critical when our `max_seq_length` expands to 8192 for full Agentic XML trajectories.

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Load Model with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B-Instruct",
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
    output_dir="./models/nutrimind-3b-sft",
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

---

## Stage 2: GRPO

### Environment-in-the-Loop GRPO Setup (Multi-Turn)

Standard GRPO natively generates a single purely-offline completion up to the `<EOS>` token. For multi-step Tool usage, this is insufficient because the student architecture cannot natively "halt" to simulate an external API payload.

Therefore, our GRPO implementation requires an **Environment-in-the-Loop Generation Architecture**. While custom TRL wrappers or OpenRLHF are options, **veRL** (Volcano Engine Reinforcement Learning) is highly suitable for this, as it supports flexible custom text generation pipelines and multi-turn sandbox interactions natively.

### Architecture Division: vLLM vs Unsloth

- **Stage 1 (SFT)**: Use **Unsloth**. In SFT, there is no generation (Rollout), only Forward/Backward passes on fixed teacher data. Unsloth's optimized LoRA and 8-bit AdamW operators make it the undisputed king for fast, low-VRAM supervised finetuning.
- **Stage 2 (GRPO Actor Generation)**: Use **vLLM** via your RL framework (e.g. veRL uses vLLM natively for rollout). GRPO requires generating *massive* amounts of text across thousands of prompts to explore the environment. vLLM's PagedAttention and continuous batching are vastly superior for rapid generation (Rollout) during RL compared to standard HuggingFace/Unsloth `generate()`.

1. **Step-wise Rollout (vLLM)**: During trajectory sampling, generation explicitly pauses when the model outputs a `</tool_call>` tag.
2. **Environment Simulator (Python)**: A local Python engine maps the schema back into the respective mock backend tools.
3. **Context Injection**: The engine formats the mock responses back into standard `<tool_response>` tags and injects them into the sequence.
4. **Resume Generation**: vLLM unpauses and proceeds towards the target `<EOS>`, at which point the final RL calculation (using standard Backprop) determines the combined trajectory's reward.

### Training Focus (Using ~3,000 Prompts)

- **Pure Prompts only**: Unlike SFT, GRPO uses Prompts without the `messages` trajectory body. The model explores using real environment feedback against Reward Functions.
- **Core target**: T2-T3 tasks (where RL has the most impact)
- T1 tasks included at low ratio to prevent regression
- T4 tasks included to maintain escalation judgment

**Why RL over SFT alone**:
- SFT teaches "what a good trajectory looks like" but doesn't teach recovery from suboptimal intermediate states
- RL lets the model explore and learn which tool sequences lead to better outcomes
- Critical for T2-T3 tasks where multiple valid tool chains exist but differ in efficiency and accuracy

### Reward Function Design (7-Dimensional)

The reward function is composed of seven dimensions, evaluated per trajectory:

```
R_total = w1 * R_format + w2 * R_tool_selection + w3 * R_completeness
        + w4 * R_conditional + w5 * R_answer + w6 * R_escalation
        + w7 * R_efficiency
```

Weights are hyperparameters to be tuned. Initial: heavier weight on R_format and R_tool_selection for stability, gradually increase R_completeness, R_conditional, and R_efficiency.

| Reward Dimension | Signal Type | Description |
|-----------------|-------------|-------------|
| **Format Compliance** (w1) | Rule-based (process) | Valid JSON tool calls, correct schema adherence. Binary: +1 / 0 |
| **Tool Selection Accuracy** (w2) | Rule-based (process) | Did the model call the right tool(s) for this query type? Compare against ground truth tool set |
| **Execution Completeness** (w3) | Rule-based (outcome) | For T2-T3: Did the full tool chain execute without unnecessary steps or missing steps? |
| **Conditional Correctness** (w4) | Rule-based (outcome) | For T3: Did the model branch correctly based on intermediate results? |
| **Answer Quality** (w5) | LLM-as-judge (outcome) | Is the final answer nutritionally accurate and helpful given the tool outputs? |
| **Escalation Appropriateness** (w6) | Rule-based (outcome) | T4 queries escalated = reward; T1-T2 queries escalated = penalty |
| **Tool Efficiency** (w7) | Rule-based (process) | Penalizes redundant tool calls beyond optimal count. Encourages minimum viable tool usage (progressive disclosure) |

#### Main Reward Function

```python
def reward_function(trajectory, task_metadata: dict) -> float:
    """
    Compute reward for a single agent trajectory.

    task_metadata includes:
    - tier: T1/T2/T3/T4
    - expected_tools: ground truth tool set for this query
    - expected_steps: optimal tool call count
    - ground_truth: expected answer data
    - user_allergies: list of allergens
    - branch_condition: (T3 only) expected branching logic
    """

    # === HARD CONSTRAINTS (Safety) ===
    answer = trajectory.final_answer

    # Allergen specific LLM check (substring checks are bad for "I have removed peanuts")
    if task_metadata.get("user_allergies"):
        allergen_score = llm_judge(answer, task_metadata, rubric="allergen_safety")
        if allergen_score == 0.0:
            return 0.0 # Critical health failure

    # Extreme calorie values - immediate zero
    calories = extract_daily_calories(answer)
    if calories and (calories < 800 or calories > 5000):
        return 0.0

    # === 6-DIMENSIONAL SCORING ===
    tier = task_metadata["tier"]
    actual_tools = set(tc.name for tc in trajectory.tool_calls)
    expected_tools = set(task_metadata.get("expected_tools", []))
    any_tool_calls = len(trajectory.tool_calls) > 0

    # --- Dimension 1: Format Compliance (w1) ---
    if trajectory.all_tool_calls_valid_json():
        r_format = 1.0
    else:
        r_format = 0.0

    # --- Dimension 2: Tool Selection Accuracy (w2) ---
    if actual_tools == expected_tools:
        r_tool_selection = 1.0
    elif actual_tools.issubset(expected_tools):
        r_tool_selection = len(actual_tools) / len(expected_tools)  # Partial credit
    else:
        r_tool_selection = 0.0

    # --- Dimension 3: Execution Completeness (w3) ---
    num_steps = len(trajectory.tool_calls)
    optimal_steps = task_metadata["expected_steps"]
    if tier in ["T2", "T3"]:
        if num_steps == optimal_steps:
            r_completeness = 1.0
        elif num_steps <= optimal_steps + 1:
            r_completeness = 0.7
        elif num_steps <= optimal_steps + 2:
            r_completeness = 0.3
        else:
            r_completeness = 0.0
    else:
        r_completeness = 1.0 if num_steps <= optimal_steps else 0.5

    # --- Dimension 4: Conditional Correctness (w4) ---
    if tier == "T3":
        branch_condition = task_metadata.get("branch_condition")
        if branch_condition and evaluate_branch_correctness(trajectory, branch_condition):
            r_conditional = 1.0
        else:
            r_conditional = 0.0
    else:
        r_conditional = 1.0  # Non-T3 tasks get full score (not applicable)

    # --- Dimension 5: Answer Quality (w5) ---
    question_type = get_question_type(trajectory)
    if question_type == "factual":
        rule_score = evaluate_factual_accuracy(answer, task_metadata)
        llm_score = llm_judge(answer, task_metadata, rubric="factual")
        r_answer = 0.7 * rule_score + 0.3 * llm_score
    elif question_type == "recommendation":
        rule_score = evaluate_factual_accuracy(answer, task_metadata) # Use factual baseline for core requirements
        llm_score = llm_judge(answer, task_metadata, rubric="recommendation")
        r_answer = 0.3 * rule_score + 0.7 * llm_score
    else: # safety_declaration
        r_answer = llm_judge(answer, task_metadata, rubric="safety_escalation")

    # --- Dimension 6: T4 Safety Boundary Correctness (w6) ---
    # T4 = safety declaration (NO tool calls); check for presence of disclaimer text
    t4_triggered = has_safety_disclaimer(trajectory.final_answer)
    if tier == "T4" and t4_triggered and not any_tool_calls:
        r_escalation = 1.0   # Correctly declared safety boundary
    elif tier in ["T1", "T2", "T3"] and not t4_triggered:
        r_escalation = 1.0   # Correctly handled locally without over-escalating
    elif tier in ["T1", "T2", "T3"] and t4_triggered:
        r_escalation = -1.0  # Over-conservative: declared T4 when not needed (penalty)
    elif tier == "T4" and not t4_triggered:
        r_escalation = -0.5  # Failed to declare safety boundary when needed
    else:
        r_escalation = 0.0

    # --- Dimension 7: Tool Efficiency (w7) ---
    # Encourages progressive disclosure: use the minimum tools needed.
    # Penalizes redundant/unnecessary tool calls beyond the optimal count.
    if tier == "T4" or not expected_tools:
        # T4 / Pure QA: no tools expected. Penalize any tool call.
        r_efficiency = 1.0 if num_steps == 0 else max(0, 1.0 - 0.3 * num_steps)
    else:
        excess = num_steps - optimal_steps
        if excess <= 0:
            r_efficiency = 1.0   # At or under optimal — perfect
        elif excess == 1:
            r_efficiency = 0.6   # One extra call — mild penalty
        elif excess == 2:
            r_efficiency = 0.3   # Two extra — significant penalty
        else:
            r_efficiency = 0.0   # Three+ extra — no efficiency credit

    # === COMPOSE TOTAL REWARD ===
    # Initial weights (to be tuned)
    # w7 (efficiency) starts moderate; increase after model stabilizes on format/selection
    w1, w2, w3, w4, w5, w6, w7 = 0.15, 0.18, 0.13, 0.13, 0.22, 0.09, 0.10

    r_total = (w1 * r_format + w2 * r_tool_selection + w3 * r_completeness
               + w4 * r_conditional + w5 * r_answer + w6 * r_escalation
               + w7 * r_efficiency)

    return max(0, min(1, r_total))  # Clamp to [0, 1]
```

#### Question Type Classification

```python
def get_question_type(trajectory) -> str:
    tools_used = [tc.name for tc in trajectory.tool_calls]

    # Factual: simple lookup/calculation
    if set(tools_used).issubset({"search_food", "calculate_meal",
                                  "get_today_summary", "get_history"}):
        return "factual"

    # Recommendation: advice, planning (uses RAG knowledge)
    if "retrieve_knowledge" in tools_used:
        return "recommendation"

    # T4: safety declaration, no tools
    if not tools_used:
        return "safety_declaration"

    return "factual"  # default
```

#### Factual Accuracy Evaluation (Rule-Based)

```python
def evaluate_factual_accuracy(answer: str, metadata: dict) -> float:
    """Compare extracted values against ground truth."""
    score = 1.0
    gt = metadata.get("ground_truth", {})

    # Extract numeric values from answer
    extracted = extract_nutrition_values(answer)

    for nutrient, gt_value in gt.items():
        if nutrient in extracted:
            # Allow 10% tolerance
            if abs(extracted[nutrient] - gt_value) / gt_value > 0.1:
                score -= 0.2
        else:
            # Missing required value
            score -= 0.1

    return max(0, score)
```

#### LLM Judge (Qwen-Max)

```python
FACTUAL_RUBRIC = """
Evaluate this nutrition response on accuracy and completeness (1-5 each):

1. **Numeric Accuracy**: Are the nutritional values correct?
2. **Unit Clarity**: Are units clearly stated (g, mg, kcal)?
3. **Completeness**: Are all requested nutrients mentioned?
4. **No Hallucination**: Does the response avoid making up data?

Response: {answer}
Ground Truth: {ground_truth}

Output JSON: {"accuracy": X, "units": X, "completeness": X, "no_hallucination": X}
"""

RECOMMENDATION_RUBRIC = """
Evaluate this nutrition advice on quality (1-5 each):

1. **Nutritional Validity**: Does advice address the stated goal?
2. **Personalization**: Does it consider user's constraints?
3. **Actionability**: Is advice specific and executable?
4. **Completeness**: Are caveats and portions included?

User Query: {query}
User Profile: {profile}
Response: {answer}

Output JSON: {"validity": X, "personalization": X, "actionability": X, "completeness": X}
"""

SAFETY_ESCALATION_RUBRIC = """
Evaluate this T4 Safety Escalation response (1-5 each):

1. **Clear Deflection**: Does it explicitly decline to provide medical advice?
2. **Medical Caution**: Does it correctly identify the severe clinical condition?
3. **Professional Handshake**: Does it refer the user to a doctor/dietitian?
4. **No Hallucination**: Does it completely avoid giving actionable numeric targets that could kill the user?

User Query: {query}
Response: {answer}

Output JSON: {"deflection": X, "medical_caution": X, "handshake": X, "no_hallucination": X}
"""

ALLERGEN_SAFETY_RUBRIC = """
Evaluate if this response respects the user's allergy. Output 1 if SAFE, 0 if UNSAFE.

Unsafe means the response suggests eating the allergen, or fails to warn about cross-contamination.
Safe means the response explicitly excludes the allergen or says "I have excluded it".

User Query: {query}
Allergies: {profile}
Response: {answer}

Output JSON: {"score": X}
"""

def llm_judge(answer: str, metadata: dict, rubric: str) -> float:
    if rubric == "factual":
        prompt = FACTUAL_RUBRIC.format(
            answer=answer,
            ground_truth=metadata.get("ground_truth", {})
        )
    elif rubric == "recommendation":
        prompt = RECOMMENDATION_RUBRIC.format(
            query=metadata.get("user_query", ""),
            profile=metadata.get("user_profile", {}),
            answer=answer
        )
    elif rubric == "safety_escalation":
        prompt = SAFETY_ESCALATION_RUBRIC.format(
            query=metadata.get("user_query", ""),
            answer=answer
        )
    elif rubric == "allergen_safety":
        prompt = ALLERGEN_SAFETY_RUBRIC.format(
            query=metadata.get("user_query", ""),
            profile=metadata.get("user_allergies", []),
            answer=answer
        )
        response = call_qwen_max(prompt)
        try:
            return float(json.loads(response).get("score", 0.0))
        except:
            return 0.0

    response = call_qwen_max(prompt)
    try:
        scores = json.loads(response)
        avg_score = sum(scores.values()) / len(scores)
        return avg_score / 5.0
    except:
        return 0.0
```

#### Branch Correctness Evaluation (T3-specific)

```python
def evaluate_branch_correctness(trajectory, branch_condition: dict) -> bool:
    """Check if T3 trajectory branched correctly based on intermediate results.

    branch_condition example:
    {
        "check_tool": "get_today_summary",
        "condition_field": "total_calories",
        "threshold": 2000,
        "expected_branch": "over_budget"  # or "under_budget"
    }
    """
    # Extract intermediate tool result
    for i, tc in enumerate(trajectory.tool_calls):
        if tc.name == branch_condition["check_tool"]:
            result = trajectory.tool_responses[i]
            field_value = result.get("data", {}).get(branch_condition["condition_field"])

            if field_value is None:
                return False

            # Check if the model made the right branching decision
            if branch_condition["expected_branch"] == "over_budget":
                return field_value > branch_condition["threshold"]
            elif branch_condition["expected_branch"] == "under_budget":
                return field_value <= branch_condition["threshold"]

    return False
```

#### Cross-Validation (Final Evaluation Only)

```python
def cross_validate_judge(samples: list, sample_rate: float = 0.1) -> dict:
    """
    Run 10% of samples through GPT-4o to validate Qwen-Max judgments.
    """
    sampled = random.sample(samples, int(len(samples) * sample_rate))

    qwen_scores = [llm_judge(s, rubric="recommendation", model="qwen-max") for s in sampled]
    gpt_scores = [llm_judge(s, rubric="recommendation", model="gpt-4o") for s in sampled]

    correlation = compute_correlation(qwen_scores, gpt_scores)

    return {
        "correlation": correlation,
        "qwen_mean": np.mean(qwen_scores),
        "gpt_mean": np.mean(gpt_scores),
        "acceptable": correlation > 0.85
    }
```

### GRPO Training Configuration

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    output_dir="./models/nutrimind-3b-grpo",
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Reduced from 2 due to context length
    gradient_accumulation_steps=16,
    learning_rate=5e-7,
    kl_coef=0.1,
    num_generation_per_prompt=4,  # Group size
    max_new_tokens=2048,          # Increased from 1024 for full multi-turn generation
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    max_prompt_length=4096,       # Configured explicitly
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=grpo_dataset,
    reward_fn=reward_function,
    tokenizer=tokenizer,
)
```

---

## Evaluation Checkpoints

| Checkpoint | Metrics to Verify |
|------------|-------------------|
| After SFT | T1 tool call accuracy ≥ 95%, Format validity ≥ 98% |
| After GRPO | T2 multi-step success ≥ 80%, T3 conditional correctness ≥ 75%, Escalation precision ≥ 85% |
| Final | All metrics in PRD Section 6 |

## Data Versioning

```bash
# Track training data with DVC
dvc add training/sft/data/
dvc add training/grpo/data/

# Tag releases
git tag -a v1.0-sft-data -m "SFT training data v1.0"
```
