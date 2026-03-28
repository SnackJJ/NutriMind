# Tool Specification

## Format Convention

The model uses XML-tag delimited tool calls:

**Model Output:**
```text
<think>
[Internal reasoning in English: intent recognition, tool selection, parameter planning]
</think>
<tool_call>
{"name": "tool_name", "arguments": {"param1": "value1"}}
</tool_call>
```

**System Response:**
```text
<tool_response>
{"status": "success", "data": { ... }}
</tool_response>
```

**Error Response:**
```text
<tool_response>
{"status": "error", "error_type": "not_found", "message": "Food not found in database."}
</tool_response>
```

### Format Rules

1. JSON inside `<tool_call>` must be valid and parseable in a single pass
2. One tool call per generation step (sequential only, no parallel)
3. Model MUST output `<think>` before `<tool_call>`
4. After `<tool_response>`, model either chains another tool call or produces final answer
5. Final answers are plain text (no tags)
6. If answerable from internal knowledge, skip tool call

---

## Tool Calling Protocol

> **Decision Reference**: [ADR-001: Pure Text Tool Calling](../decisions/001-pure-text-tool-calling.md)

### Design Rationale

We use **pure text tool calling** instead of function calling APIs (e.g., DashScope's `tools` parameter). This decision applies to:
- **Trajectory collection**: Teacher model (qwen3.5-plus) outputs tool calls as plain text
- **SFT training**: Training data contains `<tool_call>` tags as literal text tokens
- **Student inference**: Deployed model outputs the same text format

**Why not function calling API?**
1. Function calling APIs discard intermediate `<think>` content — we lose reasoning traces
2. API returns structured `tool_calls` array, which differs from our SFT target format
3. Student model deployment (vLLM/local) has no function calling layer — format gap

**Result**: Teacher output == SFT target == Student output. Zero format conversion.

### Agent-Side Text Parsing Middleware

The agent environment (both collection and inference) implements a text parsing middleware that handles the tool execution loop:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Agent Execution Loop                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │  Model   │───▶│  Detect     │───▶│  Extract &   │───▶│  Execute   │  │
│  │  Output  │    │  <tool_call>│    │  Parse JSON  │    │  Tool Fn   │  │
│  └──────────┘    └─────────────┘    └──────────────┘    └────────────┘  │
│       │                                                        │         │
│       │ (no tool call = final answer)                          │         │
│       ▼                                                        ▼         │
│  ┌──────────┐                                          ┌────────────┐   │
│  │  Return  │◀────────────────────────────────────────│  Wrap as   │   │
│  │  Answer  │         (append to context)              │<tool_resp> │   │
│  └──────────┘                                          └────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Loop steps**:
1. **Generate**: Model outputs text (may contain `<think>` and/or `<tool_call>`)
2. **Detect**: Parser scans for `<tool_call>...</tool_call>` tags
3. **Extract**: If found, extract JSON payload from within tags
4. **Validate**: Parse JSON, validate against tool schema (name, required params)
5. **Execute**: Call corresponding tool function with validated arguments
6. **Wrap**: Format tool result as `<tool_response>...</tool_response>`
7. **Append**: Add assistant message + user message (tool response) to context
8. **Loop**: Continue generation until no `<tool_call>` is detected (final answer)

### Parser Specification

**Tag detection** (regex):
```python
TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL
)
```

**JSON schema** (required fields):
```json
{
  "name": "string (must be in VALID_TOOLS)",
  "arguments": "object (tool-specific parameters)"
}
```

**Error handling**:
| Error Type | Behavior |
|------------|----------|
| No closing tag | Wait for more tokens (streaming) or reject (batch) |
| Invalid JSON | Return `<tool_response>{"status": "error", "error_type": "invalid_json", "message": "..."}</tool_response>` |
| Unknown tool name | Return `<tool_response>{"status": "error", "error_type": "unknown_tool", "message": "..."}</tool_response>` |
| Missing required param | Return `<tool_response>{"status": "error", "error_type": "missing_param", "message": "..."}</tool_response>` |

**Sequential enforcement**: If model outputs multiple `<tool_call>` tags in one generation, only the first is executed. Remaining calls are silently ignored. This enforces the single-tool-per-turn constraint.

### System Prompt Tool Definition Injection

Since we don't use the API's `tools` parameter, tool definitions must be injected into the system prompt:

```text
## Available Tools

You have access to the following tools. To use a tool, output:
<tool_call>
{"name": "tool_name", "arguments": {...}}
</tool_call>

### get_food_nutrition
Look up nutrition information for one or more foods.
Parameters:
- foods (required): Array of {food_name: string, amount_grams: number}

### log_meal
Record a meal to user history.
Parameters:
- meal_type (required): "breakfast" | "lunch" | "dinner" | "snack"
- foods (required): Array of {food_name: string, amount_grams: number}

[... remaining tools ...]
```

### Code Location

| Component | File | Shared Between |
|-----------|------|----------------|
| Parser + Executor | `src/orchestrator/tool_parser.py` | Collection + Inference |
| Tool definitions (for prompt) | `configs/tools.yaml` → `system_prompt_tools` | Collection + Inference |
| Collection loop | `src/training/sft/collect_trajectories.py` | Collection only |
| Inference loop | `src/orchestrator/orchestrator.py` | Inference only |

---

## Tool Definitions

#### Tool 1: `get_food_nutrition`

| Attribute | Value |
|-----------|-------|
| **Description** | Look up nutrition information for one or more foods from USDA database using a hybrid retrieval engine (BM25 + BGE Embeddings + Reranking). Returns detailed nutrients per food plus totals. |
| **When to use** | ANY food nutrition query — single food or multiple foods / full meal |
| **When NOT to use** | Dietary guidelines/advice (use `retrieve_knowledge`) |
| **Latency / Cost** | < 200ms / Free |

```json
{
  "name": "get_food_nutrition",
  "parameters": {
    "foods": {
      "type": "array",
      "required": true,
      "description": "List of foods to look up. For single food query, provide one item.",
      "items": { "food_name": "string", "amount_grams": "number" },
      "example": [
        {"food_name": "chicken breast", "amount_grams": 150}
      ]
    }
  },
  "returns": {
    "status": "success | partial_success | partial_failure | error",
    "data": {
      "total": { "calories_kcal": "number", "protein_g": "number", "fat_g": "number", "carbs_g": "number" },
      "breakdown": [
        {
          "food_name": "string — matched USDA description",
          "match_confidence": "high | medium | low",
          "match_score": "number — hybrid similarity score (0.0 to 1.0)",
          "match_note": "string — optional, present when confidence is low",
          "...nutrients": "calories, protein, fat, carbs, fiber, sugars, sodium, cholesterol, vitamins, etc."
        }
      ],
      "macro_ratio": { "protein_pct": "number", "fat_pct": "number", "carbs_pct": "number" },
      "failed_items": ["optional — list of foods not found, with error details"]
    }
  },
  "error_cases": ["empty_food_list", "all_foods_not_found"]
}
```

**`match_confidence` semantics**:
- `high` (Score > 0.85) — Strong match. The USDA description highly correlates with the query both lexically and semantically.
- `medium` (Score 0.5 - 0.85) — Good match. Most query components align with the USDA entry.
- `low` (Score 0.35 - 0.5) — Weak match. The tool found a potential candidate, but semantic drift is possible (e.g., query "cola" matching "chocolate pudding").

**When `match_confidence: low`**, the model MUST:
1. Examine `food_name` in the result — does it semantically match the query?
2. If yes: proceed normally.
3. If no: retry with a more specific query term (e.g. USDA-style name like `"Beverages, carbonated, cola"` instead of `"cola"`).
4. If second retry also returns `low` confidence: ask the user to clarify, do not fabricate nutrition values.

**Search behavior note**: The underlying retrieval uses a hybrid Lexical-Semantic framework (BM25 + Semantic Embeddings + Cross-Encoder Reranking). This is significantly more robust than simple keyword matching but still requires specific inputs for ambiguous items. If a score is below `0.35`, the tool returns `food_not_found` to prevent hallucination.

**Status semantics**:
- `success` — all foods found
- `partial_success` — some foods found, some failed (check `failed_items`)
- `partial_failure` — no foods found at all but list was non-empty
- `error` — system error or empty input

**Single food example**:
```json
{"name": "get_food_nutrition", "arguments": {"foods": [{"food_name": "chicken breast", "amount_grams": 100}]}}
```
Returns `breakdown[0]` with full nutrient details for that one food.

---

### Tool 2: `log_meal`

| Attribute | Value |
|-----------|-------|
| **Description** | Persist a single meal record to user history |
| **When to use** | User explicitly reports a meal they ate and wants it recorded |
| **When NOT to use** | User just asks questions; use `get_today_summary` to query history |
| **Latency / Cost** | < 50ms / Free |

```json
{
  "name": "log_meal",
  "parameters": {
    "meal_type": { "type": "string", "enum": ["breakfast", "lunch", "dinner", "snack"], "required": true },
    "foods": {
      "type": "array",
      "required": true,
      "items": { "food_name": "string", "amount_grams": "number" }
    },
    "timestamp": { "type": "string", "format": "ISO-8601", "required": false, "default": "now" }
  },
  "returns": { "status": "success", "meal_id": "string", "total_calories": "number" },
  "error_cases": ["missing_required_field", "invalid_meal_type"]
}
```

---

### Tool 3: `get_today_summary`

| Attribute | Value |
|-----------|-------|
| **Description** | Retrieve today's nutritional intake summary and remaining budget |
| **When to use** | User asks about today's totals, remaining calories, or macro progress |
| **When NOT to use** | Multi-day trends (use `get_history`) |
| **Latency / Cost** | < 50ms / Free |

```json
{
  "name": "get_today_summary",
  "parameters": {},
  "returns": {
    "date": "string",
    "total_calories": "number",
    "calorie_budget": "number",
    "remaining_calories": "number",
    "protein_g": "number",
    "fat_g": "number",
    "carbs_g": "number",
    "fiber_g": "number",
    "meal_count": "number",
    "food_summary": "string — comma-separated list of all foods eaten today",
    "meals_logged": [
      {
        "meal_id": "string",
        "meal_type": "string — breakfast | lunch | dinner | snack",
        "logged_at": "string — ISO-8601 timestamp",
        "food_names": "string — comma-separated list of foods in this meal",
        "calories_kcal": "number",
        "protein_g": "number",
        "fat_g": "number",
        "carbs_g": "number"
      }
    ]
  },
  "error_cases": ["no_profile_found"]
}
```

---

### Tool 4: `get_history`

| Attribute | Value |
|-----------|-------|
| **Description** | Query multi-day nutritional history, trends, and optionally goal adherence analysis |
| **When to use** | User asks about past N days' eating patterns, weekly averages, trends, OR goal progress / adherence rate |
| **When NOT to use** | Today only (use `get_today_summary`) |
| **Latency / Cost** | < 100ms / Free |

```json
{
  "name": "get_history",
  "parameters": {
    "days": { "type": "integer", "default": 7, "max": 90 },
    "metric": { "type": "string", "enum": ["calories", "protein", "fat", "carbs", "all"], "default": "all" },
    "compare_to_goal": { "type": "boolean", "default": false, "description": "When true, include goal adherence analysis (adherence_pct, days_within/over/under target, avg_deviation)" }
  },
  "returns": {
    "period": "string",
    "daily_averages": { "calories_kcal": "number", "protein_g": "number" },
    "trend": "string",
    "daily_breakdown": ["array of daily summaries"],
    "goal_adherence": {
      "description": "Only present when compare_to_goal=true",
      "<metric_name>": {
        "target_value": "number",
        "daily_average": "number",
        "days_within_target": "number (±10% tolerance)",
        "days_over_target": "number",
        "days_under_target": "number",
        "adherence_pct": "number",
        "avg_deviation": "number"
      }
    }
  },
  "error_cases": ["invalid_date_range", "no_data_in_range"]
}
```

**`compare_to_goal` behavior**: When `true`, fetches targets from `user_goals` table (falls back to defaults: 2000 kcal, 90g protein, 65g fat, 250g carbs). Computes adherence within ±10% tolerance window.

---

### Tool 5: `retrieve_knowledge`

| Attribute | Value |
|-----------|-------|
| **Description** | RAG search over local English-language nutrition knowledge base |
| **When to use** | Dietary guidelines, medical nutrition principles, supplement info |
| **When NOT to use** | Food facts/nutrition lookup (use `get_food_nutrition`) |
| **Latency / Cost** | < 200ms / Free |

```json
{
  "name": "retrieve_knowledge",
  "parameters": {
    "query": { "type": "string", "required": true, "example": "type 2 diabetes dietary guidelines" },
    "mode": {
      "type": "string",
      "enum": ["hybrid", "semantic", "keyword"],
      "default": "hybrid",
      "description": "Retrieval strategy. hybrid=BM25+semantic+RRF, semantic=embedding only, keyword=BM25 only"
    },
    "top_k": { "type": "integer", "default": 3, "max": 5 }
  },
  "returns": {
    "status": "success",
    "top_relevance_score": "float — highest rerank score among results",
    "data": {
      "passages": [{
        "content": "string",
        "source": "string — document title (use for relevance judgment)",
        "source_id": "string — source identifier",
        "section": "string — section heading (use for relevance judgment)",
        "url": "string — source URL",
        "relevance_score": "float — rerank score if available, else RRF score"
      }]
    }
  },
  "error_cases": ["empty_query", "no_relevant_results", "internal_error"]
}
```

**Model-driven relevance judgment** (no `retrieval_quality` field):
- Model examines `top_relevance_score` AND passage `source`/`section` to judge relevance.
- If score > 0.7 AND passage topic matches query: use the result.
- If score < 0.4 OR passage topic is off-topic (e.g., asked about VLCD risks but got calcium supplement info): reformulate query or switch mode.
- After 3 attempts with poor results: fall back to internal knowledge with disclaimer.

**Recommended retrieval strategy**:
1. Start with `mode: "hybrid"` (default).
2. If result is off-topic: switch mode (`keyword` for precise terms, `semantic` for concepts) or rephrase.
3. **Max 3 retries** before graceful degradation with explicit disclaimer.

---

### Tool 6: `set_goal`

| Attribute | Value |
|-----------|-------|
| **Description** | Set or update a specific nutrition target for the user |
| **When to use** | User wants to define or change daily calorie/macro goals, or specify weight management direction (lose/maintain/gain) |
| **When NOT to use** | User asks about goal progress (use `get_history(compare_to_goal=True)`) |
| **Latency / Cost** | < 50ms / Free |

```json
{
  "name": "set_goal",
  "parameters": {
    "metric": {
      "type": "string",
      "enum": ["calories", "protein", "fat", "carbs"],
      "required": true,
      "description": "Which nutrition metric to set a target for"
    },
    "target_value": {
      "type": "number",
      "required": true,
      "description": "Daily target value (kcal for calories, grams for macros). Calories: 1000-5000 kcal; macros: > 0g"
    },
    "goal_type": {
      "type": "string",
      "enum": ["lose", "maintain", "gain"],
      "required": false,
      "description": "Overall weight management direction. Persisted to user profile when provided."
    }
  },
  "returns": {
    "status": "success",
    "data": {
      "metric": "string — the metric that was updated",
      "previous_value": "number | null — old target value (null if first time)",
      "new_value": "number — the newly set target",
      "goal_type": "string — current goal direction (lose/maintain/gain)"
    }
  },
  "error_cases": ["invalid_metric", "value_out_of_range", "invalid_goal_type"]
}
```

**Validation rules**:
- `calories`: must be between 1000-5000 kcal
- `protein/fat/carbs`: must be > 0g
- `goal_type`: must be one of `lose`, `maintain`, `gain` (optional)

---

## T4: Safety Boundary Declaration (No Tool)

**T4 is NOT a tool call.** When a user's query involves complex medical nutrition
management beyond the agent's safe operational boundary, the model outputs a
standardized disclaimer directly without invoking any tool.

**T4 Trigger Conditions:**
- Active dialysis or post-organ transplant nutrition
- Post-surgery recovery nutrition protocols
- Complex drug-nutrient interactions (multiple medications)
- Active cancer treatment nutrition
- Multiple severe comorbidities (e.g., diabetes + CKD + heart failure combined)

**T4 Response Pattern:**
```
<think>
User's situation involves [specific conditions]. This requires clinical-level nutrition
management that exceeds my safe service boundary. I should not attempt to advise.
</think>

Your situation involves complex medical nutrition management that exceeds my safe
service boundary. Please consult your physician or a registered dietitian for
personalized guidance.
```

**Key distinctions:**
- T3: Complex but manageable → use `retrieve_knowledge` + other tools + synthesize
- T4: Medically fragile → declare boundary, no tool calls, recommend professional

---

## Tool Selection Decision Tree (Progressive Disclosure)

The model should evaluate tool needs in escalating stages, starting from the
simplest judgment and only progressing deeper when needed:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 0: Do I need a tool at all?                                  │
│                                                                     │
│   • Can I answer from internal knowledge alone?                     │
│     → YES: answer directly, no tool call (Pure QA)                  │
│     → NO: proceed to Stage 1                                        │
│                                                                     │
│   • Does this involve a medically high-risk condition?              │
│     (dialysis, post-transplant, active cancer treatment, etc.)      │
│     → YES: safety boundary declaration, no tool call (T4)           │
│     → NO: proceed to Stage 1                                        │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 1: Which single tool answers the question?                   │
│                                                                     │
│   • Food nutrition query (1 or more)?    → get_food_nutrition       │
│   • User wants to RECORD a meal?         → log_meal                 │
│   • TODAY's intake / remaining budget?   → get_today_summary        │
│   • Past N DAYS' trends / averages?      → get_history              │
│   • Goal progress / adherence rate?      → get_history(compare_to_goal=true) │
│   • User wants to SET/CHANGE a goal?     → set_goal                 │
│   • Evidence-based dietary guidelines?   → retrieve_knowledge       │
│                                                                     │
│   If a single tool suffices → call it and answer (T1)               │
│   If the answer requires more context → proceed to Stage 2          │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 2: Do I need to chain multiple tools?                        │
│                                                                     │
│   Common multi-step patterns (T2):                                  │
│   • get_food_nutrition → log_meal       (compute then persist)      │
│   • get_food_nutrition → log_meal → get_today_summary (full track)  │
│   • get_food_nutrition → retrieve_knowledge (lookup + contextualize)│
│   • set_goal → get_today_summary        (set target then show status)│
│                                                                     │
│   Plan the minimal tool sequence, execute step-by-step.             │
│   If the next step depends on a result → proceed to Stage 3         │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 3: Do I need conditional branching?                          │
│                                                                     │
│   Execute first tool → inspect result → decide next action (T3):    │
│   • get_today_summary → over budget? → retrieve_knowledge           │
│   • get_history(compare_to_goal=true) → low adherence? → retrieve_knowledge │
│   • get_history(compare_to_goal=true) → consistently over? → set_goal (lower target) │
│   • get_history → declining trend? → retrieve_knowledge             │
│   • get_food_nutrition → not found / low_confidence? → retry with USDA-style name → ask user to clarify │
│                                                                     │
│   Branch based on intermediate data, not pre-planned.               │
└─────────────────────────────────────────────────────────────────────┘
```

**Key principle**: Always start at Stage 0. Most queries resolve at Stage 0-1.
Only escalate when the current stage cannot fully satisfy the user's request.
Use the **minimum number of tools** needed to produce a complete, accurate answer.

---

## Tier-to-Tool Mapping

| Tier | Tool Pattern | Example |
|------|-------------|---------|
| **T1** | Single tool call | `get_food_nutrition` → answer; `set_goal` → answer |
| **T2** | Multi-step tool chain (2-3 tools) | `set_goal` → `get_today_summary` → answer; `get_food_nutrition` → `log_meal` → answer |
| **T3** | Multi-step with conditional branching | `get_history(compare_to_goal=true)` → *branch*: if consistently over → `set_goal` → answer |
| **T4** | Safety boundary declaration (no tool) | `<think>` → direct disclaimer → recommend professional |

---

## Error Handling Protocol

| Error / Signal | Strategy |
|----------------|----------|
| `food_not_found` | Retry with a more specific, USDA-style name; if second retry also fails, ask user to clarify |
| `match_confidence: low` | Inspect returned `food_name`; if semantically wrong, retry with category + descriptor (e.g. `"carbonated beverage cola"` not `"cola"`); max 1 retry then ask user |
| `ambiguous_food_name` | Present candidates, ask user to choose |
| `no_relevant_results` | Answer from internal knowledge + note limitation |
| Low score or off-topic passage | Examine `top_relevance_score` + `source`/`section`; if score < 0.4 or topic mismatch, switch mode or rephrase; max 3 retries then use internal knowledge with disclaimer |
| Empty passages | No RAG results; answer from internal knowledge with explicit note |

Error handling trajectories should comprise ~10% of SFT data.

---

## Invocation Patterns

### Pattern: T1 — Single-Step Tool Call

```
User: How much protein is in 100g chicken breast?

<think>
Simple food lookup (T1). User wants nutritional data for a single food.
Call get_food_nutrition with one food item.
</think>
<tool_call>
{"name": "get_food_nutrition", "arguments": {"foods": [{"food_name": "chicken breast", "amount_grams": 100}]}}
</tool_call>

<tool_response>
{"status": "success", "data": {"total": {...}, "breakdown": [{"calories_kcal": 165, "protein_g": 31, ...}]}}
</tool_response>

100g of chicken breast contains approximately 31g of protein and 165 calories...
```

### Pattern: T2 — Multi-Step Tool Chain

```
User: I just had two eggs and a bowl of oatmeal for breakfast. Log it and tell me total calories.

<think>
Multi-step task (T2): calculate nutrition totals first, then log the meal.
Step 1: get_food_nutrition for both foods at once.
Step 2: log_meal with the results.
</think>
<tool_call>
{"name": "get_food_nutrition", "arguments": {"foods": [{"food_name": "egg", "amount_grams": 120}, {"food_name": "oatmeal", "amount_grams": 250}]}}
</tool_call>

<tool_response>
{"status": "success", "data": {"total": {"calories_kcal": 390, "protein_g": 18, ...}}}
</tool_response>

<think>
Got totals: 390 kcal. Now log this as breakfast.
</think>
<tool_call>
{"name": "log_meal", "arguments": {"meal_type": "breakfast", "foods": [{"food_name": "egg", "amount_grams": 120}, {"food_name": "oatmeal", "amount_grams": 250}]}}
</tool_call>

<tool_response>
{"status": "success", "meal_id": "abc123", "total_calories": 390}
</tool_response>

Your breakfast has been logged. Total: ~390 calories (18g protein, ...)
```

### Pattern: T3 — Conditional Branching

```
User: Am I over my calorie budget today? If so, suggest a low-cal dinner.

<think>
Conditional task (T3): check today's intake first, then branch based on result.
Step 1: get_today_summary to see remaining budget.
</think>
<tool_call>
{"name": "get_today_summary", "arguments": {}}
</tool_call>

<tool_response>
{"status": "success", "data": {"total_calories": 1800, "calorie_budget": 2000, "remaining_calories": 200}}
</tool_response>

<think>
200 kcal remaining for dinner — very tight. User needs low-calorie dinner suggestions.
Retrieve knowledge about low-calorie dinner options.
</think>
<tool_call>
{"name": "retrieve_knowledge", "arguments": {"query": "low calorie dinner options under 200 calories"}}
</tool_call>

<tool_response>
{"status": "success", "top_relevance_score": 0.72, "data": {"passages": [{"source": "Dietary Guidelines", "section": "Low-calorie meal planning", ...}]}}
</tool_response>

You have 200 kcal left for the day. Here are some light dinner options that fit your budget...
```

### Pattern: T4 — Safety Boundary Declaration

```
User: I'm on dialysis. What should I eat to gain muscle while managing my potassium?

<think>
User is on dialysis — this is active renal replacement therapy. Protein, potassium,
and phosphorus management for a dialysis patient requires clinical-level nutrition
intervention that exceeds my safe service boundary. I should not attempt to advise
on specific intake targets. Declare T4 safety boundary.
</think>

Your situation involves complex medical nutrition management (dialysis with concurrent
fitness goals) that exceeds my safe service boundary. The potassium, phosphorus, and
protein requirements for dialysis patients are highly individualized and medically
regulated. Please consult your nephrologist or a renal dietitian for personalized guidance.
```

### Pattern: Pure QA — No Tool

```
User: What does Vitamin C do?

<think>
General nutrition knowledge question. Can answer directly from training knowledge.
No external data lookup needed.
</think>

Vitamin C is an essential water-soluble antioxidant that supports immune function,
collagen synthesis, and iron absorption...
```
