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

## Tool Definitions

### Tool 1: `get_food_nutrition`

| Attribute | Value |
|-----------|-------|
| **Description** | Look up nutrition information for one or more foods from USDA database. Returns detailed nutrients per food plus totals. |
| **When to use** | ANY food nutrition query — single food or multiple foods / full meal |
| **When NOT to use** | Dietary guidelines/advice (use `retrieve_knowledge`) |
| **Latency / Cost** | < 100ms / Free |

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
      "breakdown": ["per-item nutritional objects with full details (calories, protein, fat, carbs, fiber, sugars, sodium, cholesterol, vitamins, etc.)"],
      "macro_ratio": { "protein_pct": "number", "fat_pct": "number", "carbs_pct": "number" },
      "failed_items": ["optional — list of foods not found, with error details"]
    }
  },
  "error_cases": ["empty_food_list", "all_foods_not_found"]
}
```

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
    "meals_logged": ["array of meal summaries"]
  },
  "error_cases": ["no_profile_found"]
}
```

---

### Tool 4: `get_history`

| Attribute | Value |
|-----------|-------|
| **Description** | Query multi-day nutritional history and trends |
| **When to use** | User asks about past N days' eating patterns, weekly averages, trends |
| **When NOT to use** | Today only (use `get_today_summary`) |
| **Latency / Cost** | < 100ms / Free |

```json
{
  "name": "get_history",
  "parameters": {
    "days": { "type": "integer", "default": 7, "max": 90 },
    "metric": { "type": "string", "enum": ["calories", "protein", "fat", "carbs", "all"], "default": "all" }
  },
  "returns": {
    "period": "string",
    "daily_averages": { "calories_kcal": "number", "protein_g": "number" },
    "trend": "string",
    "daily_breakdown": ["array of daily summaries"]
  },
  "error_cases": ["invalid_date_range", "no_data_in_range"]
}
```

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
    "top_k": { "type": "integer", "default": 3, "max": 5 },
    "domain": {
      "type": "string",
      "enum": [
        "micronutrients", "dietary_guidelines", "sports_nutrition",
        "medical_nutrition", "life_stage", "meal_planning",
        "food_safety", "supplements", "weight_management"
      ],
      "required": false,
      "description": "Filter knowledge base by domain for more precise retrieval"
    }
  },
  "returns": {
    "status": "success",
    "data": {
      "passages": [{
        "content": "string",
        "source": "string — document title",
        "source_id": "string — source identifier",
        "section": "string — section heading",
        "url": "string — source URL",
        "relevance_score": "float — rerank score if available, else RRF score",
        "low_confidence": "boolean — optional, true when best result is below threshold"
      }]
    }
  },
  "error_cases": ["empty_query", "no_relevant_results", "internal_error"]
}
```

---

### Tool 6: `get_goal_adherence`

| Attribute | Value |
|-----------|-------|
| **Description** | Analyze adherence to nutrition goals over a specified period |
| **When to use** | User asks about goal progress, adherence rate, or whether they've been hitting targets |
| **When NOT to use** | Raw history data (use `get_history`); today only (use `get_today_summary`) |
| **Latency / Cost** | < 100ms / Free |

```json
{
  "name": "get_goal_adherence",
  "parameters": {
    "days": { "type": "integer", "default": 7, "max": 90 },
    "metric": { "type": "string", "enum": ["calories", "protein", "fat", "carbs", "all"], "default": "all" }
  },
  "returns": {
    "status": "success",
    "data": {
      "period": "string",
      "metric": "string — only present when metric != 'all'",
      "target_value": "number — only present when metric != 'all'",
      "daily_average": "number — only present when metric != 'all'",
      "days_within_target": "number — only present when metric != 'all'",
      "days_over_target": "number — only present when metric != 'all'",
      "days_under_target": "number — only present when metric != 'all'",
      "adherence_pct": "number — only present when metric != 'all'",
      "avg_deviation": "number — only present when metric != 'all'",
      "trend": "string — only present when metric != 'all'",
      "metrics": {
        "<metric_name>": {
          "target_value": "number",
          "daily_average": "number",
          "days_within_target": "number",
          "days_over_target": "number",
          "days_under_target": "number",
          "adherence_pct": "number",
          "avg_deviation": "number",
          "trend": "string"
        }
      }
    }
  },
  "error_cases": ["invalid_metric", "invalid_date_range", "no_goal_set", "no_data_in_range"]
}
```

**Key distinction from `get_history`**:
- `get_history` returns raw daily breakdown (what you ate)
- `get_goal_adherence` returns computed analysis against targets (how you did vs your goal)

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
│   • Goal progress / adherence rate?      → get_goal_adherence       │
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
│                                                                     │
│   Plan the minimal tool sequence, execute step-by-step.             │
│   If the next step depends on a result → proceed to Stage 3         │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 3: Do I need conditional branching?                          │
│                                                                     │
│   Execute first tool → inspect result → decide next action (T3):    │
│   • get_today_summary → over budget? → retrieve_knowledge           │
│   • get_goal_adherence → low adherence? → retrieve_knowledge        │
│   • get_history → declining trend? → retrieve_knowledge             │
│   • get_food_nutrition → not found? → ask user to clarify           │
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
| **T1** | Single tool call | `get_food_nutrition` → answer; `get_goal_adherence` → answer |
| **T2** | Multi-step tool chain (2-3 tools) | `get_food_nutrition` → `log_meal` → `get_today_summary` → answer |
| **T3** | Multi-step with conditional branching | `get_goal_adherence` → *branch*: if low adherence → `retrieve_knowledge` → answer |
| **T4** | Safety boundary declaration (no tool) | `<think>` → direct disclaimer → recommend professional |

---

## Error Handling Protocol

| Error | Strategy |
|-------|----------|
| `food_not_found` | Retry with simplified name OR ask user to clarify |
| `ambiguous_food_name` | Present candidates, ask user to choose |
| `no_relevant_results` | Answer from internal knowledge + note limitation |

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
{"name": "retrieve_knowledge", "arguments": {"query": "low calorie dinner options under 200 calories", "domain": "dietary"}}
</tool_call>

<tool_response>
{"status": "success", "data": {"passages": [...]}}
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
