# Phase 2.6: SFT Trajectory Collection & Validation (v2)

## Overview
This document outlines the pipeline for taking our scaled query pool (generated in Phase 2.5) and running it through a Teacher Model to collect actual tool-use trajectories. These trajectories will then be cleaned, validated, and split to form our final SFT dataset.

The pipeline consists of three stages:
1. **Pre-processing**: Query pool split
2. **Collection**: Async trajectory generation via Teacher Model
3. **Post-processing**: Normalization → Validation → Stratified split

**Design decision**: User profiles (originally planned for T3) are deferred to a future iteration. The current SFT phase focuses on the core capability — agentic tool calling and multi-step reasoning. T3 queries will be treated the same as other tiers: the Teacher Model generates a reasonable general response with proper tool usage. Personalization can be layered on top once the base capabilities are solid.

---

## 1. Pre-processing: Query Pool Split
**Script**: `scripts/split_query_pool.py`

- **Input**: `data/queries/expanded_query_pool.jsonl` (~5800 queries covering T0–T4 and Error)
- **Output**:
  - `data/queries/sft_candidate_pool.jsonl` (~2500 queries)
  - `data/queries/grpo_prompt_pool.jsonl` (~3300 queries)
- **Method**: Stratified random split by tier to ensure both pools have proportional tier distribution. Shuffle within each tier, then split and merge.
- Only the SFT Candidate Pool is used in this phase. The GRPO pool is reserved for Phase 4.

---

## 2. Trajectory Collection

### 2.1 Simulation Loop
**Script**: `src/training/sft/collect_trajectories.py`

**Mechanism**:
1. Initialize Teacher Model (qwen-max via DashScope).
2. For each query in SFT Candidate Pool:
   - Send query to Teacher Model with the standard NutriMind system prompt (full English, `<think>` tags, JSON schema tool calling).
   - Intercept tool calls locally, execute against local SQLite USDA database, return results.
   - Loop until Teacher Model yields a final text answer.
   - Serialize entire conversation as a single trajectory.

**Required implementation**:
- **Input format**: Read from `.jsonl` (not `.json`).
- **Async concurrency**: Use `asyncio.Semaphore` with a conservative limit (e.g., `semaphore=5`) to respect DashScope rate limits.
- **Rate limit handling**: Implement exponential backoff retry (base=1s, max_retries=5) on 429/rate-limit responses.
- **Resumability**: Before processing each query, check if its ID already exists in `real_trajectories.jsonl`. Skip if present.

**Output**: `data/trajectories/real_trajectories.jsonl`

### 2.2 Query-Aware Mock Data for User-State Tools

**Problem**: Tools like `get_today_summary` and `get_history` require user state data. During trajectory collection, we don't have real user data, but returning fixed mock values would teach the model to ignore tool results.

**Solution**: Generate mock data that is semantically consistent with the query, using an adapter pattern that keeps the production tool code untouched.

#### Design Principles

1. **Isolation**: Mock logic lives only in `collect_trajectories.py`, not in production tool files
2. **No cleanup needed**: Production tools are never modified, so nothing to "revert" after collection
3. **Query-aware**: Use LLM to infer reasonable user state from query context

#### Implementation

```python
# In collect_trajectories.py

from src.tools.mock_user_state import generate_mock_state, mock_today_summary, mock_history

# Tools that need query-aware mocking
MOCK_TOOLS = {"get_today_summary", "get_history"}

def execute_tool_with_mock(tool_name: str, tool_args: dict, query: str) -> dict:
    """Execute tool, using query-aware mock for user-state tools."""
    if tool_name in MOCK_TOOLS:
        state = generate_mock_state(query)  # LLM generates consistent state
        if tool_name == "get_today_summary":
            return mock_today_summary(state)
        elif tool_name == "get_history":
            return mock_history(state, **tool_args)
    else:
        # Real execution for all other tools
        return execute_tool(tool_name, tool_args)
```

#### Mock State Generation

```python
# src/tools/mock_user_state.py (new file, only used during collection)

def generate_mock_state(query: str) -> dict:
    """Use LLM to infer user state consistent with query semantics."""
    # Example: "我今天吃得有点多了" → {"today_eaten": 1850, "remaining": 150, ...}
    # Example: "这周蛋白质摄入怎么样" → {"history_trend": "stable", "protein_avg": 85, ...}

    prompt = f'''Based on this query, generate a realistic user nutrition state.
Query: "{query}"
Return JSON with: today_eaten, calorie_budget, remaining, protein_g, carbs_g, fat_g,
                  history_7d_avg_calories, history_trend ("stable"/"increasing"/"decreasing")'''

    # Call lightweight LLM (qwen-plus) to generate state
    # Falls back to random values on error
    ...

def mock_today_summary(state: dict) -> dict:
    return {
        "status": "success",
        "data": {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_calories": state["today_eaten"],
            "calorie_budget": state["calorie_budget"],
            "remaining_calories": state["remaining"],
            "protein_g": state["protein_g"],
            "carbs_g": state["carbs_g"],
            "fat_g": state["fat_g"],
            "meals_logged": state.get("meals_logged", [])
        }
    }

def mock_history(state: dict, days: int = 7, metric: str = "all") -> dict:
    return {
        "status": "success",
        "data": {
            "period": f"Last {days} days",
            "daily_averages": {"calories_kcal": state["history_7d_avg_calories"]},
            "trend": state["history_trend"],
            "daily_breakdown": _generate_daily_breakdown(state, days)
        }
    }
```

#### Effect

| Query | Generated State | Tool Returns |
|-------|-----------------|--------------|
| "我今天吃得有点多了，还剩多少卡路里" | `{today_eaten: 1850, budget: 2000}` | `{remaining: 150}` ✓ consistent |
| "这周我的蛋白质摄入够吗" | `{protein_avg: 75, trend: "stable"}` | `{avg: 75g, trend: stable}` ✓ consistent |
| "帮我看看昨天吃了什么" | `{meals_logged: [...]}` | Realistic meal history |

#### Cost

- ~1 extra LLM call per trajectory (qwen-plus, cheap)
- Only for queries that trigger `get_today_summary` or `get_history` (~700 queries, 12% of pool)

---

## 3. Post-processing

### 3.1 Normalization
**Script**: `src/training/sft/normalize.py`

Standardize message array format to conform to Qwen/OpenAI chat format:
- Remove markdown fencing from tool calls or think blocks if improperly formatted.
- Ensure all role names strictly map to `system`, `user`, `assistant`, or `tool`.
- **Reject gate**: Trajectories with unfixable structural damage (severely malformed JSON, missing required fields) are logged to `normalize_rejects.jsonl` and excluded from downstream processing.

### 3.2 Rule-based Validation
**Script**: `src/training/sft/validate_rules.py`

- Discard trajectories that call non-existent tools.
- Discard trajectories where the language is not 100% English.
- Discard trajectories that exceed the max turn limit (stuck in infinite tool loops).
- **T4 check**: Verify T4 trajectories completely bypass tool calls and directly output the safety boundary disclaimer.
- **Non-T4 tool usage check**: Flag and discard non-T4/non-T0 trajectories where the model gave a direct answer without any tool call. This prevents SFT from learning "skip tools for nutrition questions".

### 3.3 Semantic Validation
**Script**: `src/training/sft/validate_semantic.py`

LLM-as-a-Judge evaluation (use a different model from the Teacher to avoid self-judge bias):
- Evaluate whether the final response actually answers the user's initial query.
- Discard trajectories where the Teacher Model invoked a tool but ignored its result.

### 3.4 Stratified Split & Export
**Script**: `scripts/split_dataset.py`

- **Input**: Normalized, fully validated golden trajectories (~1500–2000 remaining).
- **Process**:
  - Group trajectories by tier.
  - Within each tier, shuffle and split 90/10.
  - Merge into final train/val sets.
- **Output**:
  - `data/trajectories/train_sft.jsonl`
  - `data/trajectories/val_sft.jsonl`
  - `data/trajectories/split_report.json` — class distribution (T0–T4, Error counts), average trajectory length (turns and tokens).

---

## SFT Training Data Format Reference

```json
{
  "messages": [
    {"role": "system", "content": "You are NutriMind, a nutrition assistant..."},
    {"role": "user", "content": "How much protein is in 100g chicken breast?"},
    {"role": "assistant", "content": "<think>User wants nutritional info...</think>\n{\"name\": \"get_food_nutrition\", ...}"},
    {"role": "tool", "content": "{\"protein\": 31.0, ...}"},
    {"role": "assistant", "content": "<think>Got the data...</think>\n100g of chicken breast contains approximately 31g of protein..."}
  ]
}
```

All tiers share the same system prompt and message format. The tier label is metadata only and does not appear in training data.

---

## Execution Checklist

### Pre-processing
- [ ] Run `split_query_pool.py` to produce SFT Candidate Pool and GRPO Prompt Pool.
- [ ] Verify stratified tier distribution in both pools.

### Collection
- [ ] Create `src/tools/mock_user_state.py` with query-aware state generation.
- [ ] Update `collect_trajectories.py`:
  - `.jsonl` input format
  - Async concurrency with semaphore
  - Exponential backoff retry
  - Resumability (skip existing IDs)
  - Query-aware mock adapter for `get_today_summary` / `get_history`
- [ ] Run trajectory collection on SFT Candidate Pool.

### Post-processing
- [ ] Run `normalize.py` (with reject logging).
- [ ] Run `validate_rules.py` (including non-T4 tool usage check).
- [ ] Run `validate_semantic.py` (using a different judge model).
- [ ] Run `split_dataset.py` for stratified 90/10 split.
- [ ] Review `split_report.json` to confirm class balance.