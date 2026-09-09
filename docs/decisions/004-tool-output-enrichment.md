# ADR-004: Tool Output Enrichment (Food Summary Injection)

- **Status**: accepted
- **Date**: 2026-03-21
- **Deciders**: @jzq

## Context

During Batch 4 trajectory collection for SFT training, we identified a critical data quality issue: **Teacher Model Hallucinations** in historical food queries.

**The Problem**:
- `get_history` and `get_today_summary` tools were originally designed to return only numerical aggregates (calories, protein, fats, etc.) and meal counts.
- Many user queries in the multi-step pool ask specific details about what was eaten (e.g., "What was in my lunch salad last Wednesday?").
- The Teacher model (Qwen 397B), seeing only calorie totals, would "fill in the blanks" by hallucinating plausible food names to satisfy the user's request.
- This resulted in "cheated" single-step trajectories where the model answered complex questions without actually retrieving the list of foods.

## Decision

Enrich the output of user-state tools with a `food_summary` field (a comma-separated string of food names consumed during the period/day).

### 1. Mock Tool Enrichment (`src/tools/mock_user_state.py`)
Update the Mock logic to generate query-aware `food_names` and `food_summary`. This ensures that during synthetic collection, the teacher has factual food names to ground its responses.

### 2. Real Tool Enrichment (`src/tools/get_history.py`)
Update the `daily_summary` SQL View and the Python tool mapping to include `food_summary`. Use SQLite's `GROUP_CONCAT` on the `meal_log_items` table to aggregate food names for each date.

## Consequences

### Positive
- **Eliminates Hallucinations**: Model now has access to the actual (or mock-actual) food names.
- **Enables Complex Multi-Step Reasoning**: With food names visible, we can now generate trajectories that involve analyzing specific intake patterns (e.g., "I noticed you ate steak 3 times last week...") followed by recommendations.
- **Maintains Simple Interface**: We avoid creating a new `get_meal_details` tool, keeping the toolset concise.

### Negative
- **Context Size Increase**: Tool output strings will be slightly larger (minor impact).
- **Tool Complexity**: The SQL VIEW `daily_summary` is now slightly more complex to maintain.

## Validation

**Test Query (via Mock)**: "What did I have for dinner last Wednesday?"
- **Old Result**: `{"calories_kcal": 1415, ...}` (Lead to hallucination).
- **New Result**: `{"calories_kcal": 1415, "food_summary": "Turkey sandwich, Greek yogurt", ...}` (Leads to grounded response).

## Related

- [ADR-003: Teacher Demonstration Prompting](003-teacher-demonstration-prompting.md)
- [Phase 2.6 Plan](../plans/phase2.6_trajectory_collection.md)
