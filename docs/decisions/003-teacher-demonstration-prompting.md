# ADR-003: Teacher Demonstration Prompting

- **Status**: accepted
- **Date**: 2026-03-21
- **Deciders**: @jzq

## Context

During batch3 trajectory collection with `qwen3.5-397b-a17b`, we observed that **35% of trajectories were classified as T0-qa** (no tool calls), despite the query pool containing only ~15% T0 queries. The model is "too smart" — it answers nutrition questions from memory instead of demonstrating tool usage.

**Root cause**: The model optimizes for answering correctly, not for demonstrating tool-calling workflow. When it "knows" the answer, it skips tools.

**Attempted solutions that failed**:
- `tool_choice="required"`: DashScope API does not support this parameter in thinking mode
- Tier injection (telling model the expected tier): Leaks human classification, introduces bias

## Decision

Add a **meta-prompting section** to the system prompt that frames the model's role as a "teacher demonstrating tool usage" rather than just "answering questions".

New section added to `SYSTEM_PROMPT`:

```markdown
## ROLE: TEACHING DEMONSTRATION

You are generating training data for a student model. Your task is to DEMONSTRATE the correct tool-calling workflow, not just answer questions.

Key principle: When accurate data can be obtained via tools, ALWAYS use the tool rather than answering from memory. The student model will learn from your behavior — if you skip tools, it will learn to skip tools.

Demonstrate:
1. Reasoning about which tool to use (in <think>)
2. Calling the tool with correct parameters
3. Analyzing the tool result (in <think>)
4. Synthesizing a final answer based on real data
```

**Implementation**: Create `collect_trajectories_v2.py` with the modified prompt, keeping the original script unchanged for comparison.

## Consequences

### Positive
- No tier leakage — model doesn't know the expected classification
- Natural incentive to use tools — "teaching" implies demonstrating, not shortcutting
- T0 queries should still work correctly — the original prompt already says "general/conversational queries can be answered directly"

### Negative
- May cause over-tooling on edge cases (model calls tools when truly unnecessary)
- Need to validate T0-qa rate before/after to confirm improvement

### Neutral
- Requires A/B comparison between v1 and v2 collection scripts

## Validation Results (2026-03-21)

**Test**: 100-query stratified sample from GRPO pool

| Metric | V1 (original) | V2 (teaching) | Δ |
|--------|---------------|---------------|---|
| T0-qa | 31 (31%) | 22 (22%) | **-9%** |
| T1 | 25 | 26 | +1 |
| T2 | 31 | 32 | +1 |
| T3 | 3 | 11 | **+8** |
| T4 | 4 | 5 | +1 |

- ✅ 11 queries improved (T0-qa → tool-use)
- ⚠️ 2 queries regressed (T2 → T0-qa)
- T3 improvement notable: model more willing to do multi-step reasoning

**Conclusion**: Target met (22% < 25% threshold). V2 prompt approved for production collection.

## Related

- [ADR-001: Pure Text Tool Calling](001-pure-text-tool-calling.md) — format decision
- [Phase 2.6 Plan](../plans/phase2.6_trajectory_collection.md) — collection pipeline
