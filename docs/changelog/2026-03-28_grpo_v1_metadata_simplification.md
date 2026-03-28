# GRPO Phase 4 Redesign: Route A + Metadata Simplification

**Date**: 2026-03-28
**Module**: `src/training/grpo/` + `docs/plans/phase4_grpo.md`
**Action**: Adopt Route A (controlled comparison), simplify metadata, fix internal contradictions

## Summary

1. **Adopted Route A**: All experiments (C/D/F) start from SFT with ref=SFT for clean controlled comparison
2. **Simplified v1 metadata**: Only `query`, `tier`, `difficulty` needed
3. **v2 uses intrinsic signals**: No `expected_tools` or `branch_condition` needed
4. **Fixed T0/T4 separation**: T4 only checks safety declaration, doesn't penalize tool calls

## Motivation

### Problem 1: Metadata noise
Original design required auto-generating `expected_tools`, `ground_truth`, and `branch_condition` via keyword heuristics. This introduced significant noise:
- "If I eat this chicken" has "if" but is T1, not T3
- Correct tool chains for conditional queries depend on runtime results
- Pre-annotated ground_truth may not match actual tool outputs

### Problem 2: Internal contradiction in phase4_grpo.md
The experiment matrix said D and F both start from SFT, but Task 4.3 described v1→v2→v3 iterative chains. These couldn't both be true.

**Solution: Adopt Route A** - all experiments start from SFT, enabling clean controlled comparison of GRPO vs GiGPO.

## Changes

### reward.py

1. **T0/T4 separation in `compute_tool_selection_score`**:
   - T0 (pure QA): Should NOT call tools, penalize any tool calls
   - T4 (safety): Only check final answer has safety declaration, do NOT penalize tool calls
   - T1-T3: Should call at least one valid tool

2. **T4 design rationale**:
   - Prevents "see sensitive word → refuse immediately" shortcut (over-refusal)
   - Model can explore both "check then refuse" and "refuse directly" paths
   - GRPO group comparison naturally selects optimal path

3. **`compute_outcome_score_rule_based` updated**:
   - T1: Compare answer against actual tool results (runtime ground truth)
   - Removed duplicate T4 logic (handled in tool_selection)

### prepare_prompts.py

1. **v1 minimal metadata**:
   ```json
   {"query": "...", "tier": "T1", "difficulty": "medium"}
   ```
   No expected_tools, ground_truth, or branch_condition for v1.

2. **Improved tier classification**:
   - T4: High-precision medical keywords only (not "if")
   - T2: Regex patterns for log+query combos
   - T1: Single tool patterns
   - Conservative: ambiguous "if" queries classified as T1, not T3

3. **v2 functions preserved** for future use with human-verified metadata

### phase4_grpo.md

- Task 4.2.2: Documented v1 vs v2 metadata strategy
- Reward v1 section: Explained T4 design rationale

## Files Modified

- [reward.py](../../src/training/grpo/reward.py) - T0/T4 separation, runtime ground truth
- [prepare_prompts.py](../../src/training/grpo/prepare_prompts.py) - v1/v2 split, improved tier rules
- [phase4_grpo.md](../plans/phase4_grpo.md) - Route A adoption, intrinsic signals for v2, fixed contradictions

## Key Design Decisions

### Route A (adopted)
```
All experiments: Base=SFT, Ref=SFT
C: GRPO + v1 reward (validate pipeline)
D: GRPO + v2 reward (full GRPO)
F: GiGPO + v2 reward (step-level advantage)

Core comparison: D vs F (same everything except advantage method)
```

### v2 Intrinsic Signals (no metadata needed)
```python
# Efficiency: detect repeated tool calls
r_efficiency = 1.0 - 0.2 * (len(tool_calls) - len(set(tool_calls)))

# Conditional: check if <think> references tool results
r_conditional = count_value_references(think_content, tool_results)
```

## Testing Notes

- Tier classification accuracy should be spot-checked: require >90% before training
- If "if" keyword causes T3 over-classification, review T3 samples manually
- D vs F comparison should use identical prompt pools for cleanest analysis
