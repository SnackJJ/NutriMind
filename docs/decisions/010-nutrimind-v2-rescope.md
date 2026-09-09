# ADR-010: NutriMind v2.0 — Rescope to Qwen3.5-2B on Frozen NutriEnv v1.0

- **Status**: accepted
- **Date**: 2026-09-08
- **Deciders**: zeqing

## Context

Phase-1 NutriMind (Qwen3-4B, 6 T1–T4 tools, RAG, mock `src/training/sft/evaluate.py`,
GRPO shortcut collapse in ADR-009) has almost no defensible agent Pass numbers:
RAG 23/23 is a sanity test, `verl_val` is a train val, `evaluate.py` mocks tools and
can sample the GRPO pool. The next stage changes the base model, the action space, the
evaluation, and the data pipeline at once. Continuing to call all of it "NutriMind"
conflates two projects and lets phase-1's weak metrics leak into the story.

## Decision

The project is versioned.

**NutriMind v2.0** = a small open-weight student (`Qwen/Qwen3.5-2B` Instruct) trained and
reported on the frozen **NutriEnv v1.0** exam (63 tasks, `catalog_sha256`-pinned,
`Scorer` end-state Pass) via a NutriMind-owned data factory. Locked spec:
`docs/plans/nutrienv_student.md`. Finalized data-factory design:
`docs/plans/nutrimind_v2_data_factory.md` (from `/tmp/nutrienv_student_data.md`).

**NutriMind v1** = the archived phase-1 system. Kept as engineering history, not mixed
into v2: no shared LoRA, no shared ops, no shared eval script. It survives as at most one
interview sentence (6-tool orchestrator, ~1.5k teacher traces, ADR-009 collapse).

Data rollout waves are named **Batch 1 / Batch 2**, not "第一档/第二档" — the exam's
`Task.tier` and `EVALUATE_TIERS` are unrelated authoring fields.

## Consequences

### Positive
- Honest resume framing: "multi-turn tool-calling post-training on a frozen interactive
  exam", not "nutrition chatbot".
- v1.0 is a real ruler with an existing 4-model flash/pro leaderboard in
  `../nutri-env/reports/`.
- The mock-eval leakage in `evaluate.py` is retired from the headline.

### Negative
- v1's RAG (1635 chunks, 23/23), the 6-tool orchestrator, and the ~1.5k v1 teacher
  traces drop out of the headline.
- Two version lines to keep straight in docs and on the CV.

### Neutral
- NutriEnv lives in the sibling repo `../nutri-env` and is consumed read-only
  (see ADR-012).
- GiGPO, Pro-as-target, and mixing RAG tools into the exam student stay out of scope
  for v2's first phase.

## Related

- `docs/plans/nutrienv_student.md`, `docs/handoff/2026-09_nutrienv-student.md`
- [ADR-009](009-grpo-reward-redesign-against-shortest-path-collapse.md),
  [ADR-011](011-batch1-sft-trajectory-short-plan-thinking-teacher.md),
  [ADR-012](012-nutrienv-read-only-benchmark.md)
