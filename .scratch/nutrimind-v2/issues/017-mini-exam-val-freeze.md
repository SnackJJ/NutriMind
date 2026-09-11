---
id: 017
title: mini-exam val freeze — build --freeze-mini → sft/val_mini.json (30 fresh TRAIN_ROSTER tasks)
status: CLOSED (2026-09-11)
commit: 36777a3
depends_on: [010, 004, 005, 006]
spec: ../spec.md
spec_sections: ["2", "6.5f", "9"]
---

# 017 — mini-exam val freeze

**What to build:** `build --freeze-mini` — a `target=eval` path that authors **30 fresh
TRAIN_ROSTER tasks**, runs `check_achievable` only (no teacher, no serialize, no
Pass-filter), and emits the frozen set to `sft/val_mini.json`. Deterministic enumeration
(fixed seeds), oracle-verified, disjoint from the exam (passes the spec §11 dedup gates)
and — by its reserved seed range — from Batch-1 accepted `task_id`s. This is the set the
future v2 trainer selects checkpoints against; it is never the v1.0 exam.

**Blocked by:** 010, 004, 005, 006.

**Status:** CLOSED (2026-09-11)

- [x] `build --freeze-mini` writes `sft/val_mini.json` with exactly 30 tasks and issues
      zero teacher calls
- [x] every task is `check_achievable`-reachable and passes the exam dedup gates
      (`gates.run`)
- [x] re-running `--freeze-mini` with the same config reproduces a byte-identical
      `sft/val_mini.json`
- [x] the 30 `task_id`s do not overlap the reserved Batch-1 seed range
- [x] each task reconstructs to a runnable `NutriEnv` (same env round-trip as the
      TaskPackage)

## Closure notes

- Seeds start at `900000` (`MINI_EXAM_SEED_BASE`), disjoint from Batch-1's
  `0..max_intents`. Currently log-only (author strategies for other families are
  ticket 012).
