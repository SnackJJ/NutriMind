---
id: 017
title: mini-exam val freeze — build --freeze-mini → sft/val_mini.json (30 fresh TRAIN_ROSTER tasks)
status: ready-for-agent
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

**Status:** ready-for-agent

- [ ] `build --freeze-mini` writes `sft/val_mini.json` with exactly 30 tasks and issues
      zero teacher calls
- [ ] every task is `check_achievable`-reachable and passes the exam dedup gates
      (`gates.run`)
- [ ] re-running `--freeze-mini` with the same config reproduces a byte-identical
      `sft/val_mini.json`
- [ ] the 30 `task_id`s do not overlap the reserved Batch-1 seed range
- [ ] each task reconstructs to a runnable `NutriEnv` (same env round-trip as the
      TaskPackage)
