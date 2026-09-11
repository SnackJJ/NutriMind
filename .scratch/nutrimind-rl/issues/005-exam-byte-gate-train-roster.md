---
id: 005
title: Exam byte-identical gate; train/val only TRAIN_ROSTER
status: CLOSED (2026-09-11)
commit: 2081ab5
depends_on: ["nutrimind-v2/023"]
spec: ../spec.md
spec_sections: ["D8"]
adr: [../../docs/decisions/012-nutrienv-read-only-benchmark.md]
---

# 005 — Frozen exam gate + roster isolation

**What to build:** Before any eval rollout, the v1.0 exam file is verified
byte-identical to the pinned published split; mismatch aborts. Training,
difficulty measurement, and mini-exam val read only `TRAIN_ROSTER` tasks. The 63
are never in those loops.

**Blocked by:** Data Factory 023.

**Status:** CLOSED (2026-09-11)

- [x] a mutated exam file fails before any rollout
- [x] an unmodified pin proceeds
- [x] a test asserts train/difficulty/mini-exam task ids are disjoint from
      `load_exam()` ids

## Closure notes

- `src/training/rl/exam_gate.py`: `assert_exam_byte_identical` /
  `before_eval_rollout` / `assert_disjoint_from_exam`.
- Package import does not load nutrienv.
