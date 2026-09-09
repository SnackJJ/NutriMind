---
id: 005
title: gates.run pure validation seam (Seam 2) incl. exam-collision dedup
status: CLOSED (2026-09-09) — gates.run pure seam; 14 tests green (61 stable subset)
commit: 8da38ed
depends_on: [003]
spec: ../spec.md
spec_sections: ["11", "19.2", "19.6", "22.9", "9.4"]
---

# 005 — gates.run pure validation seam (Seam 2)

**What to build:** The pure validation boundary every authored `Task` passes through
before it becomes a TaskPackage. The exam corpus is passed in **explicitly** — no
implicit module-level load, no global cache:

```
gate_ctx = GateContext.from_exam(exam_tasks)   # loads the 63 exam Tasks + precomputes
                                               # normalized queries + semantic_keys, ONCE
result   = gates.run(task, ctx=gate_ctx) -> GateResult(keep, failure_code,
                                                       reason_detail, stage)
```

`build` builds the `GateContext` once at startup and threads it through; tests build one
from a hand-picked task list. `gates.run(task, ctx)` is pure — same `(task, ctx)` →
same `GateResult` across runs. Ordered, first failure wins, per spec §11:

1. `gate.verbatim_query_collision` — candidate query casefolded + whitespace-collapsed +
   trailing-punctuation-stripped equals any of the 63 exam queries normalized the same
   way. Near-duplicates that are not verbatim are deliberately kept.
2. `gate.semantic_key_collision` — `nutrienv.bench.validator.semantic_key(task)` equals
   any exam task's `semantic_key`.
3. `gate.slot_value_overlaps_exam` — `update` / composite-with-update only.
4. `gate.stage_a` — `stage_a_code_gate(task)` non-empty.
5. `gate.draft_invalid` — `validate_draft(task)` non-empty, **with the 3-leg allow-list**
   `["update oracle ledger is missing"]` (spec §22.9 — the exam item `adr24-comp-8255`
   trips the same false-positive).
6. `gate.unachievable` — `task.id` in `check_achievable([task]).unreachable` →
   **status `indeterminate`**, not a plain drop (authoring bug, not a model failure).

Gate drops produce the `rejects/gate.jsonl` record shape (spec §9.4). Pure — no network;
tests drive `Task`s from `generate_one(expander=synthetic)` offline against a
test-constructed `GateContext`.

**Blocked by:** 003.

**Status:** CLOSED (2026-09-09)

- [x] `gates.run(task, ctx)` is pure (no I/O, no module-level state); the same
      `(task, ctx)` yields the same `GateResult` across runs
- [x] `GateContext.from_exam` reads the 63 exam `Task`s and precomputes the normalized
      queries + `semantic_key`s exactly once; `gates.run` never touches the exam file
- [x] each of the 6 checks has a test that trips exactly it with the right `failure_code` / `stage`
- [x] a near-duplicate (not verbatim) of an exam query is **kept**
- [x] an authored `Task` whose `semantic_key` equals an exam task's is **dropped**
      (`gate.semantic_key_collision`)
- [x] `gate.unachievable` yields `status=indeterminate`, routed as indeterminate not dropped
- [x] the 3-leg composite shape passes when `validate_draft` returns only
      `["update oracle ledger is missing"]`
- [x] gate ordering asserted: a `Task` failing checks 1 and 4 reports check 1

## Closure notes

- Exam-corpus facts verified against the pinned rev: 63 normalized queries but
  **62** semantic keys — the frozen exam itself contains one intra-duplicate
  pair (`adr20-rec-5019` / `adr29-conv-03`); asserted in tests.
- Gate-6 construction that works standalone: a 3-leg composite whose update
  leg carries stale kcal windows *declared in the query* ("lower my calorie
  target by 555") — passes gates 1–5 (only the allow-listed FP remains) and is
  unreachable. Fully-stale standalone `upd-weight` windows collide with the
  exam's idempotent-update `semantic_key` (empty shift) → gate 2 preempts;
  the kcal-declared composite avoids that.
- `normalize_query` strips trailing whitespace+punctuation as one run (so
  "…allergies. !!!" still collides with "…allergies."), per the spec's
  cosmetic-tail intent.
- Shared offline fixtures extracted to `tests/training/data_factory/_fixtures.py`
  (synthetic expander, seeded `generate_one` factories, public-symbol 3-leg
  assembly) for reuse by 006/007/010 tests.
