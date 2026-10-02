---
id: 004
title: TRAIN_ROSTER with provably exam-disjoint derive_profile_windows
status: CLOSED (2026-09-09) — TRAIN_ROSTER 20 people; 12 tests green (237 dir-wide)
commit: 0281856
depends_on: [003]
spec: ../spec.md
spec_sections: ["2", "US-16", "US-18", "OQ-9"]
---

# 004 — TRAIN_ROSTER with provably exam-disjoint windows

**What to build:** The NutriMind-side people the factory authors tasks for. A
`TRAIN_ROSTER` of fictional `train-*` users whose body facts are chosen so that
`derive_profile_windows` output is **provably disjoint** from every nutri-env `ROSTER`
person's windows — the single isolation point that stops a template-family oracle from
colliding with the exam. Persona split provisionally 65/20/15 everyday/gym/cut (OQ-9;
literature citation deferred, non-blocking — it affects the roster, not the pipeline).

**Blocked by:** 003.

**Status:** CLOSED (2026-09-09)

- [x] `TRAIN_ROSTER` is importable and every entry has a `train-` prefixed `user_id`
- [x] a test computes `derive_profile_windows` for every `TRAIN_ROSTER` person and every
      nutri-env `ROSTER` person and asserts the two window sets are disjoint (no shared
      window tuple) — this is the train/exam isolation guarantee
- [x] `profile_for` / persona lookups resolve for each `TRAIN_ROSTER` person via public
      nutri-env helpers
- [x] persona counts match the provisional 65/20/15 mix, documented as provisional (OQ-9)
- [x] `generate_one(family="log", person=<a TRAIN_ROSTER person>, expander=<synthetic>)`
      produces an accepted `Task` for at least one person per persona

## Closure notes

- Isolation is structural, not enumerated: half-kilo everyday weights off the
  exam integer grid (blocks protein-tuple equality incl. 2x/1/2x cross-regime),
  gym/cut integer weights off the exam set, EERs off all 23 exam EERs — proved
  exhaustively by `test_train_exam_window_isolation`.
- `sodium_mg` excluded from the contract with a live proof: constant
  `(0.0, 2300.0)` for all 43 people in both rosters.
- Allergy vocabulary finding: the catalog's `allergen_tags` is exactly the exam
  roster's 9 tags; sesame is unsupported (`no_allergen_food`, asserted live), so
  TRAIN reuses the supported vocabulary — isolation rides on the `train-*`
  prefix + window disjointness. Documented in the module docstring.
- TRAIN_ROSTER is a tuple (mirrors nutri-env's ROSTER shape).
