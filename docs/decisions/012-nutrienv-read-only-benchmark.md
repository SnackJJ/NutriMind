# ADR-012: NutriEnv Is a Read-Only Benchmark Dependency

- **Status**: accepted
- **Date**: 2026-09-08
- **Amended**: 2026-09-08 — borrowed-symbol guard scope narrowed to two classes; see the Amendment log
- **Deciders**: zeqing

## Context

The Batch-1 data factory needs task shapes that
`nutrienv.bench.pipeline.generate_one` does not author:

- 3-leg `update+log→recommend` composite — `_LEGAL_COMPOSITE_PAIRS` holds 2-tuples only.
- Batch-2 shapes: `amend_meal→recommend`, "refuse an unsafe profile edit + still
  recommend" (starve), closed-list `allowed_food_ids` — `_recommend_from_template` /
  `_log_then_recommend` never set these fields (`/tmp/nutrienv-data-review.md` P1-5).

The cheapest fix is a small patch to `../nutri-env` (one function +
`_LEGAL_COMPOSITE_PAIRS` entry).

## Decision

**NutriMind never patches NutriEnv.** `../nutri-env` is consumed as a read-only
benchmark and library: the frozen v1.0 split, `Scorer`, `NutriEnv`, the `ReActHarness`
loop shape, `_SYSTEM_V2`, `catalog.sqlite`, and the `generate_one` internals.

Every authoring gap is closed **NutriMind-side** in `src/training/data_factory/` by
composing **public** nutri-env symbols (in `nutrienv`'s `__all__`), then verifying each
result with `achievable.check_achievable`. Where a shape is only reachable through a
private helper, NutriMind reconstructs it from public symbols (the 3-leg composite is a
feasibility spike — `.scratch/nutrimind-v2/spec.md` §22.9 and ticket 002) or subclasses
the owning class; it does not import the underscore name as a long-term dependency.

Guardrails:

- `../nutri-env` is pinned to an exact git SHA in `pyproject.toml`.
- A compatibility test asserts, for the **public borrowed API only** (symbols in
  nutri-env's `__all__` that v2 calls directly): the symbol imports, and its
  `inspect.signature` is unchanged. Private helpers get an existence-only import check at
  most, and are covered indirectly by the end-to-end behaviour test. An upstream change
  to a public symbol breaks CI loudly.

## Consequences

### Positive
- The ruler cannot be bent to fit the student — no path by which a v1.0 number moves
  because NutriMind edited the harness.
- NutriEnv's CHARTER (Env + Bench, no training loop) stays intact; the training loop is
  entirely NutriMind's.

### Negative
- v2 still leans on nutri-env behaviour that is not all in `__all__`. The mitigation is
  the two-class guard (below) plus preferring public-symbol reconstruction; a genuinely
  unavoidable private dependency must be promoted to nutri-env's `__all__` upstream
  first, not imported by its underscore name.
- The 3-leg composite and all Batch-2 shapes are hand-assembled and separately
  `check_achievable`-verified rather than inheriting `generate_one`'s built-in guards.
- Whether the public API is sufficient for the 3-leg composite is unproven until ticket
  002's spike runs.

### Neutral
- If a shape proves genuinely reusable it can be contributed upstream later as a normal
  PR — but never as a prerequisite for a NutriMind batch.

## Amendment log

### 2026-09-08 — borrowed-symbol guard scope

The original Decision text read:

> Every authoring gap is closed **NutriMind-side** by composing borrowed helpers
> (`_update_from_template`, `_bind_log_foods`, `plan_windows_for_meal`,
> `compose_oracles`, …) … An import smoke test asserts every borrowed symbol still
> exists with the expected signature.

Problem: `_update_from_template` and `_bind_log_foods` are underscore-private and not in
nutri-env's `__all__`. A signature guard on them would fabricate a stability contract
nutri-env never offered, and the wording conflicts with the design constraint "do not
freeze internal helpers as a public API".

Narrowed to two classes:

- **Public borrowed API** — in nutri-env's `__all__`, a formal v2 dependency: import
  existence guard + `inspect.signature` guard + behaviour tests.
- **Private implementation detail** — underscore-prefixed, not in `__all__`: no
  signature-stability promise; not a v2 production dependency; covered indirectly by the
  end-to-end behaviour test; if a stable dependency becomes unavoidable, get it promoted
  to `__all__` upstream first.

The core decision — NutriMind never patches NutriEnv, authoring gaps are solved
NutriMind-side — is unchanged. The concrete public/private symbol split lives in
`.scratch/nutrimind-v2/spec.md` §18.

## Related

- [ADR-010](010-nutrimind-v2-rescope.md),
  [ADR-011](011-batch1-sft-trajectory-short-plan-thinking-teacher.md)
- `.scratch/nutrimind-v2/spec.md` (§18 public/private symbol split, §22.9 3-leg spike)
- `docs/plans/nutrienv_student.md` (§5 Pointers), `/tmp/nutrienv-data-review.md`
  (P0-1, P1-5)
