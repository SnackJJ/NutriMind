---
id: 029
title: Re-point the data factory to the current mill API (legacy_generate_one)
status: ready-for-agent
depends_on: []
spec: ../spec.md
spec_sections: ["2.1", "18"]
adr: [../../docs/decisions/012-nutrienv-read-only-benchmark.md, ../../docs/decisions/016-nutrienv-v1.1.1-pin-caliber.md]
---

# 029 — Re-point the data factory to the current mill API

**What to build:** `src/training/data_factory/*` stops importing the mill API that
the engine moved, and the ticket-016 signature guard is re-baselined against the
pinned tree. No ruler change: the scorer, the exam split, the catalog and the
prompt fingerprint stay exactly as pinned.

**Why now:** the factory cannot author anything. Upstream moved the old mill into
`nutrienv.bench.pipeline.legacy_generate_one` and renamed `freeze_tasks`, and
NutriMind kept the old import paths. Measured 2026-10-10 on the pinned tree
`v1.1.1` (`f4d70b2`) **and** on the earlier lab HEAD `e8508d8`, with identical
results — so this break predates the pin move and is not caused by it:

- `import src.training.data_factory.author` →
  `ImportError: cannot import name 'GRAM_UNITS' from 'nutrienv.bench.pipeline.generate_one'`
- 26 modules under `tests/training/data_factory/` fail at collection for the same
  reason.
- The files that do collect are red: `test_borrowed_api_signatures.py`
  (7 of 54 pinned symbols no longer import from the pinned module; 6 signatures
  moved), its two guard siblings, and
  `test_nutrienv_smoke.py::test_generate_one_update_needs_no_expander`
  (`TypeError: generate_one() got an unexpected keyword argument 'family'`).

**Blocked by:** None.

**Status:** ready-for-agent

## Symbol map (pinned tree `v1.1.1`)

| borrowed symbol | was | is now |
| --- | --- | --- |
| `generate_one` | `bench.pipeline.generate_one` | `bench.pipeline.legacy_generate_one` (the new `generate_one` takes `author`/`reviewer` callables) |
| `make_log_expander`, `make_unfit_rewriter`, `parse_query_foods_payload`, `search_fit_plate`, `AMOUNT_PATHS` | `bench.pipeline.generate_one` | `bench.pipeline.legacy_generate_one` |
| `_WORD`, `_speech_amount_path` | `bench.pipeline.generate_one` | `bench.pipeline.legacy_generate_one` |
| `GRAM_UNITS`, `OUNCE_UNITS`, `UNIT_SYNONYMS` | `bench.pipeline.generate_one` | `world.portions` (also re-exported by `legacy_generate_one`) |
| `KNIVES` | `bench.pipeline.generate_one` | `bench.pipeline.knives` |
| `freeze_tasks` | `bench.pipeline.freezer` | `bench.pipeline.freezer.freeze_legacy_tasks` |

Signature drift beyond the moves (`bench.Oracle`, `generate_one.generate_one`,
`realize.bind_evaluate_reasons`, `validator.fitting_plan`, `types.ledger_totals`,
`types.WorldState`, and `parse_query_foods_payload`): additive or cosmetic in
every case checked by hand, but the guard exists to make each one deliberate, so
re-capture rather than re-type them.

## Acceptance

- [ ] `import src.training.data_factory.author` and `.speech` succeed on the pin
- [ ] one task authors end-to-end (`generate_one` for a `log` family and for an
      `update` family, the second without an expander) and survives
      `check_achievable`
- [ ] `tests/training/data_factory/` collects every module again; no `skip`/`xfail`
      was added anywhere to get there
- [ ] `PUBLIC_BORROWED_API` in `test_imports.py` and `EXPECTED_SIGNATURES` in
      `test_borrowed_api_signatures.py` point at the new module paths, with the
      baseline re-captured by the snippet in that file's docstring (never typed by
      hand), and `NUTRIENV_BASELINE_REV` bumped to the pinned rev
- [ ] `scripts/setup_nutrienv.sh`, `configs/data_factory*.yaml`, the two pin tests
      and `docs/decisions/012` are untouched by this ticket — the pin is not moving
- [ ] the ruler is provably untouched: `git diff <pin> -- src/nutrienv/bench
      src/nutrienv/world data/splits data/fdc` is empty

## Notes

- Do not "fix" this by re-adding a `generate_one` shim to the pinned tree. The
  pinned tree is read-only (ADR-012), and a shim would hide the next upstream
  move instead of failing on it.
- `data/rl/grpo_v2/*.parquet` and `data/student/*` were authored under the old
  pin and stay as they are; re-authoring is a separate decision, not this ticket.
