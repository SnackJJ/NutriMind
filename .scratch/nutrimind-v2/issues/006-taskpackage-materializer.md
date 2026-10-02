---
id: 006
title: Canonical TaskPackage materializer (§9.1) + public env round-trip
status: CLOSED (2026-09-09) — 16 tests green (253 dir-wide)
commit: 25cfd46
depends_on: [003]
spec: ../spec.md
spec_sections: ["9.1", "10", "19.1", "22.2"]
---

# 006 — Canonical TaskPackage materializer + public env round-trip

**What to build:** The canonical artifact that the SFT, RLVR, and eval materializers all
read. `materialize(task, run_ctx) -> TaskPackage`, written to
`task_packages/<task_id>.json` (spec §9.1 schema):

- `environment` block via the verified public round-trip (ticket 002 Part A):
  `task_to_item(task)` → `freeze_tasks([task], output_path=<transient scratch file>)` →
  `load_split(<tmp>)`. The whole `s0` goes in the package; reconstruction needs only the
  transient file, no external dataset.
- `oracle.payload` via the `freezer` serializer (`sub_oracles` for composite).
- `verifier` reference to `nutrienv.bench.scorer.Scorer`.
- `reward_semantics` binary, `reward_version = v2-r1`,
  `{pass:1.0, fail:0.0, indeterminate:null}`.
- `termination` from `nutrienv.harness.runner.FINISH_OPS` + `FAMILY_MAX_STEPS[family]`
  (fallback `DEFAULT_MAX_STEPS`).
- full `provenance` block.

Plus the three identifiers (spec §10) as a helper: `task_key` (no seed),
`task_id = f"{task_key}--{seed:06d}"`, `attempt_id`. A `task_id` seen twice within one
run raises.

**Blocked by:** 003.

**Status:** CLOSED (2026-09-09)

- [x] every spec §9.1 required field is present; `schema_version == "nutrimind-v2-taskpackage/1"`
- [x] env round-trips: reconstruction yields the same `s0` (profile fields, ledger rows,
      `allowed_food_ids`) and the same `catalog_sha` as the source `Task` (test per §19.1)
- [x] the reconstructed task is runnable (`NutriEnv().reset`) and Pass-reachable
      (`check_achievable`)
- [x] `termination.max_steps == FAMILY_MAX_STEPS[family]` (or `DEFAULT_MAX_STEPS`) per family
- [x] `provenance` carries `nutrimind_rev` (git sha), `nutrienv_rev`, `catalog_sha`,
      `config_sha`, `seed`, `intent_ref`, `built_at`
- [x] a duplicate `task_id` within one run raises; a re-run with an existing
      `task_packages/<task_id>.json` skips it (idempotent)
- [x] the transient scratch file is deleted after materialization (no orphan temp files)

## Closure notes

- `RunContext` (defined here, not in 003's concepts) carries run-wide fields
  (catalog + sha, revs, config_sha) plus per-task fields (steps/seed/intent_ref)
  and the run-scoped `seen_task_ids` registry — so the spec §10 duplicate rule
  has exactly one home. Simple families use `steps == (family,)` → e.g. key
  `log--log--train-alba`.
- The round-trip inside `materialize` is fail-fast insurance, not just payload
  extraction: rebuilt id/query are compared and a lossy round-trip raises
  (tested via monkeypatch), so a future nutrienv break fails at build time,
  not training time.
- Scratch-file cleanup is tested by routing `tempfile.tempdir` at a probe dir
  and asserting it stays empty after materialization.
- `oracle_version` is the rev-prefixed `nutrienv-<rev[:7]>` (spec OQ "finer
  oracle_version" stays deferred).
- §19.1's package → reconstruct test rebuilds the 1-item frozen file from the
  PACKAGE blocks (persona/situations from the source Task — the package's
  reconstruction contract is the environment + catalog_sha).
