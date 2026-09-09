---
id: 006
title: Canonical TaskPackage materializer (§9.1) + public env round-trip
status: ready-for-agent
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

**Status:** ready-for-agent

- [ ] every spec §9.1 required field is present; `schema_version == "nutrimind-v2-taskpackage/1"`
- [ ] env round-trips: reconstruction yields the same `s0` (profile fields, ledger rows,
      `allowed_food_ids`) and the same `catalog_sha` as the source `Task` (test per §19.1)
- [ ] the reconstructed task is runnable (`NutriEnv().reset`) and Pass-reachable
      (`check_achievable`)
- [ ] `termination.max_steps == FAMILY_MAX_STEPS[family]` (or `DEFAULT_MAX_STEPS`) per family
- [ ] `provenance` carries `nutrimind_rev` (git sha), `nutrienv_rev`, `catalog_sha`,
      `config_sha`, `seed`, `intent_ref`, `built_at`
- [ ] a duplicate `task_id` within one run raises; a re-run with an existing
      `task_packages/<task_id>.json` skips it (idempotent)
- [ ] the transient scratch file is deleted after materialization (no orphan temp files)
