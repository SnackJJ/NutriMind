---
id: 007
title: Tri-state verifier seam (Seam 3) — VerificationResult, three axes, indeterminate
status: CLOSED (2026-09-09) — 25 tests green (278 dir-wide)
commit: 3a6a41a
depends_on: [003]
spec: ../spec.md
spec_sections: ["11", "12", "19.2", "19.3"]
---

# 007 — Tri-state verifier seam (Seam 3)

**What to build:** The pure verification boundary.

```
verify(task_package: TaskPackage, episode: EpisodeResult) -> VerificationResult
```

`EpisodeResult` is the shared type from ticket 003 — `{end_state, turns: list[TurnMeta],
reached_finish, error}`. It carries everything the three axes need, so this single
signature replaces spec §19.7's looser `verify(task_package, end_state)` (the end state
is `episode.end_state`). Optionally factor the execution axis into a pure helper
`derive_execution(episode) -> "ok"|"no_finish"|"invalid_op"|"error"` that `verify` calls —
keeps the re-parse logic testable on its own.

Three axes recorded **separately**:

- `execution` — `ok` / `no_finish` / `invalid_op` / `error` — derived from
  `episode.turns` + `episode.reached_finish` + `episode.error` (v2's own re-parse of
  `raw_action_text` vs `executed_op`).
- `oracle_exec` — `ok` / `error` / `env_mismatch` — the verifier reconstructs from
  `task_package.environment` and compares lineage to `episode.end_state`; a raise →
  `error`, a mismatch → `env_mismatch`.
- `scorer` — `pass` / `fail` / `None` (None when an axis above is not ok) — from
  nutri-env's `Scorer`.

`status` is **derived**: `pass` iff all three clean and `scorer=pass`; `fail` iff clean
and `scorer=fail`; `indeterminate` otherwise.

- A completed **legal** episode with `Scorer.passed is False` → `fail`
  (`failure_codes = ["task_fail", <Scorer tag>]`). Never promoted to `indeterminate`.
- An exception (`Scorer` or env raises) → `indeterminate`, traceback in `evidence`.
  **Never** `fail`.
- Action-legality (`teacher_invalid_op`) is computed from v2-owned per-turn trajectory
  metadata (`raw_action_text` re-parsed by v2's own parser vs `executed_op`) — never from
  `nutrienv.harness.react._parse_action`'s internal fallback path.
- `diagnostic_scores` field exists and **never** affects `status` / `reward` in v2.0.
- `reward` map `{pass:1.0, fail:0.0, indeterminate:null}`, `reward_version = v2-r1`.

Pure given the `EpisodeResult`. `TurnMeta` / `EpisodeResult` live in ticket 003; ticket
009 populates them from a real rollout. This ticket depends only on 003 — the coupling to
009 is through the shared type, not a build-order edge.

**Blocked by:** 003.

**Status:** CLOSED (2026-09-09)

- [x] `verify(task_package, episode)` is pure given the `EpisodeResult`; carries
      `oracle_version` / `rubric_version` / `reward_version`
- [x] `derive_execution(episode)` (if factored out) returns the right axis value for a
      finished / no-finish / fallback-substituted / errored episode
- [x] a test for each `indeterminate` trigger: `teacher_error`, `teacher_no_finish`,
      `teacher_invalid_op`, `oracle_error`, `env_reconstruction_mismatch`, `gate.unachievable`
- [x] a completed legal episode with `Scorer.passed is False` → `status=fail`,
      `failure_codes=["task_fail", <tag>]`; never `indeterminate`
- [x] an episode where `Scorer` raises → `status=indeterminate` with traceback in `evidence`;
      never `fail`
- [x] hard-constraint boundaries: window just inside vs just outside the ±15% tolerance;
      allergen present; off-`allowed_food_ids` food; missing required op
- [x] two different in-window, allergen-safe plans both → `pass` (order-independent, no
      verbatim match)
- [x] nonexistent `food_id` → `fail` (`wrong_goal`), never `pass`
- [x] the `teacher_invalid_op` test is built from trajectory metadata with no reference to
      `_parse_action`
- [x] a low `diagnostic_scores` value changes neither `status` nor `reward`

## Closure notes

- `derive_execution` factored out as the ticket suggested; priority order:
  error > no_finish > invalid_op (the episode is judged by its most
  fundamental defect first).
- v2's parser (`parse_action_text`) mirrors ONE executed normalization: an
  accepted submit_plan drops free-form `reasons` — otherwise a genuine plan
  would falsely mismatch the action the env received. The legal-op vocabulary
  is v2-owned (nutri-env's `OPS` is not in `__all__`; ADR-012 forbids the
  import). No reference to `_parse_action` anywhere in src or tests.
- The ±15 % spec line is the LEDGER gram tolerance (ADR-0023, inside
  `_match_ledger_multiset`), NOT the plan windows — plan windows are strict
  lo/hi. Both boundary classes are tested: ledger 1.14x passes / 1.2x fails
  (log_miss); plan total == hi passes / total > hi fails (window) with the
  window pinned to the fitting plan's exact kcal total.
- The env REJECTS malformed submit_plans without storing them, so
  wrong_goal evidence falls back to the episode's last submitted payload
  (v2's own recorded metadata) to name the nonexistent food_id.
- gate.unachievable stays routed at the gate stage (ticket 005): the §19.2
  trigger test asserts the pipeline ordering instead of calling verify.
- `diagnostic_scores` is always None from verify in v2.0; the test also
  proves a hand-set value cannot flip status/reward (§19.3).
