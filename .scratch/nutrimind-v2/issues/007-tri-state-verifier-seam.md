---
id: 007
title: Tri-state verifier seam (Seam 3) — VerificationResult, three axes, indeterminate
status: ready-for-agent
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

**Status:** ready-for-agent

- [ ] `verify(task_package, episode)` is pure given the `EpisodeResult`; carries
      `oracle_version` / `rubric_version` / `reward_version`
- [ ] `derive_execution(episode)` (if factored out) returns the right axis value for a
      finished / no-finish / fallback-substituted / errored episode
- [ ] a test for each `indeterminate` trigger: `teacher_error`, `teacher_no_finish`,
      `teacher_invalid_op`, `oracle_error`, `env_reconstruction_mismatch`, `gate.unachievable`
- [ ] a completed legal episode with `Scorer.passed is False` → `status=fail`,
      `failure_codes=["task_fail", <tag>]`; never `indeterminate`
- [ ] an episode where `Scorer` raises → `status=indeterminate` with traceback in `evidence`;
      never `fail`
- [ ] hard-constraint boundaries: window just inside vs just outside the ±15% tolerance;
      allergen present; off-`allowed_food_ids` food; missing required op
- [ ] two different in-window, allergen-safe plans both → `pass` (order-independent, no
      verbatim match)
- [ ] nonexistent `food_id` → `fail` (`wrong_goal`), never `pass`
- [ ] the `teacher_invalid_op` test is built from trajectory metadata with no reference to
      `_parse_action`
- [ ] a low `diagnostic_scores` value changes neither `status` nor `reward`
