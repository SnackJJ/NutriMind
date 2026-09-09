---
id: 011
title: build --target sft end-to-end for the log family (Seam 1)
status: CLOSED (2026-09-09) — 8 end-to-end tests green (336 dir-wide), fully offline
commit: ff875f9
depends_on: [010, 007, 008, 009]
spec: ../spec.md
spec_sections: ["4.1", "6", "8", "19.7", "11", "16", "22"]
---

# 011 — build --target sft end-to-end for `log` (Seam 1)

**What to build:** The first full path — `build --target sft` for the `log` family.
After materialize, per `task_id`:

- run the teacher rollout (ticket 009) for attempts 1..k — attempt 1 at
  `temperature_first` (0.0), attempts 2..k at `temperature_retry` (0.7), stop at the
  first Pass. `k` counts total attempts, retries included (spec §16).
- score the end state with the tri-state verifier (ticket 007).
- `status=pass` → **Pass-filter** → `serialize` (ticket 008) → accepted.
- accepted records sorted by `task_id`, written to `sft/train.jsonl` via temp + atomic
  rename; `run_manifest.json` written.
- `status=fail` → `rejects/teacher.jsonl` with **all k attempts** + their `failure_codes`
  (`["task_fail", <tag>]`) as an analysis candidate — **never** an RLVR negative.
- `status=indeterminate` / teacher error / no-finish → `rejects/indeterminate.jsonl`.
- `rollouts/cache/<task_id>.json` is a **`RolloutCache`** (ticket-003 type, pinned so
  ticket 014's `--from-stage serialize` can re-select without re-running the teacher):
  `{task_id, attempts: [{attempt_id, EpisodeResult, VerificationResult}], selected_attempt}`.
  It is a **multi-attempt container**, not one bare episode — every attempt 1..k that ran
  is recorded (each `EpisodeResult` carrying its messages / `content` / `reasoning_content`
  / `finish_reason` / `usage` / `end_state`); `selected_attempt` is the first Pass, or
  `null` if none passed.

Seam 1 = `build(config, *, expander, teacher_complete)`: inject a synthetic `expander`
and a scripted `teacher_complete` (queue of `(content, reasoning_content)`).

**Blocked by:** 010, 007, 008, 009.

**Status:** CLOSED (2026-09-09)

- [x] Seam-1 test: synthetic `expander` + scripted `teacher_complete` → `build --target sft`
      for `log` produces a deterministic, byte-identical `sft/train.jsonl` across runs
- [x] a scripted Pass episode → exactly one record in `sft/train.jsonl`;
      `meta.verification.status == "pass"`, `reward == 1.0`
- [x] a scripted Fail episode → nothing in `sft/train.jsonl`; one `rejects/teacher.jsonl`
      line with an `attempts` array (length ≤ k) and
      `failure_codes == ["task_fail", <Scorer tag>]`; no field marks it an RLVR negative
- [x] a scripted teacher error / no-finish → one `rejects/indeterminate.jsonl` line, nothing
      accepted
- [x] the teacher stops retrying at the first Pass (`len(cache.attempts) ≤ k`;
      `selected_attempt` points at that Pass)
- [x] one bad intent among several good → the good ones land, the run completes
- [x] `rollouts/cache/<task_id>.json` deserializes to a `RolloutCache`: one entry per
      attempt that ran, each with its `EpisodeResult` (messages per step, assistant
      `content` + `reasoning_content` + `finish_reason` + `usage`, `end_state`) and its
      `VerificationResult`; `selected_attempt` is the first Pass or `null`

## Closure notes

- `selected_attempt` is the 0-based index into `attempts` of the first Pass
  (per the ticket-003 concepts docstring; `accepted_from_attempt` in the
  record is 1-based n).
- Task-level fail/indeterminate routing after k attempts: any fail attempt →
  `rejects/teacher.jsonl` (the line carries the FIRST failing attempt's
  failure_codes + a compact per-attempt summary; full EpisodeResult/
  VerificationResult live in the cache); no completed legal attempt at all →
  `rejects/indeterminate.jsonl` with the last attempt's codes. No field
  anywhere marks a teacher fail as an RLVR negative (tested).
- Serialize is cache-authoritative: the record is built from the RELOADED
  on-disk cache, so run-1 (live episode) and run-2 (cache hit) produce
  byte-identical `sft/train.jsonl` by construction. `serialize` reads
  `persona` from a dict-backed episode (the cache round-trip has no live
  Task).
- Cache JSON drops exactly the non-serializable leaves — the live
  FoodCatalog inside WorldState (huge; re-derivable from the pinned
  catalog_sha). Everything else (profiles, ledger rows, plans, all TurnMeta
  fields incl. reasoning_content/usage/finish_reason) survives verbatim.
- Terminal semantics refined per spec §8: a materialized package alone is
  NOT terminal when the run includes the teacher stage (the task may still
  need its episode); `--stop-after gate` re-runs keep skipping on packages.
- `teacher_k` for the log family = 6 family-wide (provisional, spec §2.1's
  3-leg value applied family-wide pending the design-doc reconstruction,
  ticket 021); the offline tests use k=1/2 to keep scripts short.
- The 7:2:1 train/holdout/loss_val hash split (§6 step 6) is deferred to
  ticket 014/015 with the rest of the hardening; 011 writes sft/train.jsonl
  only.
- CLI: `--teacher ark` wires `make_ark_teacher_client` (NUTRIMIND_ALLOW_
  NETWORK=1 + ARK_API_KEY guarded at call time); the offline tests inject
  ScriptedTeacher directly through the Seam-1 signature.
