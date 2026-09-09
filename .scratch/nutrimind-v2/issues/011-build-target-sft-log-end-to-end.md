---
id: 011
title: build --target sft end-to-end for the log family (Seam 1)
status: ready-for-agent
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

**Status:** ready-for-agent

- [ ] Seam-1 test: synthetic `expander` + scripted `teacher_complete` → `build --target sft`
      for `log` produces a deterministic, byte-identical `sft/train.jsonl` across runs
- [ ] a scripted Pass episode → exactly one record in `sft/train.jsonl`;
      `meta.verification.status == "pass"`, `reward == 1.0`
- [ ] a scripted Fail episode → nothing in `sft/train.jsonl`; one `rejects/teacher.jsonl`
      line with an `attempts` array (length ≤ k) and
      `failure_codes == ["task_fail", <Scorer tag>]`; no field marks it an RLVR negative
- [ ] a scripted teacher error / no-finish → one `rejects/indeterminate.jsonl` line, nothing
      accepted
- [ ] the teacher stops retrying at the first Pass (`len(cache.attempts) ≤ k`;
      `selected_attempt` points at that Pass)
- [ ] one bad intent among several good → the good ones land, the run completes
- [ ] `rollouts/cache/<task_id>.json` deserializes to a `RolloutCache`: one entry per
      attempt that ran, each with its `EpisodeResult` (messages per step, assistant
      `content` + `reasoning_content` + `finish_reason` + `usage`, `end_state`) and its
      `VerificationResult`; `selected_attempt` is the first Pass or `null`
