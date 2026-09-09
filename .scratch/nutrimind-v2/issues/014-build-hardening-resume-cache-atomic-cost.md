---
id: 014
title: build hardening — resume / rollouts cache reuse / --from-stage / atomic writes / cost budget
status: ready-for-agent
depends_on: [011]
spec: ../spec.md
spec_sections: ["8", "22.14", "22.15", "16", "US-14", "US-15"]
---

# 014 — build hardening: resume / cache / atomic writes / cost budget

**What to build:** Making a real run safe to interrupt and cheap to re-run.

- **Resume:** a re-run in the same `output_dir` skips any `task_id` already present in
  `sft/train.jsonl` or any `rejects/*.jsonl`; reuses `task_packages/` + `rollouts/cache/`
  when present. `--from-stage {author,gate,materialize,rollout,serialize}` forces re-run
  of terminal tasks from that stage. `--from-stage serialize` reads each
  `rollouts/cache/<task_id>.json` as a `RolloutCache` (ticket-003 type), re-runs
  `serialize` on `attempts[selected_attempt]` (or re-picks the first Pass if
  `selected_attempt` is stale), and issues **zero teacher calls**.
- **Atomic writes:** `task_packages/`, `rollouts/cache/`, `sft/*.jsonl`,
  `run_manifest.json` via temp + `os.replace`. `rejects/*.jsonl` append-only with
  `flush()` and a finalize pass that drops a trailing partial line.
- **Cost budget:** `usd_budget`; `on_budget: warn` (log at 80%, continue) or `stop`
  (halt cleanly at 100%, write the manifest with what completed).

**Blocked by:** 011.

**Status:** ready-for-agent

- [ ] killing a run mid-task leaves the previous `sft/train.jsonl` intact and no
      half-written JSON line anywhere (test via a crash injected between write and rename)
- [ ] a second run over an unchanged cache issues zero teacher calls and reproduces the
      same `sft/train.jsonl`
- [ ] `--from-stage serialize` re-serializes from `RolloutCache.attempts[selected_attempt]`
      for every cached task and issues zero teacher calls; a cache with `selected_attempt =
      null` (no Pass) stays a reject, not a spurious accept
- [ ] `--from-stage rollout` re-runs the teacher for terminal tasks and leaves non-targeted
      stages alone
- [ ] `on_budget: stop` halts at 100% of `usd_budget` and still writes a valid
      `run_manifest.json` with the completed counts
- [ ] `on_budget: warn` logs at 80% and runs to completion
- [ ] a `task_id` seen twice within one run raises (bad enumerator)
