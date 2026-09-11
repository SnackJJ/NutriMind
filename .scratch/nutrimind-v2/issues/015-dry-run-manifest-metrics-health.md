---
id: 015
title: --dry-run projection + run_manifest.json metrics + §20 health checks + reject histograms
status: CLOSED (2026-09-11)
commit: 36777a3
depends_on: [011]
spec: ../spec.md
spec_sections: ["9.5", "17", "20", "US-11", "US-13", "US-26"]
---

# 015 — --dry-run + manifest metrics + health checks

**What to build:** Being able to tell whether a run worked without opening the data.

- **`--dry-run`:** authors and gates every intent **without** the teacher and writes
  `dry_run_report.json` with projected accept counts and a per-`failure_code` reject
  histogram (US-13).
- **`run_manifest.json`:** counts split by `status` (accepted / fail / indeterminate) and
  by `failure_code` (`indeterminate` a separate bucket from `fail`); `family_mix` target
  vs actual; `cost`; a `versions` block (`oracle_version`, `rubric_version`,
  `reward_version`, `environment_version`, `task_schema_version`).
- **§20 health metrics** computed into the manifest: `catalog_sha_match`;
  `serialization_success_rate` (`serialized / status=pass`) ≥ 0.98;
  `teacher_completion_rate` (`completed / (completed + teacher_error + teacher_no_finish)`)
  ≥ 0.9; `teacher_pass_rate` (`pass / completed`); run-level `indeterminate_rate`
  (`indeterminate_task_ids / attempted_task_ids`) ≤ 0.05 — **skipped when
  `attempted_task_ids < 40`**, raw counts reported instead; a reject-histogram shape check
  (dominated by `author.*` + `task_fail`; a large `gate.draft_invalid` /
  `gate.unachievable` share flags the run).

**Blocked by:** 011.

**Status:** CLOSED (2026-09-11)

- [x] `--dry-run` issues zero teacher calls and writes `dry_run_report.json` with projected
      accepts + reject-reason histogram
- [x] `run_manifest.json` splits counts by `status` and by `failure_code`, `indeterminate`
      a separate bucket from `fail`
- [x] the `versions` block is present and matches the per-record `meta` versions
- [x] `serialization_success_rate`, `teacher_completion_rate`, `teacher_pass_rate` are
      computed and match a hand-count on a scripted run
- [x] `indeterminate_rate` is a ratio only when `attempted_task_ids ≥ 40`, else raw counts
      with a note
- [x] `family_mix` shows target vs actual per family
- [x] the reject-histogram shape check flags a synthetic run with an inflated
      `gate.draft_invalid` share

## Closure notes

- Health flags live on `run_manifest.json` (`health.reject_histogram_ok`); they do not
  abort the run. `catalog_sha_match` remains the preflight hard gate.
