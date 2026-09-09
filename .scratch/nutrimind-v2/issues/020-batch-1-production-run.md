---
id: 020
title: Batch-1 production run — ~420 accepted Pass at the design §7 family mix; validate §20; 7:2:1 split
status: ready-for-agent
depends_on: [013, 014, 015, 017]
spec: ../spec.md
spec_sections: ["2", "US-1", "6", "20", "9"]
adr: [../../docs/decisions/011-batch1-sft-trajectory-short-plan-thinking-teacher.md]
---

# 020 — Batch-1 production run

**What to build:** The actual Batch-1 **data factory** run. `build` for the full design
§7 family mix (composite ~57% / recommend ~17% / evaluate ~13% / log ~10% / update ~3%,
≈ 420 accepted **Pass** total) with the real `ark/` expander and the real teacher,
Pass-filtered, serialized. The accepted set is sorted by `task_id`, written atomically,
and split 7:2:1 by a `task_id` hash into `sft/train.jsonl` / `sft/holdout.jsonl` /
`sft/loss_val.jsonl`. The run is validated against spec §20 and the reject histogram is
reviewed.

**Blocked by:** 013, 014, 015, 017.

**Status:** ready-for-agent

- [ ] `run_manifest.json` shows `catalog_sha_match == true`, `counts.accepted ≥ ~380`
      (target ~420), and every family's actual within ±15% of its design §7 target
- [ ] `sft/train.jsonl` has **zero** `verbatim_query_collision` / `semantic_key_collision`
- [ ] `serialization_success_rate ≥ 0.98`, `teacher_completion_rate ≥ 0.9`, run-level
      `indeterminate_rate ≤ 0.05`
- [ ] the 7:2:1 split is keyed by a `task_id` hash and reproducible; the three files
      partition the accepted set with no overlap
- [ ] `cost.est_usd ≤ cost.budget_usd`; `mean_seconds_per_accepted` and
      `mean_teacher_tokens_per_accepted` recorded
- [ ] the reject histogram is dominated by `author.*` bind reasons + `task_fail`; a large
      `gate.draft_invalid` / `gate.unachievable` share is investigated, not ignored
- [ ] the mini-exam val `task_id`s (ticket 017) are absent from the accepted set
