---
id: 013
title: 3-leg composite update+log→recommend family + OQ-7 Part C acceptance gate (40 accepted Pass)
status: CLOSED (2026-09-11)
depends_on: [012, 025, 026]
spec: ../spec.md
spec_sections: ["22.9", "23-OQ-7-Part-C", "6", "10", "16", "20"]
adr: [../../docs/decisions/012-nutrienv-read-only-benchmark.md]
prior: ["002 step (c)"]
---

# 013 — 3-leg composite family + Part C acceptance gate

**What to build:** The `composite update+log→recommend` (3-leg) family.

**Authoring** — the spec §22.9 public-symbol recipe only (no `_update_from_template` /
`_bind_log_foods`): `generate_one(family="update", …)` + `generate_one(family="log", …)`;
`update_sub = dataclasses.replace(u.oracle, ledger=None, ledger_tail=None)` (strip the
standalone update oracle's stale `ledger=()`); `log_sub` = `Oracle(ledger_tail=tail,
ledger=(*s0.ledger,*tail), profile=deepcopy(expected))`; `rec_sub` = `Oracle(
profile=deepcopy(expected), last_plan=[], ledger=(*s0.ledger,*tail),
plan_must_be_safe=True, plan_must_fit_windows=True,
plan_windows=plan_windows_for_meal(...))`; `Task(id, "composite", query, s0,
compose_oracles(update_sub, log_sub, rec_sub), …)`.

**Gate policy for this shape:** `stage_a_code_gate == []` **and**
`validate_draft ∈ ([], ["update oracle ledger is missing"])` **and** `check_achievable`
reachable **and** a correct-replay `Scorer` Pass.

**§6 expander ladder is engaged from the start** — `gram_anchor` on, a persona-derived
`amount_path` mix (**not** forced `named_measure`), and a `qwen3.8-max` expander fallback.
Ticket 002 step (c) measured `bind_fail_rate ≈ 0.75–0.83` with a naive `ark` expander →
the ladder is mandatory, not a contingency. Candidate sizing:
`candidate_count = min(max_candidate_limit, max(120, ceil(40 / p · 1.5)))` where `p` =
`estimated_unique_accepted_rate` measured with the tuned expander.

**Hard acceptance gate:** a qualification run reaches **40 distinct-`task_id`,
intra-family `semantic_key`-deduped accepted Pass** under the §6 ladder, with `k = 6`
teacher attempts per `task_id`, run-level `indeterminate_rate ≤ 0.05`, and a reject
histogram dominated by `author.*` bind reasons + `task_fail`. If 40 is **not** reached the
ticket does **not** silently accept a lower number — it records exactly one of: a
documented ADR-012 Amendment-log exception (private `_bind_log_foods`), an upstream
nutri-env `__all__` promotion PR, or an explicit design §7 re-size.

**Blocked by:** 012, 025, 026 (no qualification run on the retired text-op path).

**Status:** ready-for-agent

- [x] the 3-leg authoring path imports no `_update_from_template` / `_bind_log_foods`
      (asserted via `co_names`, as in `test_three_leg_public_assembly.py`)
- [x] one assembled 3-leg task passes the full gate policy and correct-replays to a `Scorer`
      Pass with `sub_tags == ("pass","pass","pass")`; wrong end states tag `log_miss` /
      `window` / `update_miss` correctly
- [x] the §6 expander ladder is wired and on by default for this family (`gram_anchor` +
      persona `amount_path` mix + `qwen3.8-max` fallback); a test asserts `named_measure`
      is not force-forced
- [x] `candidate_count` is computed from a measured `estimated_unique_accepted_rate`,
      recorded in the run manifest with `seed_start`
- [x] a qualification run reaches **40** distinct-`task_id`, `semantic_key`-deduped
      accepted Pass with `indeterminate_rate ≤ 0.05` and a healthy reject histogram — **or**
      the chosen fallback is written into ADR-012's Amendment log / a `__all__` PR / a
      design §7 change
- [x] N is counted on `task_id` after intra-family `semantic_key` dedup (lowest seed kept);
      multiple `attempt_id`s of one `task_id` contribute at most one
