---
id: 002
title: Feasibility spike — env reconstruction (A), public-API 3-leg (B), N=40 batch qualification (C)
type: prototype
status: CLOSED (2026-09-09) — A=verdict A, B=YES; C is implementation-gated (not a spike)
depends_on: [001]
branch: nutrimind-v2/ticket-002-composite-feasibility
spec: ../spec.md
adr: [../../docs/decisions/011-batch1-sft-trajectory-short-plan-thinking-teacher.md, ../../docs/decisions/012-nutrienv-read-only-benchmark.md]
resolves: [OQ-2, OQ-7-partB]
---

## Outcome (2026-09-09)

- **Part A / OQ-2 — verdict A (self-contained).** Public temp-file round-trip
  `task_to_item` → `freeze_tasks` → `load_split` is lossless + runnable + Pass-reachable.
  Caveat: no in-memory public entry. Test: `test_env_reconstruction.py`.
- **Part B / OQ-7 — YES.** 3-leg `update+log→recommend` assembles from public symbols
  only (shape-equal to exam `adr24-comp-8255`); `stage_a_code_gate` + `check_achievable`
  + Scorer all correct; wrong-state tags (`log_miss`/`window`/`update_miss`) correct.
  `validate_draft` false-positive `"update oracle ledger is missing"` allow-listed (the
  exam item trips it too). Test: `test_three_leg_public_assembly.py`.
- **Side finding — ADR-011 amended.** `ark/deepseek-v4-flash` (`api/plan/v3`) returns
  `reasoning_content` by default, multi-turn, under `_SYSTEM_V2` → teacher endpoint moves
  from `deepseek/` direct to `ark/`; teacher + expander share one endpoint/key. But
  ReAct-turn reasoning is long/variable (≤~2.3k tok) → the ~80-tok plan cap is
  load-bearing.
- **Part C — NOT a spike.** The 3-leg yield is gated by log-leg colloquialization bind
  rate; the naive ark expander gives `bind_fail_rate ≈ 0.75–0.83` (> design §6's 60 %
  threshold), thinking on/off doesn't help. Every assembled+gated task did correct-replay
  to a Pass (16/16). Part C needs a tuned expander (§6 `gram_anchor` ladder + §8
  `amount_path` mix) + the teacher-rollout machinery + real spend → it becomes a **hard
  acceptance gate on the 3-leg implementation ticket**, not a pre-`to-tickets` spike.
- **Next:** `to-tickets`. Merge `nutrimind-v2/ticket-002-composite-feasibility` (which
  contains ticket 001's commit too) as the baseline.

# 002 — Three-part feasibility spike

Three distinct proofs at different levels. **Part B (single-instance feasibility +
~20-seed rate estimate) does NOT substitute for Part C (N=40 accepted-Pass batch
qualification).**

## Prerequisite

Ticket 001 closed: `nutrienv` importable, public-API smoke test green locally and in CI.

## Design provenance (copied here so this ticket does not depend on `/tmp`)

From the finalized design doc (currently `/tmp/nutrienv_student_data.md` §7, pending move
to `docs/plans/nutrimind_v2_data_factory.md`):

- Batch-1 overall target ≈ **420 accepted Pass traces** (table column: "Pass traces";
  total row: "~420").
- `composite update+log→recommend` (3-leg) family target = **40**, teacher **k = 6**.
- **N = 40 is an accepted-Pass count, not a candidate count.** Nothing in the code or an
  existing spec contradicts this. Do not reinterpret it.

---

## Part A — single-item environment reconstruction (OQ-2)

Prove:

```
one frozen item  →  reconstruct environment  →  NutriEnv().reset / oracle / Scorer all run
```

Steps: take one `Task` from `generate_one` (synthetic expander) → `task_to_item(task)` →
find a path back to a runnable `NutriEnv` / `Task` / `WorldState`. Try `load_split`
against a 1-item file; look for any other public entry; if only `split._item` / `_s0` /
`_oracle` (private) works, record that. Assert reconstructed `s0` (profile fields, ledger
rows, `allowed_food_ids`) and `catalog_sha` equal the originals.

**Verdict options:** A self-contained · B needs an external file · C needs a new public
API · D not viable.

### VERDICT — A (self-contained), with a temp-file caveat  ·  2026-09-09

Spike: `.scratch/nutrimind-v2/spikes/002a_env_reconstruction.txt`;
regression test: `tests/training/data_factory/test_env_reconstruction.py` (4 passed).

- **Public round-trip works and is lossless.** For `update`, `recommend`, and
  `composite(update, recommend)` tasks (all no-expander paths):
  `task_to_item(task)` → `freeze_tasks([task], catalog=, catalog_sha=, output_path=<tmp>)`
  → `load_split(<tmp>, catalog=)` → `Task'`.
  Reconstructed `s0` (user_id / allergies / windows / phase / activity / weight / ledger
  rows / `allowed_food_ids`), the full oracle signature (incl. `sub_oracles`,
  `plan_windows`, `last_plan` sentinel, `update_band`, `ledger_tail`), and
  `(family, persona, tier, query)` all equal the originals.
- Reconstructed task is **runnable** (`NutriEnv().reset(rt.s0)` → dict) and still
  **Pass-reachable** (`check_achievable([rt]).unreachable` empty).
- **No external dataset dependency.** The TaskPackage `environment` block carries the
  whole `s0`; reconstruction needs only a **transient scratch file**, not
  `nutrienv-v1.0.json` or any shared split. `task_id ↔ world state` binding is stable.
- **Caveat — no in-memory public entry.** `nutrienv.bench.split.__all__` =
  `["GOLD_SPLIT_PATH", "EXAM_SPLIT_PATH", "load_split", "load_exam"]`; `load_split`
  requires a file on disk (`target.is_file()`). `_item` / `_s0` / `_oracle` are private.
  So the RLVR runner / `build` must write a 1-item split to a temp path and `load_split`
  it. Functional, but an implementation wart.

**Non-blocking upstream nice-to-have (not a Batch-1 blocker):** ask nutri-env to add
`item_to_task(item, catalog)` (or `load_split_payload(payload, catalog)`) to
`nutrienv.bench.split.__all__` so reconstruction is in-memory. Until then, the temp-file
round-trip stands.

**Consequence for the spec:** §9.1 `environment` moves from "CANDIDATE, unverified" to
"verified — public temp-file round-trip; self-contained". OQ-2 is resolved. Proceed to
Part B.

---

## Part B — single-instance public-API feasibility (OQ-7)

Prove, **for one assembled task**, using **only public symbols** (`generate_one(family="update")`
+ `generate_one(family="log")` + `plan_windows_for_meal` + `Oracle` + `compose_oracles`;
**no** `_update_from_template` / `_bind_log_foods`):

```
assemble 3-leg composite Task
→ check_achievable([task]).unreachable == []
→ stage_a_code_gate(task) == []
→ validate_draft(task) == []
→ Scorer().score(correct_end_state, oracle).passed is True
→ Scorer tags wrong end states correctly:
    log leg missing        → log_miss
    out-of-window plan      → window
    profile not updated     → update_miss
→ task_to_item(task) round-trips env + composite oracle   (uses Part A's path)
```

Then a **~20-seed sweep** for **preliminary rate estimates only**. Report, over the
sweep (denominator = attempted `task_id`s):

- `raw_pass_rate` — attempts that reach `Scorer.passed is True`
- `distinct_task_survival` — accepted that are distinct `task_id`s
- `semantic_dedup_survival` — accepted `task_id`s surviving intra-family `semantic_key`
  dedup
- `indeterminate_rate` — for information; **do not gate Part B on it** (the run-level
  `≤ 0.05` threshold, spec §20, is applied in Part C; 20 seeds is below its minimum
  sample)
- **`estimated_unique_accepted_rate`** = (unique, `semantic_key`-deduped accepted
  `task_id`s) / (attempted `task_id`s) — the single number that drives Part C's
  `candidate_count`.

> **The ~20-seed estimates are not the N=40 acceptance.** They only tell us whether
> Part C is worth attempting and size its `candidate_count`. Passing Part B ⇏ passing
> Part C. If `estimated_unique_accepted_rate ≤ 0`, Part C does not start (verdict: not
> viable for batch qualification).

### VERDICT — YES, public composition assembles and verifies  ·  2026-09-09

Spike: `.scratch/nutrimind-v2/spikes/002b_three_leg.txt`;
regression test: `tests/training/data_factory/test_three_leg_public_assembly.py` (6 passed).

- **Assembled from public symbols only** — `generate_one(family="update")` +
  `generate_one(family="log")` + `plan_windows_for_meal` + `ledger_totals` + `Oracle` +
  `Task` + `compose_oracles` + `recommend_query` + `speakable_tracer_food`. No
  `_update_from_template` / `_bind_log_foods` (asserted via `co_names`).
- **Shape-equivalent to the frozen exam item `adr24-comp-8255`**: 3 sub-oracles —
  update (`profile` only, `ledger=None`), log (`ledger` + `ledger_tail` + `profile`),
  recommend (`last_plan=[]` + `plan_windows` + `plan_must_*` + `profile`).
- **One gotcha, matched by the exam**: reusing the standalone update oracle needs its
  stale `ledger=()` stripped (`dataclasses.replace(u.oracle, ledger=None,
  ledger_tail=None)`), else the update sub scores `log_miss` once the log leg populates
  the ledger.
- **`validate_draft` is unreliable for this shape** — it returns
  `["update oracle ledger is missing"]`, and **`load_exam()`'s `adr24-comp-8255` returns
  the identical string**. So the 3-leg gate policy is: `stage_a_code_gate(task) == []`
  **and** `validate_draft(task) in ([], ["update oracle ledger is missing"])` **and**
  `check_achievable([task]).unreachable == []`.
- **Scorer matrix (seed 101)**: correct replay → Pass `("pass","pass","pass")`;
  `skip_log` → `log_miss`; `over_window` (safe plan ×4) → `window`; `skip_update` →
  `update_miss`. All correct.
- **~20-seed sweep** (crude synthetic tracer expander, no retry-on-gate-fail):
  `assembled+gated 11/20`, `correct-replay Pass 11/11`, `semantic_key-unique 11/11`,
  `estimated_unique_accepted_rate ≈ 0.55`. Rejects are authoring
  (`author.log.small_grams`/`unresolvable`/`schema` from the crude tracer) +
  `check_achievable` UNREACHABLE (no safe in-window dinner after the update) — normal
  attrition. A real expander (Part C) does better.

**Consequences for the spec:**
- §22.9 — 3-leg authoring path is confirmed viable from public symbols; add the
  `ledger=None` strip and the `validate_draft` allow-list to the recipe.
- §23 OQ-7 Part B — resolved YES. Proceed to Part C (needs the real teacher; separate
  go-ahead — it costs API tokens).
- Part C `candidate_count` seed from `estimated_unique_accepted_rate` measured with the
  **real** expander, not this 0.55.

### Side finding — `ark/deepseek-v4-flash` returns the plan (2026-09-09, 3 live calls)

Probe: `.scratch/nutrimind-v2/spikes/002_ark_reasoning_probe.txt`.

- `ark/deepseek-v4-flash` on **`api/plan/v3/chat/completions`** returns **both**
  `message.content` and `message.reasoning_content` **by default** (no `thinking` param).
  Reasoning length at `usage.completion_tokens_details.reasoning_tokens` (~60–220 tok);
  top-level `usage.reasoning_tokens` is `None` — which is why the published v1.0 reports
  showed `reasoning_tokens: 0` (they read the top-level field). The flash model was
  reasoning all along.
- **Consequence: ADR-011 amended (2026-09-09)** — teacher endpoint moves from
  `deepseek/` direct to `ark/deepseek-v4-flash` `api/plan/v3`. Teacher + expander now
  share one endpoint + one `ARK_API_KEY`.
- **Confirmed (spike (c) part 1):** `reasoning_content` stays populated **multi-turn**
  under `react_manual("v2")` (3 turns, `reasoning_tokens` 739 / 180 / 2289; clean single
  JSON `content` each turn). But ReAct-turn reasoning is **long and variable** (up to
  ~2.3k tok/turn), not the 60–220 the single-Q probe showed → the ~80-tok plan
  truncation (ADR-011) is load-bearing and the teacher call needs a length control.
  Expander calls should set `thinking: {"type": "disabled"}`.

### Step (c) — real ark expander yield on the 3-leg log leg (2026-09-09)

Spikes: `.scratch/nutrimind-v2/spikes/002c_*.txt`. Log leg = `generate_one(family="log",
expander=make_log_expander(complete=<ark>, model_id="deepseek-v4-flash"))`, forced
`amount_path="named_measure"`, `parse_retries=1`, no `gram_anchor`.

| expander | seeds | assembled+gated | correct-replay Pass | est. unique accepted rate | ark tokens |
|---|---|---|---|---|---|
| crude synthetic tracer | 20 | 11 | 11/11 | ~0.55 | 0 |
| ark, thinking DISABLED | 12 | 3 | 3/3 | ~0.25 | 25k (0 reasoning) |
| ark, thinking ENABLED | 12 | 2 | 2/2 | **~0.17** | 34k (8k reasoning) |

- The 3-leg's yield is **dominated by the log-leg colloquialization bind rate**, not by
  the composition (which is solid — Part B). Rejections are all `_bind_log_foods`:
  `ambiguous` / `amount_path` / `unresolvable` / `small_grams` — the LLM's spoken portion
  phrase does not back-resolve.
- **Thinking on/off does not help** — thinking-enabled is slightly *worse* (0.17 vs
  0.25) and ~5× the token cost. The bind failures are a portion-phrasing problem, not a
  reasoning problem.
- `bind_fail_rate ≈ 0.75–0.83` with the default `LogExpander` prompt + `ark` +
  forced `named_measure` is **above design §6's 60 % fallback threshold** → design §6's
  expander ladder (`gram_anchor`, then fall back to `qwen3.8-max`) is **confirmed
  mandatory for this family, not optional**. Forcing `named_measure` is the hard path;
  `explicit_grams` binds trivially → the persona-derived `amount_path` mix (§8) is a
  real lever.
- **Every task that *did* assemble + gate also correct-replayed to a Pass** (16/16
  across all three runs) — the 3-leg oracle/gate contract is sound; the risk is purely
  authoring throughput.

**Consequence:** Part C's `estimated_unique_accepted_rate` cannot be measured in a spike
— it needs a *tuned* expander (`gram_anchor` wired + `amount_path` mix + possibly
`qwen3.8-max`) **and** the teacher-rollout machinery (OQ-16) **and** real teacher spend.
Part C is therefore an **implementation-gated acceptance criterion**, not a spike:
proceed to `to-tickets` with the 3-leg family carrying "40 accepted Pass under the §6
expander ladder; if it can't reach 40, re-size design §7" as a hard gate on its
implementation ticket.

---

## Part C — N=40 accepted-Pass batch qualification (only if Part B passes)

v2.0 requires **40 accepted Pass** for this family (design doc §7; not negotiable in this
ticket). Prove it under the fixed config below, or declare acceptance failure. These are
**rules, not recommendations** — write them into `configs/data_factory.yaml` for the
qualification run:

- **The unit of N is `task_id`** (spec §10: `task_id = f"{task_key}--{seed:06d}"`,
  `task_key` carries no seed). "40 accepted Pass" = **40 distinct accepted `task_id`s**,
  after intra-family `semantic_key` dedup. Different seeds are different `task_id`s and
  count separately unless they collide on `semantic_key`. Multiple `attempt_id`s of one
  `task_id` contribute **at most one**. N is **not** counted on `task_key` (the design
  doc's "Pass traces" are per instance, not per logical task).
- **Intra-family `semantic_key` dedup.** Before counting, deduplicate the family's
  accepted `task_id`s on `nutrienv.bench.validator.semantic_key` (against each other, not
  the exam — the exam collision is already a `gate` drop). Two accepted `task_id`s with
  the same `semantic_key` → keep the lowest seed, count one.
- **`k = 6` = max teacher rollout attempts per `task_id`.** One rollout = one full ReAct
  episode (internally bounded by `nutrienv.harness.runner.FAMILY_MAX_STEPS["composite"]`
  = 30 steps). Attempt 1 (temp 0.0), then up to 5 retries (temp 0.7), stop at first Pass.
  **`k` counts total attempts, retries included** — not "6 retries", not "6 steps".
  (Spec §16, ADR-011.)
- **A `task_id` with no Pass in 6 attempts** → `task_fail` if ≥1 attempt was a completed
  legal episode (Scorer said no), else `indeterminate` (spec §11). `task_fail` is a
  teacher-generation outcome kept for analysis in `rejects/teacher.jsonl` — **not** an
  RLVR negative, **not** an SFT accept.
- **candidate cap** —
  `candidate_count = min(max_candidate_limit, max(120, ceil(40 / p * 1.5)))`, where
  `p = estimated_unique_accepted_rate` from Part B (see below) and `max_candidate_limit`
  is a config hard ceiling (e.g. 2000). Edge cases:
  - **`p ≤ 0`** → Part C does **not** start; Part B verdict becomes "not viable for
    batch qualification".
  - **`candidate_count` hits `max_candidate_limit` and 40 accepted Pass are still not
    reached** → Part C **fails**.
- **`estimated_unique_accepted_rate`** (the rate that drives sizing — **not** raw
  `accepted / attempted`). Part B must report, over its ~20-seed sweep:
  `raw_pass_rate`, `distinct_task_survival` (accepted that survive `task_id` uniqueness),
  `semantic_dedup_survival` (accepted that survive intra-family `semantic_key` dedup),
  `indeterminate_rate`. Then
  `estimated_unique_accepted_rate =
   (unique, semantic_key-deduped accepted task_ids) / (attempted task_ids)`
  measured directly over the sweep.
- **seed range** — contiguous `[seed_start, seed_start + candidate_count)`; record
  `seed_start`.
- **teacher config** — `deepseek/deepseek-v4-flash`, `reasoning_effort=low`,
  `temperature_first=0.0`, `temperature_retry=0.7`, `per_turn_timeout_s=60`.
- **run-level `indeterminate_rate` ceiling** — `indeterminate_task_ids /
  attempted_task_ids` (**not** `/ accepted`) must be **≤ 0.05** (spec §20's *v2-r1
  operational health threshold*, not a domain fact). If `attempted_task_ids < 40`, do
  not apply the ratio — report raw counts and re-run with a larger `candidate_count`.
- **max `reject`** — no hard ceiling; the reject histogram must be dominated by
  `author.*` bind reasons + `task_fail`. A large `gate.draft_invalid` /
  `gate.unachievable` share means the authoring path is broken → fail Part C regardless
  of the Pass count.

**Acceptance:**

- **40 distinct-`task_id`, `semantic_key`-deduped accepted Pass** under the config above,
  with `indeterminate_rate ≤ 0.05` and a healthy reject histogram → Part C passes.
- **Fewer than 40** (or an unhealthy histogram / high `indeterminate_rate`) →
  **acceptance failure**, not "an acceptable result at this config". Record the chosen
  fallback:
  1. a documented exception to ADR-012 (private `_bind_log_foods` path — written into
     ADR-012's Amendment log), **or**
  2. an upstream nutri-env `__all__` promotion PR (normal work, not a Batch-1 blocker),
     **or**
  3. dropping / re-sizing the 3-leg family for Batch 1 (explicit design-doc §7 change).

## Out of scope

- The full `build` pipeline, `gates.run`, materializers, teacher rollout wiring.
- Batch-2 shapes (amend / starve / closed-list).
- Generating the other ~380 Batch-1 traces.

## Acceptance evidence (ticket closes only when these exist)

- A runnable spike script/notebook covering Part A and Part B.
- Part A verdict (A / B / C / D) with the reconstruction path used.
- Part B verdict (assembles+verifies: yes/no) + the ~20-seed preliminary rate.
- If Part B passes: a Part C run with the pinned config and its result (40 reached / not
  reached + fallback).
- Spec §9.1 / §22.9 / §23 (OQ-2, OQ-7) and, if a fallback is chosen, ADR-012's Amendment
  log or design-doc §7 updated to match.

## Decisions to record on close

- OQ-2 verdict (A/B/C/D) and the public function used for single-item reconstruction.
  If B or C: the TaskPackage/RLVR schema changes needed before implementation.
- OQ-7 Part B verdict (assembles+verifies: yes/no) + the ~20-seed preliminary rate and
  `indeterminate` rate.
- Part C: the pinned config values used (`max_intents`, `seed_start`, teacher config),
  and 40 distinct-`task_id` `semantic_key`-deduped accepted Pass reached — or the
  fallback taken.
- The `k = 6` reading is **fixed** (max 6 teacher rollout attempts per `task_id`, retries
  included; spec §16). Record only if the design doc / a run shows a different intent.
- The identifier + N-counting rules are **fixed** (spec §10: `task_key` / `task_id` /
  `attempt_id`; N = distinct accepted `task_id` after intra-family `semantic_key` dedup;
  `task_fail` ≠ RLVR negative). Record only deviations found in practice.
- Part B rate components reported (`raw_pass_rate`, `distinct_task_survival`,
  `semantic_dedup_survival`, `indeterminate_rate`, `estimated_unique_accepted_rate`) and
  the `candidate_count` / `max_candidate_limit` used in Part C.
