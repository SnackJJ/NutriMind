# ADR-013: Failure-recovery coverage as a required property of SFT data

- **Status**: proposed
- **Date**: 2026-09-10
- **Deciders**: zeqing
- **Ticket**: 022 (cut with this ADR)

## Context

ADR-009 diagnosed v1's shortest-path collapse; ADR-010 answered it by giving v2 a
**verifiable** world — a `TaskPackage` pins `s0`, the `Oracle` pins the end state, and the
tri-state verifier (spec §12) replaces v1's weighted syntactic rubric. That fixes **what we
score**.

It does not fix **which states the student ever sees**. Batch-1's keep rule is the
Pass-filter: a teacher trajectory is retained iff the end state Passes (CONTEXT.md,
ADR-011). The teacher is `ark/deepseek-v4-flash`; a strong teacher's Pass trajectories are
almost by definition *clean* — they contain no `ActionError` turn. So the v2 corpus
inherits a structural blind spot that has nothing to do with the reward.

The environment does produce those states, and the student will meet them.
`NutriEnv.step` returns `{ok: False, observation: None, error: exc.as_dict(), done: False}`
with **the world unchanged** (`nutrienv/env/nutri_env.py:63-71`), and the v2 rollout driver
feeds that error back as the next observation (`data_factory/rollout.py:245`). Deployment's
most common non-trivial state is exactly "the last observation was an error".

In an agent loop this is compounding error, not ordinary exposure bias: one illegal action
puts every subsequent turn off-distribution, and the episode runs on to the step budget. A
student that has never seen an error observation has no prior for the rest of the episode.

Measured on v1: the GRPO prompt pool carried an `error_recovery` bucket (66 of 1597
attempted prompts), and `TIER_TOOL_MAPPING_V2` had an `error-recovery` row — the *prompts*
existed, but there was no SFT data in which recovery is demonstrated, and the reward could
not have scored it (ADR-009's outcome component fell through to its syntactic fallback on
1597/1597 rows).

## Decision

**Batch-1, and every later SFT batch, must contain a deliberate and measured fraction of
recovery-positive traces.**

1. **Definition — recovery-positive.** An accepted trace is recovery-positive iff

   - at least one `TurnMeta.observation` carries a **semantic** error code — one of
     `unknown_food`, `implausible_quantity`, `bad_index` — **and**
   - the episode's `VerificationResult.status == "pass"`.

   `bad_schema` and `unknown_op` do **not** count. Those are syntax repairs the student
   already learns from ordinary SFT; counting them would inflate the metric while adding no
   new behaviour.

2. **Target band — 15–25 % of accepted traces**, reported as
   `recovery_fraction = recovery_positive / accepted`. *Provisional* (the same provenance
   class as `over_generate_x` in `configs/data_factory.yaml`): the lower bound is "enough to
   establish a prior"; the upper bound keeps repair from dominating the corpus and teaching
   the student to *seek* errors. Re-pin after Batch 1 measures the natural rate.

3. **Recovery is an attribute, not a family.** It does not appear in the §7 family mix and
   gets no `target_n` of its own. A family determines the **oracle shape**; an error trap
   does not change the end state — `NutriEnv.step` leaves the world unchanged on an
   `ActionError` — so the §5 gate policy and `check_achievable` are untouched. Recovery
   composes with every family.

4. **Authoring.** The trap is chosen at the `intent` stage (pure code, spec §4.1), before
   any LLM call: pick a task shape whose *natural first action* is illegal but *recoverable*
   — e.g. a spoken food that `search_foods` cannot resolve directly, or a portion that trips
   `implausible_quantity`. `check_achievable` must still Pass: the oracle is reachable
   *despite* the trap.

5. **Metric and gate.** `recovery_fraction` is written to `run_manifest.json` (spec §9.5,
   §20) and surfaced by the dry-run manifest health check (ticket 015). Out of band is a
   **health warning, not a hard fail** — Batch 1 is throughput-limited, and the correct
   response to a shortfall is over-generation, never a lower `counts.accepted`.

6. **Syntax errors are handled by measurement and constraint, not by injection.**
   `bad_schema` / `unknown_op` are *protocol* failures, and ordinary SFT already covers that
   ability densely — every assistant turn of every accepted trace is a positive example of a
   well-formed op. Injecting malformed ops on purpose would spend corpus on an ability the
   corpus already teaches. So:

   - `metrics.schema_error_rate` is reported **separately** (spec §20) as the student's
     protocol-mastery signal. A high rate is a *curriculum* problem for the SFT stage, not a
     recovery problem for the corpus.
   - A syntax error followed by a successful recovery is recorded but **not** counted toward
     `recovery_fraction`: it cannot distinguish "the student learned to repair" from "the
     student sampled one bad token".
   - The durable fix for the class is **constrained decoding on the op**, not more data — an
     error a grammar can make impossible should not be taught with examples. ADR-011's shape
     (free-text plan, then one JSON op, `_parse_action` taking the first JSON object) bounds
     how far the constraint can reach without changing the protocol; scope any
     constrained-decoding work against that bound.

## Consequences

### Positive
- Closes the one state-distribution gap a *verifier* cannot close. ADR-010 bought "we can
  tell right from wrong"; this buys "the student has seen wrong".
- Backs the RL `error_recovery` family (v1's 66-prompt bucket) with SFT data instead of
  leaving it a family the student has no prior for.
- Gives the data factory a measurable *corpus property*, not just a yield number — the first
  v2 metric that describes the data rather than the run.
- Cheap to measure: both halves of the predicate are already recorded
  (`TurnMeta.observation`, `EpisodeResult.verification`). No new plumbing.

### Negative
- Lower teacher Pass rate on trapped tasks → lower yield against the ~420 target and
  `usd_budget`. Must be paid by over-generation.
- The teacher usually recovers in one turn, so the signal per trace is thin. Requiring ≥1
  recovery rather than several is deliberate; forcing the teacher to *not* avoid the trap
  needs prompt-side control and is out of scope here.
- Adds an acceptance property to tickets 012/013/020, which were cut against the old
  criteria. The change is additive, but those tickets need a re-read.

### Neutral
- Does not change the trajectory shape, the plan truncation, `max_seq_tokens`, the RLVR
  export (ticket 018), or exam isolation. Recovery-positive tasks are ordinary train tasks
  that obey the same gates.
- 15–25 % is a judgment, not a measurement; the first Batch-1 run replaces it with a number.

## Alternatives considered

- **A sixth family, `error-recovery`.** Rejected — a family *is* an oracle shape (spec §5,
  §22.9). A trap changes the path, not the end state, so a family would fragment the §2.1
  mix, need its own gate policy, and have to be duplicated across every other family to
  cover the shapes.
- **Splice a synthetic error observation into a passing trace.** Rejected — the observation
  must be produced by the world. A spliced error teaches the student to react to a string
  that cannot follow that prefix: a distributional artefact RL then has to unlearn.
- **Leave it to RL / OPD.** Rejected as the *sole* answer. RL under a binary verifier gives
  no per-step credit for recovery — that is exactly the credit-assignment problem GiGPO
  exists for. On-policy distillation from the student's own failure state is the
  *complementary* mechanism, and it needs recovery-positive teacher traces to distil from.
  Neither substitutes for the SFT corpus.

## Open questions

- Which traps are authorable through `generate_one`'s public API alone (ADR-012 read-only),
  without patching nutri-env.
- The **natural** recovery rate: measure how many teacher Pass trajectories already contain
  a semantic error before forcing anything. If it is already ≥15 %, this ADR reduces to
  "measure and report" and no authoring change is needed.
- Whether the student's inference-time error rate tracks the training fraction — a corpus
  property is only useful if it moves the deployment distribution.

## Related

- [ADR-009](009-grpo-reward-redesign-against-shortest-path-collapse.md) (v1 reward collapse),
  [ADR-010](010-nutrimind-v2-rescope.md) (v2 re-scope),
  [ADR-011](011-batch1-sft-trajectory-short-plan-thinking-teacher.md) (trajectory shape,
  Pass-filter), [ADR-012](012-nutrienv-read-only-benchmark.md) (nutri-env read-only)
- spec §2.1, §4.1, §9.5, §12, §19.4, §20; tickets 015 (manifest metrics) and 022
- `nutrienv/env/nutri_env.py` (an illegal action leaves the world unchanged),
  `src/training/data_factory/rollout.py:245`, `src/training/data_factory/concepts.py`
  (`TurnMeta`, `EpisodeResult`)
