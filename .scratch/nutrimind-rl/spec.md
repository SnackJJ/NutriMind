# NutriMind v2.0 RL — Spec

Status: **ready-for-agent**
Tracker: local Markdown (`docs/agents/issue-tracker.md`)
Domain vocabulary: `CONTEXT.md` (repo root) — used verbatim below
Governing ADRs: [ADR-010](../../docs/decisions/010-nutrimind-v2-rescope.md),
[ADR-011](../../docs/decisions/011-batch1-sft-trajectory-short-plan-thinking-teacher.md)
(protocol half superseded by ADR-014),
[ADR-012](../../docs/decisions/012-nutrienv-read-only-benchmark.md) (amended 2026-09-11),
[ADR-014](../../docs/decisions/014-native-tool-calling-v2-protocol.md),
[ADR-015](../../docs/decisions/015-v2-post-training-stack-trl-sft-verl-rl.md)
Sibling spec: `.scratch/nutrimind-v2/spec.md` (Data Factory)

> **Scope**: the post-SFT **RL** stage of the v2.0 student — GRPO (default) and DAPO
> (comparison arm) on verified tasks, with measured difficulty and a shared native
> tool-calling rollout.
>
> **Sibling spec**: `.scratch/nutrimind-v2/spec.md`. This spec **consumes**
> `TaskPackage` and `rlvr/<task_id>.json` and never authors tasks. Teacher collection
> stays in the factory (ADR-014): same lab loop and schema, factory-owned tickets.
>
> **Deferred**: OPD (sequence-level DAgger and token-level GKD), localization /
> counterfactual replay, GiGPO, DPO, PPO.
>
> **Pilot query budget (2026-09-12):** `.scratch/nutrimind-pilot/spec.md`. RL starts
> from ~200 unique query identities. OPD's unique-query budget is 0 until selection
> from student-induced failure states. Binary Pass/Fail reward is unchanged. A staged
> reward / credit-assignment roadmap is not this spec.

## Problem Statement

The v2.0 Data Factory spec ends at `sft/train.jsonl` and an RLVR export. Everything
after SFT was unspecified: how the student is rolled out inside NutriEnv, which tasks
are worth training on, what the reward is, and how any of it is scored. Without a spec,
the RL stage would be rebuilt from v1 habits — which is the failure mode ADR-010 exists
to prevent.

The v1 RL phase failed for architectural reasons: a stateless tool-responder, a
syntactic rubric with no verifier, a dispatch that fell through on every row, and
difficulty that was never measured (`"medium"` on the whole pool). The model that ran
2700 steps into a single-tool-call shortcut was optimizing exactly what that reward
asked for.

Two facts change what is now possible. A **native tool-calling harness** exists in
`nutri-env-lab` and has been run against the real teacher. The reward is a **verifier**
(`Scorer`) over a pinned oracle: `{passed, tag, sub_tags}`.

What was missing is the specification that turns those into a stage that can be built
test-first, that cannot silently degrade, and whose numbers mean something.

## Solution

Specify RL as a **consumer of TaskPackages**, sharing **one** new seam: a student
rollout driver whose model endpoint is injected. The driver reuses the lab's FC episode
loop and schema; it does not reimplement them. The same driver serves student RL
rollouts, difficulty measurement, and the exam comparator. Teacher collection uses that
loop from the Data Factory, not from this spec.

**RL**: veRL GRPO over verified tasks with a binary Pass reward. Task difficulty is
**measured** (`p̂` from k rollouts at a checkpoint hash), not labelled. Zero-variance
groups and `indeterminate` episodes are dropped at train time, not patched with a
stale static pre-filter. DAPO is one comparison arm behind the same rollout and
reward.

Both arms report into the factory's manifest discipline, and every arm asserts its
configuration at startup.

## User Stories

### Rolling the student out

1. As the maintainer, I want a single function that rolls a policy out inside NutriEnv
   for k episodes on one TaskPackage and returns `EpisodeResult`s, so that RL,
   difficulty measurement, and evaluation all consume the same artifact type.
2. As the maintainer, I want the policy to be injected as an endpoint specification, so
   that the student and the exam comparator are the same code path.
3. As the maintainer, I want that path to reuse the lab FC loop and tool schema the
   factory uses for the teacher, so that "the student is measured the way the teacher
   was" is structural.
4. As a test author, I want the rollout driver to accept a scripted policy, so that a
   full multi-turn FC episode can be driven deterministically with no network.
5. As the maintainer, I want a rollout that hits the step budget without an explicit
   finish to be recorded as such, so that truncation is visible rather than blended
   into failure.

### Native tool calling

6. As the maintainer, I want the student trained and evaluated under native tool
   calling, so that the protocol at test time is the protocol the student was trained
   on.
7. As the maintainer, I want the tool schema and the system prompt to come from one
   shared lab definition, so that train and eval cannot diverge in tool naming or
   argument shape.
8. As a test author, I want an assertion that the prompt sent during training is
   token-identical to the prompt sent at eval time for the same TaskPackage.
9. As the maintainer, I want `parallel_tool_calls=false` in train and eval, so that a
   turn is one tool call, matching the lab's serial-FC leaderboard.
10. As the maintainer, I want a turn in which the model returns no tool call to be
    recorded distinctly, so that "talked instead of acting" is not conflated with a
    legal action.

### Selecting tasks for RL

11. As the maintainer, I want the student's pass rate on every training task measured
    from k rollouts, so that task difficulty is a number rather than a label.
12. As the maintainer, I want the measured pass-rate distribution reported per family,
    so that a global band cannot silently select the curriculum away from the exam's
    wall families.
13. As the maintainer, I want zero-variance groups detected and dropped at train time,
    so that compute is not spent on groups that produce no gradient.
14. As the maintainer, I want `indeterminate` episodes excluded from `p̂` and from
    advantage, so that oracle noise is not treated as Fail or as zero-variance.
15. As the maintainer, I want difficulty re-measured whenever the policy checkpoint
    changes, so that a stale band cannot silently empty the training set.
16. As the maintainer, I want the measured pass rate stored with the checkpoint hash it
    was measured against.

### Reward and advantage

17. As the maintainer, I want the RL reward to be derived from the verifier's tri-state
    result, so that no reward term can be computed from a signal the oracle did not
    produce.
18. As a test author, I want a test asserting that the reward function reads no field
    other than the verification result.
19. As the maintainer, I want a group with fewer than two valid (`pass`/`fail`) samples,
    or with zero reward variance, dropped and counted in the effective-gradient
    fraction denominator, without resampling inside the group.
20. As the maintainer, I want GRPO and DAPO switchable behind the same reward and the
    same rollout, so that the comparison isolates the algorithm.

### Evaluation and isolation

21. As the maintainer, I want every reported number to name the exact exam revision it
    was measured on.
22. As the maintainer, I want the exam file verified byte-identical to the pinned
    published v1.0 before any run, aborting if the working tree differs.
23. As the maintainer, I want training, difficulty measurement, and early stopping to
    read only `TRAIN_ROSTER` tasks, so that the 63-task exam is never in the loop.
24. As the maintainer, I want `mini-exam val` to remain the only checkpoint-selection
    signal.
25. As the maintainer, I want the student reported with-reasoning and tools-only.
26. As the maintainer, I want pass@1 and pass@k reported together.

### Experiment discipline

27. As the maintainer, I want every arm to print and assert its reward version,
    reference-model revision, task-selection policy and band, advantage estimator,
    rollout k, parallel-tool policy, and exam revision at startup.
28. As the maintainer, I want an arm whose assertion fails to abort before spending
    rollout compute.
29. As the maintainer, I want each arm's metrics written to a manifest with the same
    provenance fields the Data Factory uses.
30. As the maintainer, I want the effective-gradient step fraction reported per arm.

## Implementation Decisions

### D1 — Pin `nutri-env-lab`, read-only (ADR-012 amended)

v2 depends on `../nutri-env-lab` at
`0ee68eaa6c246e8079915761c95fc986c53d4979` (FC harness committed). Read-only still
holds: the lab is consumed, never patched. NutriMind wraps the loop with an injected
policy; it does not copy `run_episode_tool_call` into this repo.

### D2 — Native tool calling (ADR-014)

Every stage speaks FC. Schema and system prompt come from the lab. `parallel_tool_calls`
is false. A no-tool-call turn is recorded as such, not coerced into an op.

This supersedes ADR-011's text-op shape. Short plan remains as truncated
`reasoning_content`.

### D3 — The seam: one injected student rollout driver

```
student_rollout(policy_spec, task_package, *, k, seed) -> list[EpisodeResult]
```

- `policy_spec` is an endpoint handle (url, model, api key, temperature,
  `parallel_tool_calls=false`).
- The driver **reuses** the lab episode loop and tool schema.
- It returns the factory `EpisodeResult`, extended for FC (`tool_calls`,
  `tool_call_id`; `executed_op` is still what `NutriEnv.step` received;
  `reasoning_content` is the truncated plan). Verifier still depends only on
  `end_state`.

Consumers: RL rollouts, difficulty measurement, the exam harness.

Teacher collection is **not** this seam's job (factory tickets). If a second seam is
proposed, it must beat injection of a scripted policy at this one.

### D4 — TaskPackage ↔ Task adaptation

The lab runner takes a `Task`; inputs are `TaskPackage`s. Adaptation is factory-side
(lossless round-trip already asserted) and is exercised through the same
injected-policy test as D3.

### D5 — Reward is the verifier, and nothing else

`pass → 1.0`, `fail → 0.0`, `indeterminate → excluded`. No proxy terms (length, tier,
format, tool-call validity as a reward). No string dispatch with a silent default:
closed tables, unknown raises.

`indeterminate` is not Fail and is not "zero variance". It is excluded from `p̂`
(numerator and denominator) and from the GRPO/DAPO group.

### D6 — Difficulty is measured per checkpoint; zero-variance groups are dropped

- `p̂(task, checkpoint) = n_pass / (n_pass + n_fail)` from k rollouts. `indeterminate`
  counted separately, not in `p̂`.
- Sweet-spot band is configuration, **per family**.
- Rows stored with the checkpoint hash; a band for one policy is never applied to
  another.
- Default: **dynamic sampling**. Tasks outside the band are not trained this round.
  At train time, a group with `< 2` valid samples or zero reward variance is **dropped**
  (counted in the effective-gradient fraction denominator). Do **not** resample inside
  the group to manufacture variance; draw the next batch of tasks.

### D7 — Advantage estimator: GRPO default, DAPO comparison (ADR-015)

Both consume the identical rollout and reward. veRL is the RL trainer. GiGPO, DPO, and
PPO are out of this spec's first phase.

### D8 — Exam contract

Evaluation is on the frozen published v1.0 exam. The runner verifies the exam file is
byte-identical to the pin and **fails before any rollout** if not. No v1.1 in this spec.

Training / difficulty / mini-exam val / optional internal held-out test are
`TRAIN_ROSTER` only and disjoint from the 63.

### D9 — Every arm asserts its own configuration

On startup: reward version, reference-model revision, task-selection policy and band,
advantage estimator, rollout k, `parallel_tool_calls`, exam revision. Mismatch aborts
before rollout spend.

### D10 — Framework (ADR-015)

RL = veRL (GRPO / DAPO) behind D3. SFT = TRL `SFTTrainer` (factory / SFT tickets), same
FC template. v1 `train_grpo.py`, `gigpo_trainer.py`, and TRL `environment_factory`
Python-method tools are archive.

### D11 — OPD is deferred

Sequence-level DAgger (remote teacher API) and token-level GKD (hosted larger
same-family teacher) are future work. This spec does not specify localization, fork
state, or correction records. `p̂ ≈ 0` tasks are counted and reported, not consumed
here.

## Testing Decisions

**What makes a good test here**: only external behaviour at the seam, and purity
everywhere below it. A scripted policy drives a full multi-turn FC episode with no
network. Everything downstream is a pure function of
`(TaskPackage, list[EpisodeResult])`.

- **Rollout driver** — scripted policy + TaskPackage → deterministic `EpisodeResult`
  list; k episodes independent; step-budget exhaustion distinct from finish; error
  observation preserved; no-tool-call turn distinct; a second tool_call in one turn is
  not executed (`parallel_tool_calls=false`).
- **Prompt construction** — token-identical train vs eval for the same TaskPackage.
- **Reward** — tri-state mapping; `indeterminate` excluded; no proxy field on the
  dependency surface.
- **Difficulty** — `p̂` arithmetic excluding `indeterminate`; per-family aggregation;
  band refused on checkpoint-hash mismatch; a zero-variance group is dropped not
  resampled in-place.
- **Arm assertions** — mismatched configuration aborts before any rollout.

**Prior art**: Data Factory seam tests (`test_rollout.py` scripted queue;
`test_build_sft.py` injected teacher; `test_gates.py` pure tables). Same injected-policy
pattern; the loop is the lab FC loop, not `ReActHarness`.

**Seam count**: one new. No test may require a second injection point.

## Out of Scope

- **OPD** (sequence-level DAgger, token-level GKD, localization, counterfactual replay).
- **GiGPO, DPO, PPO** as first-phase algorithms.
- **Parallel tool calls** as a training or headline-eval setting.
- **The SFT trainer implementation** (ADR-015; factory / SFT tickets).
- **Batch 2** and any authoring change.
- **Changing the Data Factory's gates, families, or the §7 mix** — except the protocol
  switch the factory spec records under ADR-014 (new factory tickets, not this spec).
- **Deployment, serving, and the product surface.**
- **The lab's own benchmark suite** — consumed, not owned.
- **Declaring a v1.1 exam.**

## Further Notes

### ADRs

- **ADR-014** — native tool calling (this spec's protocol).
- **ADR-015** — TRL SFT + veRL GRPO/DAPO.
- **ADR-012 amended** — pin `nutri-env-lab@0ee68ea`.
- **ADR-011** — teacher / plan / length kept; text-op superseded.
- **ADR-007** — v1 only.

### Vocabulary this spec uses (CONTEXT.md)

student rollout · sweet-spot band · arm · effective-gradient fraction · tool call ·
plan (`reasoning_content`) · v1.0 (exam) · TRAIN_ROSTER · mini-exam val · Pass ·
unique query identity · pilot query budget

### The v1 four, restated

| v1 failure | Rule here |
|---|---|
| Reward had no verifier | D5 — projection of the verifier, dependency-surface test |
| Dispatch fell through | D5 — closed tables, unknown raises |
| Environment had no state | Pinned NutriEnv is deterministic and steppable |
| Difficulty was never measured | D6 — `p̂` per checkpoint hash; zero-variance groups dropped |
