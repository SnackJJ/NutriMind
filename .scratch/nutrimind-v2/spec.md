# NutriMind v2.0 Data Factory — Spec

Status: draft for review · not for implementation
Tracker: local Markdown (`docs/agents/issue-tracker.md`)
Domain vocabulary: `CONTEXT.md` (repo root) — used verbatim below
Governing ADRs: [ADR-010](../../docs/decisions/010-nutrimind-v2-rescope.md),
[ADR-011](../../docs/decisions/011-batch1-sft-trajectory-short-plan-thinking-teacher.md),
[ADR-012](../../docs/decisions/012-nutrienv-read-only-benchmark.md) (amended 2026-09-08)
Design doc: `docs/plans/nutrimind_v2_data_factory.md` — the `/tmp` original is lost;
reconstructed by ticket 021 from §2.1 + the ADRs + ticket 002
Open decision/prototype tickets: `.scratch/nutrimind-v2/issues/001`, `002`

This spec turns the finalized design into a verifiable local project spec. It pins the
**build** boundary, the canonical **TaskPackage**, the SFT and RLVR flows that derive
from it, the tri-state verification contract, the rubric layering, the v1/v2 boundary,
and the test seams. It does not implement anything.

---

## 1. Problem Statement

We are starting **NutriMind v2.0**: a Qwen3.5-2B student trained and scored on the frozen
**v1.0 (exam)**. The student needs Pass-filtered SFT **trajectory** data, and later RLVR
tasks, both derived from the same authored task material. The **NutriMind v1** collection
code targets a different base model, a different action space (Qwen-native `<tool_call>`
over 6 tools), and a mock evaluator — none of it applies. Today `nutrienv` is not even
importable in this repo. The design as discussed still has gaps in: the boundary between
a data factory and an experiment runner; a canonical task artifact that SFT and RLVR can
both consume; the oracle's return contract (bare bool loses the "the oracle could not
decide" case); rubric scope; failure taxonomy; the loss-mask storage decision; and the
v1/v2 project boundary.

## 2. Goals

- A **build** step that is a *task/data artifact factory*: from **intents** it authors
  tasks, validates them, materializes a canonical **TaskPackage**, and — per target —
  emits an SFT record, an RLVR task export, or an evaluation task.
- A **canonical TaskPackage** as the single source for SFT, RLVR, and evaluation
  artifacts. SFT trajectories and RLVR tasks are different artifacts derived from it,
  never reverse-inferred from each other.
- A tri-state verification contract: `pass` → reward 1.0, `fail` → reward 0.0,
  `indeterminate` → reward `null` (excluded from normal reward statistics).
- A two-layer rubric: a small **hard contract** fixed before bulk generation, and a
  **soft rubric** that is diagnostic-only in v2.0.
- **Batch 1** ≈ 420 accepted **Pass** SFT traces at the design-doc §7 family mix, from
  off-exam tasks on the same world / `catalog_sha` / `Scorer` / ReAct-v2 as the exam.
- A frozen **mini-exam val** (30 fresh `TRAIN_ROSTER` tasks, oracle-verified, not
  teacher-rolled).
- Two injected external dependencies (**expander**, **teacher**), never implicitly
  constructed in core logic.
- Full provenance (versions of oracle, rubric, reward, environment, catalog, schema,
  code) on every record and in a `run_manifest.json`.
- Test seams that assert external behaviour and do not freeze internal `nutrienv`
  helpers as public API.
- A v2 SFT data loader that is separate from v1's, with the v1 loader untouched.

### 2.1 Design provenance (copied here — do not depend on `/tmp`)

Key numbers below come from the finalized design doc that was at
`/tmp/nutrienv_student_data.md` and is now **lost** (ticket 021 reconstructs it at
`docs/plans/nutrimind_v2_data_factory.md` from this section + the ADRs + ticket 002).
Recorded here so this spec is self-sufficient regardless:

- Batch-1 overall target ≈ **420 accepted Pass traces** (design §7 table column
  "Pass traces"; total row "~420"). Family mix: composite ~57 % / recommend ~17 % /
  evaluate ~13 % / log ~10 % / update ~3 %.
- `composite update+log→recommend` (3-leg) family target = **40**, teacher **k = 6**.
  **N = 40 is an accepted-Pass count, not a candidate count.** Nothing in the code or an
  existing spec contradicts this; do not reinterpret it.
- Teacher = `ark/deepseek-v4-flash` on `api/plan/v3`, `thinking: {"type": "enabled"}` as
  the length control (ADR-011 **amended 2026-09-09** — was `deepseek/` direct +
  `reasoning_effort=low`). Expander = `ark/deepseek-v4-flash` with
  `thinking: {"type": "disabled"}`. One provider, one credential (`ARK_API_KEY`) for both.
  `max_seq_length` 20k, `context_limit=None` train + eval.
- The design doc's §5 still sketches the 3-leg assembly via **private** helpers
  (`_update_from_template` / `_bind_log_foods`) and a `compose3.py`. That is the
  pre-spec draft. **This spec and ADR-012 (amended) require public symbols only**; the
  final approach is decided by ticket 002. When the design doc moves into `docs/plans/`,
  reconcile its §5/§8 to match.

## 3. Non-goals / Out of Scope

**build does not** train models, start an RL loop, run experiment sweeps, compare
models, manage checkpoints, aggregate full experiment statistics, or decide training
hyper-parameters. Those belong to an **experiment / training / evaluation runner**
(`src/training/sft/train.py` for v1; a future v2 trainer and GRPO/eval runners), which
is out of scope here.

Also out of scope:
- The v2 SFT training run and the v2 trainer itself (the *loader contract* is in scope;
  the loader and trainer implementation are not).
- GRPO / GiGPO / reward *shaping* / 0.8B.
- **Batch 2** shapes (`amend_meal→recommend`, starve-refuse+recommend, closed-list
  `allowed_food_ids`) and full-length CoT distillation (gated on the B3 diagnostic,
  ADR-011).
- Any change to `../nutri-env` (ADR-012).
- Any change to v1 artefacts, v1 code paths, or v1 data files.
- `evaluate+recommend` composite (`generate_one` returns
  `Rejected("", "unfit_substitute")`).
- Complex graded reward, the full soft rubric, advanced reward shaping, path-quality
  scoring as a training signal, distributed generation, a multi-provider unified model
  adapter, advanced run recovery. Deferred; not added now for a future maybe.
- Persona-ratio literature citation (Open Question; provisional 65/20/15
  everyday/gym/cut does not block authoring).

## 4. Solution Overview

### 4.1 build vs runners

```
build  =  task / data artifact factory          (this spec)
           author → validate → materialize TaskPackage → per-target materialize

runner =  experiment / training / evaluation     (out of scope)
           consumes SFT records / RLVR tasks / eval tasks; trains; sweeps; compares
```

`build` has one top-level entry, `build(config, *, expander, teacher_complete)`, which
is a **thin orchestrator**:

```
build
├── author task            (intent → nutrienv Task, or public-API composition)
├── validate task          (gates.run; ordered, first failure wins)
├── materialize TaskPackage (canonical; env reconstruction + oracle + reward semantics + termination + provenance)
├── [target=sft]  collect teacher trajectory → validate trajectory → serialize v2 SFT record
├── [target=rlvr] export RLVR task (env reconstruction + verifier config + reward adapter spec)
└── [target=eval] export evaluation task (--freeze-mini: frozen TRAIN_ROSTER mini-exam)
```

`--stop-after author` and `--stop-after gate` are debug / staged-artifact capabilities
(write `tasks/` or `task_packages/` and stop), **not** a second pipeline.

Rules:
- **Single task failure** → record a `reject` or `indeterminate` outcome and continue
  with the other tasks.
- **Config error, schema error, or a dependency that cannot be imported** → the whole
  run fails immediately (non-zero exit, partial `run_manifest.json` written).
- Every stage has an explicit input, output, and on-disk boundary (§8, §9).
- Retry policy is explicitly attributed: **teacher** call retries belong to the
  `teacher_complete` adapter's caller inside `build` (attempt 1..k, §16); transport
  retries (HTTP) belong to the LLM client the adapter wraps; `build` itself does not
  retry gates or serialization (those are deterministic — a failure is a real defect).

### 4.2 TaskPackage → materializers

```
                       TaskPackage (canonical)
                      /        |         \
          SFT materializer  RLVR materializer  evaluation materializer
                 |                |                    |
          v2 SFT record     RLVR task export      eval task
```

The **TaskPackage** is authored once. An SFT trajectory and an RLVR task are **siblings**
derived from it. Do not build an RLVR task from an SFT trajectory, and do not synthesize
an SFT trajectory from an RLVR task. A teacher **trajectory** may serve as SFT data, a
reference trajectory, oracle/environment test data, or an evaluation baseline — but it is
never a required part of an RLVR task. In particular, a teacher episode that **failed**
(`status=fail` / `task_fail`) is **not** an RLVR negative sample: RLVR negatives are
produced only by the RLVR runner executing a *model* rollout and the verifier returning
`fail`. Teacher non-Pass episodes are SFT-reject / analysis material (EEF / SRFT / DPO
candidates) only.

**Unresolved (P0, ahead of module-naming questions):** whether the TaskPackage is truly
**self-contained** for RLVR depends on being able to reconstruct one frozen environment
item to a runnable `NutriEnv` through a *public* nutri-env entry. The candidate path
(`freezer.task_to_item` to serialize; a public single-item reconstruction; `NutriEnv().reset`)
is **not verified** — `load_split` reads a file of items, not one in-memory item. If
public single-item reconstruction does not exist, the RLVR runner would depend on an
external dataset file and the package is not self-contained. Ticket 002 resolves this.

### 4.3 Per-task pipeline

`author` → `validate` → `materialize TaskPackage` → (`target` branch). For `target=sft`:
`teacher rollout` → score with the tri-state **verifier** → `serialize`. `author` for
shapes `generate_one` supports directly calls `generate_one`; for 3-leg
`update+log→recommend` it composes **public** `nutrienv` symbols (ticket 002 spike).
Teacher rollout is a `nutrienv.harness.ReActHarness` subclass whose completion is the
injected `teacher_complete`, capturing `reasoning_content` per turn.

## 5. User Stories

1. As the maintainer, I want one `build` command to produce a Batch-1 SFT set, so that I
   can cold-start the Qwen3.5-2B student without hand-authoring traces.
2. As the maintainer, I want `build` to refuse to run when the catalog SHA does not
   match the exam's, so that train and exam never diverge on world facts.
3. As the maintainer, I want every authored task materialized as a canonical
   **TaskPackage**, so that SFT and RLVR consume the same task definition.
4. As the maintainer, I want an RLVR task exportable from a TaskPackage **without** a
   teacher trajectory, so that RL does not inherit a teacher dependency.
5. As the maintainer, I want the verifier to return `pass` / `fail` / `indeterminate`
   with `failure_codes` and `evidence`, so that an oracle problem is not miscounted as a
   model failure.
6. As the maintainer, I want `indeterminate` outcomes kept out of the normal
   pass/fail statistics, so that oracle noise does not create false negatives in the
   reward signal.
7. As the maintainer, I want a hard-contract rubric fixed before bulk generation and a
   soft rubric that is diagnostic-only, so that an unvalidated quality heuristic never
   fails a valid answer.
8. As the maintainer, I want binary reward in v2.0 (`1.0` / `0.0` / `null`), with graded
   reward explicitly deferred and versioned, so that Batch-1 data is not tied to an
   unproven reward shape.
9. As the maintainer, I want `oracle_version`, `rubric_version`, `reward_version`,
   `environment_version`, `catalog_sha`, `task_schema_version`, `seed`, and (if allowed)
   `git_sha` on every record, so that a later rubric change can re-evaluate old data
   instead of silently reinterpreting it.
10. As the maintainer, I want failed **teacher** traces kept with a stable reason code,
    so that I can mine them later and diagnose authoring bugs.
11. As the maintainer, I want a per-run `run_manifest.json` with counts, cost, and
    provenance, so that I can tell whether a run succeeded without opening the data.
12. As the maintainer, I want the accepted family mix reported against the target, so
    that composite is not silently under-represented.
13. As the maintainer, I want a `--dry-run` that authors and gates without the teacher,
    so that I can see projected accept counts and reject reasons cheaply.
14. As the maintainer, I want to re-run after a serializer fix and reuse cached teacher
    episodes, so that I do not pay for the teacher twice.
15. As the maintainer, I want an interrupted run to leave the previous outputs intact and
    no half-written JSON lines, so that a crash is safe.
16. As the maintainer, I want `TRAIN_ROSTER` people whose derived windows are provably
    disjoint from nutri-env `ROSTER`, so that template-family oracles cannot collide
    with the exam.
17. As the maintainer, I want any generated query matching one of the 63 exam queries
    verbatim (normalized), or sharing an exam task's `semantic_key`, to be dropped.
18. As the maintainer, I want `amount_path` chosen from the persona, so that speech
    realism matches who is speaking.
19. As the maintainer, I want the **expander** and **teacher** injected, so that tests
    run offline and deterministically.
20. As a test author, I want `gates.run`, the verifier, and `serialize` to be pure
    functions, so that I can assert outcomes without a network.
21. As a test author, I want a synthetic **expander** and a scripted `teacher_complete`,
    so that I can drive Pass / Fail / indeterminate episodes through `build`.
22. As the maintainer, I want v2 SFT records to carry explicit `segment` and `train_on`
    semantics (not a tokenizer-bound token mask), so that the v2 loader derives the
    token-level mask after applying the v2 chat template.
23. As the maintainer, I want a separate v2 SFT loader with the v1 loader untouched, so
    that v1 and v2 schemas and loss semantics never mix.
24. As the maintainer, I want v2 outputs under a new directory with a `schema_version`,
    so that v1 artefacts are never read or overwritten.
25. As the maintainer, I want a documented rollback (point the trainer back at the v1 SFT
    set), so that a v2 failure does not strand the project.
26. As the maintainer, I want per-`failure_code` reject histograms, so that I can tell an
    expander problem from a teacher-capability wall from an oracle problem.
27. As the maintainer, I want a cost budget with a soft warning and optional hard stop.
28. As a security reviewer, I want confirmation that only synthetic data leaves the
    machine.
29. As the maintainer, I want `build` to keep processing after a single task fails.
30. As the maintainer, I want the compatibility guard to protect only the public borrowed
    `nutrienv` API, so that an upstream break fails CI without freezing internal helpers.
31. As the maintainer, I want `nutrienv` importable and a minimal public-API smoke test
    green **before** any implementation work starts.
32. As the maintainer, I want the 3-leg composite treated as a feasibility spike, not a
    settled fact, until a public-API composition is shown to assemble and verify.

## 6. End-to-end Workflow

1. Load `configs/data_factory.yaml`. Load the catalog via `load_catalog`; assert
   `catalog_digest(catalog)` equals the exam split's `catalog_sha256`; else abort the
   run.
2. Assert the installed `nutrienv` package rev equals `config.nutrienv_rev`; else abort.
3. Load the 63 exam `Task`s once (`nutrienv.bench.load_exam`); precompute their
   normalized queries and `semantic_key`s.
4. Enumerate **intents** per family up to its target `N` × the over-generation
   multiplier. Each intent fully specifies
   `(family, TRAIN_ROSTER person, seed, occasion, scene, shell/slots, amount_path,
   knife, steps, tier)` plus a deterministic `task_id`. Sort by `task_id`. Write
   `intents/<family>.jsonl`.
5. For each intent (skip if `task_id` already terminal in this output dir unless
   `--force`):
   a. **author** → a `nutrienv` `Task`, or an `author`-stage reject.
   b. **validate** (`gates.run`) → keep, or a `gate`-stage reject.
   c. **materialize TaskPackage** → `task_packages/<task_id>.json` (§9). `--stop-after`
      {`author`|`gate`} stops here.
   d. `target=sft`: if `rollouts/cache/<task_id>.json` exists, load it; else run the
      teacher rollout (attempt 1 at temperature 0.0; attempts 2..k at 0.7) and write the
      episode atomically. Then run the **verifier** (§11–12) on the end state →
      `VerificationResult`.
      - `status="pass"` → **serialize** → `sft/train.jsonl`.
      - `status="fail"` → `rejects/teacher.jsonl` (`failure_code="task_fail"`; a completed
        legal teacher episode that missed the hard contract; stored with all `k` attempts
        as an analysis candidate — **not** an RLVR negative, §11).
      - `status="indeterminate"` → `rejects/indeterminate.jsonl` (no completed legal
        attempt at all — teacher error / no-finish / invalid-op — or oracle/env
        exception / unreachable oracle).
   e. `target=rlvr`: **export RLVR task** (§9) from the TaskPackage. No teacher.
   f. `target=eval` / `--freeze-mini`: `check_achievable` only, then emit the frozen
      task; no teacher, no serialize.
6. Sort accepted SFT records by `task_id`; write `sft/train.jsonl` via temp + atomic
   rename; split 7:2:1 by a `task_id` hash into `train` / `holdout` / `loss_val`.
7. Write `run_manifest.json` (temp + rename).

## 7. Input Contract

- **`configs/data_factory.yaml`** (new). Keys (names indicative, shapes fixed):
  `nutrienv_rev`, `catalog_path`, `exam_split_path`, `target` (`sft`|`rlvr`|`eval`|`all`),
  `teacher` (`{model_id, thinking, temperature_first, temperature_retry,
  per_turn_timeout_s}`), `expander` (`{model_id, thinking, timeout_s, parse_retries}`),
  `families` (per family `{target_n, teacher_k, over_generate_x, amount_path_weights?,
  gram_anchor: bool}`), `max_seq_tokens` (20000), `plan_max_tokens` (~80),
  `tokenizer_name` (student tokenizer id or `null`), `max_intents`, `usd_budget`,
  `on_budget` (`warn`|`stop`), `output_dir` (default `data/student/`),
  `rubric_version` (`v2-r1`), `reward_version` (`v2-r1`).
- **`intents/*.jsonl`**: produced by `build`, or hand-supplied to re-run a fixed set.
  One JSON object per line = the intent tuple + `task_id` + `schema_version`.
- **Injected `expander`**: `callable(pool, *, persona, family, amount_path) ->
  {"query": str, "foods": [str]}` — the `generate_one` `expander` contract (prior art:
  `nutri-env/scripts/generate_one_cli.py::make_synthetic_query_foods_expander`).
- **Injected `teacher_complete`**: `callable(model_id: str, messages: Sequence[Mapping])
  -> {"content": str, "reasoning_content": str | None, "finish_reason": str,
  "usage": {"prompt_tokens": int, "completion_tokens": int, "reasoning_tokens": int}}`.
- **Environment**: `ARK_API_KEY` + `ARK_BASE_URL` — one credential for **both** the
  teacher and the expander (ADR-011 amended: `ark/deepseek-v4-flash` on `api/plan/v3`).
  Add both to `.env.example`. Real network calls only in production wiring; a guard
  requires `NUTRIMIND_ALLOW_NETWORK=1`.
- **Dependency**: `pyproject.toml` must add `nutrienv` as an editable path dependency
  pinned to an exact git rev. Ticket 001.

## 8. State Model

Per intent / task:

```
pending
 ├─ author  ─► authored ─► validate ─► gated ─► materialized(TaskPackage)
 │                                                     │
 │              target=sft ─► rollout ─► verified ─────┤
 │                              │            ├─ pass ─────────► serialized ─► accepted   → sft/train.jsonl
 │                              │            ├─ fail ─────────────────────────► rejected → rejects/teacher.jsonl
 │                              │            └─ indeterminate ────────────────► rejected → rejects/indeterminate.jsonl
 │                              └─ teacher_error / teacher_no_finish ─────────► rejected → rejects/indeterminate.jsonl
 │              target=rlvr ─► rlvr_exported                                            → rlvr/<task_id>.json
 │              target=eval ─► eval_frozen                                              → sft/val_mini.json
 ├─► author reject   (GenerateOneResult.rejected, incl. speech/bind)        → rejects/author.jsonl
 └─► gate reject     (first failing check)                                  → rejects/gate.jsonl
```

- Terminal accept: `accepted` (`sft/train.jsonl`), `rlvr_exported`, `eval_frozen`.
- Terminal reject: `rejects/{author,gate,teacher,indeterminate,serialize}.jsonl`.
- On disk by stage: `intents/*.jsonl` · `tasks/*.jsonl` (authored+gated `Task`) ·
  `task_packages/<task_id>.json` (canonical) · `rollouts/cache/<task_id>.json`
  (full teacher episode, atomic) · `sft/train.jsonl` (accepted, written once at end) ·
  `rlvr/<task_id>.json` · `rejects/*.jsonl` (append-only).
- Resume: a re-run in the same `output_dir` skips any `task_id` already present in
  `sft/train.jsonl` or any `rejects/*.jsonl`; reuses `task_packages/` and
  `rollouts/cache/` when present. `--from-stage {author,gate,materialize,rollout,serialize}`
  forces re-run of terminal tasks from that stage (e.g. re-serialize all cached episodes
  after a serializer fix).
- A `task_id` seen twice within a run → raise (bad enumerator). Across runs → skipped
  unless `--force`.

## 9. Output Contract

All under `output_dir` (default `data/student/`, already git-ignored via `data/`):

```
data/student/
  intents/<family>.jsonl
  tasks/<family>.jsonl
  task_packages/<task_id>.json
  rollouts/cache/<task_id>.json
  rlvr/<task_id>.json                 # target=rlvr
  rejects/{author,gate,teacher,indeterminate,serialize}.jsonl
  sft/train.jsonl
  sft/holdout.jsonl
  sft/loss_val.jsonl
  sft/val_mini.json                   # --freeze-mini
  run_manifest.json
  dry_run_report.json                 # --dry-run
```

### 9.1 TaskPackage (canonical) — `task_packages/<task_id>.json`

```json
{
  "schema_version": "nutrimind-v2-taskpackage/1",
  "task_key": "composite--update+log+recommend--train-ada",
  "task_id": "composite--update+log+recommend--train-ada--000191",
  "query": "Please add milk to my allergies. For lunch I had two slices of white bread. What should I have for dinner?",
  "family": "composite",
  "steps": ["update", "log", "recommend"],
  "tier": "",
  "environment": {
    "note": "VERIFIED 2026-09-09 (ticket 002 Part A): self-contained via a transient temp-file round-trip; no in-memory public entry — see below",
    "s0": { "profile": { "...": "..." }, "ledger": [], "allowed_food_ids": null },
    "reconstruct_with": "CANDIDATE: nutrienv.bench.pipeline.freezer.task_to_item(task) to serialize; then a public path from one in-memory item back to a Task/WorldState, then NutriEnv().reset(task.s0). load_split reads a FILE of items; whether a single in-memory item round-trips through a public entry is NOT yet confirmed."
  },
  "catalog": { "catalog_sha": "57184b2b…", "nutrienv_rev": "203d807" },
  "oracle": {
    "note": "payload of nutrienv Oracle (sub_oracles for composite), via freezer",
    "payload": { "sub_oracles": [ { "...": "..." } ] },
    "oracle_version": "nutrienv-203d807"
  },
  "verifier": { "kind": "nutrienv.bench.scorer.Scorer", "call": "Scorer().score(end_state, oracle)" },
  "reward_semantics": {
    "reward_version": "v2-r1",
    "kind": "binary",
    "map": { "pass": 1.0, "fail": 0.0, "indeterminate": null }
  },
  "rubric_version": "v2-r1",
  "termination": { "finish_ops": ["finish", "done", "stop"], "max_steps": 30 },
  "seed": 191,
  "provenance": {
    "nutrimind_rev": "<sha>", "nutrienv_rev": "203d807", "catalog_sha": "57184b2b…",
    "config_sha": "<sha>", "intent_ref": "intents/composite.jsonl#191", "built_at": "…"
  }
}
```

Field readiness against the current code:

| Field | Now | Source |
|---|---|---|
| `task_key`, `task_id`, `query`, `family`, `steps`, `tier`, `seed` | yes | intent + `Task` (§10 identifiers) |
| `environment` — serialize (`freezer.task_to_item`) | yes | public |
| `environment` — reconstruct one item → runnable `NutriEnv` | **VERIFIED, self-contained** (2026-09-09, ticket 002 Part A) | Public round-trip `task_to_item` → `freeze_tasks([task], output_path=<tmp>)` → `load_split(<tmp>)` is lossless for `s0` (profile/ledger/`allowed_food_ids`), oracle (incl. `sub_oracles`), and meta; result is runnable + Pass-reachable. **No in-memory public entry** (`split.__all__` is all file-based; `_item`/`_s0`/`_oracle` private) → reconstruction writes a **transient scratch file**. No external dataset dependency. Non-blocking upstream ask: add `item_to_task(item, catalog)` to `nutrienv.bench.split.__all__`. |
| `catalog` | yes | `catalog_digest`, config `nutrienv_rev` |
| `oracle.payload` | yes | `freezer` oracle payload of `Task.oracle` (`sub_oracles` for composite) |
| `verifier` | yes | `nutrienv.bench.scorer.Scorer` reference |
| `reward_semantics` (binary) | yes | fixed v2-r1 |
| `rubric_version` | yes | fixed v2-r1 |
| `termination` | yes | `nutrienv.harness.runner.FINISH_OPS` + `FAMILY_MAX_STEPS[family]` (`DEFAULT_MAX_STEPS=12`; `update 6 / log 12 / evaluate 12 / recommend 30 / composite 30`) |
| `provenance` | yes | run context |
| `oracle_version` finer than the nutri-env rev | Open Q (not blocking) | a hash of the oracle-scoring code path; deferred, use the rev for now |
| graded `reward_semantics` schema | Open Q (not blocking) | deferred; binary only in v2.0 |

> **Open Question 2 — RESOLVED (2026-09-09, ticket 002 Part A): verdict A, self-contained.**
> A single frozen item round-trips losslessly to a runnable, Pass-reachable `NutriEnv`
> through the public API (`task_to_item` → `freeze_tasks` → `load_split`), via a
> transient scratch file (no public in-memory entry). No external dataset dependency;
> `task_id` ↔ world-state binding is stable. The RLVR materializer therefore writes the
> `environment` block into a temp 1-item split and `load_split`s it. A non-blocking
> upstream ask (add `item_to_task` to `split.__all__`) would remove the temp file.

### 9.2 v2 SFT record — `sft/train.jsonl` (one JSON object per line)

```json
{
  "schema_version": "nutrimind-v2-sft/1",
  "task_key": "log--log--train-ada",
  "task_id": "log--log--train-ada--000042",
  "accepted_from_attempt": 1,
  "task_package_ref": "task_packages/log--log--train-ada--000042.json",
  "messages": [
    {"role": "system",    "content": "<react_manual('v2') text>"},
    {"role": "user",      "content": "Task:\nFor lunch I had a bowl of oatmeal and a banana."},
    {"role": "user",      "content": "Step budget: 12 action(s) remaining, including this turn.\nObservation:\n{...}"},
    {"role": "assistant", "content": "Plan: find oatmeal and banana in the catalog, then log both to today-lunch.\n{\"op\": \"search_foods\", \"q\": \"oatmeal\"}"},
    {"role": "user",      "content": "Step budget: 11 ...\nObservation:\n{...}"},
    {"role": "assistant", "content": "Plan: oatmeal is 2708110; read its portion table.\n{\"op\": \"get_food\", \"food_id\": \"2708110\"}"},
    {"role": "user",      "content": "Step budget: 10 ...\nObservation:\n{...}"},
    {"role": "assistant", "content": "Plan: log oatmeal 235 g and banana 118 g to today-lunch, then finish.\n{\"op\": \"log_meal\", \"food_id\": \"2708110\", \"grams\": 235, \"eaten_at\": \"today-lunch\"}"},
    {"role": "user",      "content": "Step budget: 9 ...\nObservation:\n{...}"},
    {"role": "assistant", "content": "Plan: banana next.\n{\"op\": \"log_meal\", \"food_id\": \"2709158\", \"grams\": 118, \"eaten_at\": \"today-lunch\"}"},
    {"role": "user",      "content": "Step budget: 8 ...\nObservation:\n{...}"},
    {"role": "assistant", "content": "Plan: both rows in the ledger; hand in.\n{\"op\": \"finish\"}"}
  ],
  "segments": ["system", "task", "observation", "step", "observation", "step", "observation", "step", "observation", "step", "observation", "final"],
  "train_on":  [false,    false,  false,        true,   false,        true,   false,        true,   false,        true,   false,        true],
  "meta": {
    "family": "log", "steps": ["log"], "tier": "", "persona": "everyday", "batch": 1,
    "seed": 42,
    "teacher": "ark/deepseek-v4-flash",
    "teacher_params": {"thinking": {"type": "enabled"}, "temperature_first": 0.0, "temperature_retry": 0.7},
    "expander": "ark/deepseek-v4-flash",
    "verification": {"status": "pass", "reward": 1.0, "failure_codes": [], "evidence": []},
    "oracle_version": "nutrienv-203d807",
    "rubric_version": "v2-r1",
    "reward_version": "v2-r1",
    "environment_version": "nutrienv-203d807",
    "task_schema_version": "nutrimind-v2-taskpackage/1",
    "catalog_sha": "57184b2b…",
    "nutrienv_rev": "203d807",
    "nutrimind_rev": "<sha>",
    "n_steps": 5, "n_turns_without_plan": 0, "plan_truncation": "token"
  }
}
```

Rules:
- `messages`: OpenAI-shaped. `system` first, exactly once. Then the `Task:` user turn.
  Then strictly alternating `user` (observation) / `assistant` (plan + op). The last
  message is an `assistant` turn whose op is in `nutrienv.harness.runner.FINISH_OPS`.
- `segments`: parallel to `messages`, one of `system` | `task` | `observation` | `step`
  | `final`. `step` and `final` are assistant turns; `final` is the last.
- `train_on`: parallel to `messages`, per-message bool. **The v2 loader must use this
  array**, not re-derive from role. For Batch 1, `train_on[i]` is true exactly where
  `segments[i] ∈ {step, final}`. A future ablation (e.g. mask the plan, keep only the
  op) would change the data — a per-segment sub-split of the assistant content — not the
  loader.
- **No token-level `loss_mask` is stored.** The v2 loader applies the v2 chat template,
  tokenizes, and for each message with `train_on[i]` true sets `labels` on that
  message's content token span; everything else `-100`. The tokenizer id and chat
  template are the **v2 loader's** configuration, not the record's. This is the explicit
  reversal of the earlier draft.
- `assistant.content` = `f"{plan}\n{op_json}"`. `plan` is the teacher `reasoning_content`
  for that turn, truncated to `plan_max_tokens` (token-exact with the injected student
  tokenizer if `tokenizer_name` is set, else a `~4 chars/token` char heuristic; recorded
  in `meta.plan_truncation`). `op_json` is the compact JSON of the action actually
  executed against `NutriEnv` for that turn (recorded as `executed_op` in the episode).
  If v2's own parse of `raw_action_text` does not yield that exact op (i.e. the harness
  substituted a fallback), the episode is `indeterminate` / `teacher_invalid_op` (§11–12)
  and is never serialized. The record does not depend on `nutrienv.harness.react._parse_action`'s
  internal behaviour.
- `meta.tier` is `""` for `log`/`recommend`/`update`/`composite`, or one of
  `nutrienv.bench.quality_gates.EVALUATE_TIERS`
  (`single`/`pair`/`triple`/`long`/`explicit_grams`/`synonym`) for `evaluate`. It is
  **not** the batch number and **not** v1's T1–T4.
- `observations` are copied verbatim from the episode (already capped by
  `ReActHarness.act` at 6000 chars each).

The v2 loader rejects a record that: lacks `schema_version` / `segments` / `train_on`;
has `len(messages) != len(segments) != len(train_on)`; has `segments[-1] != "final"`;
has a `system`/`observation` message with `train_on = true`; has any `<tool_call>` /
`<think>` / `<|im_start|>` marker in an `assistant.content` (that is a v1 record).

### 9.3 RLVR task export — `rlvr/<task_id>.json` (schema pinned; not built here)

```json
{
  "schema_version": "nutrimind-v2-rlvr/1",
  "task_id": "...",
  "task_package_ref": "task_packages/<task_id>.json",
  "prompt": { "system": "<react_manual('v2')>", "task": "Task:\n<query>" },
  "environment": { "...": "same env reconstruction block as the TaskPackage" },
  "verifier": { "kind": "nutrienv.bench.scorer.Scorer", "oracle": { "...": "oracle payload" }, "oracle_version": "nutrienv-203d807" },
  "reward": { "adapter": "binary", "reward_version": "v2-r1", "map": { "pass": 1.0, "fail": 0.0, "indeterminate": null } },
  "termination": { "finish_ops": ["finish", "done", "stop"], "max_steps": 30 },
  "seed": 191,
  "meta": { "...": "provenance, versions" }
}
```

### 9.4 Reject record — `rejects/<stage>.jsonl`

```json
{
  "schema_version": "nutrimind-v2-reject/1",
  "task_id": "composite--log+recommend--train-ben--000191",
  "stage": "teacher",
  "status": "fail",
  "failure_codes": ["task_fail", "window"],
  "reason_detail": "Scorer tag=window on sub_oracle[1] (recommend leg): protein_g 12.1 outside [0.0, 41.48]",
  "evidence": [{"sub_tags": ["pass", "window"], "failing_sub_oracle": 1, "note": "analysis candidate, not an RLVR negative"}],
  "attempts": [{"attempt_id": "…--attempt-01", "execution": "ok", "scorer": "fail", "failure_codes": ["window"]}, {"...": "up to k"}],
  "intent": { "...": "the full intent tuple" },
  "query": "I had a plate of chicken fried rice for lunch. What should I eat for dinner?",
  "task_package_ref": "task_packages/…json",
  "episode_ref": "rollouts/cache/…json",
  "meta": { "family": "composite", "steps": ["log", "recommend"], "seed": 191,
            "catalog_sha": "57184b2b…", "nutrienv_rev": "203d807", "nutrimind_rev": "<sha>",
            "oracle_version": "nutrienv-203d807", "rubric_version": "v2-r1",
            "reward_version": "v2-r1", "batch": 1 }
}
```

- `stage` ∈ `{author, gate, teacher, indeterminate, serialize}`.
- `status` ∈ `{fail, indeterminate}` (an `author`/`gate` reject is a pre-verification
  drop; `status` is omitted or `"dropped"` there).
- `failure_codes`: list of stable slugs (§11). `reason_detail`: human text, not parsed.

### 9.5 `run_manifest.json`

As in the previous draft, plus per-code counts split by `status`, an `indeterminate`
bucket separate from `fail`, and `versions` (`oracle_version`, `rubric_version`,
`reward_version`, `environment_version`, `task_schema_version`).

## 10. Reproducibility and Rerun Semantics

**Three identifiers** (distinct, do not conflate):

- **`task_key`** = `f"{family}--{'+'.join(steps)}--{person.user_id}"` — the *logical task
  identity*; **no seed**. Two intents with the same `task_key` are the same person doing
  the same family+steps with different sampled content.
- **`task_id`** = `f"{task_key}--{seed:06d}"` — one *concrete task instance*. Different
  seeds are different `task_id`s. This is the resume key, the filename stem, and the
  sort/dedup key throughout §8–§10.
- **`attempt_id`** = `f"{task_id}--attempt-{n:02d}"` — one teacher rollout of a `task_id`
  (`n` in `1..k`).

**N counting** (fixes the earlier contradiction): **N counts distinct accepted
`task_id`s** (not `task_key`, not `attempt_id`), after removing any accepted `task_id`
whose `semantic_key` duplicates another accepted `task_id` **in the same family** (keep
the lowest seed). The multiple `attempt_id`s of one `task_id` contribute **at most one**
accepted trace. Different seeds → different `task_id`s → counted separately *unless* they
collide on `semantic_key`. So "N = 40" = 40 distinct, `semantic_key`-deduped accepted
`task_id`s for the family. (If the intent were "40 distinct *logical* tasks", N would
count `task_key` — it does **not**; the design doc's "Pass traces" are per-instance.)

**Deterministic** (identical across runs given the same inputs): intent enumeration and
ordering; `task_key` / `task_id` / `attempt_id`; the `catalog_sha` assertion; `author`
for `generate_one`-supported shapes given a fixed `expander` output (the RNG inside
`generate_one` is seeded by the intent `seed`); all `gates.run` verdicts; the
**verifier** given a fixed end state; `serialize`; output ordering (`sft/train.jsonl`
sorted by `task_id`; the 7:2:1 split keyed by a `task_id` hash).

**Non-deterministic**: the **expander** LLM and the **teacher** completion. The
reproducibility boundary is `rollouts/cache/` — run 1 is non-deterministic; runs 2..N
off an unchanged cache are deterministic. Retry temperature 0.7 (attempts 2..k) means
even a single teacher task is not bit-reproducible without the cache; this is accepted.

**Scripted tests**: `build(config, expander=fake, teacher_complete=fake)` with canned
responses is fully deterministic and asserted byte-identical across runs.

**Provenance recorded**: `run_manifest.json` plus, on every record, `meta` versions
(`oracle_version`, `rubric_version`, `reward_version`, `environment_version`,
`task_schema_version`, `catalog_sha`, `nutrienv_rev`, `nutrimind_rev`, `seed`). `git_sha`
recording is allowed for this repo (it is a git repo with a remote) and is the
`nutrimind_rev` field.

## 11. Failure Model

**Pre-verification drops** (no `VerificationResult`):

- `author.*` — pass through `GenerateOneResult.rejected.reason` verbatim, namespaced
  `author.<reason>`. Stable reason vocabulary from nutri-env includes `schema`,
  `empty_pool`, `illegal_pair`, `unfit_substitute`, `steps`, `rec_foods`, `no_ledger`,
  `empty_windows`, `duplicate`, `not_in_pool`, `ambiguous`, `omitted_food`, `repeat`,
  `unresolvable`, `amount_path`, `small_grams`, `over_cap`, `not_gym_persona`,
  `template`, `template_occasion`, `slot_conflict`, `no_allergen_dish`, `no_deficit`.
  Speech / portion-bind failures are a subset (`ambiguous`, `omitted_food`,
  `unresolvable`, `amount_path`, `repeat`, `duplicate`).
- `gate.*`, ordered, first failure wins:
  1. `gate.verbatim_query_collision` — candidate query, casefolded + whitespace-collapsed
     + trailing-punctuation-stripped, equals any of the 63 exam queries normalized the
     same way. Near-duplicates that are not verbatim are deliberately allowed.
  2. `gate.semantic_key_collision` — `nutrienv.bench.validator.semantic_key(task)` equals
     any exam task's `semantic_key`. Note: `semantic_key`'s structural branches
     (multi_item_log, fuzzy_portion, recommend, update, evaluate, leftover) are
     text-independent; its fallback branch keys on the raw, case-sensitive query. Both
     behaviours are intended; the gate uses `semantic_key` as-is.
  3. `gate.slot_value_overlaps_exam` — `update` / composite-with-update only: a slot
     value equals a value used by an exam `update` item.
  4. `gate.stage_a` — `nutrienv.bench.pipeline.review_harness.stage_a_code_gate(task)`
     returned a non-empty list (`reason_detail` = the list).
  5. `gate.draft_invalid` — `nutrienv.bench.validator.validate_draft(task)` returned a
     non-empty list (`reason_detail` = the list).
  6. `gate.unachievable` → **status `indeterminate`** — `task.id` is in
     `nutrienv.bench.achievable.check_achievable([task]).unreachable`. This is an
     authoring bug, not a model failure.

**Verification outcomes** (`VerificationResult`, §12):

The two axes are independent and must not be conflated:
- **Did the episode run legally?** (completed with a FINISH op, every action a genuine
  parse of the assistant text) — a factory/environment concern.
- **Did the end state satisfy the hard contract?** (`Scorer.passed`) — a model concern.

Mapping:

- `pass` — episode ran legally **and** `Scorer.score(end_state, oracle)["passed"] is True`.
- `fail` (code **`task_fail`**) — a completed, legal teacher episode (reached a FINISH
  op, all actions genuine) whose end state does **not** satisfy the hard contract
  (`Scorer` `passed is False`). This stays `fail` — it is never re-labelled
  `indeterminate`. `failure_codes` = `["task_fail"]` plus the `Scorer` tag (`log_miss` /
  `window` / `wrong_goal` / `inventory_miss` / `allergy` / `update_miss`); `evidence`
  carries `sub_tags` and the failing sub-oracle index for composite.
  **`task_fail` is a teacher-generation outcome, not a curated negative.** It is stored
  in `rejects/teacher.jsonl` with all `k` attempts + their `failure_codes` as an
  **analysis candidate** (EEF / SRFT / DPO). It is **not** an RLVR negative sample —
  RLVR negatives come only from the RLVR runner executing a *model* rollout and having
  the verifier return `fail`; a teacher's non-Pass episode is SFT-reject / reference
  material only (§4.2, §9.3).
- `indeterminate` — reward `null`, kept out of pass/fail stats. **Only** these triggers,
  and an exception is never turned into `fail`:
  - `teacher_error` — API error, timeout, or attempts (`k`) exhausted with no completed
    episode.
  - `teacher_no_finish` — hit the max-steps cap without a FINISH op.
  - `teacher_invalid_op` — an executed action was not a genuine parse of that turn's
    assistant text (§12, "Action legality"). Detected from **v2-owned trajectory
    metadata** (`raw_action_text` + `executed_op` per turn, recorded by v2's
    `ReActHarness` subclass) re-parsed by v2's **own** parser — **not** by inspecting
    `nutrienv.harness.react._parse_action`'s internal fallback path.
  - `oracle_error` — `Scorer` or the env raised (catch, record the traceback in
    `evidence`, continue; never scored as `fail`).
  - `env_reconstruction_mismatch` — reconstructed `s0` / catalog does not match the
    TaskPackage.
  - `gate.unachievable` (above).

**Serialize failures** (`serialize` stage, status `indeterminate`):
`serialize.empty_episode`, `serialize.no_system_turn`, `serialize.consecutive_assistant`,
`serialize.missing_observation`, `serialize.turn_count_mismatch`,
`serialize.last_turn_not_finish`, `serialize.too_long` (record tokens > `max_seq_tokens`),
`serialize.no_plan_any_turn` (every assistant turn lacked `reasoning_content`; one
missing plan is tolerated with `plan=""`).

**False-positive / false-negative priority ladder** (Oracle design + Testing):

```
env / input / dependency error        → indeterminate
clear hard-constraint violation       → fail
all hard constraints satisfied        → pass
low soft-rubric (diagnostic) score    → pass, record diagnostic_scores
```

- **False positive** (invalid answer → `pass`) is the high-risk case. nutri-env's
  `Scorer` already guards it: nonexistent `food_id` → `wrong_goal`; non-positive /
  non-finite grams → `wrong_goal`; allergen in plan → `allergy`; window breach →
  `window`; off-`allowed_food_ids` → `inventory_miss`; missing required op → `log_miss`
  / `update_miss`. v2 adds the `teacher_invalid_op` check (§12, "Action legality") so a
  fabricated fallback action — one the harness substituted because the assistant text did
  not parse — cannot ride into an accepted trace.
- **False negative** (valid answer → `fail`): nutri-env already mitigates with
  order-independent multiset matching (`_match_ledger_multiset`, `_match_plan_items`),
  ±15 % gram tolerance (ADR-0023), the `last_plan=[]` free-recommendation sentinel,
  `allowed_food_ids` category-synonym matching, and update-band scoring. v2's obligation
  is to **not add** verbatim matching on top, and to boundary-test the dedup gates so a
  legitimately new task is not dropped as an exam collision.

## 12. Oracle & Verification Contract

The **verifier** wraps nutri-env's `Scorer` (which returns binary `passed` + a `tag`)
and adds the `indeterminate` state that a data factory needs. Prior art: the eval suite
already separates `void` tasks (harness/env error) from pass/fail and reports a
`clean_pass_rate`; `indeterminate` is that concept, named for v2.

Concept interface (follow project naming if one emerges; nutri-env has no tri-state
type, so this is v2-owned):

```
VerificationResult(
    status: "pass" | "fail" | "indeterminate",   # derived from the three axes below
    execution: "ok" | "no_finish" | "invalid_op" | "error",
    oracle_exec: "ok" | "error" | "env_mismatch",
    scorer: "pass" | "fail" | None,              # None when an axis above is not ok
    reward: float | None,              # 1.0 | 0.0 | None
    failure_codes: list[str],          # stable slugs, §11
    evidence: list[str | dict],        # Scorer tag/sub_tags, computed totals vs window, offending food_id, ...
    oracle_version: str,               # "nutrienv-<rev>" for v2.0
    rubric_version: str,               # "v2-r1"
    reward_version: str,               # "v2-r1"
    diagnostic_scores: dict | None,    # soft rubric; NEVER affects status or reward in v2.0
)
```

Definitions — `Scorer`'s binary result is one input, not the whole `status`:

- **pass** — the episode ran legally **and** `Scorer` `passed is True`.
- **fail** — the episode ran legally (completed with a FINISH op; every action a genuine
  parse of the assistant text) **and** `Scorer` `passed is False`. A plain
  `Scorer.passed is False` on a legal episode is a **real model failure and stays
  `fail`** — it is never promoted to `indeterminate`, or the true failure rate would be
  hidden.
- **indeterminate** — the oracle cannot reliably judge, or the environment / input /
  dependency is abnormal (teacher error / no-finish / invalid op, oracle exception,
  unreachable oracle, env-reconstruction mismatch). An exception is **never** turned into
  `fail` — that would fabricate a false negative. Reward `null`; excluded from pass/fail
  statistics.

So: not every `Scorer=False` is `indeterminate`, and not every exception is `fail`.

**Action legality** (feeds `teacher_invalid_op`, without touching nutri-env internals):
v2's `ReActHarness` subclass records per turn `raw_action_text` (the assistant message it
sent) and `executed_op` (the action `NutriEnv.step` actually received). A **v2-owned**
parser re-parses `raw_action_text`. The episode is `teacher_invalid_op` (→
`indeterminate`) when, for any turn, v2's parse yields no well-formed
`{"op": <legal op>, ...}` **or** yields an op that differs from `executed_op` (the
harness substituted a fallback). The check reads only the public assistant text and v2's
own recorded metadata — never `nutrienv.harness.react._parse_action`'s return path or its
`{"op": "get_profile"}` fallback constant.

**Three axes recorded separately** (so `Scorer=False` and an execution exception are
never conflated again, in v2.0 or later). The episode / `VerificationResult` carries all
three; `status` is derived from them, not from `Scorer` alone:

| Axis | Field | Values | Feeds |
|---|---|---|---|
| Execution legality | `execution` | `ok` / `no_finish` / `invalid_op` / `error` | `indeterminate` unless `ok` |
| Oracle executability | `oracle_exec` | `ok` / `error` / `env_mismatch` | `indeterminate` unless `ok` |
| Scorer judgment | `scorer` | `pass` / `fail` (only meaningful when the two above are `ok`) | `pass` / `fail` |

`status` = `pass` iff `execution=ok ∧ oracle_exec=ok ∧ scorer=pass`; `fail` iff
`execution=ok ∧ oracle_exec=ok ∧ scorer=fail`; `indeterminate` otherwise.

Hard contract fixed for v2.0 (before bulk generation):

- Hard constraints = exactly what `Scorer` enforces for the task's `oracle`
  (`_score_verdict`, `_score_plan`, `_score_composite`, ledger/profile equality).
- Always-pass: all three axes clean, `scorer=pass`.
- Always-fail: `execution=ok`, `oracle_exec=ok`, `scorer=fail`. Never promoted to
  `indeterminate`.
- Indeterminate: any `execution` ≠ `ok` or `oracle_exec` ≠ `ok`. An exception never
  becomes `fail`.
- Reward map: `{pass: 1.0, fail: 0.0, indeterminate: null}`, `reward_version = v2-r1`.
- Oracle exception handling: catch, set `oracle_exec=error`, record the traceback in
  `evidence`, `status=indeterminate`, continue the run.
- Evidence always recorded: the `Scorer` `tag` (and `sub_tags` for composite), and for a
  `fail` the concrete number that missed (e.g. `protein_g 12.1 outside [0, 41.48]`).

## 13. Rubric: Hard Contract vs Soft Rubric

**Hard contract** (v2.0, immutable within `rubric_version = v2-r1`): §12. This is what
decides `status` and `reward`.

**Soft rubric** (diagnostic-only in v2.0, may iterate freely): partial constraint
satisfaction, path length / conciseness, recommendation quality, explanation quality,
preference among multiple valid answers, other reward-shaping candidates. In v2.0 these
populate `diagnostic_scores` and the run stats **only**. A low soft score never changes
`status` or `reward`. Early uses: data diagnostics, human review, run reports, quality
ranking, informing a future graded reward.

**Evolution**: a graded reward, if introduced later, gets a new `reward_version` and a
new `rubric_version`; old records keep their original `status` / `reward` / versions; a
re-evaluation is an explicit, separate pass that writes new records (or a sidecar), never
an in-place overwrite; and it ships with a test that asserts reward monotonicity
(a strictly-better trajectory never scores lower).

## 14. Version and Migration Boundary

### 14.1 Record / data boundary

- **Not compatible with v1.** v1 SFT records are `{query, tier(T1–T4), messages}` with
  Qwen-native `<tool_call>` / `<think>` / `<tool_response>` content and **no stored
  loss mask** — `src/training/sft/train.py::custom_loss_masking` derives it at tokenize
  time by scanning for `<|im_start|>assistant` (`151644, 77091`) … `<|im_start|>user`
  (`151644, 872`) spans, bound to the Qwen3-4B tokenizer. v2 records are §9.2, with
  `segments` + `train_on` and a `nutrimind-v2-*` `schema_version`.
- **v1 data never enters the v2 pipeline**; v2 never reads or writes `data/trajectories/*`
  or `data/training/*`. v2 writes only under `output_dir` (default `data/student/`,
  which does not exist yet).
- **No migration.** Different base model, different action space; converting v1 → v2 is
  meaningless.
- **Loader isolation risk**: v1's `load_trajectory_data` only needs `messages` and would
  silently mis-train on a v2 record (it ignores unknown keys and applies
  `enable_thinking=True` + the `<|im_start|>` scan). Mitigations: v2 records keep
  `family`/`tier` under `meta` (no bare top-level `tier`); the two loaders are never
  pointed at the same directory; the v2 loader hard-rejects a v1-shaped record and vice
  versa (§9.2).
- **Rollback**: a v2 failure falls back to v1 by pointing the trainer at
  `data/training/sft_train_trajectory.jsonl` (untouched) with the v1 eval. No data
  migration either direction.

### 14.2 Project / code boundary

Do **not** copy the project into a `v2/` tree. One repository, explicit namespaces:

| | v1 (frozen) | v2 (new) |
|---|---|---|
| Data-gen code | `src/training/sft/collect_trajectories*.py`, `normalize.py`, `validate_rules.py`, `validate_semantic.py` | `src/training/data_factory/` (build, author, gates, materializers, teacher client) |
| SFT loader / trainer | `src/training/sft/train.py` | a new module under `src/training/data_factory/` (e.g. `sft_v2_loader.py` or `sft_v2/`); pick inline per the existing layout — **not a blocker** (Open Question 4). **Out of scope to build**; contract in §9.2. |
| Eval | `src/training/sft/evaluate.py` (mock) | v2 uses nutri-env's `Scorer` + a v2 runner (out of scope) |
| GRPO | `src/training/grpo/` (veRL/TRL, ADR-002/005/006/007) | out of scope for this spec |
| Artifacts | `data/trajectories/`, `data/training/` | `data/student/` |
| Schema | positional `{query, tier, messages}` | `schema_version`-tagged `nutrimind-v2-*` |
| Loss semantics | derived from `<\|im_start\|>` token ids at train time | `segments` + `train_on` in the record; token mask derived in the v2 loader |
| Shared | `src/utils/`, low-level helpers, `configs/` conventions | same |

Archival — **not blocking** (Open Question 13), do it in this order, none of it gates v2
implementation:
- Now: nothing moves. v1 code stays where it is — `tests/test_tools.py`,
  `tests/training/grpo/*`, `tests/test_*` and the v1 training entry still import it.
- Now: the v1↔v2 boundary is captured by this section (§14) and by the invariants
  "v1 is not rewritten by the v2 loader" and "v2 output never overwrites a v1 artifact".
- After the v2 loader is stable **and** the working tree is clean: cut a git tag
  `nutrimind-v1-final`, and add `docs/archive/v1/` recording the v1 SFT record schema, the
  v1 loader's `<|im_start|>`-scan masking, the v1 training entry, and v1 known
  limitations (mock `evaluate.py`, GRPO collapse ADR-009, no stored mask, GRPO-pool leak
  in `evaluate.py`). Do **not** rush a `-final` tag over uncommitted changes.

## 15. Reproducibility fields — see §10.

## 16. Resource and Cost Limits

- **expander**: `timeout_s` 60; attempts `parse_retries + 1` (nutri-env default 1 → 2).
- **teacher**: `k` (per-family, 1–6) = **max teacher rollout attempts per `task_id`**.
  One rollout = one full ReAct episode. Attempt 1 at `temperature_first` (0.0), attempts
  2..k at `temperature_retry` (0.7); stop at the first Pass. **`k` counts total attempts,
  retries included** — not "k retries", not "k steps". A `task_id` with no Pass in `k`
  attempts is `task_fail` if ≥1 attempt was a completed legal episode, else
  `indeterminate` (§11); `task_fail` is a teacher-generation outcome kept for analysis,
  never an RLVR negative or an SFT accept. Retry attribution:
  `build` loops attempts 1..k; the `teacher_complete` adapter's HTTP client owns
  transport retries; `build` does not retry gates or serialization. Design-doc `teacher
  k` (§7 table) maps directly to this `k`.
- **max teacher turns / episode** (the *step* budget, distinct from `k`):
  `nutrienv.harness.runner` policy — `DEFAULT_MAX_STEPS = 12`,
  `FAMILY_MAX_STEPS = {update:6, log:12, evaluate:12, recommend:30, composite:30}`. Cap
  hit within an attempt without a FINISH op → that attempt is `no_finish`; all `k`
  attempts `no_finish` → `teacher_no_finish` (indeterminate).
- **per-turn `timeout_s`** 60.
- **per-record token cap**: `max_seq_tokens` = 20000 (ADR-011). Over → `serialize.too_long`.
- **per-run cap**: `max_intents`.
- **cost budget**: `usd_budget`; `on_budget: warn` (log at 80 %, continue) or `stop`
  (halt cleanly at 100 %, write the manifest with what completed).
- **network boundary**: real `expander` / `teacher` calls only in production wiring; a
  guard requires `NUTRIMIND_ALLOW_NETWORK=1`. Every test injects fakes.

## 17. Observability and Provenance

- `run_manifest.json` — the single "did the run work" artefact; counts split by
  `status` (accepted / fail / indeterminate) and by `failure_code`; `family_mix`
  target vs actual; `cost`; `versions`.
- Every accepted / reject record carries the `meta` version block (§9.2).
- `task_packages/<task_id>.json` — the canonical task, replayable by nutri-env tooling.
- `rollouts/cache/<task_id>.json` — a `RolloutCache`: `{task_id, attempts:
  [{attempt_id, EpisodeResult, VerificationResult}], selected_attempt}`. One entry per
  teacher attempt 1..k that ran; each `EpisodeResult` carries the messages sent per step,
  each assistant turn's `content` + `reasoning_content` + `finish_reason` + `usage` +
  latency, the resolved `Task`, and the produced `end_state`. `selected_attempt` is the
  first Pass (or `null`). Multi-attempt by construction — this is what
  `--from-stage serialize` re-reads without re-paying the teacher.
- `loguru` logs (existing dep): task-level INFO outcomes, WARNING on retries and cost
  thresholds, ERROR on serialize failures. API keys and full prompts never logged
  (prompts live in the cache files).

## 18. Compatibility with NutriEnv (ADR-012, amended)

ADR-012 is amended (see its Amendment log, 2026-09-08) to two symbol classes:

### Public borrowed API — in nutri-env `__all__`, guarded by import + signature + behaviour tests

| Module | Symbols |
|---|---|
| `nutrienv.bench` (re-exports) | `Oracle`, `Task`, `Scorer`, `check_achievable`, `load_split`, `load_exam`, `EXAM_SPLIT_PATH` |
| `nutrienv.bench.pipeline.generate_one` | `generate_one`, `make_log_expander`, `make_unfit_rewriter`, `parse_query_foods_payload`, `search_fit_plate`, `AMOUNT_PATHS`, `KNIVES` |
| `nutrienv.bench.realize` | `compose_oracles`, `scored_oracles`, `realize_evaluate`, `bind_evaluate_reasons` |
| `nutrienv.bench.validator` | `validate_draft`, `semantic_key`, `fitting_plan` |
| `nutrienv.bench.pipeline.review_harness` | `stage_a_code_gate` |
| `nutrienv.bench.quality_gates` | `EVALUATE_TIERS` |
| `nutrienv.bench.pipeline.freezer` | `freeze_tasks`, `task_to_item` |
| `nutrienv.bench.pipeline.templates` | `RECOMMEND_SHELLS`, `UPDATE_SHELLS`, `recommend_query`, `update_query` |
| `nutrienv.bench.pipeline.roster` | `ROSTER`, `RosterPerson`, `profile_for`, `sample_roster_person` |
| `nutrienv.bench.pipeline.types` | `catalog_digest` |
| `nutrienv.world.daily_windows` | `plan_windows_for_meal`, `derive_profile_windows`, `meal_slot_and_remainder` |
| `nutrienv.world.types` | `ledger_totals`, `WorldState`, `Profile`, `LedgerRow`, `MAX_ITEM_GRAMS` |
| `nutrienv.world.catalog_store` | `load_catalog` |
| `nutrienv.harness` | `ReActHarness`, `ScriptHarness` (only these two are re-exported here) |
| `nutrienv.harness.react` | `react_manual`, `context_messages` (in that module's `__all__`; **not** re-exported from `nutrienv.harness` — ticket 001 finding) |
| `nutrienv.harness.runner` | `DEFAULT_MAX_STEPS`, `FAMILY_MAX_STEPS`, `FINISH_OPS` |
| `nutrienv.env` | `NutriEnv` |

### Private implementation detail — NOT frozen

`_update_from_template`, `_bind_log_foods`, `_log_then_recommend`,
`_update_then_recommend`, `_composite_speech_spans`, `_recommend_from_template`,
`_evaluate_from_bound`, `_anchored_bind_grams`, `_parse_action`, `_SYSTEM_V2`,
`split._item` / `_s0` / `_oracle`.

- No signature-stability promise.
- Not a v2 production dependency. Where a capability is only reachable through one,
  reconstruct it from public symbols (3-leg composite — ticket 002) or subclass the
  owner (`ReActHarness` for `_parse_action` and the loop; `react_manual("v2")` instead
  of `_SYSTEM_V2`).
- Covered indirectly by the Seam-1 end-to-end behaviour test.
- If a stable dependency on one becomes unavoidable, the correct move is to get it
  promoted to nutri-env's `__all__` first, not to import the underscore name.

### Import prerequisite

`nutrienv` is currently **not importable** in this repo. Before any implementation:
establish a reproducible NutriEnv dependency (ticket 001 — source, revision pinning, and
install style chosen per the project's package manager, **not** pre-locked to a specific
syntax), then a minimal public-API smoke test must pass: `import nutrienv`; construct one
`Task` via `generate_one` with a synthetic expander; `NutriEnv().reset(task.s0)`;
`Scorer().score(...)`; `load_exam()`; `task_to_item(task)`; read
`nutrienv.harness.runner.FAMILY_MAX_STEPS` / `FINISH_OPS`; `catalog_digest(catalog)`.
Ticket 001 owns this, and must close only on that test passing locally **and** in CI.

## 19. Testing Decisions

A good test asserts **external behaviour** — the returned value or the files/records
produced — not internal call order.

### 19.1 TaskPackage
- All required fields present; `schema_version` correct.
- `environment` round-trips: `task_to_item(task)` → package → reconstruct → the same
  `s0` (profile fields, ledger rows, `allowed_food_ids`) and the same `catalog_sha`.
- `provenance` complete; `seed` and `catalog_sha` present.
- `task_id` uniqueness within a run (duplicate → raise); idempotent skip across runs.
- `termination.max_steps` equals `FAMILY_MAX_STEPS[family]` (or `DEFAULT_MAX_STEPS`).

### 19.2 Oracle / verifier
- Normal `pass`; clear `fail`; `indeterminate` for each trigger (teacher error,
  no-finish, invalid-op fallback, oracle exception, env-reconstruction mismatch,
  `gate.unachievable`).
- Hard-constraint boundaries: window edge (just inside / just outside the ±15 %
  tolerance), allergen present, off-`allowed_food_ids` food, missing required op.
- Multiple valid answers: two different in-window, allergen-safe plans both `pass`
  (order-independent, no verbatim match).
- Nonexistent `food_id` → `fail` (`wrong_goal`), never `pass`.
- Non-bindable portion in an authored task → `author.unresolvable` (dropped before
  verification), never a silent `pass`.
- A completed, legal episode with `Scorer.passed is False` → `fail` (never
  `indeterminate`); an episode where `Scorer` raises → `indeterminate` (never `fail`).
- Fabricated action: an episode whose recorded `raw_action_text` for some turn does not
  v2-re-parse to its `executed_op` → `indeterminate` (`teacher_invalid_op`). The test
  builds this from the trajectory metadata, without reference to `_parse_action`.
- End-state read by the verifier equals the env state the rollout produced (no drift).

### 19.3 Rubric
- `rubric_version` on every `VerificationResult` and record.
- A low `diagnostic_scores` value does **not** change `status` or `reward`.
- Hard-contract outcome is independent of `diagnostic_scores`.
- Evidence is present and points at the concrete miss for every `fail`.
- A simulated `rubric_version` bump does not alter existing records in place.

### 19.4 SFT
- Scripted Pass episode → one record in `sft/train.jsonl`; `segments` / `train_on` /
  `messages` lengths equal; `train_on` true exactly on `step`/`final`; `meta` version
  block complete.
- Scripted Fail episode → nothing in `sft/train.jsonl`; one line in
  `rejects/teacher.jsonl` with `failure_codes=["task_fail", "<Scorer tag>"]` and an
  `attempts` array; nothing marks it an RLVR negative.
- Scripted teacher error / no-finish → `rejects/indeterminate.jsonl`.
- One bad intent among several good → the good ones still land; the run completes.
- `serialize` edges (each its own test): no `system` turn; consecutive `assistant`;
  missing observation; empty episode; last turn not a FINISH op; one turn without
  `reasoning_content` → `plan=""` + `meta.n_turns_without_plan==1`; every turn without a
  plan → `serialize.no_plan_any_turn`; `reasoning_content` over `plan_max_tokens` →
  truncated, `meta.plan_truncation` set, op still parseable; record over `max_seq_tokens`
  → `serialize.too_long`.
- No `<tool_call>` / `<think>` / `<|im_start|>` markers in any assistant content.
- **v2 loader**: derives a token-level mask from `train_on` after
  `apply_chat_template` with the v2 tokenizer; masks system/observation to `-100`;
  unmasks `step`/`final` content spans; mask length equals the token-id length; rejects a
  v1-shaped record; rejects `len` mismatches and `segments[-1] != "final"`.
- **v1/v2 isolation**: the v1 loader is unchanged; a test asserts the v1 loader and the
  v2 loader reject each other's record shape.

### 19.5 RLVR (schema-level; export not built here)
- A TaskPackage → RLVR export has `environment`, `verifier.oracle`, `reward.map`,
  `termination`; no `messages`, no teacher fields.
- `environment` reconstructs to a runnable `NutriEnv`.
- `reward.map` is `{pass:1.0, fail:0.0, indeterminate:null}` with `reward_version`.
- `termination.max_steps` and `finish_ops` match the TaskPackage.
- An RLVR export is never derived from an SFT record and carries no trajectory.

### 19.6 Compatibility
- `test_imports.py` — every symbol in §18's public table resolves.
- `test_borrowed_api_signatures.py` — `inspect.signature` assertions **only** for that
  public set. No signature assertion for any underscore-private helper.
- A test asserting the 3-leg composite is assembled through public symbols (no import of
  `_update_from_template` / `_bind_log_foods`) — behaviour, via Seam 1.
- A test asserting ADR-012's amended two-class rule matches what the guard files
  actually check (public → signature-guarded; private → not).

### 19.7 Seams and prior art
- **Seam 1** — `build(config, *, expander, teacher_complete)`: single end-to-end entry;
  inject a synthetic `expander`
  (`nutri-env/scripts/generate_one_cli.py::make_synthetic_query_foods_expander`) and a
  scripted `teacher_complete` (queue of `(content, reasoning_content)`; prior art
  `tests/training/grpo/test_environment_execute.py` mock registry).
- **Seam 2** — `gates.run(task, ctx=GateContext.from_exam(exam_tasks)) ->
  GateResult(keep, failure_code, reason_detail, stage)` (pure; the exam corpus is passed
  in, never loaded implicitly). Build `Task`s with `generate_one(expander=synthetic)`
  offline.
- **Seam 3** — the **verifier** `verify(task_package, episode: EpisodeResult) ->
  VerificationResult` (pure given the `EpisodeResult`; `EpisodeResult` bundles
  `end_state` + per-turn `TurnMeta` + `reached_finish` + `error`, so the verifier can
  derive all three axes of §12).
- **Seam 4** — `serialize(task_package, episode: EpisodeResult) -> record` (pure).
- **Seam 5** — the compatibility guard (§19.6), split from behaviour tests.
- pytest: no config today; add `[tool.pytest.ini_options] testpaths=["tests"]` or rely
  on discovery. Tests under `tests/training/data_factory/`. Prior art
  `tests/test_tools.py` (temp SQLite + `unittest.mock.patch`).

## 20. Evaluation Metrics

The factory is not "successful" merely by producing `sft/train.jsonl`. From
`run_manifest.json`:

- `catalog_sha_match == true` (hard gate).
- `counts.accepted` ≥ ~380 (target ~420); per-family `actual` within ±15 % of `target`
  for every family in design §7.
- `sft/train.jsonl` has **zero** `verbatim_query_collision` / `semantic_key_collision`
  (drop conditions; nonzero here is a bug).
- `serialization_success_rate` (`serialized / status=pass`) ≥ 0.98.
- `teacher_completion_rate`
  (`completed / (completed + teacher_error + teacher_no_finish)`) ≥ 0.9.
- `teacher_pass_rate` (`pass / completed`) ≈ 0.7 overall; flagged if 3-leg composite
  < 0.35.
- **`indeterminate_rate`** — a **v2-r1 operational health threshold** (not a domain
  fact; revisable with `reward_version`). Denominator is **attempted `task_id`s**, not
  accepted:
  `indeterminate_rate = indeterminate_task_ids / attempted_task_ids ≤ 0.05`.
  Applied **run-level** (whole run). Do **not** apply the ratio when
  `attempted_task_ids < 40` — report raw counts and re-run with a larger candidate cap.
  Above 0.05 the factory (not the model) has a problem — inspect
  `env_reconstruction_mismatch`, `oracle_error`, `gate.unachievable`. Part B (§ticket 002)
  **reports** this number but does not gate on it; Part C **applies** it.
- Reject histogram dominated by `author.*` bind reasons and `task_fail`; a large
  `gate.draft_invalid` / `gate.unachievable` share fails the run's health check.
- `cost.est_usd` ≤ `cost.budget_usd`.
- `mean_seconds_per_accepted`, `mean_teacher_tokens_per_accepted` recorded.

No like-for-like metric vs v1 (different task space / base model). The only v1 comparison
is qualitative: the v2 reject profile should not be dominated by authoring-validity
failures.

## 21. Data Safety

v2 processes only synthetic data — `TRAIN_ROSTER` (fictional `train-*` people), USDA
catalog foods, LLM-authored queries. **No real user data enters the pipeline.** Prompts
sent to Volcengine (`ark/deepseek-v4-flash` on `api/plan/v3`, one endpoint for teacher +
expander) contain only synthetic roster + catalog content; nothing sensitive to redact.
`rollouts/cache/` and logs may hold raw (synthetic) teacher responses under the
git-ignored `data/`. No de-identification required. The only external data boundary:
synthetic prompts leave the machine to one model provider.

## 22. Implementation Decisions

1. **build is a task/data artifact factory, not an experiment runner.** It does not
   train, sweep, compare models, manage checkpoints, aggregate experiment stats, or
   decide hyper-parameters. One thin top-level entry
   `build(config, *, expander, teacher_complete)` delegating to per-stage callables;
   `--stop-after {author,gate}` are debug/staged-artifact flags, not a second pipeline.
2. **Canonical TaskPackage** (§9.1) is authored once and is the sole source for SFT,
   RLVR, and evaluation materializers. Artifacts are never reverse-inferred from each
   other. Env reconstruction uses public `freezer.task_to_item` + `load_split` +
   `NutriEnv().reset`; the oracle payload uses the `freezer` serializer; termination
   uses `runner.FINISH_OPS` + `FAMILY_MAX_STEPS`.
3. **Tri-state verification.** `VerificationResult` (§12) wraps `Scorer`'s binary result
   and adds `indeterminate` (nutri-env's eval-suite `void` concept). `pass→1.0`,
   `fail→0.0`, `indeterminate→null`; `indeterminate` is excluded from pass/fail stats
   and from the SFT accept path.
4. **Binary reward, `reward_version = v2-r1`.** Graded reward is deferred; if added it
   gets a new version, does not mutate old records, ships a monotonicity test, and
   re-evaluation is an explicit separate pass.
5. **Rubric layering** (§13): a fixed hard contract; a diagnostic-only soft rubric that
   never changes `status`/`reward` in v2.0.
6. **Versioning on every record + manifest**: `oracle_version`, `rubric_version`,
   `reward_version`, `environment_version`, `catalog_sha`, `task_schema_version`, `seed`,
   `nutrimind_rev` (git sha), `nutrienv_rev`.
7. **Two injected dependencies** (§7): `expander` (the `generate_one` contract) and
   `teacher_complete` (returns `content` + `reasoning_content` separately). Production
   `teacher_complete` is a thin client against `ark/deepseek-v4-flash` on
   `api/plan/v3/chat/completions` (`ARK_API_KEY`), reading `message.content` +
   `message.reasoning_content` + `usage.completion_tokens_details.reasoning_tokens`, with
   `thinking: {"type": "enabled"}` as the length control (ADR-011 amended — replaces the
   DeepSeek-direct `reasoning_effort`); `nutrienv.io.chat.complete_chat` is **not** used
   for rollouts (it sends no thinking param and `_message_text` collapses
   `content`/`reasoning_content`).
8. **Teacher rollout** subclasses `nutrienv.harness.ReActHarness`, overriding the
   completion to use `teacher_complete` and keep `reasoning_content`, and **recording per
   turn** a v2-owned trajectory-metadata structure: at minimum `raw_action_text` (the
   assistant message sent) and `executed_op` (the action `NutriEnv.step` received);
   also `parse_status` / `fallback_used` / `fallback_reason` as computed by v2's own
   re-parse. The base `_parse_action`, message assembly, and `context_messages` are
   reused to *drive* the env; action-legality detection (§12) uses this metadata, not
   `_parse_action`'s internal fallback. `version="v2"`, `context_limit=None`, max steps
   per `runner` policy.
   **This instrumented subclass + its trajectory-metadata schema is a discrete
   implementation component**, not a serializer detail — it will need its own
   implementation ticket, upstream of `serialize` and the verifier.
9. **3-leg `update+log→recommend`** — **VERIFIED viable from public symbols**
   (ticket 002 Part B, 2026-09-09). Recipe:
   `u = generate_one(family="update", person=P, seed=S, shell="upd-add-allergy-short", …)`;
   `l = generate_one(family="log", person=P, seed=S, expander=…)`;
   `update_sub = dataclasses.replace(u.oracle, ledger=None, ledger_tail=None)` (strip the
   standalone update oracle's stale `ledger=()`, else the update sub scores `log_miss`);
   `log_sub = Oracle(ledger_tail=tail, ledger=(*s0.ledger,*tail), profile=deepcopy(expected))`;
   `rec_sub = Oracle(profile=deepcopy(expected), last_plan=[], ledger=(*s0.ledger,*tail),
   plan_must_be_safe=True, plan_must_fit_windows=True,
   plan_windows=plan_windows_for_meal(expected.windows, eaten_after_log, "dinner"))`;
   `Task(id, "composite", query, s0, compose_oracles(update_sub, log_sub, rec_sub), …)`.
   The result is shape-equivalent to the frozen exam item `adr24-comp-8255`.
   **Gate policy for this shape**: `stage_a_code_gate == []` **and**
   `validate_draft ∈ ([], ["update oracle ledger is missing"])` (the exam item trips the
   same false-positive) **and** `check_achievable` reachable **and** a correct-replay
   `Scorer` Pass. No `_update_from_template` / `_bind_log_foods` import.
10. **v2 SFT loader is separate**; v1's `src/training/sft/train.py` is untouched. v2
    records carry `segments` + `train_on` (per-message); the v2 loader derives the
    token-level mask after `apply_chat_template` with the **v2** tokenizer. No
    tokenizer-bound token mask is stored.
11. **`plan` truncation**: token-exact with an injected student tokenizer
    (`tokenizer_name`; `transformers` is a core dep) to `plan_max_tokens`; char
    heuristic fallback with `meta.plan_truncation="char"`.
12. **`amount_path`** derived per persona (design §6/§8): `gym` → `explicit_grams` ~60 %;
    `everyday`/`cut` → `named_measure` + `unspecified`, `explicit_grams` ~15 %; ~15 %
    ounce phrasing within `named_measure`; `unspecified` ≤ 20 %.
13. **`gram_anchor`** off by default; per-family on from config when a 50-draft probe
    shows `bind_fail_rate` 35–60 %. `enable_semantic_vote` stays `False`.
    **Ticket 002 step (c) finding:** for the 3-leg `update+log→recommend` family, the
    naive `LogExpander` + `ark/deepseek-v4-flash` + forced `named_measure` gives
    `bind_fail_rate ≈ 0.75–0.83` (thinking on/off both) — **above** the fallback
    threshold. So for the 3-leg (and likely all expander-backed log families) the §6
    ladder — `gram_anchor`, a persona-derived `amount_path` mix (not forced
    `named_measure`), then `qwen3.8-max` fallback — is **mandatory from the start, not a
    contingency**. Every 3-leg task that *did* bind + gate correct-replayed to a Pass
    (16/16 in the spike), so the risk is authoring throughput, not correctness.
14. **Retry attribution** (§16): attempts 1..k in `build`; transport retries in the LLM
    client; no `build` retry of gates or serialization.
15. **Atomic writes**: `task_packages/`, `rollouts/cache/`, `sft/*.jsonl`,
    `run_manifest.json` via temp + `os.replace`; `rejects/*.jsonl` append-only with
    `flush()` and a finalize pass that drops a trailing partial line.
16. **Determinism**: intents sorted by `task_id`; `sft/train.jsonl` sorted by `task_id`;
    7:2:1 split by `task_id` hash.
17. **Config** in `configs/data_factory.yaml`; resolved-config SHA in the manifest.
18. **ADR-012 amended** to the two-class rule (§18); the amendment is recorded in
    ADR-012's own Amendment log, preserving the original text.

## 23. Open Questions

Ordered by priority. **P0 items block cutting implementation tickets**; the rest are
tuning decisions that can be made inline or carried on a ticket.

### P0 — blocking (resolve before `to-tickets`)

- **OQ-1 · NutriEnv installability** — `nutrienv` is not importable today. Ticket 001:
  establish a reproducible NutriEnv dev/test dependency (source, revision pinning, and
  editable/path/workspace style per the project's package manager — **not** pre-locked to
  `git+https editable`) so `import nutrienv` and a minimal public-API smoke test pass
  locally and in CI.
- **OQ-2 · Single-item environment reconstruction — RESOLVED 2026-09-09** (ticket 002
  Part A, verdict A). `task_to_item` → `freeze_tasks` → `load_split` round-trips
  losslessly to a runnable, Pass-reachable `NutriEnv`; the TaskPackage is **self-contained**
  (no external dataset dependency). Caveat: no public in-memory entry, so reconstruction
  goes through a transient scratch file. Non-blocking upstream ask: `item_to_task` in
  `nutrienv.bench.split.__all__`. §9.1 updated.
- **OQ-7 · Public-API 3-leg composite feasibility + N=40 batch qualification** — ticket
  002 splits this into three levels that are **not interchangeable**:
  - **Part B — RESOLVED YES (2026-09-09).** Public symbols assemble one 3-leg composite
    (shape-equivalent to exam `adr24-comp-8255`); `stage_a_code_gate == []`,
    `check_achievable` reachable, correct-replay `Scorer` Pass, wrong-state tags
    (`log_miss` / `window` / `update_miss`) all correct. `validate_draft` trips a known
    false-positive (`"update oracle ledger is missing"`) the exam item trips too →
    allow-listed. Recipe in §22.9. **Every task that bound + gated correct-replayed to a
    Pass (16/16).**
  - **Part C — NOT a spike; becomes a hard acceptance gate on the 3-leg implementation
    ticket.** Ticket 002 step (c) established that the 3-leg yield is gated by the log-leg
    *colloquialization* bind rate: naive `LogExpander` + `ark/deepseek-v4-flash` gives
    `bind_fail_rate ≈ 0.75–0.83` (thinking on/off both). Reaching **40 accepted Pass**
    (N counts distinct `task_id` after intra-family `semantic_key` dedup; `k = 6` teacher
    attempts; §20 `indeterminate_rate` ≤ 0.05) requires a **tuned** expander (§6
    `gram_anchor` ladder + §8 persona `amount_path` mix + `qwen3.8-max` fallback) **and**
    the teacher-rollout machinery (OQ-16) **and** real teacher spend — all implementation.
    The gate on the 3-leg ticket: **40 accepted Pass under the §6 ladder; if it can't
    reach 40, re-size design §7** (an explicit, recorded change — not a silent accept).
    Fallbacks if the ladder still can't: a documented ADR-012 exception (private
    `_bind_log_foods`), or an upstream `__all__` PR.

### P1 — decide inline, not blocking

- **OQ-3 · `oracle_version` granularity** — `"nutrienv-<rev>"` (fine for v2.0) vs. a hash
  of the oracle-scoring code path (later).
- **OQ-4 · v2 SFT loader module location** — put it under `src/training/data_factory/`
  (`sft_v2_loader.py` or `sft_v2/`); pick per the existing layout. **Not a blocker**;
  the §9.2 contract holds regardless; the module can be renamed later.
- **OQ-5 · store `train_on`** — yes, store it (for the plan-masking ablation and the
  RLVR/eval materializers), even though for Batch 1 it equals `role=="assistant"`.
- **OQ-6 · missing `reasoning_content`** — one turn: `plan=""` + counter; every turn:
  `serialize.no_plan_any_turn`. Confirm the threshold.
- **OQ-8 · `output_dir`** — `data/student/` (no v1 collision; covered by the `data/`
  gitignore).
- **OQ-9 · persona-ratio citation** — NHANES adult activity distribution or a diet-app
  survey; provisional 65/20/15 everyday/gym/cut. Affects `TRAIN_ROSTER`, not the pipeline.
- **OQ-10 · exact per-family N** — design §7 sums to ~420; `build` targets from config
  and reports actual vs target. For **every** family, "N accepted Pass" now means
  **N distinct `task_id`s**, deduplicated on `semantic_key` within the family first
  (rule fixed in ticket 002 Part C; applies family-wide, not just the 3-leg).
- **OQ-11 · first-attempt teacher temperature** — 0.0 for attempt 1, 0.7 for retries.
- **OQ-12 · `loss_val` file** — a third file `sft/loss_val.jsonl` vs. a slice of
  `sft/holdout.jsonl`. (`mini-exam val` is separate.)
- **OQ-13 · v1 archive + tag** — **not blocking.** Nothing moves now; the boundary note
  is §14; cut `nutrimind-v1-final` and add `docs/archive/v1/` only after the v2 loader is
  stable and the tree is clean.
- **OQ-14 · `.env.example`** — add `ARK_API_KEY` / `ARK_BASE_URL`.
- **OQ-15 · RLVR verifier execution model** — in-process `Scorer` per rollout vs.
  subprocess/service. The RLVR schema (§9.3) does not depend on the answer.
- **OQ-16 · instrumented `ReActHarness` subclass** — the teacher-rollout subclass plus
  its per-turn trajectory-metadata schema (`raw_action_text`, `executed_op`,
  `parse_status`, `fallback_used`, `fallback_reason`) is its own implementation ticket
  (upstream of `serialize` and the verifier), not folded into the serializer. Non-blocking
  for this spec; flagged so `to-tickets` produces it as a discrete unit.

## 24. Further Notes

- `docs/specs/training.md` is v1-specific (`apply_chat_template(enable_thinking=True)`,
  `<tool_call>` single-token vocab, Qwen3-4B) and does not govern v2. A v2 training spec
  is a separate future doc.
- nutri-env's `EXPANDER_MODELS["deepseek-v4-flash-0731"]` (DashScope snapshot) is a
  different id and route from the v2 teacher `ark/deepseek-v4-flash` on `api/plan/v3`
  (ADR-011 amended). Do not conflate.
- nutri-env's eval suite already has a `void` / `is_void` / `clean_pass_rate` concept —
  v2's `indeterminate` is the same idea, named.
- The finalized design doc `/tmp/nutrienv_student_data.md` (which ADR-010/011 reference as
  `docs/plans/nutrimind_v2_data_factory.md`) is **lost** — no copy survives. Its
  load-bearing numbers are preserved in §2.1 above and in ticket 002. **Ticket 021**
  reconstructs the doc at that path from §2.1 + the ADRs + ticket 002.
- `nutrienv` is currently **not importable** in this repo. Ticket 001 is a hard
  prerequisite for all implementation.

## 25. Readiness for `to-tickets`

**Core behaviour seams are decided; implementation boundaries remain constrained by
OQ-2 (single-item environment reconstruction), OQ-7 / ticket 002 (public-API 3-leg
composition), and NutriEnv installability (OQ-1 / ticket 001).**

Decided and stable:
- `build` is a task/data artifact factory, not an experiment runner; one thin top-level
  entry delegating to per-stage callables.
- The **TaskPackage** is the shared source for SFT / RLVR / evaluation; artifacts are
  never reverse-inferred from each other.
- `gates.run` is a pure validation boundary; the tri-state **verifier** is a pure
  boundary; `serialize` is a pure boundary.
- `pass` / `fail` / `indeterminate` and binary reward `v2-r1` are an explicit contract;
  a legal `Scorer=False` stays `fail`, an exception never becomes `fail`.
- The v2 SFT loader is separate from v1; the record carries `segments` + `train_on`; the
  token-level mask is derived in the v2 loader, never stored.
- Public vs private nutri-env API boundary (§18, ADR-012 amended).
- Action-legality detection uses v2-owned trajectory metadata, not `_parse_action`
  internals.

Resolved:
- **OQ-1** — `nutrienv` installs (strict-editable, pinned) and CI is green
  ([run 34299424567](https://github.com/SnackJJ/NutriMind/actions/runs/34299424567),
  9 passed). Ticket 001 CLOSED.
- **OQ-2** — single frozen item round-trips losslessly to a runnable, Pass-reachable
  `NutriEnv` through the public API (via a transient scratch file); TaskPackage is
  self-contained, RLVR needs **no** external dataset file. Ticket 002 Part A, verdict A.

Resolved (cont.):
- **OQ-7 Part B** — public symbols assemble + verify one 3-leg composite
  (shape-equivalent to `adr24-comp-8255`); gate policy pinned in §22.9; every
  bound+gated task correct-replays to a Pass. Ticket 002 Part B, verdict YES. Ticket 002
  CLOSED.
- **ADR-011 amended** — teacher endpoint `deepseek/` direct → `ark/deepseek-v4-flash`
  (`api/plan/v3`); it returns `reasoning_content` multi-turn under `_SYSTEM_V2` by
  default. ReAct-turn reasoning is long/variable → the ~80-tok plan cap is load-bearing.

Carried into implementation (not a pre-`to-tickets` blocker):
- **OQ-7 Part C — the 3-leg family reaching 40 accepted Pass** is a **hard acceptance
  gate on the 3-leg implementation ticket**, not a spike. Ticket 002 step (c) showed the
  bind rate is the constraint (`bind_fail_rate ≈ 0.75–0.83` with a naive expander), so
  the ticket must engage the §6 expander ladder from the start; if it still can't reach
  40, re-size design §7 (recorded, not silent). §22.9 / §23 OQ-7 / §6 updated.
- **OQ-16** — the instrumented `ReActHarness` subclass + teacher client (now an `ark/`
  call) is its own implementation ticket, upstream of `serialize`/verifier.

Dependency chain (complete): **001 ✅ → 002-A ✅ → 002-B ✅** → spec updated (§6, §9.1,
§16, §22.9, §23) → **`to-tickets`**.

**Ready for `to-tickets`.** Tickets 001 and 002 are CLOSED with acceptance evidence
(001: CI run 34299424567; 002: `test_env_reconstruction.py` + `test_three_leg_public_assembly.py`,
19 data-factory tests green). Cut implementation tickets, with the 3-leg ticket carrying
the OQ-7 Part C acceptance gate. OQ-4, OQ-13, OQ-16 do not block and can be decided
inline or carried on a ticket.
