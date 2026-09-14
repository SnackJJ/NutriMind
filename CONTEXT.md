# NutriMind

NutriMind v2.0 trains a small open-weight student (Qwen3.5-2B) to act inside NutriEnv and
reports it on the same frozen exam as the public flash/pro leaderboard. This glossary
covers the v2 student and its data factory; phase-1 (v1) vocabulary is archived with the
code.

## Language

### Project & world

**NutriMind v2.0**:
The current project — a Qwen3.5-2B student trained and scored on frozen NutriEnv v1.0 via
a NutriMind-owned data factory.
_Avoid_: "the student project", "phase 2"

**NutriMind v1**:
The archived phase-1 system (Qwen3-4B, 6 T1–T4 tools, RAG, mock `evaluate.py`).
Engineering history; never mixed into v2.
_Avoid_: "old NutriMind", "phase 1" as a live thing

**NutriEnv**:
The sibling-repo benchmark: the steppable world, `Scorer`, the frozen v1.0 split, and
the native tool-calling harness. v2 consumes the lab checkout (`../nutri-env-lab`)
read-only (ADR-012 amended) — NutriMind never patches it.
_Avoid_: "the environment repo", "the gym", "nutri-env" as a second living pin

**v1.0 (exam)**:
The frozen 63-task NutriEnv split, `catalog_sha256`-pinned. Never touched for training,
early-stop, or prompt search; reported once at the end. Not the train / val / mini-exam
splits (those are `TRAIN_ROSTER` and disjoint from these 63).
_Avoid_: "the test set", "the benchmark" (ambiguous)

**Pass**:
`Scorer.score(end_state, oracle)["passed"] is True` — the finished world matches the
oracle's end state. The only headline metric (plus pass@k).
_Avoid_: "correct", "success" (unqualified)

### Data factory

**Data factory**:
The NutriMind-side loop (`src/training/data_factory/`) that builds off-exam tasks, runs
the teacher, Pass-filters, and serializes SFT traces. Borrows `generate_one` internals;
is not nutri-env's retired `legacy_run_batch`.
_Avoid_: "the mill" (that names the retired nutri-env batch layer), "generator"

**Batch 1 / Batch 2**:
Successive waves of factory output. Batch 1 ≈ 420 Pass traces over the shapes
`generate_one` can author. Batch 2 = the escalation set (amend / starve / closed-list;
full CoT), authored only if a Batch-1 diagnostic demands it. Unrelated to the exam's
`Task.tier` and `EVALUATE_TIERS`.
_Avoid_: "第一档/第二档", "tier 1/tier 2", "phase A/B"

**intent**:
The pure-code seed tuple `(family, person, seed, occasion, shell/slots, amount_path,
knife, steps, tier)` enumerated before any LLM call. First of the three authoring stages
(intent → Oracle → speech).
_Avoid_: "spec", "config", "prompt"

**speech**:
The one LLM pass that turns a code intent into a colloquial `{query, foods}` via the
expander. The task remains a **single query**: situation lives inside that utterance
("stopped by the market this morning", "for dinner"), not in a user-dialogue history.
`resolve_portion` must back-bind every spoken portion or the draft is dropped.
_Avoid_: "the query gen", "paraphrase", "multi-turn user dialogue" as a factory task type

**expander**:
The model that writes `speech` (`LogExpander` / `UnfitRewriter`). It sees a
**semantic brief**, not a dumped catalog row. Batch 1 uses
`deepseek/deepseek-v4.1-flash` on the Command Code Provider API, not
`qwen3.8-max`.
_Avoid_: "the writer", "paraphraser"

**semantic brief**:
The small fact set the expander is allowed to see for one intent: situation, time/meal,
source, persona, family intent, and a natural handle for the already-chosen canonical
entity. Canonical binding is done in code before this brief exists.
_Avoid_: "prompt dump", "catalog fields", "metadata list"

**query/entity consistency**:
The post-speech check that the query and structured `foods` bind uniquely to the
intended canonical entity and match the intent's meal, amount, preparation, and venue.
Fail → regenerate or reject; the Scorer never guesses among catalog variants.
_Avoid_: "quality filter", "semantic vote"

**unique query identity**:
One distinct natural-language user query after expansion and collision checks. The
unit of the SFT / RL / OPD query budget. Not a trajectory, not a teacher attempt, not
a GRPO group, not a token.
_Avoid_: "sample count", "query" as a synonym for traces

**pilot query budget**:
Provisional unique-query counts for a feasibility run: SFT cold start 100, RL pool
~200, OPD 0 until student-induced failures. Not Batch 1, not a proven minimum, and
not a change to the production factory family mix.
_Avoid_: "Batch 1", "minimum data", "theoretical floor"

**teacher**:
The model whose Pass-filtered trajectories become SFT data —
`deepseek/deepseek-v4.1-flash` on `https://api.commandcode.ai/provider/v1`,
`thinking: {"type": "enabled"}` as the length control (ADR-011 amended
2026-09-13). Same endpoint and `COMMANDCODE_API_KEY` as the `expander`. Same
model as the eval comparator; the comparison is end-state Pass, not style.
_Avoid_: "the expert"; do not call it "the oracle" (that is the Scorer's gold state)

**Pass-filter**:
The keep rule — retain a teacher trajectory iff it Passes. RFT / rejection sampling, not
unfiltered distillation.
_Avoid_: "success filter", "quality filter"

### Trajectory

**trajectory**:
One full teacher episode = one multi-turn chat (system + `Task:` + assistant
`tool_calls` + `tool` observations + terminal `finish`). The SFT training unit; loss is
masked to assistant turns, all at once.
_Avoid_: "sample", "conversation"; "rollout" is the act of running one, not the artifact

**tool call**:
The v2 protocol channel: a declared `tools` schema, an assistant turn that carries
`tool_calls`, and the environment observation returned as a `tool` message keyed by
`tool_call_id`. Train and eval use this channel. `finish` terminates without an
environment step. Parallel tool calls are off (one call per assistant turn).
_Avoid_: "op" as the protocol, "text-op", "ReAct JSON", "function call" as a second name

**plan**:
The ≤ ~2-sentence reasoning for an assistant turn, taken from the teacher's
`reasoning_content` and hard-truncated. Stored on the turn as `reasoning_content`, not
as a text prefix in front of JSON. Comparability reports **with-reasoning** vs
**tools-only** (tool_calls kept, reasoning stripped).
_Avoid_: "think block", "rationale"; "CoT" is the general concept, not this field

**op**:
Retired as a v2 *protocol* word. Still the key on the dict `NutriEnv.step` consumes
(`{"op": <tool name>, ...}`) — an environment API, not what the student emits.
_Avoid_: using "op" for a student turn; "action JSON"

**TRAIN_ROSTER**:
The NutriMind-side people (`roster_train.py`), `train-*` user_ids, body facts chosen so
`derive_profile_windows` output is provably disjoint from nutri-env's `ROSTER`. The
single isolation point between train and exam.
_Avoid_: "our roster", "synthetic users"

### Validation

**mini-exam val**:
30 frozen fresh TRAIN_ROSTER tasks, oracle-verified but not teacher-rolled. The student
is run through them and scored with `Scorer`; drives checkpoint selection. Never v1.0.
_Avoid_: "the val set" (ambiguous), "dev set"

**loss val**:
A 7:2:1 hold-out of SFT traces for a next-token loss curve. A weak early-stop signal,
secondary to the mini-exam.
_Avoid_: "validation split"

### RL

**student rollout**:
One student episode inside NutriEnv, driven by an injected policy spec, returning the
factory's `EpisodeResult`.
_Avoid_: "generation", "sample"

**sweet-spot band**:
The configured pass-rate interval a task must fall in to be selected for RL, set per
family, measured against a checkpoint hash.
_Avoid_: "medium", "difficulty label"

**arm**:
One experiment configuration. It asserts its own settings at startup: reward version,
reference-model revision, task-selection policy and band, advantage estimator, rollout
k, exam revision, parallel-tool policy.
_Avoid_: "run", "setting"

**effective-gradient fraction**:
The share of sampled groups that produce a non-zero advantage (enough valid samples and
non-zero reward variance).
_Avoid_: "useful batch rate"
