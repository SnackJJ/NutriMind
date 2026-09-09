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
The sibling-repo benchmark (`../nutri-env`): the steppable world, `Scorer`, the frozen
v1.0 split, the ReAct harness. Consumed read-only — NutriMind never patches it (ADR-012).
_Avoid_: "the environment repo", "the gym"

**v1.0 (exam)**:
The frozen 63-task NutriEnv split, `catalog_sha256`-pinned. Never touched for training,
early-stop, or prompt search; reported once at the end.
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
expander. `resolve_portion` must back-bind every spoken portion or the draft is dropped.
_Avoid_: "the query gen", "paraphrase"

**expander**:
The model that writes `speech` (`LogExpander` / `UnfitRewriter`). Batch 1 uses
`ark/deepseek-v4-flash`, not `qwen3.8-max`.
_Avoid_: "the writer", "paraphraser"

**teacher**:
The model whose Pass-filtered trajectories become SFT data —
`deepseek/deepseek-v4-flash` **direct**, thinking-on at `reasoning_effort=low`. Same
model family as the eval comparator `ark/deepseek-v4-flash`; the comparison is end-state
Pass, not style.
_Avoid_: "the expert"; do not call it "the oracle" (that is the Scorer's gold state)

**Pass-filter**:
The keep rule — retain a teacher trajectory iff it Passes. RFT / rejection sampling, not
unfiltered distillation.
_Avoid_: "success filter", "quality filter"

### Trajectory

**trajectory**:
One full teacher episode = one multi-turn chat (system + `Task:` + alternating
observation / assistant + terminal `finish`). The SFT training unit; loss is masked to
assistant turns, all at once.
_Avoid_: "sample", "conversation"; "rollout" is the act of running one, not the artifact

**plan**:
The ≤ ~2-sentence reasoning prefix on an assistant turn, taken from the teacher's
`reasoning_content` and hard-truncated. `_parse_action` strips it, so NutriEnv sees only
the op.
_Avoid_: "think block", "rationale"; "CoT" is the general concept, not this field

**op**:
One `{"op": ...}` JSON action per assistant turn (ReAct v2, `_SYSTEM_V2`). The only thing
NutriEnv consumes.
_Avoid_: "tool call" (there is no tool-call channel here), "action JSON"

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
