# ADR-014: Native Tool Calling Is the v2 Protocol

- **Status**: accepted
- **Date**: 2026-09-11
- **Deciders**: zeqing
- **Supersedes**: the text-op half of [ADR-011](011-batch1-sft-trajectory-short-plan-thinking-teacher.md)
  (short plan, teacher endpoint, truncation, full-log 20k **kept**)

## Context

ADR-011 stored each Batch-1 assistant turn as a short plan concatenated with a JSON
`{"op": …}` blob, parsed out of prose by the ReAct harness. That matched the published
v1.0 *text* protocol. It does not match how agents are trained now: a declared `tools`
schema, structured `tool_calls`, and observations as `tool` messages.

The Data Factory had not yet spent Batch-1 collection (ticket 020 unrun). Switching
after that ticket would mean re-collecting. `nutri-env-lab` already runs a native
tool-calling episode loop against the real teacher endpoint.

Keeping SFT on text-op and RL/eval on tool calling would be the classic train/eval
template mismatch.

## Decision

Every v2 stage — teacher collection, SFT records, student rollout, evaluation — speaks
**native tool calling**:

- tools are declared once, from the lab schema (`NUTRIENV_TOOLS` / `TOOL_SYSTEM_PROMPT`);
- an assistant turn carries `tool_calls`; the observation is a `tool` message keyed by
  `tool_call_id`;
- the environment still consumes `{"op": <name>, …}` inside `NutriEnv.step` — that is
  the env API, not the student protocol;
- `finish` terminates without an environment step;
- **`parallel_tool_calls` is false** in train and eval (one tool call per assistant
  turn). NutriEnv is stateful; lab parallel runs applied multiple writes in one turn
  without mid-turn observations and collapsed Pass (~84 % serial FC → ~33 % parallel
  on v1.0 for the same teacher). A read-only-parallel arm is out of scope for this
  phase.

The short plan remains: it is the truncated teacher `reasoning_content` on that turn,
stored as `reasoning_content`, not as a text prefix in front of JSON. Headline
comparability is **with-reasoning** vs **tools-only**.

Teacher collection stays in the Data Factory. The factory reuses the lab episode loop
with an injected completion; it does not reimplement the loop. Ticket 009's
`TeacherReActHarness` stays CLOSED as history; new tickets replace the teacher path
**before any `target=sft` accepted-record run** (tickets 012 / 013 / 014 / 020 all
consume that path — not only 020).

## Consequences

### Positive
- Train, teacher, and exam share one protocol.
- Aligns with current agent post-training (OpenAI-shaped tools, Qwen native FC).
- Lab already measured the teacher on this channel.

### Negative
- Serialize, RLVR export, and the v2 loader must change before Batch-1 collection.
- Published v1.0 *text ReAct* numbers are a different protocol; FC numbers are compared
  to the lab's serial-FC leaderboard, not to text ReAct.

### Neutral
- ADR-011's teacher pin (Command Code `deepseek/deepseek-v4.1-flash`), ~80-token
  plan cap, Pass-filter, and `max_seq_length=20k` are unchanged.

## Related

- [ADR-011](011-batch1-sft-trajectory-short-plan-thinking-teacher.md),
  [ADR-012](012-nutrienv-read-only-benchmark.md),
  [ADR-015](015-v2-post-training-stack-trl-sft-verl-rl.md)
- `.scratch/nutrimind-v2/spec.md`, `.scratch/nutrimind-rl/spec.md`
