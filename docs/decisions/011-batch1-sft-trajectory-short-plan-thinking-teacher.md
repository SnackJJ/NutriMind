# ADR-011: Batch-1 SFT Trajectory — Short Plan + JSON Op, Thinking Teacher

- **Status**: accepted — **protocol half superseded by [ADR-014](014-native-tool-calling-v2-protocol.md)**
- **Date**: 2026-09-08
- **Amended**: 2026-09-09 — teacher endpoint `deepseek/` direct → `ark/deepseek-v4-flash` (`api/plan/v3`); 2026-09-11 — text-op shape superseded by native tool calling (ADR-014); 2026-09-13 — teacher + expander → Command Code Provider API `deepseek/deepseek-v4.1-flash`; see the Amendment log
- **Deciders**: zeqing
- **Supersedes**: reverses the non-thinking default in `docs/plans/nutrienv_student.md:37`

## Context

`docs/plans/nutrienv_student.md:37` sets the student's default to **non-thinking**
("small Qwen3.5 thinking traces are extremely long; NutriEnv needs one JSON `op` per
turn"), with thinking listed only as an ablation. That was one line, no ADR, and it
contradicts phase-1 (ADR-001 `enable_thinking=True`) and the RFT / STaR / FireAct
literature the same plan cites (all keep rationales).

Two facts from the code:

1. The published v1.0 flash/pro reports ran every model with `reasoning_tokens: 0`.
   Under that bare `ark/` call, `deepseek-v4-flash` emits ~13 tokens/step — pure JSON,
   no rationale (63 tasks, 9,076 completion tokens total). "Distil the teacher's chain
   of thought" is not free from that endpoint.
2. `nutrienv.io.chat.complete_chat` sends a minimal payload (no thinking param) and
   `_message_text` returns `content` **or** `reasoning_content`, never both. The
   DeepSeek **direct** API (`api.deepseek.com`,
   `extra_body={"thinking":{"type":"enabled"}}`, `reasoning_effort`) does return
   `message.reasoning_content` alongside `content`.

The reasoning-heavy families (recommend / evaluate / composite) are 64% of the exam and
100% of the expected wall; asking a 2B to do multi-constraint arithmetic in one forward
pass with no scratchpad is the failure mode. A short scratchpad is also what keeps
`pass@8` on composite non-zero so GRPO has something to run on (ADR-009 collapse risk).

## Decision

Each Batch-1 SFT assistant turn is **a short plan followed by one JSON op** — the
four-part loop shape `plan / tool_call / tool_result / final`:

- **plan** — ≤ ~2 sentences / ~80 tokens, sourced from the teacher's `reasoning_content`,
  hard-truncated. Trained on (not masked).
- **tool_call** — one `{"op": ...}`. `nutrienv.harness.react._parse_action` extracts the
  first JSON object regardless of the plan prefix, so `NutriEnv` and the exam are
  unaffected.
- **tool_result** — the Env observation, delivered as the next `user` turn
  (`Observation:\n{...}`, the harness's 6000-char cap left as-is).
- **final** — the trajectory must end in an explicit `submit_plan`/`finish`; a teacher
  trace that hits the step limit without `finish` is dropped.

**Teacher** = `deepseek/deepseek-v4-flash` **direct**, `thinking` enabled at
`reasoning_effort="low"`, `temperature=0.7` for `k>1` retries. NutriMind runs its own
teacher client (OpenAI SDK; `content` and `reasoning_content` stored as separate
fields). `complete_chat` / `_message_text` are not used for rollouts.
*(Superseded by the 2026-09-09 amendment below — teacher endpoint is now
`ark/deepseek-v4-flash` on `api/plan/v3`.)*

**Context window** — training and evaluation both use `context_limit=None` (full log, the
published口径); `max_seq_length = 20k`; whole-trajectory SFT with the loss mask over all
assistant turns at once; episodes over 20k are dropped from SFT, never slid.

This **reverses `nutrienv_student.md:37`**: short thinking is the Batch-1 default; bare
non-thinking is the ablation. Full-length CoT distillation (`ark/deepseek-v4-pro` or
flash + `<think>` prompt, `rewrite_think` compression to ≤100 tok/turn, observation
trimming, 24k) is a **Batch-2 escalation**, taken only if the B3 diagnostic shows
`pass@8(composite) − pass@1(composite) < ε` **and** `pass@8(composite) < ~0.15`.

## Consequences

### Positive
- Preserves the RFT / STaR rationale-keeping the plan's own literature calls for.
- Gives the 2B a scratchpad on the families that are the wall.
- Keeps `pass@8` on composite alive → GRPO gate is not dead on arrival.
- Short plan holds `max_seq_length` at 20k with no compression infrastructure.

### Negative
- Student output is no longer byte-identical to the published bare-JSON protocol.
  Comparability rests on: `_parse_action` strips the plan, so the Env trajectory and
  end-state Pass are the same. Report the student both ways (with plan / plan-stripped)
  on the 63.
- +~80 tokens/turn accumulates in full-log rollouts, GRPO included.
- Teacher is `deepseek-v4-flash` **direct + thinking-low** — same model family as the
  eval comparator `ark/deepseek-v4-flash`, not the identical configuration. Resume
  wording: the comparison is end-state Pass, not "we matched flash".

### Neutral
- `reasoning_effort="low"`, not `"high"`; the ~80-token cap is a hard truncation, not a
  prompt request.
- non-thinking vs thinking on the SFT checkpoint is already a plan §D ablation; this
  only flips which one is the default.

## Amendment log

### 2026-09-13 — teacher + expander: Command Code Provider API (`deepseek/deepseek-v4.1-flash`)

Live probe (2026-09-13; `.scratch/nutrimind-v2/spikes/029_commandcode_v41_flash.txt`):

- Base `https://api.commandcode.ai/provider/v1`, OpenAI-compatible
  `/chat/completions`. Credential `COMMANDCODE_API_KEY`.
- Wire model id is `deepseek/deepseek-v4.1-flash` (the catalog name). Bare
  `deepseek-v4.1-flash` is rejected (`unsupported_model`).
- Plan text is `message.reasoning` (not `reasoning_content`). Usage still
  reports `completion_tokens_details.reasoning_tokens`. Native tool calling
  works (`finish_reason=tool_calls`, OpenAI-shaped `tool_calls`).
- Python's default urllib User-Agent is Cloudflare 1010; the factory client
  sends `NutriMind-data-factory/1.0`.
- `thinking: {"type": "disabled"}` does **not** drop `reasoning` on this
  model. Expander still consumes `content` only.

**Amended decision:** teacher and expander share this provider and model.
`configs/data_factory.yaml` pins the full completions URL and
`COMMANDCODE_API_KEY`. The client forwards `tools` / `parallel_tool_calls`
and maps `reasoning` → `reasoning_content`. Unchanged: thinking as the
teacher length-control flag, ~80-token plan truncation, Pass-filter,
native tool calling (ADR-014), `max_seq_length=20k`.

### 2026-09-11 — protocol half superseded (native tool calling)

The text-op shape (`plan` concatenated with `{"op": …}`, parsed by ReAct) is
**superseded by [ADR-014](014-native-tool-calling-v2-protocol.md)**. Assistant turns
now carry `tool_calls`; observations are `tool` messages; the plan is stored as
truncated `reasoning_content`.

**Unchanged by that supersession:** teacher endpoint and credential; thinking as
length control; ~80-token plan cap; Pass-filter; whole-trajectory SFT;
`max_seq_length=20k` / `context_limit=None`; non-thinking as the ablation.

Comparability is **with-reasoning / tools-only**, not "plan prefix stripped by
`_parse_action`".

### 2026-09-09 — teacher endpoint: `deepseek/` direct → `ark/deepseek-v4-flash` (`api/plan/v3`)

The original Context and Decision chose the **DeepSeek direct** API for the teacher
because it was the only *verified* way to get `message.reasoning_content` (the "plan"),
and the published v1.0 reports showed `reasoning_tokens: 0` from `ark/deepseek-v4-flash`.

A live probe (2026-09-09, ticket 002; 3 calls) established:

- `ark/deepseek-v4-flash` on the **`api/plan/v3/chat/completions`** endpoint returns
  **both** `message.content` (the op JSON) **and** `message.reasoning_content` (the CoT)
  **by default** — no `thinking` param required. `thinking: {"type": "enabled"}` is
  accepted and may modulate length.
- Reasoning length is reported at
  `usage.completion_tokens_details.reasoning_tokens` (top-level `usage.reasoning_tokens`
  is `None`). The published reports read the top-level field, so their "reasoning_tokens:
  0" was a **measurement artifact** — the flash model was reasoning all along; the report
  did not count it and `_message_text` discarded it.

**Amended decision:**

- **Teacher endpoint = `ark/deepseek-v4-flash` on `api/plan/v3`**, `ARK_API_KEY`. Same
  endpoint and key as the **expander** — one provider, one credential (resolves the
  Q6/Q7 endpoint-consolidation intent).
- The teacher client reads `message.content` + `message.reasoning_content` +
  `usage.completion_tokens_details.reasoning_tokens`. `thinking: {"type": "enabled"}` is
  the length control (replaces `reasoning_effort` from the DeepSeek-direct shape);
  observed reasoning is already short (~60–220 tok/turn in the probe).
- **Unchanged:** NutriMind still runs its own thin client (nutri-env's
  `complete_chat` / `_message_text` collapse `content`/`reasoning_content` and cannot be
  used); the instrumented `ReActHarness` subclass still captures `reasoning_content` +
  `raw_action_text` + `executed_op` per turn; the four-part trajectory shape; the ~80-tok
  plan truncation; full-log 20k context.
- **Confirmed (2026-09-09, spike (c)):** `reasoning_content` **stays populated in a
  multi-turn ReAct episode** under `react_manual("v2")` (3 turns, all populated:
  `reasoning_tokens` 739 / 180 / 2289; `content` a clean single JSON op each turn).
- **New constraint from the same spike:** ReAct-turn reasoning is **long and highly
  variable** (up to ~2.3k tokens/turn observed), not the ~60–220 tok the single-Q probe
  suggested. So the raw teacher `reasoning_content` **cannot be used as the plan
  verbatim** — the ~80-token hard truncation (already in this ADR) is load-bearing, and
  the teacher call should pass a length control (`thinking` budget / a "reason in ≤N
  tokens" system-prompt line) so cost and latency stay bounded. The **expander** call
  should set `thinking: {"type": "disabled"}` — its output is structured `{query, foods}`
  JSON, no reasoning needed.

## Related

- `docs/plans/nutrienv_student.md` (§2.1, §D), `docs/plans/nutrimind_v2_data_factory.md`
- [ADR-001](001-pure-text-tool-calling.md) (phase-1 used `enable_thinking=True`)
- [ADR-009](009-grpo-reward-redesign-against-shortest-path-collapse.md)
  (`<think>` stripping in `compute_state_key` already precedented)
- [ADR-010](010-nutrimind-v2-rescope.md), [ADR-012](012-nutrienv-read-only-benchmark.md),
  [ADR-014](014-native-tool-calling-v2-protocol.md)
