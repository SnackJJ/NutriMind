---
id: 009
title: Real teacher rollout — instrumented ReActHarness subclass + ark/ teacher client (OQ-16)
status: ready-for-agent
depends_on: [003]
spec: ../spec.md
spec_sections: ["4.3", "7", "16", "22.7", "22.8", "OQ-16", "18"]
adr: [../../docs/decisions/011-batch1-sft-trajectory-short-plan-thinking-teacher.md]
---

# 009 — Instrumented ReActHarness subclass + ark/ teacher client

**What to build:** The machinery that turns a TaskPackage into a real teacher
**trajectory**, upstream of the verifier and `serialize`.

**(a) Instrumented `nutrienv.harness.ReActHarness` subclass** — completion via the
injected `teacher_complete`; keeps `reasoning_content` per turn; **populates the shared
`TurnMeta` and `EpisodeResult` types from ticket 003** — per turn at minimum
`raw_action_text` (assistant message sent) and `executed_op` (action `NutriEnv.step`
received), plus `parse_status` / `fallback_used` / `fallback_reason` from v2's **own**
re-parse; per episode `end_state`, `turns`, `reached_finish`, `error`. Reuses the base
`_parse_action` / message assembly / `context_messages` only to *drive* the env.
`version="v2"`, `context_limit=None`, max steps per `nutrienv.harness.runner` policy.

**(b) Thin production `teacher_complete`** — against `ark/deepseek-v4-flash`
`api/plan/v3/chat/completions` (`ARK_API_KEY`; ADR-011 amended — **same endpoint and key
as the expander**). Reads `message.content` + `message.reasoning_content` +
`usage.completion_tokens_details.reasoning_tokens`, keeping `content` and
`reasoning_content` **separate** (nutri-env's `complete_chat` / `_message_text` collapse
them). `thinking: {"type": "enabled"}` is the length control (replaces the DeepSeek-direct
`reasoning_effort`) — ReAct-turn reasoning is long / variable (≤ ~2.3k tok/turn observed;
the ~80-tok `plan` cap is load-bearing downstream). Retry attribution: transport (HTTP)
retries in this client; attempts 1..k belong to `build` (ticket 011). Real calls guarded
by `NUTRIMIND_ALLOW_NETWORK=1`; `ARK_API_KEY` from env, never logged.

**Blocked by:** 003. (Produces the `TurnMeta` / `EpisodeResult` that ticket 007 consumes —
the shared types live in 003, so there is no 007↔009 build-order edge.)

**Status:** ready-for-agent

- [ ] a scripted `teacher_complete` (queue of `(content, reasoning_content)`) drives Pass /
      Fail / no-finish / invalid-op episodes through the subclass deterministically
- [ ] a produced `EpisodeResult` validates against the ticket-003 type: every `TurnMeta`
      has `raw_action_text` + `executed_op`; a turn where the harness substituted a
      fallback is flagged (`fallback_used=True`); the episode carries `end_state` /
      `reached_finish` / `error`
- [ ] the production client returns `content` and `reasoning_content` as separate fields;
      a test asserts they are not collapsed
- [ ] no real network call happens without `NUTRIMIND_ALLOW_NETWORK=1`; the API key never
      appears in logs or the episode cache
- [ ] one live smoke call against `ark/` `api/plan/v3` returns non-empty `reasoning_content`
      (skipped in CI, runs locally behind the guard)
- [ ] the `thinking` length control and the per-turn timeout are read from config; the
      expander client sends `thinking: {"type": "disabled"}`
