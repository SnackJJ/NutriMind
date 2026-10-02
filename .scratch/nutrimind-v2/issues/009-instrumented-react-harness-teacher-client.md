---
id: 009
title: Real teacher rollout — instrumented ReActHarness subclass + ark/ teacher client (OQ-16)
status: CLOSED (2026-09-09) — 18 offline tests green (296 dir-wide); live smoke behind guard
commit: 4cecaa0
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

**Status:** CLOSED (2026-09-09)

- [x] a scripted `teacher_complete` (queue of `(content, reasoning_content)`) drives Pass /
      Fail / no-finish / invalid-op episodes through the subclass deterministically
- [x] a produced `EpisodeResult` validates against the ticket-003 type: every `TurnMeta`
      has `raw_action_text` + `executed_op`; a turn where the harness substituted a
      fallback is flagged (`fallback_used=True`); the episode carries `end_state` /
      `reached_finish` / `error`
- [x] the production client returns `content` and `reasoning_content` as separate fields;
      a test asserts they are not collapsed
- [x] no real network call happens without `NUTRIMIND_ALLOW_NETWORK=1`; the API key never
      appears in logs or the episode cache
- [x] one live smoke call against `ark/` `api/plan/v3` returns non-empty `reasoning_content`
      (skipped in CI, runs locally behind the guard)
- [x] the `thinking` length control and the per-turn timeout are read from config; the
      expander client sends `thinking: {"type": "disabled"}`

## Closure notes

- The base class's private `_parse_action` is reused ONLY through
  `super().act()` to drive the env (the ticket's sanctioned use); all
  legality metadata comes from v2's own re-parse in `verify.py`.
- The base `act()` fallback returns `{"op": "get_profile"}` for
  unparsable/unknown ops — the subclass detects the substitution by comparing
  its own re-parse against the executed action (`fallback_used=True`,
  `fallback_reason="parse:<status>"`).
- The driver deliberately drops the eval runner's idle-read/submit_plan early
  breaks: a v2 training trajectory must end in an explicit FINISH op
  (ADR-011); `serialize.last_turn_not_finish` (ticket 008) enforces it
  downstream.
- Config's endpoint is a PATH (`api/plan/v3/chat/completions`); the client
  resolves it against `ARK_BASE_URL` (host-only / host+plan / full-URL
  variants all tested — no segment duplication).
- Retry temperature (0.7 for attempts 2..k) rides the base `extra_body`
  (`{"temperature": 0.7}` merges after the default 0.0) — build constructs
  harnesses per attempt; the transport retries stay in the client.
- urllib (stdlib) — no new dependency; the fake-urlopen tests capture the
  request body, headers, and timeout, and never touch the network.
