---
id: 024
title: Extend TurnMeta/EpisodeResult for native tool calling
status: CLOSED (2026-09-11)
commit: 5468a80
depends_on: [003, 023]
spec: ../spec.md
spec_sections: ["9.2", "12"]
adr: [../../docs/decisions/014-native-tool-calling-v2-protocol.md]
---

# 024 — EpisodeResult / TurnMeta for FC

**What to build:** The factory concept types carry a native tool-calling turn. Verifier
still depends only on `end_state`. This is the type RL-001 consumes — do not fork a
second episode type in the RL tree.

`reasoning_content` already exists on `TurnMeta` (ticket 003). This ticket adds the FC
channel and records the no-tool-call case.

**Blocked by:** 003, 023.

**Status:** CLOSED (2026-09-11)

- [x] each turn can store `tool_calls` (list) and the `tool_call_id` used on the
      following `tool` message
- [x] `executed_op` remains what `NutriEnv.step` received (`{"op": name, ...}`) or
      `None` if nothing was stepped
- [x] a turn with no `tool_calls` is representable and distinct (not coerced into an
      op, not a parse-fallback `get_profile`)
- [x] ReAct-only fields `raw_action_text` / `parse_status` / `fallback_used` /
      `fallback_reason` are unused on the FC path (null/false); they are not a second
      protocol
- [x] `EpisodeResult.to_dict` / `from_dict` round-trips the new fields
- [x] existing verifier tests still pass: `verify` reads `end_state`, not tool_calls

## Closure notes

- `TurnMeta.tool_calls` / `tool_call_id` added; empty `tool_calls` + `executed_op is
  None` is the no-call case. RL-001 must reuse this type (no fork).
- `verify` is unchanged; a regression test mutates `tool_calls` and checks the
  verdict is stable.
