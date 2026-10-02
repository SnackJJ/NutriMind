---
id: 026
title: Serialize v2 SFT records as native tool calling (§9.2)
status: CLOSED (2026-09-11)
commit: 54117e2
depends_on: [008, 024, 025]
spec: ../spec.md
spec_sections: ["9.2"]
adr: [../../docs/decisions/014-native-tool-calling-v2-protocol.md]
---

# 026 — Serialize FC SFT records

**What to build:** `serialize` writes the ADR-014 §9.2 shape: system =
`TOOL_SYSTEM_PROMPT`, assistant turns with `tool_calls` + truncated
`reasoning_content`, observations as `role=tool` keyed by `tool_call_id`. Rejects the
retired text-op blob (`plan\n{"op": …}` with no `tool_calls`).

Ticket 008 stays CLOSED as the old serializer; this ticket replaces its production
path.

**Blocked by:** 008, 024, 025.

**Status:** CLOSED (2026-09-11)

- [x] an accepted Pass episode serializes to §9.2: `segments` use `step` / `tool` /
      `final`; `train_on` is true only on assistant turns
- [x] `reasoning_content` is truncated to `plan_max_tokens` (same cap as ADR-011)
- [x] a text-op assistant message without `tool_calls` is not written as accepted
- [x] `finish` is a tool call in `FINISH_OPS` and is the last assistant turn
- [x] loader-facing reject rules in this ticket's tests match §9.2 (028 implements
      the loader)

## Closure notes

- Production `serialize()` is FC. Ticket 008 stays CLOSED as the old text-op
  serializer. `build --target sft` uses `rollout_tool_call`.
