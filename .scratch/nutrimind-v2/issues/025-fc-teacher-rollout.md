---
id: 025
title: Teacher rollout via lab FC loop (injected completion)
status: CLOSED (2026-09-11)
commit: 01ef381
depends_on: [023, 024]
spec: ../spec.md
spec_sections: ["4.3", "7", "16"]
adr: [../../docs/decisions/014-native-tool-calling-v2-protocol.md]
---

# 025 — Teacher rollout via lab FC loop

**What to build:** Production teacher collection uses `run_episode_tool_call` +
`NUTRIENV_TOOLS` with the injected `teacher_complete`. `parallel_tool_calls=false`.
Ticket 009's `TeacherReActHarness` stays CLOSED; this path replaces it for any new
`target=sft` run.

**Blocked by:** 023, 024.

**Status:** CLOSED (2026-09-11)

- [x] a scripted completion queue of `(reasoning_content, tool_calls)` drives Pass /
      Fail / no-finish / no-tool-call / invalid-tool episodes with no network
- [x] each produced `EpisodeResult` validates against ticket 024: `tool_calls` +
      `executed_op`; no-tool-call turns are distinct; a second tool_call in one
      assistant message is not executed
- [x] `reasoning_content` is captured per turn and not collapsed into `content`
- [x] the lab loop is reused, not copied into NutriMind
- [x] no real network call without `NUTRIMIND_ALLOW_NETWORK=1`

## Closure notes

- `src/training/data_factory/rollout_fc.py` injects `teacher_complete` into
  `run_episode_tool_call` (patches the lab's `post_chat_completion_raw` + `NutriEnv`).
- `TeacherReActHarness` is untouched (ticket 009 CLOSED).
