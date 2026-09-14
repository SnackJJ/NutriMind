---
id: 027
title: RLVR export prompt = lab tool schema (supersedes 018)
status: CLOSED (2026-09-11)
depends_on: [006, 010, 023]
spec: ../spec.md
spec_sections: ["9.3", "19.5"]
adr: [../../docs/decisions/014-native-tool-calling-v2-protocol.md]
---

# 027 — RLVR export native-FC prompt

**What to build:** The `target=rlvr` branch writes `rlvr/<task_id>.json` with
`prompt.system = TOOL_SYSTEM_PROMPT`, `prompt.tools = NUTRIENV_TOOLS`,
`prompt.task = "Task:\n<query>"`. No teacher, no messages. Ticket **018 is
SUPERSEDED** (`react_manual("v2")` is not the v2 prompt).

**Blocked by:** 006, 010, 023.

**Status:** ready-for-agent

- [x] `build --target rlvr` writes `rlvr/<task_id>.json` with zero teacher calls
- [x] `prompt` has `system`, `tools`, and `task`; it has no `react_manual` text
- [x] `tools` is the lab schema (same object the teacher/student loops declare)
- [x] `environment` reconstructs to a runnable `NutriEnv`; `reward.map` is the
      tri-state binary map; `termination` matches the TaskPackage
- [x] the export path takes a TaskPackage, never an SFT record
