---
id: 001
title: student_rollout seam — scripted FC policy, no network
status: CLOSED (2026-09-11)
commit: 91eeaa3
depends_on: ["nutrimind-v2/023", "nutrimind-v2/024"]
spec: ../spec.md
spec_sections: ["D3"]
adr: [../../docs/decisions/014-native-tool-calling-v2-protocol.md]
---

# 001 — student_rollout seam

**What to build:** `student_rollout(policy_spec, task_package, *, k, seed) ->
list[EpisodeResult]`. Reuses the lab FC loop. Injected policy (scripted in tests).
Returns the **factory** `EpisodeResult` from Data Factory ticket 024 — do not define
a second type.

**Blocked by:** Data Factory 023, 024.

**Status:** CLOSED (2026-09-11)

- [x] k episodes on one TaskPackage are independent given seed
- [x] step-budget exhaustion is distinct from `finish`
- [x] a no-tool-call turn is recorded distinctly
- [x] `parallel_tool_calls=false`: a second tool_call in one assistant message is
      not executed
- [x] an error observation is preserved as an error observation
- [x] no network; scripted policy only in this ticket

## Closure notes

- `src/training/rl/rollout.py` wraps factory `rollout_tool_call`. Returns factory
  `EpisodeResult` (ticket 024). No second type.
