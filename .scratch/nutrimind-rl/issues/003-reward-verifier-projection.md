---
id: 003
title: RL reward is a pure projection of the tri-state verifier
status: CLOSED (2026-09-11)
depends_on: [001]
spec: ../spec.md
spec_sections: ["D5"]
---

# 003 — Reward = verifier projection

**What to build:** `pass → 1.0`, `fail → 0.0`, `indeterminate → excluded` from the
group and from `p̂`. No other fields.

**Blocked by:** 001.

**Status:** ready-for-agent

- [x] a test asserts the reward function's dependency surface is only the
      verification result
- [x] `indeterminate` is not mapped to 0.0 or 1.0
- [x] closed-table dispatch: an unknown attribute raises (no silent default)
