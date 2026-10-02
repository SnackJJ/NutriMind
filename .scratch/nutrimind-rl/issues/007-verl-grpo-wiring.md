---
id: 007
title: veRL GRPO wiring — reward, config, arm assert (no weight sync yet)
status: CLOSED (2026-09-11)
depends_on: [001, 003, 004, 006, "nutrimind-v2/027"]
spec: ../spec.md
spec_sections: ["D7", "D10"]
adr: [../../docs/decisions/015-v2-post-training-stack-trl-sft-verl-rl.md]
---

# 007 — veRL GRPO wiring (testable)

**What to build:** veRL GRPO consumes this spec's rollout results and ticket-003
rewards. Arm assertions from 006 apply. Config names GRPO, G, the band, and
`parallel_tool_calls=false`. v1 `train_grpo.py` / `gigpo_trainer.py` are not
extended.

This ticket does **not** invent how a training checkpoint becomes
`policy_spec.url` — that is ticket 008. Tests inject a frozen `policy_spec` (the
D3 seam).

**Blocked by:** 001, 003, 004, 006, Data Factory 027.

**Status:** ready-for-agent

- [x] GRPO advantage uses only ticket-003 rewards; `indeterminate` and
      zero-variance groups are excluded as in 004
- [x] a config mismatch hits the 006 abort path
- [x] a dry/scripted group produces a non-zero-advantage step when rewards
      differ, and a dropped step when they do not
- [x] v1 GRPO entry points are untouched
