---
id: 009
title: DAPO comparison arm — same rollout, reward, and task pool
status: CLOSED (2026-09-11)
depends_on: [008]
spec: ../spec.md
spec_sections: ["D7"]
adr: [../../docs/decisions/015-v2-post-training-stack-trl-sft-verl-rl.md]
---

# 009 — DAPO comparison arm

**What to build:** One arm that swaps the advantage estimator to DAPO and nothing
else. Same `student_rollout`, same ticket-003 reward, same measured task pool,
same 006 assertions (estimator name differs).

**Blocked by:** 008.

**Status:** ready-for-agent

- [x] switching the arm flag to DAPO does not change reward or rollout code
      paths (test: dependency surface)
- [x] a GRPO arm and a DAPO arm with identical inputs differ only in the
      asserted estimator and the resulting advantages
- [x] GiGPO is not implemented
