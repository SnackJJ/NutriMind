---
id: 005
title: SFT go/no-go — protocol and mixed Pass/Fail on held-out queries, not loss
status: CLOSED (2026-09-12)
depends_on: [004]
spec: ../spec.md
spec_sections: ["US-29", "US-37", "US-38"]
adr: [../../docs/decisions/013-sft-failure-recovery-coverage.md, ../../docs/decisions/014-native-tool-calling-v2-protocol.md]
design_note: ../../docs/research/post-training-query-budget-pilot.md
---

# 005 — SFT cold-start go/no-go

**What to build:** Before outcome-only GRPO, a go/no-go report on held-out
**unique query identities**. SFT loss alone is not a go. The cold start is
insufficient when trajectories are almost all invalid or no-finish, or groups
are all-zero. It is usable for an RL pilot when the student executes the
protocol and produces both Pass and Fail on a meaningful subset of query
groups.

**Blocked by:** 004

**Status:** CLOSED (2026-09-12)

- [x] the report includes schema/tool-call legality, finish vs no-finish, execution and oracle-reconstruction health, Pass@1 by family, family and composite-type coverage, recovery-positive rate where applicable, and mixed-reward group rate at the planned `G`
- [x] a scripted all-invalid or all-no-finish set is insufficient
- [x] a scripted all-zero-group set is insufficient
- [x] a scripted set with mixed Pass/Fail on more than one family is usable
- [x] SFT loss is not a go/no-go input
