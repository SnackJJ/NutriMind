---
id: 006
title: RL unique-query pool ~200, expansion rule, OPD budget 0 until student failures
status: CLOSED (2026-09-12)
depends_on: [005, "nutrimind-v2/027"]
spec: ../spec.md
spec_sections: ["US-39", "US-42", "US-43", "US-45", "US-46"]
adr: [../../docs/decisions/015-v2-post-training-stack-trl-sft-verl-rl.md]
design_note: ../../docs/research/post-training-query-budget-pilot.md
---

# 006 — RL pool, expansion rule, OPD unique-query budget 0

**What to build:** An RL **pilot query budget** of about **200 unique query
identities**, consumed from factory TaskPackages (controlled SFT overlap plus
held-out). Reward stays binary: Pass = 1, Fail = 0, Indeterminate excluded.
Grow the pool only when the listed diagnostics fire — not because the training
curve is noisy. OPD's unique-query budget is **0**; later selection is from
student-induced failure states where teacher and verifier agree and the world
is not catalog-ambiguous. No OPD training in this ticket. No staged reward.

**Blocked by:** 005, Data Factory 027 (RLVR export already CLOSED).

**Status:** CLOSED (2026-09-12)

- [x] the RL pool is counted in unique query identities (~200), not traces or group size
- [x] a held-out unique-query set exists; overlap with SFT is explicit rather than accidental
- [x] reward mapping is Pass=1 / Fail=0 / Indeterminate excluded; no new reward version
- [x] a noisy-curve fixture with healthy mixed groups does not expand the pool
- [x] too few mixed-reward groups, empty family/composite coverage, train-up/held-out-flat, or verifier-shortcut dominance does expand (or request expansion)
- [x] OPD unique-query budget is 0; a catalog-ambiguous or teacher/verifier-disagreement state is refused by the selector
- [x] this ticket does not author tasks and does not train OPD
