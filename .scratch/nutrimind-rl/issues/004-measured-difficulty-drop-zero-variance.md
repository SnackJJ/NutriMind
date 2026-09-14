---
id: 004
title: Measured p̂ per checkpoint, per-family band, drop zero-variance groups
status: CLOSED (2026-09-11)
depends_on: [001, 003]
spec: ../spec.md
spec_sections: ["D6"]
---

# 004 — Difficulty measurement and group drop

**What to build:** `p̂(task, checkpoint) = n_pass / (n_pass + n_fail)` from k
rollouts. `indeterminate` counted separately. Sweet-spot band is per family and
stored with the checkpoint hash. At train time, groups with fewer than two valid
samples or zero reward variance are dropped (counted in the effective-gradient
fraction denominator), not resampled in-place.

**Blocked by:** 001, 003.

**Status:** ready-for-agent

- [x] `p̂` arithmetic excludes `indeterminate` from numerator and denominator
- [x] a band measured under checkpoint A is refused for checkpoint B
- [x] per-family aggregation is reported
- [x] a zero-variance group is dropped; the next batch draws new tasks, not
      extra rollouts of the same group
