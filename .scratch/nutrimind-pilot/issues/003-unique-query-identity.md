---
id: 003
title: Unique query identity — count queries apart from traces, k, G, steps, tokens
status: CLOSED (2026-09-12)
depends_on: [002]
spec: ../spec.md
spec_sections: ["US-22", "US-23", "US-24"]
design_note: ../../docs/research/post-training-query-budget-pilot.md
---

# 003 — Unique query identity

**What to build:** The factory reports **unique query identity** as its own
counter. When several surface realizations are sampled, select by consistency,
unique binding, diversity, then exam/query collision. Teacher `k`, GRPO group
size, rollout count, optimizer steps, and token budget do not inflate this
counter.

**Blocked by:** 002

**Status:** CLOSED (2026-09-12)

- [x] the run manifest reports unique query identity count separately from accepted-trace count and teacher attempts
- [x] two realizations of the same identity count as one; two distinct identities count as two
- [x] sampled realizations are ranked consistency → unique bind → diversity → exam/verbatim/semantic collision
- [x] raising teacher `k` does not raise the unique-query counter
