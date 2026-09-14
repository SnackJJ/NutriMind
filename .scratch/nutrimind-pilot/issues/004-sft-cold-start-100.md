---
id: 004
title: SFT cold-start overlay — 100 unique queries; production family mix untouched
status: CLOSED (2026-09-12)
depends_on: [003]
spec: ../spec.md
spec_sections: ["US-25", "US-26", "US-27", "US-28", "US-49", "US-50"]
adr: [../../docs/decisions/011-batch1-sft-trajectory-short-plan-thinking-teacher.md]
design_note: ../../docs/research/post-training-query-budget-pilot.md
---

# 004 — SFT cold-start 100 unique queries

**What to build:** A **pilot query budget** overlay that authors and
Pass-filters an SFT cold start of **100 unique query identities**,
coverage-balanced across families. Teacher `k = 6` stays a teacher setting.
The production factory family mix (Batch-1 ≈ 421 accepted Pass, ticket
`nutrimind-v2/020`) is not edited. This overlay is not a substitute for 020.

**Blocked by:** 003

**Status:** CLOSED (2026-09-12)

- [x] the overlay run stops on 100 unique query identities, coverage-balanced, not on 100 traces
- [x] production family `target_n` values are byte-identical before and after the overlay
- [x] teacher `k = 6` can still apply per task without being treated as the query budget
- [x] ticket `nutrimind-v2/020` remains the production ≈ 420 accepted-Pass run and is not closed or resized by this overlay
