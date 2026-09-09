---
id: 021
title: Reconstruct the design doc at docs/plans/nutrimind_v2_data_factory.md (the /tmp original is lost)
status: ready-for-agent
depends_on: []
spec: ../spec.md
spec_sections: ["2.1", "24"]
adr: [../../docs/decisions/010-nutrimind-v2-rescope.md, ../../docs/decisions/011-batch1-sft-trajectory-short-plan-thinking-teacher.md]
---

# 021 — Reconstruct the design doc at docs/plans/nutrimind_v2_data_factory.md

**Context — the original is gone.** ADR-010 / ADR-011 and spec §2.1 / §24 all reference a
finalized design doc at `/tmp/nutrienv_student_data.md`, to be moved to
`docs/plans/nutrimind_v2_data_factory.md`. As of 2026-09-09 that `/tmp` file **no longer
exists** (nor `/tmp/nutrienv-data-review.md`, the prior data-factory code review), and no
copy survives anywhere on this machine. So this is a **reconstruction**, not a move.

**What to build:** `docs/plans/nutrimind_v2_data_factory.md`, reconstructed from the
sources that captured its load-bearing content:

- spec §2.1 "Design provenance" — the Batch-1 ≈ 420 target, the family mix (composite
  ~57% / recommend ~17% / evaluate ~13% / log ~10% / update ~3%), the 3-leg target = 40 /
  `k = 6`, `max_seq_length` 20k / `context_limit=None`.
- ADR-010 (v2.0 rescope), ADR-011 (trajectory shape + amended teacher endpoint),
  ADR-012 (NutriEnv read-only).
- ticket 002 step (c) — the §6 expander ladder (`gram_anchor` + persona `amount_path`
  mix + `qwen3.8-max` fallback) and the measured `bind_fail_rate ≈ 0.75–0.83`.

Give it the section numbers the other tickets cite: a **§6** (expander ladder), a **§7**
(per-family N table, summing to ~420), a **§8** (persona / `amount_path`). Write the
3-leg assembly recipe (its §5 / §8 in the old draft) using **public** nutrienv symbols
per spec §22.9 — never `_update_from_template` / `_bind_log_foods` / a `compose3.py`.
Mark every reconstructed number with its source (spec §2.1, an ADR, or ticket 002); flag
anything that cannot be sourced as an open item rather than inventing a value.

**Note on status:** `docs/plans/` is git-ignored (`docs/agents/issue-tracker.md`), so this
doc is **local-only** — not committed, not a team-shared canonical artifact. The
committed sources of truth remain the spec and the ADRs; this doc is a convenience
reference so "design §7" in tickets 012 / 013 / 020 resolves to something concrete.

**Blocked by:** None — can start immediately (all sources already exist in-repo).

**Status:** ready-for-agent

- [ ] `docs/plans/nutrimind_v2_data_factory.md` exists with a §6 / §7 / §8 the other
      tickets can cite
- [ ] the §7 per-family N table sums to ~420 and matches spec §2.1 and ticket 002
- [ ] the 3-leg recipe uses only public nutrienv symbols (spec §22.9); no
      `_update_from_template` / `_bind_log_foods` / `compose3.py`
- [ ] the teacher endpoint reads `ark/deepseek-v4-flash` on `api/plan/v3` (ADR-011 amended),
      not `deepseek/` direct
- [ ] every reconstructed figure carries a source tag; unsourced items are listed as open,
      not guessed
- [ ] ADR-010 / ADR-011's "Related" links to the doc path now resolve
