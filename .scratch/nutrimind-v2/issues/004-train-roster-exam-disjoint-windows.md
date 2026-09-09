---
id: 004
title: TRAIN_ROSTER with provably exam-disjoint derive_profile_windows
status: ready-for-agent
depends_on: [003]
spec: ../spec.md
spec_sections: ["2", "US-16", "US-18", "OQ-9"]
---

# 004 — TRAIN_ROSTER with provably exam-disjoint windows

**What to build:** The NutriMind-side people the factory authors tasks for. A
`TRAIN_ROSTER` of fictional `train-*` users whose body facts are chosen so that
`derive_profile_windows` output is **provably disjoint** from every nutri-env `ROSTER`
person's windows — the single isolation point that stops a template-family oracle from
colliding with the exam. Persona split provisionally 65/20/15 everyday/gym/cut (OQ-9;
literature citation deferred, non-blocking — it affects the roster, not the pipeline).

**Blocked by:** 003.

**Status:** ready-for-agent

- [ ] `TRAIN_ROSTER` is importable and every entry has a `train-` prefixed `user_id`
- [ ] a test computes `derive_profile_windows` for every `TRAIN_ROSTER` person and every
      nutri-env `ROSTER` person and asserts the two window sets are disjoint (no shared
      window tuple) — this is the train/exam isolation guarantee
- [ ] `profile_for` / persona lookups resolve for each `TRAIN_ROSTER` person via public
      nutri-env helpers
- [ ] persona counts match the provisional 65/20/15 mix, documented as provisional (OQ-9)
- [ ] `generate_one(family="log", person=<a TRAIN_ROSTER person>, expander=<synthetic>)`
      produces an accepted `Task` for at least one person per persona
