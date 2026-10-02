---
id: 001
title: Semantic brief expander — single-query speech from situation, not a catalog dump
status: CLOSED (2026-09-12)
depends_on: []
spec: ../spec.md
spec_sections: ["Problem Statement", "Solution", "US-1", "US-4", "US-5", "US-6", "US-20"]
adr: [../../docs/decisions/012-nutrienv-read-only-benchmark.md]
design_note: ../../docs/research/single-query-contextual-expander.md
---

# 001 — Semantic brief expander

**What to build:** The expander writes one natural **single query** from a
**semantic brief** (situation, time/meal, source, persona, family intent, natural
handle for an already-chosen canonical entity). Canonical binding stays in code.
The live model does not receive a mechanical catalog-field list. The
`generate_one` expander contract still holds. The lab is not patched.

**Blocked by:** None — can start immediately.

**Status:** CLOSED (2026-09-12)

- [x] an authored task remains one user utterance; no user-dialogue history is introduced
- [x] the LLM-facing payload is a brief (situation / time / source / persona / intent / natural entity handle), not a dumped catalog-field list
- [x] the expander does not invent foods, quantities, preparations, or venues, and preserves the intent's amount path and meal semantics
- [x] the synthetic expander still returns bindable `{query, foods}` with no network
- [x] `generate_one`'s expander callable shape is unchanged; nutri-env-lab is untouched

## Closure notes

- `src/training/data_factory/speech.py`: `SemanticBrief`, `make_brief_expander`, `bind_speech_context`.
- Canonical `food_id` is code-side; the LLM-facing payload is `render_semantic_brief`.
- `generate_one` expander contract unchanged; lab untouched.
