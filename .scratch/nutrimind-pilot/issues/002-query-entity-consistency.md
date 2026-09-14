---
id: 002
title: Query/entity consistency — unique bind or regenerate/reject; Scorer never guesses
status: CLOSED (2026-09-12)
depends_on: [001]
spec: ../spec.md
spec_sections: ["Solution", "US-11", "US-12", "US-17", "US-18"]
adr: [../../docs/decisions/012-nutrienv-read-only-benchmark.md]
design_note: ../../docs/research/single-query-contextual-expander.md
---

# 002 — Query/entity consistency validator

**What to build:** After **speech**, a **query/entity consistency** check. The
query and structured `foods` must bind uniquely to the intended canonical entity
and match the intent's meal, amount, preparation, and venue. On failure: consume
the expander retry budget, then author-reject. The Scorer is never asked to pick
among catalog variants.

**Blocked by:** 001

**Status:** CLOSED (2026-09-12)

- [x] foods outside the allowed binding, an unselected/conflicting variant, missing required disambiguation, intent-conflicting wording, query/`foods` disagreement, or multi-entity ambiguity each produce a distinct `author.*` reject
- [x] a uniquely binding utterance with matching `foods` is kept
- [x] regeneration uses the existing expander retry budget and then rejects; it does not loop
- [x] an ambiguous two-variant query never becomes a gated TaskPackage
- [x] the reject histogram can tell consistency failure apart from portion-bind failure

## Closure notes

- `src/training/data_factory/consistency.py` is the author-stage check.
- Bind/schema rejects do not consume the consistency retry budget.
- Recovery traps rewrite speech *after* the check so ADR-013 traps are not scored as mismatch.
