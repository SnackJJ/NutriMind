---
id: 012
title: Widen authoring — update, recommend, evaluate (EVALUATE_TIERS), amount_path / gram_anchor
status: CLOSED (2026-09-11)
depends_on: [011, 025, 026]
spec: ../spec.md
spec_sections: ["6.4", "9.2", "22.12", "22.13", "US-12", "18"]
---

# 012 — Widen authoring: update, recommend, evaluate

**What to build:** Authoring for the remaining `generate_one`-supported shapes —
`update`, `recommend`, `evaluate` — through the same Seam-1 path as `log`.

- `evaluate` records carry `meta.tier` from `nutrienv.bench.quality_gates.EVALUATE_TIERS`
  (`single` / `pair` / `triple` / `long` / `explicit_grams` / `synonym`). It is **not**
  the batch number and **not** v1's T1–T4. `log` / `recommend` / `update` / `composite`
  carry `tier == ""`.
- `amount_path` derived per persona (spec §22.12): `gym` → `explicit_grams` ~60%;
  `everyday` / `cut` → `named_measure` + `unspecified`, `explicit_grams` ~15%; ~15% ounce
  phrasing within `named_measure`; `unspecified` ≤ 20%.
- `gram_anchor` off by default, per-family on from config (spec §22.13);
  `enable_semantic_vote` stays `False`.
- intent enumeration covers each family up to `target_n × over_generate_x`.
- the manifest reports accepted family mix vs target (US-12).

**Blocked by:** 011, 025, 026 (FC teacher + serialize must land before any
`target=sft` accepted-record run — ADR-014).

**Status:** ready-for-agent

- [x] `build --target sft` produces ≥ 1 accepted record for each of `log`, `update`,
      `recommend`, `evaluate`
- [x] every `evaluate` record's `meta.tier` is one of `EVALUATE_TIERS`; no non-`evaluate`
      record has a non-empty `tier`
- [x] the `amount_path` distribution over a run matches the per-persona weights within
      tolerance (test over a synthetic run)
- [x] setting `gram_anchor: true` for a family visibly changes authored speech; default off
- [x] `run_manifest.json` has `family_mix` with target vs actual per family
- [x] intent count per family respects `target_n × over_generate_x` and `max_intents`
