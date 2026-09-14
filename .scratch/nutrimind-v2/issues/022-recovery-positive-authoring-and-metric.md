---
id: 022
title: Recovery-positive authoring + recovery_fraction metric
status: CLOSED (2026-09-11)
depends_on: [010, 011, 015]
spec: ../spec.md
spec_sections: ["4.1", "9.5", "19.4", "20"]
---

# 022 — Recovery-positive authoring + `recovery_fraction`

**What to build:** the machinery for [ADR-013](../../../docs/decisions/013-sft-failure-recovery-coverage.md)
— make "the student has seen an error observation" a measured property of the corpus.

1. **`is_recovery_positive(result: EpisodeResult) -> bool`** — a pure predicate at the
   verifier seam (§19.7 Seam 3): given the same `EpisodeResult`, the same bool. True iff some `TurnMeta.observation` carries a **semantic** `ActionError` code
   (`unknown_food`, `implausible_quantity`, `bad_index`) **and**
   `result.verification.status == "pass"`. `bad_schema` / `unknown_op` are syntax repairs
   and do **not** count. No I/O, no globals, same inputs → same bool.

2. **The error-code partition is table-driven.** One table maps every code raised by
   `nutrienv.actions.dispatch` to `semantic | syntax`. A test enumerates the codes the
   installed nutri-env can raise and asserts each is classified — so a new upstream code
   cannot silently land in neither bucket and quietly deflate the metric. (Same failure
   class as the v1 bug this ADR exists because of: a dispatch that matches on strings and
   falls through to a default that nobody notices.)

3. **Metrics in `run_manifest.json`** (spec §9.5): `metrics.recovery_fraction`,
   `metrics.recovery_positive`, and `metrics.recovery_by_code` — the last one partitioned
   into semantic and syntax, so an inflated syntax count is visible rather than folded into
   the headline.

4. **A health check, not a gate** (beside the ticket-015 checks): warn when
   `recovery_fraction` is outside `[0.15, 0.25]`, and stay silent inside it. The run still
   succeeds; the shortfall is fixed with over-generation, never by lowering
   `counts.accepted`.

5. **One authoring lever, scoped to `log` first.** An `intent`-stage (pure code, spec §4.1)
   task shape whose *natural first action* is illegal but recoverable — a spoken food that
   `search_foods` cannot resolve directly, or a portion that trips
   `implausible_quantity`. `check_achievable` must still Pass: the oracle is reachable
   *despite* the trap. `recommend` follows once `log` is measured.

**Blocked by:** 010, 011, 015.

**Status:** ready-for-agent

- [x] `is_recovery_positive` is a pure function of an `EpisodeResult`; unit tests over
      scripted episodes: semantic error + Pass → true; `bad_schema` + Pass → false;
      semantic error + Fail → false; no error + Pass → false
- [x] a scripted episode with an `unknown_food` turn and a Passed end state lands in
      `sft/train.jsonl` **and** increments both `recovery_positive` and `accepted`
- [x] `run_manifest.json` carries `metrics.recovery_fraction`, `metrics.recovery_positive`,
      `metrics.recovery_by_code`
- [x] the health check warns at 0.00 and 0.30 and is silent at 0.20; it never fails the run
- [x] the code partition is table-driven and a test asserts every `ActionError` code
      reachable from the installed nutri-env is classified semantic or syntax
- [x] at least one authored `log` task reconstructs to a `NutriEnv` whose natural first
      action raises the intended code, and whose oracle still Passes on the correct path
- [x] a test asserts a syntax-only recovery (`bad_schema` then Pass) does **not** count
      toward `recovery_fraction`

**Not in scope:** changing the §7 family mix or `target_n` (ADR-013: recovery is an
attribute, not a family); the `recommend` lever; any nutri-env change (ADR-012).
