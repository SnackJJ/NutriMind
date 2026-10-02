---
id: 023
title: Re-pin NutriEnv to nutri-env-lab@0ee68ea + native FC smoke
status: CLOSED (2026-09-11) — local green (348 data-factory tests)
commit: 2d744c8
depends_on: []
spec: ../spec.md
spec_sections: ["18"]
adr: [../../docs/decisions/012-nutrienv-read-only-benchmark.md, ../../docs/decisions/014-native-tool-calling-v2-protocol.md]
---

# 023 — Re-pin to nutri-env-lab + FC smoke

**What to build:** Move the v2 NutriEnv pin from `../nutri-env@203d807` to
`../nutri-env-lab` at `0ee68eaa6c246e8079915761c95fc986c53d4979`, still read-only
strict-editable. A smoke test asserts the **native tool-calling** public symbols import
and the exam file at that rev is the published v1.0 (byte-identical to the pin).

Ticket 001 stays CLOSED (history of the old pin). This ticket is the v2 pin.

**Blocked by:** None — can start immediately.

**Status:** CLOSED (2026-09-11)

- [x] `configs/data_factory.yaml` `nutrienv.rev` (or equivalent) is the lab SHA above;
      a test asserts installed HEAD == that pin (single-sourced)
- [x] `from nutrienv.harness.tool_call import run_episode_tool_call` succeeds
- [x] `NUTRIENV_TOOLS` and `TOOL_SYSTEM_PROMPT` import from
      `nutrienv.harness.tools_schema`
- [x] `load_exam()` still returns 63 tasks; catalog digest matches the exam split
- [x] the exam file bytes match the committed v1.0 at the pin (no working-tree drift
      in CI)
- [x] NutriMind still does not patch the lab tree

## Closure notes

- Pin is `../nutri-env-lab` @ `0ee68eaa6c246e8079915761c95fc986c53d4979`. Ticket 001
  stays CLOSED (old pin history).
- Spec §18 FC symbols (`run_episode_tool_call`, `NUTRIENV_TOOLS`,
  `TOOL_SYSTEM_PROMPT`) are in the 016 guard. Those lab modules have no `__all__`;
  the import/`__all__` test allows that per ADR-012 ("listed explicitly in the v2
  guard").
- Local: 348 passed, 2 skipped in `tests/training/data_factory/`.
- GitHub `SnackJJ/NutriEnv` `main` is still `203d807` (lab SHA not pushed). CI
  checkout of the pin will fail until the lab commit is on that remote. Do not
  patch the lab to publish it.
