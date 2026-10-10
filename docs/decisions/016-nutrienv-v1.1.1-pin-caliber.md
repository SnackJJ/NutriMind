# ADR-016: Pin the Released NutriEnv Tree (`v1.1.1`) and Its Ruler

- **Status**: accepted
- **Date**: 2026-10-10
- **Deciders**: zeqing
- **Amends**: [ADR-012](012-nutrienv-read-only-benchmark.md) (consumed tree and pinned SHA)

## Context

ADR-012 has NutriMind consume NutriEnv read-only at an exact SHA. That SHA was
`47367d9` in the **lab** checkout (`../nutri-env-lab`). The lab is the experiment
tree: it keeps moving, and its history forked from the public repository on
2026-08-14, so the same content exists there under different commits. Two
problems followed.

- The lab moved to `e8508d8` on 2026-10-05, so the installed tree stopped
  matching the configured pin and `test_pin_is_single_sourced` went red.
- A lab checkout is an experiment scratch pad. The gate (`exam_gate`) checks only
  `git rev-parse HEAD`, so an uncommitted edit to the scorer or the catalog under
  the pin would still pass the gate and change the score silently.

Meanwhile the public repository already carries everything NutriMind consumes:
`174dcef` on `main` publishes the v1.1 contracts, the v1.1 mass-envelope exam,
and catalog v3.

## Decision

The consumed tree is the **released** one, pinned at a tag rather than a lab
branch tip:

- `../nutri-env-pin` is a detached `git worktree` of the public repo at tag
  `v1.1.1` (`f4d70b2e9affc277aa1afd8e83d26bce65a20b04`), installed
  strict-editable. A publish to `main` no longer moves the pin.
- `pyproject.toml` and `configs/data_factory*.yaml` carry that path and that SHA.
  The smoke test asserts the installed tree's git HEAD equals it.
- Exam entry points default to the pin: `scripts/run_exam_baseline.sh` takes
  `NUTRIENV_SRC` from `../nutri-env-pin`.
- `scripts/setup_nutrienv.sh` exits non-zero when the tree's HEAD is not the pin,
  instead of warning and installing anyway.

The plan to pin `../nutri-env` (the writable public clone) directly was rejected:
a moving `main` plus no dirty-tree check is the failure mode this ADR exists to
prevent.

## Consequences
**`v1.1.0` is superseded by `v1.1.1`.** The `v1.1.0` tree carries
`version = "1.1.0"` in `pyproject.toml` but `__version__ = "1.0.0"` in the
module, so every manifest `harness/runner.py` writes is labelled
`"env": "nutrienv-1.0.0"`. `v1.1.1` changes only those two strings; the exam,
scorer, catalog and prompt digests are identical, so the table below describes
both tags. Pin `v1.1.1`.


**Numbers taken before 2026-10-10 are not comparable with numbers taken after
it.** The ruler moved with the pin:

| | before (`47367d9`) | after (`v1.1.1` = `f4d70b2`) |
|---|---|---|
| `SCORER_VERSION` | `s7-amdr-windows` | `s10-meal-mass-envelope` |
| `PROMPT_VERSION` / fingerprint | `p6-amdr-window-ranges` / `c4526440…` | `p8-published-meal-mass-limits` / `616dea69…` |
| exam split | `nutrienv-v1.1.json` blob `a4475c04…`, `version: nutrienv-v1.1-gold` | blob `a7bba6d1…`, `version: nutrienv-v1.1-mass-envelopes-20261005` |
| catalog the exam reads | `data/fdc/catalog.sqlite` (sha256 `57184b2b…`) | `data/fdc/catalog-v3.sqlite` (sha256 `63e5da2c…`) |
| `NUTRIENV_TOOLS` | `get_profile` without plan-mass limits | with plan-mass limits |

The v1.0 split (blob `78cea3ce…`) and `catalog.sqlite` are identical in both
trees. `TOOL_SYSTEM_PROMPT` is byte-identical; what changed is the `get_profile`
tool description and the frozen prompt fingerprint.

Recorded baselines (`data/eval/exam/baseline_qwen35_2b_think/`,
`baseline_qwen35_2b_nothink/`; `lab_head=47367d9…`, `exam_blob=a4475c04…`,
pass@1 12.17% / 6.35%) stay on the old ruler. Re-measure before comparing them
with any new number.

Entry points that still name the old rev — `data/eval/exam/baseline_env.sh`
(`EXPECTED_REV=47367d9…`), `scripts/check_grpo_v2.py` (`ENV_COMMIT`),
`infra/grpo/Dockerfile` (`NUTRIENV_ROOT=/opt/nutri-env-pin`), and the AutoDL
checkouts `/root/nutri-env-pin` and `/root/nutri-env-runner` — reproduce the runs
recorded before this date. They are historical reproducers, not the path for new
numbers.

**The data factory is not re-pointed here.** `src/training/data_factory/*` still
imports the pre-`47367d9` mill API (`generate_one(family=…)`, `freeze_tasks`),
which the engine moved to `nutrienv.bench.pipeline.legacy_generate_one` and
`freeze_legacy_tasks`. Measured 2026-10-10, the results are identical under
`47367d9` and under the released tree, so this break predates the pin move: 26
`tests/training/data_factory/` modules fail at collection with
`ImportError: cannot import name 'GRAM_UNITS'`, and 28 tests are red in the files
that do collect, including
`test_nutrienv_smoke.py::test_generate_one_update_needs_no_expander`
(`TypeError: generate_one() got an unexpected keyword argument 'family'`). The
repair is deferred as its own task and is deliberately not silenced with
skip/xfail.
