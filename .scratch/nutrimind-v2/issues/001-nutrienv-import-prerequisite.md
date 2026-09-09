---
id: 001
title: Establish a reproducible NutriEnv dev/test dependency + public-API smoke test
type: prototype
status: CLOSED (2026-09-09) — local + CI green
blocks: [002, "all v2 data-factory implementation"]
depends_on: []
spec: ../spec.md
adr: [../../docs/decisions/012-nutrienv-read-only-benchmark.md]
branch: nutrimind-v2/ticket-001-nutrienv-dep
commit: 3bd4b2d
ci_run: https://github.com/SnackJJ/NutriMind/actions/runs/34299424567
---

# 001 — NutriEnv dependency prerequisite

## Findings (2026-09-09)

- **Package manager**: `uv` 0.10.5 (`[tool.uv]` in `pyproject.toml`; `.venv` is a uv
  venv; `uv.lock` is git-ignored and not on disk). No poetry/pdm/pip-tools.
- **Python**: 3.12.11 (miniforge). `requires-python = ">=3.10"`.
- **`uv sync` of the whole project is already broken** (pre-existing, unrelated):
  `vllm==0.12.0` in the `inference` / `training` extras does not resolve. So the
  declarative "declare + `uv sync`" flow cannot be used as-is; `uv pip install` into the
  existing `.venv` is the working path.
- **`../nutri-env` IS a real package** (`name = "nutrienv"`, `version = "1.0.0"`,
  hatchling, `packages = ["src/nutrienv"]`, zero runtime deps, `requires-python
  >=3.11`). GitHub remote `github.com/SnackJJ/NutriEnv` is **public** and reachable
  anonymously; HEAD = the pin below.
- **PACKAGING DEFECT**: nutri-env's `.gitignore` contains a bare `env/` line. Hatchling
  honours it and **excludes `src/nutrienv/env/` from the built wheel**, even though the
  files are committed. Any wheel-based install (`pip install`, `git+https@sha`,
  `uv pip install -e` in the default *compat* editable mode) yields a broken package:
  `import nutrienv.bench` → `ModuleNotFoundError: No module named 'nutrienv.env'`
  (`nutrienv/bench/achievable.py` imports `from nutrienv.env import NutriEnv` at module
  top level). ADR-012 says NutriMind never patches NutriEnv, so this is **not** fixed
  here — see "Upstream fix (recommended, not done)" below.
- **Working install**: strict PEP 660 editable —
  `uv pip install -e ../nutri-env --config-settings editable_mode=strict` — redirects
  imports to the source tree and includes every module. Verified: all smoke assertions
  pass. `scripts/setup_nutrienv.sh` wraps this.

## Decision recorded

- **Dependency source**: local sibling repo `../nutri-env`, pinned to rev
  `203d807b19953a86b5486303ba6f7dd3b9cf7bb6`.
- **Install mechanism**: strict-editable (`editable_mode=strict`), **not** a `git+`
  wheel, **not** default editable — both drop `nutrienv/env/`.
- **Declared** in `pyproject.toml`: `data-factory` optional-dep group +
  `[tool.uv.sources] nutrienv = { path = "../nutri-env", editable = true }`, with a
  comment pointing at the strict-mode requirement and the `vllm` blocker.
- **Pin** in `configs/data_factory.yaml` (`nutrienv.rev`), asserted by the smoke test.
- **Local dev vs CI**: same mechanism. CI (`.github/workflows/nutrienv-smoke.yml`)
  checks out `SnackJJ/NutriEnv` at the pin as a sibling and runs
  `scripts/setup_nutrienv.sh`.

## Upstream fix (recommended, not done — needs owner sign-off)

Two-line nutri-env change would make a clean wheel / `git+` install work and let us drop
the strict-mode requirement:

- anchor the ignore: `.gitignore` `env/` → `/env/` (only ignore a root-level `env/`), **or**
- `pyproject.toml`: `[tool.hatch.build] artifacts = ["src/nutrienv/env/**"]`.

Fixing packaging so the ruler is *installable* is not "bending the ruler" (ADR-012), but
it is still a NutriEnv change and is left to the owner.

**Interim posture (record):**

- A wheel / `git+` / default-editable `nutrienv` dependency is **not usable** until the
  upstream `env/`-exclusion is fixed.
- `editable_mode=strict` against a pinned local sibling checkout is NutriMind's
  **temporary compatibility path**.
- Once NutriEnv fixes its packaging, switch to a normal pinned package
  (`nutrienv @ git+https://github.com/SnackJJ/NutriEnv@<fixed-sha>` or a released
  version) and drop the strict-mode requirement + `scripts/setup_nutrienv.sh`.

## Evidence (local, 2026-09-09)

- package manager + install command: `uv` 0.10.5;
  `uv pip install -e ../nutri-env --config-settings editable_mode=strict`
- NutriEnv source + pinned SHA: `../nutri-env` (== `github.com/SnackJJ/NutriEnv`),
  `203d807b19953a86b5486303ba6f7dd3b9cf7bb6`
- Python: 3.12.11
- `nutrienv.__version__` = `1.0.0`; `__file__` = `../nutri-env/src/nutrienv/__init__.py`
- `catalog_digest` = `57184b2bbce4519076b4238a8d64861950db46fdc793d0e43055f07f43c28b5f`
  == exam split `catalog_sha256` ✅
- `load_exam()` → **63** tasks ✅
- `NutriEnv().reset(task.s0)` → dict; `Scorer().score(...)` → `{passed, tag}`;
  `check_achievable([t])` → reachable ✅
- `generate_one(family="update", …)` → accepted; `task_to_item` → dict ✅
- `FAMILY_MAX_STEPS` / `FINISH_OPS` importable ✅
- `pytest tests/training/data_factory/test_nutrienv_smoke.py` → **9 passed**
  (added `test_pin_is_single_sourced`: `configs/data_factory.yaml` rev == test constant
  == installed source HEAD).

### CI (GitHub Actions)

- Workflow `.github/workflows/nutrienv-smoke.yml`, run
  [`34299424567`](https://github.com/SnackJJ/NutriMind/actions/runs/34299424567) —
  **success**, job `smoke` in 10s, on commit `3bd4b2d`.
- Clean env: `uv venv --python 3.12` (fresh), installs only `pytest` + `nutrienv`
  strict-editable; **no `uv sync`**, no repo `.venv`.
- Pin single-sourced: CI reads `nutrienv.rev` from `configs/data_factory.yaml`
  (`203d807…`), checks out `SnackJJ/NutriEnv` at it, and asserts
  `git -C nutri-env rev-parse HEAD == 203d807…` before install. Log:
  `checked-out HEAD: 203d807b19953a86b5486303ba6f7dd3b9cf7bb6`.
- `nutrienv 1.0.0 OK …/nutri-env/src/nutrienv/__init__.py` (strict editable → source tree).
- `9 passed in 0.36s`.
- Non-fatal annotation only: GitHub auto-upgraded the pinned actions from Node 20 → 24.

## Close conditions

| condition | status |
|---|---|
| passes locally | ✅ 9/9 |
| passes in CI | ✅ run 34299424567 success, 9 passed |
| `import nutrienv` succeeds | ✅ |
| every public-API smoke assertion passes | ✅ 9/9 |
| dependency revision reproducible | ✅ pin single-sourced in `configs/data_factory.yaml`; CI checks out + asserts HEAD == pin; fresh strict-editable install reproduces |

**CLOSED 2026-09-09.** Next: merge `3bd4b2d` into the 002 baseline, then start 002-A
(single frozen item → runnable NutriEnv, spec §9.1 / OQ-2).


## Why

`nutrienv` is **not importable** in this repo today (`ModuleNotFoundError: No module
named 'nutrienv'`). Every part of the v2 data factory imports it. This is a hard
prerequisite for any implementation work and for ticket 002.

## Scope (prototype / decision only — no v2 business code)

1. **Confirm the dependency shape before writing config** — do not pre-lock a syntax:
   - Is `../nutri-env` an installable Python package? Is its `pyproject.toml` /
     `setup.py` complete (name, packages, entry points)?
   - Dependency source: local sibling path, a Git URL at a pinned commit, or a workspace
     member?
   - This project's package manager: `pip`, `uv`, Poetry, PDM, other? Is there a
     lockfile? (`.gitignore` currently ignores `uv.lock`.)
   - Do local dev and CI need different strategies (editable local vs. pinned commit)?
   - Pin a **commit**, not a branch/tag, for reproducibility.
2. Add the dependency in whatever form the answers above dictate (local path / editable
   install / `git+…@<sha>` / workspace member), pinned to an exact commit of
   `../nutri-env`. **Do not describe a `pyproject.toml` Git URL dependency as an
   "editable dependency"** — a `git+https@<sha>` entry installs a fixed revision and is
   not editable; editable is `pip install -e` / a path/workspace dependency. Use the term
   that matches what is actually configured, per the project's real tool (pip / uv /
   Poetry / PDM / workspace). Write the chosen rev and mechanism into the spec and
   `configs/data_factory.yaml`'s `nutrienv_rev`.
3. Install into the NutriMind environment.
4. Add `tests/training/data_factory/test_nutrienv_smoke.py` (or equivalent) asserting the
   **acceptance evidence** below.
5. Wire it so the smoke test runs in **CI**, not only a local invocation.

## Acceptance evidence (all must actually pass — the ticket does not close on the spec)

**Smoke assertions:**

```
import nutrienv                                            # succeeds
generate_one(...) with a synthetic expander                # returns a Task
NutriEnv().reset(task.s0)                                   # returns an observation
Scorer().score(end_state, task.oracle)                      # returns {"passed": ...}
load_exam()                                                 # returns exactly 63 Tasks
catalog_digest(catalog) == exam split catalog_sha256        # True (record both values)
task_to_item(task)                                          # returns a dict
nutrienv.harness.runner.FAMILY_MAX_STEPS / FINISH_OPS       # importable
```

**Evidence to keep with the closed ticket:**

- package manager + exact install command used
- NutriEnv source (local path / git URL) + pinned commit SHA
- Python version
- local smoke-test output (captured)
- CI smoke-test output (captured)
- `catalog_digest` computed value vs the exam split's `catalog_sha256`
- `load_exam()` count (must be 63)
- the minimal call results for `generate_one` / `reset` / `Scorer.score` / `task_to_item`

**Close only when ALL hold:**

- passes **locally**
- passes **in CI**
- `import nutrienv` succeeds
- every public-API smoke assertion passes
- the dependency revision is reproducible (a fresh checkout + install reproduces it)

If 001 fails, fix the dependency / package-manager boundary first — **do not start v2
business implementation.**

## Out of scope

- Any data-factory module (`build`, gates, materializers, teacher client).
- Signature guards beyond "these public symbols import and are callable".
- Changing anything in `../nutri-env`.

## Decisions to record on close

- Chosen dependency source + mechanism + pinned `nutrienv_rev`.
- Whether a single in-memory frozen item can be reconstructed through a public entry, or
  whether only file-based `load_split` is public (hands OQ-2 / ticket 002 a starting
  point).
