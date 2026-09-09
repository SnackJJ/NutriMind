#!/usr/bin/env bash
# Ticket 001 — install the NutriEnv benchmark/library into the NutriMind venv.
#
# NutriEnv is consumed read-only (ADR-012). It is a local sibling repo. The wheel
# build at the pinned rev DROPS `nutrienv/env/` (nutri-env's .gitignore has a bare
# `env/` line that hatchling honours), so a plain wheel / `git+` / default-editable
# install produces a broken package (`import nutrienv.bench` -> ModuleNotFoundError:
# nutrienv.env). Strict PEP 660 editable mode redirects imports to the source tree
# and includes every module.
#
# Usage:  scripts/setup_nutrienv.sh [PATH_TO_NUTRI_ENV]
set -euo pipefail

NUTRI_ENV="${1:-../nutri-env}"
PIN="203d807b19953a86b5486303ba6f7dd3b9cf7bb6"   # keep in sync with configs/data_factory.yaml

if [[ ! -d "${NUTRI_ENV}/src/nutrienv" ]]; then
  echo "ERROR: ${NUTRI_ENV}/src/nutrienv not found. Pass the nutri-env checkout path." >&2
  exit 1
fi

HEAD="$(git -C "${NUTRI_ENV}" rev-parse HEAD)"
if [[ "${HEAD}" != "${PIN}" ]]; then
  echo "WARNING: ${NUTRI_ENV} HEAD ${HEAD} != pinned ${PIN}" >&2
  echo "         checkout the pin:  git -C ${NUTRI_ENV} checkout ${PIN}" >&2
fi

# Target the project venv explicitly (works whether or not it is activated).
VENV_PY="${VIRTUAL_ENV:-$(pwd)/.venv}/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
  echo "ERROR: venv python not found at ${VENV_PY}. Run 'uv venv' first." >&2
  exit 1
fi

uv pip install --python "${VENV_PY}" -e "${NUTRI_ENV}" --config-settings editable_mode=strict
"${VENV_PY}" -c "import nutrienv, nutrienv.env, nutrienv.bench; print('nutrienv', nutrienv.__version__, 'OK', nutrienv.__file__)"
