#!/usr/bin/env bash
# Ticket 023 — install the NutriEnv lab pin into the NutriMind venv.
#
# NutriEnv is consumed read-only (ADR-012). v2 uses the local sibling
# `../nutri-env-lab`. The wheel build at the pinned rev DROPS `nutrienv/env/`
# (the lab's .gitignore has a bare `env/` line that hatchling honours), so a
# plain wheel / `git+` / default-editable install produces a broken package
# (`import nutrienv.bench` -> ModuleNotFoundError: nutrienv.env). Strict PEP 660
# editable mode redirects imports to the source tree and includes every module.
#
# Usage:  scripts/setup_nutrienv.sh [PATH_TO_NUTRI_ENV_LAB]
set -euo pipefail

NUTRI_ENV="${1:-../nutri-env-lab}"
PIN="0ee68eaa6c246e8079915761c95fc986c53d4979"   # keep in sync with configs/data_factory.yaml

if [[ ! -d "${NUTRI_ENV}/src/nutrienv" ]]; then
  echo "ERROR: ${NUTRI_ENV}/src/nutrienv not found. Pass the nutri-env-lab checkout path." >&2
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
