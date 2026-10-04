#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE="${1:-4090}"
shift || true
case "$PROFILE" in 4090|a800|a800_fast|a800_bs16) ;; *) echo "profile must be 4090, a800, a800_fast or a800_bs16" >&2; exit 2 ;; esac
export PATH="${GRPO_VENV:-/root/autodl-tmp/venvs/grpo}/bin:$PATH"
export PYTHONPATH="$PWD:${MIMO_VERL_ROOT:-/root/autodl-tmp/mimo-verl}:${NUTRIENV_ROOT:-/root/nutri-env-pin}/src:${FLA_ROOT:-/root/autodl-tmp/pydeps/fla}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export NUTRIMIND_ALLOW_NETWORK=1
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR:-$PWD/data/rl_runs/wandb}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/uv-cache}"
export RAY_TMPDIR="${RAY_TMPDIR:-/root/autodl-tmp/NutriMind/ray}"
mkdir -p "$WANDB_DIR" "$UV_CACHE_DIR" "$RAY_TMPDIR"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export HYDRA_FULL_ERROR=1
export VERL_FILE_LOGGER_ROOT="$PWD/data/rl_runs/metrics"
UV_BIN="${UV_BIN:-/root/miniconda3/bin/uv}"
if [[ "${GRPO_PREFLIGHT:-1}" == 1 ]]; then
  "$UV_BIN" run --no-project --python "${GRPO_VENV:-/root/autodl-tmp/venvs/grpo}/bin/python" \
    python scripts/check_grpo_v2.py --output "data/rl_runs/grpo_v2_${PROFILE}/preflight.json"
fi
exec "$UV_BIN" run --no-project --python "${GRPO_VENV:-/root/autodl-tmp/venvs/grpo}/bin/python" \
  python -m src.training.rl.train_verl --config-path "$PWD/configs" --config-name "grpo_v2_${PROFILE}" \
    'hydra.searchpath=[pkg://verl.trainer.config]' "$@"
