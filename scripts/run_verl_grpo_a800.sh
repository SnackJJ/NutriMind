#!/bin/bash
# =============================================================================
# NutriMind GRPO Training — A800 80GB Single-GPU
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

REWARD_VERSION="${1:-v2}"

echo "=========================================="
echo "NutriMind GRPO — A800 Single-GPU"
echo "=========================================="
echo "GPU 0: Hybrid Engine (vLLM + FSDP)"
echo "Reward: ${REWARD_VERSION}"
echo "=========================================="

# Pre-checks
python -c "import verl" 2>/dev/null || { echo "ERROR: veRL not installed"; exit 1; }
[[ -f configs/verl_grpo_a800.yaml ]] || { echo "ERROR: a800 config not found"; exit 1; }

# Prepare data
[[ -f data/grpo/verl_train.parquet ]] || python scripts/prepare_verl_data.py

# GPU memory pre-check
echo ""
echo "Pre-flight GPU memory check:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits | while IFS=, read -r idx name total used free; do
    echo "  GPU $idx ($name): ${total}MB total, ${used}MB used, ${free}MB free"
    if (( free < 70000 )); then
        echo "  WARNING: GPU $idx has less than 70GB free!"
    fi
done

echo ""

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="${WANDB_PROJECT:-nutrimind-grpo}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export NUTRIMIND_REWARD_VERSION="${REWARD_VERSION}"

# Launch training
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

python -m src.training.grpo.train_verl \
    --config configs/verl_grpo_a800.yaml \
    --reward_version "${REWARD_VERSION}" \
    trainer.experiment_name="nutrimind_grpo_a800_${REWARD_VERSION}_${TIMESTAMP}"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
