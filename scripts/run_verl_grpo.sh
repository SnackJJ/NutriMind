#!/bin/bash
# =============================================================================
# NutriMind veRL GRPO Training Script
# =============================================================================
#
# Usage:
#   ./scripts/run_verl_grpo.sh [v1|v2|v3] [options]
#
# Examples:
#   ./scripts/run_verl_grpo.sh v2                    # Standard training with v2 reward
#   ./scripts/run_verl_grpo.sh v1 --dry_run          # Dry run with v1 reward
#   ./scripts/run_verl_grpo.sh v2 --consumer         # Consumer GPU mode
#   ./scripts/run_verl_grpo.sh v2 data.train_batch_size=16
#
# Environment Variables:
#   CUDA_VISIBLE_DEVICES: GPU selection (default: 0,1)
#   WANDB_PROJECT: W&B project name (default: nutrimind-grpo)
#   WANDB_MODE: W&B mode (online, offline, disabled)
#
# =============================================================================

set -euo pipefail

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# =============================================================================
# Parse Arguments
# =============================================================================

REWARD_VERSION="${1:-v2}"
shift || true

# Check for --consumer flag
CONFIG="configs/verl_grpo.yaml"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --consumer)
            CONFIG="configs/verl_grpo_consumer.yaml"
            shift
            ;;
        --dry_run)
            EXTRA_ARGS+=("--dry_run")
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# =============================================================================
# Environment Setup
# =============================================================================

# GPU configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# W&B configuration
export WANDB_PROJECT="${WANDB_PROJECT:-nutrimind-grpo}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Python path
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Timestamp for experiment naming
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo "=========================================="
echo "NutriMind veRL GRPO Training"
echo "=========================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Config: ${CONFIG}"
echo "Reward version: ${REWARD_VERSION}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Timestamp: ${TIMESTAMP}"
echo "=========================================="

# Check if veRL is installed
if ! python -c "import verl" 2>/dev/null; then
    echo "ERROR: veRL is not installed. Please run:"
    echo "  pip install verl"
    exit 1
fi

# Check if config exists
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi

# Check/prepare data
if [[ ! -f "data/grpo/verl_train.parquet" ]]; then
    echo "Preparing veRL training data..."
    python scripts/prepare_verl_data.py
fi

# =============================================================================
# Launch Training
# =============================================================================

echo ""
echo "Starting training..."
echo ""

python -m src.training.grpo.train_verl \
    --config "${CONFIG}" \
    --reward_version "${REWARD_VERSION}" \
    trainer.experiment_name="nutrimind_grpo_${REWARD_VERSION}_${TIMESTAMP}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
