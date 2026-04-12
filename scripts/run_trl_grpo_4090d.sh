#!/bin/bash
# =============================================================================
# NutriMind GRPO Training — TRL + 4090D 48GB x2 GPU Isolation
# =============================================================================
# GPU 0: Training (FSDP/LoRA via TRL GRPOTrainer)
# GPU 1: vLLM Server (inference, generation)
#
# Usage:
#   bash scripts/run_trl_grpo_4090d.sh          # default: v2 reward
#   bash scripts/run_trl_grpo_4090d.sh v2       # explicit reward version
#   bash scripts/run_trl_grpo_4090d.sh --stop   # stop vLLM server
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

REWARD_VERSION="${1:-v2}"
MODEL_PATH="${MODEL_PATH:-models/nutrimind-4b-sft-merged}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_URL="http://localhost:${VLLM_PORT}"

# Handle --stop
if [[ "${1:-}" == "--stop" ]]; then
    echo "Stopping vLLM server..."
    pkill -f "trl vllm-serve" 2>/dev/null || echo "No vLLM server running"
    exit 0
fi

echo "=========================================="
echo "NutriMind GRPO — TRL GPU Isolation"
echo "=========================================="
echo "GPU 0: Training (LoRA + GRPOTrainer)"
echo "GPU 1: vLLM Server (generation)"
echo "Reward: ${REWARD_VERSION}"
echo "Model:  ${MODEL_PATH}"
echo "=========================================="

# Pre-checks
python -c "import trl" 2>/dev/null || { echo "ERROR: TRL not installed. Run: uv pip install trl"; exit 1; }
[[ -d "${MODEL_PATH}" ]] || { echo "ERROR: Model not found at ${MODEL_PATH}"; exit 1; }

# Prepare data if needed
if [[ ! -d data/grpo/trl_train ]]; then
    echo "Preparing TRL training data..."
    python scripts/prepare_trl_data.py
fi

# GPU memory pre-check
echo ""
echo "Pre-flight GPU memory check:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits | while IFS=, read -r idx name total used free; do
    echo "  GPU $idx ($name): ${total}MB total, ${used}MB used, ${free}MB free"
done

echo ""

# Environment
export WANDB_PROJECT="${WANDB_PROJECT:-nutrimind-grpo}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ─── Step 1: Start vLLM server on GPU 1 ─────────────────────────────────────
echo "Starting vLLM server on GPU 1 (port ${VLLM_PORT})..."

# Check if already running
if curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
    echo "  vLLM server already running at ${VLLM_URL}"
else
    CUDA_VISIBLE_DEVICES=1 trl vllm-serve \
        --model "${MODEL_PATH}" \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        --port "${VLLM_PORT}" \
        > logs/vllm_server.log 2>&1 &

    VLLM_PID=$!
    echo "  vLLM server started (PID: ${VLLM_PID})"

    # Wait for server to be ready
    echo "  Waiting for vLLM server..."
    for i in $(seq 1 60); do
        if curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
            echo "  vLLM server ready!"
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM server failed to start. Check logs/vllm_server.log"
            exit 1
        fi
        sleep 2
    done

    if ! curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
        echo "ERROR: vLLM server did not become ready in 120s"
        kill "${VLLM_PID}" 2>/dev/null
        exit 1
    fi
fi

# ─── Step 2: Run training on GPU 0 ──────────────────────────────────────────
echo ""
echo "Starting GRPO training on GPU 0..."

mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python -m src.training.grpo.train_trl \
    --model_path "${MODEL_PATH}" \
    --reward_version "${REWARD_VERSION}" \
    --vllm_server_url "${VLLM_URL}" \
    2>&1 | tee "logs/trl_grpo_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=========================================="
echo "Training complete!"
echo "Note: vLLM server may still be running."
echo "  Stop it with: bash scripts/run_trl_grpo_4090d.sh --stop"
echo "=========================================="
