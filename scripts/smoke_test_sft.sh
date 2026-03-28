#!/bin/bash

# Smoke Test Script for NutriMind SFT Training
# Purpose: Verify dependencies, data, and model loading for Qwen3-4B

echo "==========================================="
echo "   NutriMind SFT Smoke Test Start"
echo "==========================================="

# 1. Environment Check
echo "[1/4] Checking Environment..."
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found."
    exit 1
fi

python3 -c "import torch; print(f'  - Torch version: {torch.__version__}'); print(f'  - CUDA Available: {torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    echo "Error: Torch or CUDA not working properly."
    exit 1
fi

python3 -c "import unsloth; print(f'  - Unsloth version: {unsloth.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Unsloth not found. You may need to run: pip install unsloth"
fi

# 2. Data Check
echo "[2/4] Checking Training Data..."
DATA_PATH="data/trajectories/sft_train_trajectory.jsonl"
if [ -f "$DATA_PATH" ]; then
    NUM_LINES=$(wc -l < "$DATA_PATH")
    echo "  - Found trajectory data: $DATA_PATH ($NUM_LINES records)"
else
    echo "Error: $DATA_PATH not found. Please ensure data is uploaded to NutriMind/data/trajectories/"
    exit 1
fi

# 3. Model Loading & Label Verification (Dry Run)
echo "[3/4] Verifying Loss Mask Labels & Model Loading (Dry Run)..."
python3 src/training/sft/train.py \
    --verify_labels \
    --data_path "$DATA_PATH" \
    --model_name "Qwen/Qwen3-4B" \
    --max_seq_length 2048 \
    --batch_size 1

if [ $? -eq 0 ]; then
    echo "  - Label verification successful!"
else
    echo "Error: Label verification failed. Check model path or data format."
    exit 1
fi

# 4. Short Batch Exercise (Formal Test)
echo "[4/4] Running 10-step smoke training exercise..."
# Only 10 steps to verify backprop and gradients
python3 src/training/sft/train.py \
    --data_path "$DATA_PATH" \
    --model_name "Qwen/Qwen3-4B" \
    --output_dir "models/smoke_test" \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 1024 \
    --learning_rate 2e-5

if [ $? -eq 0 ]; then
    echo "  - Smoke training exercise complete!"
    echo "==========================================="
    echo "   NutriMind SFT Smoke Test: PASSED"
    echo "==========================================="
else
    echo "Error: Smoke training exercise failed."
    exit 1
fi
