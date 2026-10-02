#!/usr/bin/env bash
# B5 (training half): v2 SFT (Qwen3.5-2B + LoRA, TRL SFTTrainer; ADR-015) ->
# optional merge -> exam on the result (runs=3) -> compare to a baseline dir.
#
#   EXAM=... EXPECTED_REV=... NUTRIENV_SRC=../nutri-env-lab BASELINE_DIR=... \
#     bash scripts/run_sft_v2.sh
#
# Env:
#   CONFIG        SFT config                      (configs/sft_v2_lora.yaml)
#   MERGE         1 = eval the merged bf16 model; 0 = eval base + ADAPTER   (1)
#   RUNS          exam runs                       (3)
#   ENABLE_THINKING  1 = exam with Qwen enable_thinking (the header SFT trains:
#                 assistant\n<think>\n + plan); 0 = empty-think header     (1)
#   EXAM          exam split path                 (required; passed through)
#   EXPECTED_REV  expected NutriEnv rev for EXAM  (required; passed through)
#   NUTRIENV_SRC  lab source tree                 (required; passed through)
#   BASELINE_DIR  eval output dir to compare against (optional; skip compare if unset)
#   EVAL_OUT      where the exam writes           (<output_dir>/exam)
#   SKIP_TRAIN    1 = reuse an existing <output_dir> (0)
#
# Runtime: transformers >= 5.2 (qwen3_5), flash-linear-attention + causal-conv1d
# recommended for the Gated-DeltaNet layers; the exam's vLLM must support Qwen3.5.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/sft_v2_lora.yaml}
MERGE=${MERGE:-1}
RUNS=${RUNS:-3}
ENABLE_THINKING=${ENABLE_THINKING:-1}
SKIP_TRAIN=${SKIP_TRAIN:-0}
: "${EXAM:?set EXAM (exam split path)}"
: "${EXPECTED_REV:?set EXPECTED_REV (NutriEnv rev the exam was frozen at)}"
: "${NUTRIENV_SRC:?set NUTRIENV_SRC (lab source tree)}"
PY=${PY:-.venv/bin/python}

read -r BASE_MODEL OUTPUT_DIR < <("$PY" -c '
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c["model"]["id"], c["output_dir"])' "$CONFIG")
EVAL_OUT=${EVAL_OUT:-$OUTPUT_DIR/exam}

echo "[1/3] dry-run (identity + token stats)"
"$PY" -m src.training.sft.train_v2 --config "$CONFIG" --dry-run

if [ "$SKIP_TRAIN" != "1" ]; then
  echo "[2/3] train -> $OUTPUT_DIR"
  MERGE_FLAG=""
  [ "$MERGE" = "1" ] && MERGE_FLAG="--merge"
  "$PY" -m src.training.sft.train_v2 --config "$CONFIG" $MERGE_FLAG
fi

if [ "$MERGE" = "1" ]; then
  MODEL="$OUTPUT_DIR/merged"; ADAPTER=""
else
  MODEL="$BASE_MODEL"; ADAPTER="$OUTPUT_DIR"
fi

echo "[3/3] exam (runs=$RUNS) on MODEL=$MODEL ADAPTER=${ADAPTER:-<none>} -> $EVAL_OUT"
MODEL="$MODEL" ADAPTER="$ADAPTER" RUNS="$RUNS" OUT_DIR="$EVAL_OUT" ENABLE_THINKING="$ENABLE_THINKING" \
  EXAM="$EXAM" EXPECTED_REV="$EXPECTED_REV" NUTRIENV_SRC="$NUTRIENV_SRC" \
  bash scripts/run_exam_baseline.sh

if [ -n "${BASELINE_DIR:-}" ]; then
  "$PY" -m src.training.rl.compare_exam --baseline "$BASELINE_DIR" --candidate "$EVAL_OUT"
fi
