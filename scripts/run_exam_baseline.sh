#!/usr/bin/env bash
# B2 / B5 — k independent exam runs of a policy served by vLLM (native FC, ADR-014).
#
# One command once the GPU is up:
#   scripts/run_exam_baseline.sh                         # zero-shot Qwen3.5-2B, k=3
#   ADAPTER=data/student/models/sft_v2_lora/final scripts/run_exam_baseline.sh   # SFT LoRA
#   MODEL=/path/to/merged_sft scripts/run_exam_baseline.sh                        # SFT merged
#
# Order: env checks -> model present -> vLLM up (/health) -> exam gate -> eval -> stop vLLM.
# The gate runs again inside eval_exam before any rollout and when the report is built.
#
# Variables (all optional):
#   MODEL          HF id or local dir            (Qwen/Qwen3.5-2B)
#   SERVED_NAME    name the eval asks for        (basename of MODEL, or "sft" with ADAPTER)
#   ADAPTER        LoRA dir -> --enable-lora --lora-modules sft=$ADAPTER
#   LORA_RANK      --max-lora-rank               (32, configs/sft_v2_lora.yaml)
#   NUTRIENV_SRC   lab src/ dir put on PYTHONPATH (/home/jzq/Projects/nutri-env-lab-pin/src)
#   EXPECTED_REV   full 40-hex lab SHA            (0ee68ea…, the ADR-012 pin)
#   EXAM_PATH      exam JSON                      (the lab's EXAM_SPLIT_PATH)
#   RUNS CONCURRENCY THINKING(0|1) SEED_BASE LIMIT EXTRA_EVAL_ARGS
#   PORT MAX_MODEL_LEN GPU_MEM_UTIL LANGUAGE_MODEL_ONLY(1|0) EXTRA_VLLM_ARGS
#   VLLM_BIN       vllm executable. Qwen3.5 (Qwen3_5ForConditionalGeneration) is NOT in
#                  vllm 0.16.0's registry; use a newer vLLM (own venv) and point VLLM_BIN at it.
#   PY             python for eval/gate           (.venv/bin/python)
#   HF_ENDPOINT    download mirror                (https://hf-mirror.com)
#   OUT_DIR        eval output                    (data/eval/exam/<served>_<thinking>_<timestamp>)
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
ADAPTER="${ADAPTER:-}"
LORA_RANK="${LORA_RANK:-32}"
if [[ -n "${ADAPTER}" ]]; then
  SERVED_NAME="${SERVED_NAME:-sft}"
else
  SERVED_NAME="${SERVED_NAME:-$(basename "${MODEL}")}"
fi
NUTRIENV_SRC="${NUTRIENV_SRC:-/home/jzq/Projects/nutri-env-lab-pin/src}"
EXPECTED_REV="${EXPECTED_REV:-0ee68eaa6c246e8079915761c95fc986c53d4979}"
EXAM_PATH="${EXAM_PATH:-}"
RUNS="${RUNS:-3}"
CONCURRENCY="${CONCURRENCY:-16}"
THINKING="${THINKING:-0}"
SEED_BASE="${SEED_BASE:-0}"
LIMIT="${LIMIT:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PY="${PY:-${ROOT}/.venv/bin/python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MODE=$([[ "${THINKING}" == "1" ]] && echo think || echo nothink)
OUT_DIR="${OUT_DIR:-data/eval/exam/${SERVED_NAME}_${MODE}_$(date +%Y%m%dT%H%M%S)}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

die() { echo "ERROR: $*" >&2; exit 1; }

# ---- env checks -------------------------------------------------------------
command -v nvidia-smi >/dev/null || die "nvidia-smi not found (no GPU on this machine?)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
command -v "${VLLM_BIN}" >/dev/null || [[ -x "${VLLM_BIN}" ]] || die "vllm not found: ${VLLM_BIN}"
[[ -x "${PY}" ]] || die "python not found: ${PY}"
[[ -d "${NUTRIENV_SRC}/nutrienv" ]] || die "NUTRIENV_SRC=${NUTRIENV_SRC} has no nutrienv/"
[[ "${EXPECTED_REV}" =~ ^[0-9a-f]{40}$ ]] || die "EXPECTED_REV must be a full 40-hex SHA"
[[ -z "${ADAPTER}" || -f "${ADAPTER}/adapter_config.json" ]] || die "ADAPTER=${ADAPTER} has no adapter_config.json"
export PYTHONPATH="${NUTRIENV_SRC}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NUTRIMIND_ALLOW_NETWORK=1   # localhost vLLM counts as network (eval_exam refuses otherwise)

VLLM_PY="$(head -1 "$(command -v "${VLLM_BIN}")" | sed -n 's/^#!//p')"
if [[ -n "${VLLM_PY}" ]]; then
  "${VLLM_PY}" - <<'EOF' || die "this vLLM cannot serve Qwen3.5: Qwen3_5ForConditionalGeneration missing from its model registry (vllm 0.16.0 lacks it; install a newer vLLM and set VLLM_BIN)"
import vllm
from vllm.model_executor.models.registry import ModelRegistry
print("vllm", vllm.__version__)
raise SystemExit(0 if "Qwen3_5ForConditionalGeneration" in ModelRegistry.get_supported_archs() else 1)
EOF
fi

# ---- model present ----------------------------------------------------------
if [[ ! -d "${MODEL}" ]]; then
  echo "[run_exam] fetching ${MODEL} via ${HF_ENDPOINT}"
  if command -v hf >/dev/null; then hf download "${MODEL}" >/dev/null
  else huggingface-cli download "${MODEL}" >/dev/null; fi
fi

# ---- exam gate FIRST (before the GPU is spent) ------------------------------
"${PY}" - "${EXAM_PATH}" "${EXPECTED_REV}" <<'EOF' || die "exam gate failed; no eval"
import sys
from src.training.rl.exam_gate import assert_exam_byte_identical, assert_lab_at_rev
exam = sys.argv[1] or None
assert_exam_byte_identical(exam, expected_rev=sys.argv[2])
import nutrienv.bench as bench
print("[run_exam] gate ok: lab", assert_lab_at_rev(sys.argv[2]), "exam", exam or bench.EXAM_SPLIT_PATH)
EOF

# ---- vLLM up ----------------------------------------------------------------
mkdir -p "${OUT_DIR}"
VLLM_ARGS=(
  serve "${MODEL}"
  --port "${PORT}" --host 127.0.0.1
  --dtype bfloat16
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder        # Qwen3.5 emits <tool_call><function=…><parameter=…> (XML)
  --reasoning-parser qwen3
)
[[ "${LANGUAGE_MODEL_ONLY}" == "1" ]] && VLLM_ARGS+=(--language-model-only)
if [[ -n "${ADAPTER}" ]]; then
  VLLM_ARGS+=(--enable-lora --max-lora-rank "${LORA_RANK}" --lora-modules "${SERVED_NAME}=${ADAPTER}")
else
  VLLM_ARGS+=(--served-model-name "${SERVED_NAME}")
fi
# shellcheck disable=SC2206
VLLM_ARGS+=(${EXTRA_VLLM_ARGS})

echo "[run_exam] ${VLLM_BIN} ${VLLM_ARGS[*]}"
"${VLLM_BIN}" "${VLLM_ARGS[@]}" >"${OUT_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null && break
  kill -0 "${VLLM_PID}" 2>/dev/null || { tail -40 "${OUT_DIR}/vllm.log" >&2; die "vLLM exited during startup"; }
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || die "vLLM not healthy after 15 min (see ${OUT_DIR}/vllm.log)"

# ---- eval -------------------------------------------------------------------
EVAL_ARGS=(
  -m src.training.rl.eval_exam
  --base-url "${BASE_URL}" --model "${SERVED_NAME}"
  --expected-rev "${EXPECTED_REV}"
  --runs "${RUNS}" --concurrency "${CONCURRENCY}" --seed-base "${SEED_BASE}"
  --out-dir "${OUT_DIR}" --resume
)
[[ -n "${EXAM_PATH}" ]] && EVAL_ARGS+=(--exam-path "${EXAM_PATH}")
[[ -n "${LIMIT}" ]] && EVAL_ARGS+=(--limit "${LIMIT}")
[[ -n "${ADAPTER}" ]] && EVAL_ARGS+=(--adapter "${ADAPTER}")
if [[ "${THINKING}" == "1" ]]; then EVAL_ARGS+=(--enable-thinking); else EVAL_ARGS+=(--no-thinking); fi
# shellcheck disable=SC2206
EVAL_ARGS+=(${EXTRA_EVAL_ARGS})

"${PY}" "${EVAL_ARGS[@]}"

echo "[run_exam] report: ${OUT_DIR}/report.md"
