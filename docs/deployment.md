# NutriMind Agent — Deployment Guide

## Prerequisites

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU w/ 8GB VRAM | RTX 4090 (24GB) |
| RAM | 16GB | 32GB |
| Disk | 20GB free | 50GB free |
| CUDA | 12.0+ | 12.4+ |

### Software

```bash
# Python 3.10+
python --version  # Should be 3.10+

# Install dependencies
pip install -r requirements.txt

# Optional: Install vLLM for high-performance serving
pip install vllm>=0.4.0

# Optional: Install Unsloth for fast inference
pip install unsloth
```

---

## Quick Start (3 Steps)

### 1. Initialize Data

```bash
# Download and process USDA food database
python scripts/download_usda.py

# Verify
ls -la data/usda.db
```

### 2. Place Trained Model

Copy the trained model to the expected location:

```bash
# SFT model (Phase 3)
# models/nutrimind-3b-sft-merged/

# GRPO model (Phase 4) — recommended
# models/nutrimind-3b-grpo-merged/
```

### 3. Start the Service

```bash
# Option A: Mock mode (no GPU needed — for testing)
bash scripts/start_service.sh mock

# Option B: Direct inference with transformers
python scripts/demo_cli.py --backend transformers --model models/nutrimind-3b-grpo-merged

# Option C: High-performance vLLM server
bash scripts/start_service.sh vllm
# Then in another terminal:
python scripts/demo_cli.py --backend vllm --vllm_url http://localhost:8000/v1
```

---

## Configuration

### Model Configuration (`configs/model.yaml`)

```yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"  # or path to fine-tuned model
  max_seq_length: 2048

inference:
  backend: "vllm"            # "mock" | "transformers" | "vllm"
  vllm_base_url: "http://localhost:8000/v1"
  temperature: 0.1
  max_new_tokens: 1024
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUTRIMIND_MODEL_PATH` | `models/nutrimind-3b-grpo-merged` | Path to trained model |
| `NUTRIMIND_VLLM_PORT` | `8000` | vLLM server port |
| `NUTRIMIND_GPU_MEM` | `0.9` | GPU memory utilization (0-1) |
| `QWEN_API_KEY` | — | API key for expert LLM (Qwen-Max) |
| `OPENAI_API_KEY` | — | API key for GPT-4o (ablation only) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 NutriMind Agent System                    │
│                                                          │
│  User Query                                              │
│     ↓                                                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │           Orchestrator (State Machine)             │   │
│  │  START → INFERENCE → TOOL_CALL → TOOL_EXEC → ...  │   │
│  │                ↓                    ↓              │   │
│  │          Model Response       Tool Results         │   │
│  │                     ↓                              │   │
│  │                  ANSWER → END                      │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  Inference Backend:        Tools:                        │
│  ├─ vLLM (production)      ├─ search_food (USDA)        │
│  ├─ transformers           ├─ calculate_meal             │
│  └─ mock (testing)         ├─ log_user_data              │
│                            ├─ retrieve_knowledge (RAG)   │
│                            └─ call_expert_nutritionist   │
└─────────────────────────────────────────────────────────┘
```

---

## Model Training Pipeline

### Phase 3: SFT Training

```bash
# Generate training data
python src/training/sft/data_generation.py

# Validate data
python src/training/sft/validate_rules.py data/sft/train.jsonl

# Train (GPU required)
python src/training/sft/train.py --config configs/sft_training.yaml

# Merge adapter
python src/training/sft/merge_adapter.py \
    --adapter models/nutrimind-3b-sft \
    --output models/nutrimind-3b-sft-merged
```

### Phase 4: GRPO Training

```bash
# Generate prompts
python src/training/grpo/generate_prompts.py --offline

# Dry run validation
python src/training/grpo/train.py --config configs/grpo_training.yaml --dry_run

# Train (GPU required)
python src/training/grpo/train.py --config configs/grpo_training.yaml
```

### Phase 5: Ablation Experiments

```bash
# Generate test set (320 samples)
python src/training/ablation/evaluate_unified.py --generate_test_set

# Run all experiments
python src/training/ablation/run_experiments.py --experiments A B C D

# Generate comparison report
python src/training/ablation/run_experiments.py --compare_only
```

---

## Performance Targets

| Metric | Target |
|--------|--------|
| T1 Single-step accuracy | ≥ 95% |
| T2 Multi-step success | ≥ 80% |
| T3 Conditional correctness | ≥ 75% |
| T4 Escalation precision | ≥ 85% |
| Format validity | ≥ 98% |
| P50 latency (local path) | < 1s |
| P95 latency (expert path) | < 5s |
| Expert API call rate | ≤ 25% |

---

## Troubleshooting

### vLLM server fails to start
- Check CUDA version: `nvcc --version`
- Check GPU memory: `nvidia-smi`
- Try reducing `NUTRIMIND_GPU_MEM` to `0.8`

### Model outputs are garbled
- Ensure you're using the **merged** model (not the adapter-only checkpoint)
- Check `tokenizer_config.json` exists in the model directory

### Tool calls fail
- Verify `data/usda.db` exists: `python -c "import sqlite3; print(sqlite3.connect('data/usda.db').execute('SELECT COUNT(*) FROM food_nutrients').fetchone())"`
- Check tool schemas: `cat configs/tool_schemas.json | python -m json.tool`
