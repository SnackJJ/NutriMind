# ADR-008: GRPO Training Hyperparameter Tuning

- **Status**: accepted
- **Date**: 2026-04-17
- **Deciders**: zeqing

## Context

Two consecutive GRPO training runs on A800 80GB produced no learning:
- **Run 1** (reward_v2 only): `frac_reward_zero_std = 1.0` across all 1596 steps. Reward variance was zero in every group, so advantage = 0, gradient = 0. Model weights unchanged.
- **Run 2** (reward_v3 with LLM judge): `frac_reward_zero_std` dropped to ~0.5 average — the judge successfully created variance in ~50% of groups. However, the model still didn't learn: reward_mean flat at ~0.6 for 1596 steps, loss oscillating around 0.

### Diagnosis from Run 2 wandb

| Metric | Observation | Problem |
|--------|------------|---------|
| **frac_reward_zero_std** | Discrete values {0, 0.5, 1.0} — ~50% of groups still zero variance | G=4 too few; 4 rollouts often get identical scores by chance |
| **reward_mean** | Flat ~0.6 across all 1596 steps | No learning signal reaching the model |
| **loss** | Oscillating around 0, no trend | GRPO loss is mean-zero by design, but combined with flat reward = no useful gradient |
| **entropy** | Dropped from 0.9 → 0.26 | Free entropy collapse with beta=0 — model becoming deterministic without improving |
| **learning_rate** | Cosine decay, ~6e-9 at end | Most LR budget spent when frac_zero_std was high (wasted steps) |

### Root Cause Analysis

1. **G=4 too small**: With only 4 rollouts per group, the probability that all 4 get the same reward is ~50% even with the LLM judge (the rule-based component still dominates at 0.5 weight and often gives identical scores).

2. **beta=0 (no KL penalty)**: Entropy collapsed freely from 0.9 to 0.26. The model became more deterministic without becoming better — it collapsed into a mode rather than exploring.

3. **Cosine LR schedule**: LR decays to near-zero by the end of training. But the first half of training was largely wasted (high zero-std fraction), so by the time the judge was contributing meaningful signal, the LR was already too low to act on it.

4. **LR=5e-6 too conservative**: Combined with cosine decay and wasted steps, the effective learning was negligible.

## Decision

Retrain from the SFT checkpoint (not from the collapsed run 2 checkpoint) with the following changes:

### Changes

| Parameter | Before (Run 2) | After (Run 3) | Rationale |
|-----------|----------------|---------------|-----------|
| `num_generations` | 4 | **8** | Drops all-same-score probability from ~50% to ~15%. |
| `beta` | 0.0 | **0.01** | Prevents entropy collapse. 0.04 is DeepSeek-R1 (70B); smaller models need lighter penalty. Tiny-R1 (8B) uses 0.001. Start at 0.01 for 4B. |
| `learning_rate` | 5e-6 | **1e-5** | With beta as safety net, can afford larger steps. Tiny-R1 single-GPU uses 5e-6; TRL Gemma-1B uses 2e-5. Split the difference. |
| `lr_scheduler_type` | cosine (default) | **constant_with_warmup** | Constant LR doesn't waste budget on early zero-signal steps. 3% warmup for stability. |
| `num_train_epochs` | 2 | **3** | More training time with constant LR. |
| `max_completion_length` | 4096 | **2048** | Nutrition queries rarely need >2K tokens. Saves ~4-5 GB VRAM (critical for G=8). |
| `lora_r` | 32 | **16** | Saves ~2 GB. r=16 is sufficient for domain-specific RL fine-tuning (not learning new capabilities, just improving behavior). |
| `lora_alpha` | 64 | **32** | Maintain alpha/r = 2 ratio. |
| `vllm_gpu_memory` | 0.5 | **0.35** | Make room for G=8 training-side memory. |
| `vllm_enable_sleep_mode` | (not set) | **True** | **Critical**: offloads vLLM weights + KV cache to CPU during training phase, saves ~10-12 GB. Required for G=8 on single GPU. |

### References for Hyperparameter Choices

| Project | Model | G | beta | LR | Schedule | GPU |
|---------|-------|---|------|----|----------|-----|
| DeepSeek-R1 | 70B+ | 16-64 | 0.04 | N/A | N/A | Multi-GPU |
| Tiny-R1 (single) | 8B | 6 | 0.001 | 5e-6 | cosine | 1×A100 |
| TRL + Gemma-3-1B | 1B | 8 | 0.04 | 2e-5 | N/A | GPU |
| TRL default | varies | 8 | 0.04 | — | — | — |
| **NutriMind Run 3** | **4B** | **8** | **0.01** | **1e-5** | **constant+warmup** | **1×A800** |

### What NOT Changed

- **reward_v3 design** — Judge is working (API 200 OK, produces variance). The problem was training config, not reward.
- **grad_accum=8** — Effective batch = 8 groups × 8 rollouts = 64 rollouts per update. Sufficient.

## Expected Behavior

After these changes:
- `frac_reward_zero_std` should stay below 0.2 consistently (8 rollouts + judge)
- `entropy` should stabilize around 0.5-0.7 (beta prevents collapse)
- `reward_mean` should show upward trend within first 500 steps
- `loss` will still oscillate around 0 (that's normal for GRPO) but with smaller amplitude as the model converges

## Monitoring Criteria

Stop training early if:
- `entropy` < 0.3 for 100+ consecutive steps (still collapsing despite beta)
- `reward_mean` flat for 500+ steps (need to investigate further)
- `frac_reward_zero_std` > 0.5 consistently (G=8 not enough, consider G=16)

## Memory Estimate (A800 80GB)

With `vllm_enable_sleep_mode=True`, vLLM and training **time-share** the GPU:

**During generation phase** (vLLM active, training paused):
```
vLLM model weights (bf16):    ~8 GB
KV cache (0.35 util):         ~20 GB
Prompt/completion buffers:     ~4 GB
Total:                         ~32 GB
```

**During training phase** (vLLM weights offloaded to CPU):
```
Training model + LoRA (bf16):  ~10 GB
Activations (grad ckpt):      ~12 GB
Optimizer states (AdamW):      ~6 GB
Logits for G=8 × 2048 tokens: ~16 GB  (the bottleneck, see TRL #2709)
Buffer:                        ~4 GB
Total:                         ~48 GB
```

**Peak**: ~48 GB during training phase (well within 80 GB).
Without sleep mode: ~48 + 28 = ~76 GB (too close to limit).

If OOM: first reduce `max_completion_length` to 1536, then reduce G to 6.

## Files Modified

- `src/training/grpo/train_grpo.py` — Updated defaults and GRPOConfig

## Consequences

- Training time ~1.5x longer (3 epochs × 2x rollouts per step)
- Judge API cost ~2x (8 candidates per call instead of 4)
- Must restart from SFT checkpoint, not from run 2 (entropy already collapsed)
