# ADR 006: Revert to veRL on Single A800 80GB

## Status
accepted

## Context
Our previous approach (ADR-005) migrated GRPO training from veRL to TRL to enable server-based vLLM rollout and mitigate OOM issues on dual 4090Ds by attempting to isolate the rollout generation and FSDP training onto separate GPUs. However, we encountered significant issues:
1. vLLM server with `gpu-memory-utilization=0.85` pre-allocated ~43GB.
2. TRL weight synchronization led to subsequent OOM issues during the training and sync loop.
3. Managing multi-GPU isolation and vGPU overhead added too much complexity for the 4B model scale.

We have access to a single A800 (80GB) instance, which possesses enough VRAM to handle the entire training lifecycle for a 4B model without splitting workloads across GPUs or heavily compressing memory usage. 

## Decision
We will revert to using **veRL** with its **Hybrid Engine** for GRPO training, deployed on a **single A800 (80GB)** GPU.

Specific configurations:
- All `n_gpus_per_node` set to `1`.
- Use veRL's hybrid engine for generation and training on the same GPU.
- Increase vLLM `gpu_memory_utilization` to `0.85` for generation, safely releasing and switching to FSDP without OOM risk.

## Consequences
- **Simplified Architecture**: No need for explicit weight synchronization across isolated GPUs. Single GPU training is straightforward.
- **Eliminated OOM**: The 80GB VRAM comfortably fits the 4B model weights, FSDP gradients, optimizer states, and large KV caches.
- **Cost & Availability**: Transitioning from 2x4090D to 1xA800 shifts our hardware requirements but stabilizes our iterative training pipeline.
