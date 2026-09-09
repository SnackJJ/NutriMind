# ADR-002: Migrate GRPO Training from veRL to TRL

- **Status**: partially-superseded (by ADR-007)
- **Date**: 2026-04-08
- **Deciders**: zeqing

## Context

NutriMind Phase 4 GRPO training was initially built on veRL (v0.7.x). On 2×RTX 4090D (48GB each), we discovered that veRL **cannot separate rollout (vLLM) and training (FSDP) onto different GPUs**:

1. `resource_pool_spec` in YAML config is completely ignored — `init_resource_pool_mgr` only reads `n_gpus_per_node` and creates a single `global_pool` for all roles.
2. Actor and Rollout are fused into one `Role.ActorRollout` worker — there is no code path to split them.
3. `hybrid_engine: false` only affects internal worker implementation, not GPU assignment.

Result: both GPUs always run in hybrid time-sharing mode (generate → train → generate → ...), wasting inter-GPU communication overhead on a 4B model that fits on one card.

## Decision

**Migrate GRPO training from veRL to TRL v1.0 GRPOTrainer**, using:

- **`vllm_mode="server"`**: GPU 1 runs a standalone vLLM server for generation, GPU 0 runs training. True GPU isolation with overlap potential.
- **`rollout_func`** (TRL's official escape hatch): Bypasses TRL's built-in tool loop (which hardcodes HF standard tool_calling format) and runs our own multi-turn agentic loop with `<tool_call>` XML parsing (preserving ADR-001).
- **`env_mask`**: Tool response tokens are marked 0 (excluded from policy gradient loss), model-generated tokens marked 1. The model still sees tool responses via causal attention for correct conditioning.

### Why not use TRL's built-in `tools=` / `environment_factory`?

TRL's `_tool_call_loop` parses tool calls via HF chat template `response_schema`, expecting structured `tool_calls` dicts. Our SFT model outputs `<tool_call>` XML tags (ADR-001). Switching to HF format would require re-doing SFT — far more work than using `rollout_func`.

## Alternatives Considered

| Alternative | Rejected Because |
|---|---|
| veRL single card | Wastes one 4090D; no GPU isolation benefit |
| veRL fake 2-node (`nnodes:2, n_gpus_per_node:1`) | Still one `ActorRollout` worker spanning both cards; worse communication |
| Patch veRL `init_resource_pool_mgr` | Actor+Rollout role is architecturally fused; pool split alone doesn't help |
| OpenRLHF | More complex (Ray + DeepSpeed), overkill for single-node 2-GPU |
| TRL `tools=` with HF format | Requires re-doing SFT to change tool format; breaks ADR-001 |
| TRL single-turn reward only | Loses multi-turn signal; model can't learn from tool interactions |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  TRL GRPOTrainer                      │
│                                                       │
│  rollout_func(prompts, trainer)                       │
│       │                                               │
│       ▼                                               │
│  ┌─────────────────────────────────────┐             │
│  │  NutriMindEnv + vLLM Server (GPU 1) │             │
│  │                                     │             │
│  │  1. vLLM generate (stop=</tool_call>)│            │
│  │  2. ToolParser parse XML             │            │
│  │  3. NutriMindEnv.step() execute tool │            │
│  │  4. Inject tool_response             │            │
│  │  5. Loop until done or max_rounds    │            │
│  └─────────────────────────────────────┘             │
│       │                                               │
│       ▼  returns {completion_ids, logprobs, env_mask} │
│                                                       │
│  reward_fn(completion_text, **metadata)               │
│       │                                               │
│       ▼  _build_trajectory_from_solution → reward_v2  │
│                                                       │
│  Policy gradient update (GPU 0)                       │
│       env_mask=0 tokens excluded from loss            │
└──────────────────────────────────────────────────────┘
```

## Consequences

### Positive

- True GPU isolation: GPU 0 training, GPU 1 inference — no communication overhead for 4B model
- Preserves ADR-001 `<tool_call>` format — no SFT redo
- Full multi-turn trajectory available for policy gradient (not just first turn)
- `env_mask` correctly excludes tool tokens from loss while preserving causal conditioning
- Reward function (v2) fully reused, zero changes to scoring logic
- All 6 tools and NutriMindEnv reused unchanged

### Negative

- `rollout_func` is marked "experimental" in TRL — API may change
- Custom vLLM server calls add complexity vs. TRL's native vLLM integration
- rollout_func bypasses TRL's weight sync mechanism — need to verify LoRA adapter updates propagate to vLLM server between training steps
- Token-level logprob alignment between vLLM server and training model needs careful validation

### Risks

- **Weight sync**: vLLM server serves base model; training updates LoRA adapter. TRL's native vLLM mode syncs weights automatically, but `rollout_func` may bypass this. Mitigation: verify `trainer.accelerator.unwrap_model()` exports updated weights, or manually trigger sync.
- **Logprob mismatch**: vLLM server's sampling logprobs may differ from training model's forward-pass logprobs due to LoRA delta. TRL handles this via importance sampling correction — confirm this still works with `rollout_func`.

## Files Changed

### New

| File | Purpose |
|------|---------|
| `src/training/grpo/trl_environment.py` | `make_nutrimind_rollout()` rollout_func + `make_multiturn_reward_fn()` + vLLM server calls |
| `src/training/grpo/train_trl.py` | TRL GRPOTrainer entry point with LoRA + rollout_func |
| `scripts/prepare_trl_data.py` | JSONL → HF Dataset conversion |
| `scripts/run_trl_grpo_4090d.sh` | Dual-GPU launch script (vLLM server + training) |
| `docs/decisions/002-verl-to-trl-migration.md` | This ADR |

### Modified

| File | Change |
|------|--------|
| `src/training/grpo/reward.py` | Added `trl_reward_wrapper()` TRL-compatible reward entry point |

### Deprecated (kept for reference)

| File | Reason |
|------|--------|
| `src/training/grpo/train_verl.py` | Replaced by `train_trl.py` |
| `src/training/grpo/verl_interaction.py` | Replaced by `trl_environment.py` |
| `configs/verl_grpo_4090d.yaml` | Replaced by TRL GRPOConfig in `train_trl.py` |
| `scripts/run_verl_grpo_4090d.sh` | Replaced by `run_trl_grpo_4090d.sh` |
| `scripts/prepare_verl_data.py` | Replaced by `prepare_trl_data.py` |
