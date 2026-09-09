# ADR-007: Migrate from rollout_func to TRL environment_factory

- **Status**: accepted
- **Date**: 2026-04-15
- **Deciders**: zeqing
- **Supersedes**: ADR-002 (partially)

## Context

ADR-002 chose `rollout_func` over TRL's built-in `tools=`/`environment_factory=` based on the belief that:

> "TRL's `_tool_call_loop` parses tool calls via HF chat template `response_schema`, expecting structured `tool_calls` dicts. Our SFT model outputs `<tool_call>` XML tags (ADR-001). Switching to HF format would require re-doing SFT."

**This premise was wrong.** Investigation reveals:

1. **Our format IS Qwen3's native format.** Qwen3's chat template generates `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` — exactly what our model outputs and `tool_parser.py` parses.
2. **TRL's agent training now natively supports Qwen3.** The `tools=` and `environment_factory=` parameters work with Qwen3's chat template, which produces the same `<tool_call>` XML format.
3. **ADR-001 was never created as a file.** The "pure text tool calling" protocol referenced throughout the codebase is simply Qwen3's standard tool calling format, not a custom format.

The current `rollout_func` implementation (`trl_environment.py`, 406 lines) manually handles:
- vLLM server HTTP calls
- Multi-turn generation with stop token management
- Logprob extraction and stitching across turns
- env_mask construction for tool response tokens
- Token ID re-encoding and length alignment

All of this is handled automatically by TRL when using `environment_factory`.

### Hardware Change

Additionally, the training environment has changed from 2x RTX 4090D to **1x A800 (80GB)**. This eliminates:
- The need for GPU isolation (ADR-002's primary motivation for vLLM server mode)
- The dual-GPU launch script complexity
- Weight sync concerns between training and inference GPUs

## Decision

**Migrate from `rollout_func` to `environment_factory`**, leveraging TRL's native multi-turn agent training loop.

### New Architecture

```python
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

class NutriMindEnv:
    """TRL-compatible environment. Public methods become tools automatically."""

    def reset(self, **kwargs) -> str | None:
        self.meal_log = []
        self.tool_calls = 0
        return None

    def get_food_nutrition(self, foods: list) -> str:
        """Look up nutrition data for foods from the USDA database. ..."""

    def log_meal(self, meal_type: str, foods: list) -> str:
        """Record a food entry to the user's history. ..."""

    def get_today_summary(self) -> str:
        """Check today's totals and progress against goals. ..."""

    def get_history(self, days: int, compare_to_goal: bool = False) -> str:
        """Analyze multi-day trends and goal adherence. ..."""

    def retrieve_knowledge(self, query: str, mode: str = "hybrid", top_k: int = 3) -> str:
        """Search nutrition knowledge base. ..."""

    def set_goal(self, nutrient: str, value: float, goal_type: str) -> str:
        """Set or update a daily nutritional target. ..."""

def reward_func(environments, completions, **kwargs):
    """Reward function that reads environment state directly."""
    rewards = []
    for env, completion in zip(environments, completions):
        score = compute_reward(env, completion)
        rewards.append(score)
    return rewards

trainer = GRPOTrainer(
    model="Qwen/Qwen3-4B",
    environment_factory=NutriMindEnv,
    reward_funcs=reward_func,
    train_dataset=prompt_dataset,
    peft_config=LoraConfig(...),
    args=GRPOConfig(
        num_generations=4,
        max_completion_length=4096,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        use_vllm=True,
        vllm_mode="colocate",  # Single A800, no need for server mode
    ),
)
trainer.train()
```

### What TRL Handles Automatically

| Concern | rollout_func (ADR-002) | environment_factory (this ADR) |
|---------|----------------------|-------------------------------|
| Multi-turn loop | Manual (60 lines) | TRL built-in |
| Tool call parsing | Manual ToolParser | TRL via Qwen3 chat template |
| Tool execution | Manual dispatch | TRL calls env methods |
| Tool response injection | Manual formatting | TRL handles |
| Logprob computation | Manual vLLM HTTP + stitching | TRL built-in |
| env_mask / tool token masking | Manual construction | TRL built-in |
| Stop token handling | Manual client-side truncation | TRL built-in |
| vLLM integration | Manual HTTP requests | TRL colocate mode |
| Weight sync | Manual concern (ADR-002 risk) | TRL handles automatically |

## Alternatives Considered

| Alternative | Rejected Because |
|---|---|
| Keep `rollout_func` | 406 lines of complexity solving a problem that doesn't exist (format mismatch); `rollout_func` is marked "experimental" in TRL |
| `tools=` (stateless) | `environment_factory` allows stateful tracking (meal_log, tool_calls count) needed for reward computation |
| Self-built GRPO loop | Unnecessary now that TRL natively supports the exact workflow |

## Consequences

### Positive

- **~400 lines of code eliminated** (`trl_environment.py` rollout_func, vLLM HTTP client, logprob stitching)
- **No more "experimental API" risk** — `environment_factory` is the intended path for agent training
- **Automatic weight sync** — colocate mode shares GPU memory, no sync issues
- **Simpler launch** — single process, no dual-GPU orchestration script
- **TRL handles tool token masking** — no manual env_mask construction
- **Future-proof** — TRL's agent training is actively developed; `rollout_func` may be removed

### Negative

- **Less control over generation** — can't fine-tune vLLM HTTP parameters (but colocate mode is faster anyway)
- **System prompt** — must be embedded via dataset or chat_template_kwargs rather than env; verify Qwen3 template injects tool schemas correctly
- **Reward function refactor** — current `reward_v2` expects `_build_trajectory_from_solution(completion_text)`; new version reads `env` state directly

### Migration Risk

- **Low** — the format is identical; only the orchestration layer changes
- **Rollback** — ADR-002's `rollout_func` code is preserved in git history

## Files Impact

### New

| File | Purpose |
|------|---------|
| `src/training/grpo/trl_env_factory.py` | New `NutriMindEnv` class implementing TRL `environment_factory` protocol |
| `src/training/grpo/train_grpo.py` | New simplified training entry point for single A800 |
| `docs/decisions/007-environment-factory-migration.md` | This ADR |

### Modified

| File | Change |
|------|--------|
| `src/training/grpo/reward.py` | Add `reward_from_env()` that reads environment state instead of parsing completion text |

### Obsolete (after migration complete)

| File | Reason | Status |
|------|--------|--------|
| `src/training/grpo/trl_environment.py` | `rollout_func` replaced by `environment_factory` | **OBSOLETE** |
| `src/training/grpo/train_trl.py` | References rollout_func + dual-GPU setup | **OBSOLETE** |
| `scripts/run_trl_grpo_4090d.sh` | Dual-GPU launch script, not needed for single A800 | **OBSOLETE** |
| `scripts/prepare_trl_data.py` | May need update for new dataset format | **REVIEW** |
| `src/training/grpo/train_verl.py` | Already obsolete per ADR-002 | **OBSOLETE** (delete) |
| `src/training/grpo/verl_interaction.py` | Already obsolete per ADR-002 | **OBSOLETE** (delete) |
| `src/training/grpo/verl_agent_env.py` | Already obsolete per ADR-002 | **OBSOLETE** (delete) |
| `configs/verl_grpo_4090d.yaml` | Already obsolete per ADR-002 | **OBSOLETE** (delete) |
| `scripts/run_verl_grpo_4090d.sh` | Already obsolete per ADR-002 | **OBSOLETE** (delete) |

### Preserved (still needed)

| File | Why |
|------|-----|
| `src/training/grpo/environment.py` | `NutriMindEnv`, `RolloutTrajectory`, `TaskMetadata`, `RolloutGroup` — core logic reused in new env factory; `DeterministicToolCache` still useful for GiGPO |
| `src/training/grpo/reward.py` | `reward_v2` scoring logic reused; add new env-aware wrapper |
| `src/orchestrator/tool_parser.py` | Still used by orchestrator for production inference; reward function may use for validation |
| `src/orchestrator/orchestrator.py` | Production inference path unchanged |
| `src/tools/` | All 6 tools unchanged; wrapped by new env factory methods |
| `src/training/grpo/gigpo.py` | GiGPO logic preserved, may need adapter for new env |
| `src/training/grpo/monitor.py` | Training monitoring preserved |
| `src/training/grpo/prepare_prompts.py` | Prompt preparation preserved |
| `src/training/grpo/label_difficulty.py` | Difficulty labeling preserved |

## ADR Status Updates

| ADR | New Status | Note |
|-----|-----------|------|
| ADR-001 | **Never existed as file** | Format is Qwen3 native; no custom protocol. Remove references or create a brief note documenting this. |
| ADR-002 | **Partially superseded** | GPU isolation decision (veRL→TRL) still valid; `rollout_func` choice superseded by this ADR |
