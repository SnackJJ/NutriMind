# ADR-015: v2 Post-Training Stack — TRL SFT, veRL GRPO/DAPO

- **Status**: accepted
- **Date**: 2026-09-11
- **Deciders**: zeqing
- **Supersedes for v2**: [ADR-007](007-environment-factory-migration.md) (v1-only:
  TRL `environment_factory` wrapping Python methods as tools). [ADR-005](005-verl-to-trl-migration.md)
  remains the v1 history of leaving veRL.

## Context

v1 moved GRPO from veRL to TRL (`environment_factory` + a 6-tool Python env) after a
broken dual-GPU veRL setup and a custom XML `<tool_call>` loop. That stack does not
fit v2: the student acts in NutriEnv through the lab's OpenAI-shaped tools, the
reward is a verifier over end state, and wrapping `NutriEnv.step` as TRL env methods
would invent a second tool schema.

v2 first phase needs an online RLVR loop (group-relative, binary Pass) plus one
current comparison recipe. GiGPO is a single-paper estimator and stays out (ADR-010).
OPD (sequence-level DAgger / token-level GKD) is deferred.

## Decision

- **SFT**: Hugging Face **TRL `SFTTrainer`** (LLaMA-Factory is an acceptable
  equivalent) on Data Factory records, using the **same** chat template and `tools`
  schema as eval. Not Unsloth as the v2 main path. Not veRL-SFT as a requirement.
- **RL**: **veRL**, with the lab FC episode loop producing rollouts and rewards;
  veRL sees trajectories and scalar rewards, not a second env. Default algorithm
  **GRPO**; **DAPO** is the one comparison arm (identical rollout, reward, and task
  pool). DPO and PPO are later options, not this phase (DPO is offline preference;
  PPO needs a critic).
- **GiGPO**: out of v2 first phase (ADR-010 unchanged).
- **OPD**: deferred. Sequence-level would call a remote teacher API; token-level
  would host a larger same-family teacher (e.g. Qwen3.5-27B) on separate GPUs.
- **Task selection**: measure `p̂` per task per checkpoint; train only the per-family
  sweet-spot band; drop groups with fewer than two valid samples or zero reward
  variance; do not resample inside a group to manufacture variance.
- **v1 `train_grpo.py` / `gigpo_trainer.py` / TRL `NutriMindToolEnv`**: archive.
  v2 does not extend them.

## Consequences

### Positive
- RL uses the stack labs actually publish GRPO/DAPO with (veRL).
- SFT stays on the common HF path; one protocol (ADR-014) still binds SFT and RL.
- DAPO is a real comparison, not a custom estimator.

### Negative
- Two trainers (TRL and veRL) to keep in the repo. Protocol identity is the binding
  constraint, not a shared Trainer class.
- veRL is heavier than TRL to operate; a 2B LoRA on one A800 is still in range.

### Neutral
- ADR-008's G / KL / LR numbers were for v1's rubric reward. v2 re-tunes under the
  verifier; 008 is not a v2 hyperparameter pin.

## Related

- [ADR-007](007-environment-factory-migration.md),
  [ADR-010](010-nutrimind-v2-rescope.md),
  [ADR-014](014-native-tool-calling-v2-protocol.md)
- `.scratch/nutrimind-rl/spec.md`
