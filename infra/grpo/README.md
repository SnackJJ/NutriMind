# NutriMind v2 GRPO

The baseline uses XiaomiMiMo/verl commit
`a2ad9f6160b03ff2d47e59832bfb6b289f37c917`, its synchronous trainer, and the
NutriEnv native function-calling harness. It does not use the archived v1
NutriMind trainer. `scripts/run_grpo_v2.sh` is the entry point.
The local trainer adapter submits one current-policy batch at every synchronous
step, including the terminal step. The GRPO loss and weight synchronization
remain upstream implementations.

## Data and policy

`scripts/prepare_grpo_v2.py` exports 286 mixed-outcome SFT training tasks and
104 unique SFT holdouts after excluding 26 SFT-seen problems and two duplicates
from the original 132 records. Identity is query + initial state + oracle, not
just task ID. Query, initial state, oracle and lineage come from
TaskPackages, not teacher trajectories. Checksums bind selection to the SFT
checkpoint and Pass@8 probe. The exporter refuses an existing output directory.

Rewards are final-state pass=1 and fail=0. Infrastructure-invalid episodes use
MiMo's -999 sentinel and receive zero policy advantage. A token-budget stop is
a model rollout limit, not an infrastructure error. Generated token IDs are
never re-encoded; injected tool/user observations receive loss mask zero.
The native initial-state observation is included in the prompt.

The policy starts from the merged SFT checkpoint with a new rank-32 RL LoRA.
The A800 KL reference is the merged SFT policy, not the original pretrained
model. Weight transfer merges the RL adapter for vLLM; this avoids requiring
native vLLM LoRA support for the hybrid attention projections.

## Server entry points

```bash
bash scripts/run_grpo_v2.sh 4090
bash scripts/run_grpo_v2.sh a800
bash scripts/run_grpo_v2.sh a800_fast
```

Defaults:

- `GRPO_VENV=/root/autodl-tmp/venvs/grpo`
- `MIMO_VERL_ROOT=/root/autodl-tmp/mimo-verl`
- `NUTRIENV_ROOT=/root/nutri-env-pin`
- `UV_BIN=/root/miniconda3/bin/uv`
- `FLA_ROOT=/root/autodl-tmp/pydeps/fla`

`a800_fast` keeps the A800 algorithm and data settings and changes only throughput:
64 concurrent rollouts, CUDA graphs, prefix caching and a resident reference policy.
On 2026-10-02 it cut the A800 step from about 900 s to 150-250 s with unchanged
rollout/actor probability agreement.

Training needs `flash-linear-attention==0.5.2`; without it transformers falls back to the
torch gated-delta-rule kernel (about 4x slower forward/backward, same numerics against an
fp32 reference). Install it outside the venv and the launcher adds it to `PYTHONPATH`:
`uv pip install --python $GRPO_VENV/bin/python --no-deps --target $FLA_ROOT flash-linear-attention fla-core`.
Preflight fails if the kernels are missing. Keep `use_remove_padding` off: without
flash-attn, packed sequences leak across sequence boundaries in the SDPA attention layers.

The 4090 profile is a two-step, four-task hardware smoke with a 4096-token
response budget and no KL reference. Its results are not a training benchmark.
The A800 profile is a 20-step pilot with G=8, a 20480-token response budget,
KL regularization, independent validation and online constant-reward group
filtering. It still needs A800 memory/throughput validation before a longer run.
The A800 actor keeps parameters and optimizer state on GPU; the reference
policy retains CPU offload. Rollout GPU memory utilization is 0.5 and the
training microbatch is one trajectory until long-trajectory peaks are measured.
W&B runs offline alongside console and JSONL logging, with artifacts under
`data/rl_runs/wandb/`. The launcher puts uv cache and Ray temporary files on
the data disk by default.

The pinned NutriEnv revision is
`47367d9c569d0a46cbd1c97d5f08afb3a7d573ac`. Keep that checkout separate from the
current benchmark checkout. Neither tools nor scorer are modified by this integration.

## Compatibility and migration

The isolated GRPO environment leaves `/root/venvs/sft` and `/root/venvs/vllm`
unchanged. Dependencies must satisfy the MiMo fork's Transformers <5.11 constraint.
The vLLM plugin registers its existing pure-text Qwen3.5 implementation in 0.24
and exposes upstream hybrid-cache and text M-RoPE metadata; it does not replace a model forward pass.
SDPA and veRL's chunked PPO projection
avoid a FlashAttention build and full-sequence vocabulary logits.

Move the RL Parquet files and manifest, merged SFT model, original SFT adapter
and run manifest, code, pinned NutriEnv checkout, and pinned MiMo checkout to the
A800 host. Set the four path variables above if paths change. The launcher checks
environment/catalog revisions, dataset checksums, checkpoint identity and a
single visible CUDA GPU before training.

The environment currently uses CUDA 13.0 wheels. The A800 host driver must
support that runtime; a container cannot upgrade the host driver. A800 80GB is
the intended pilot target. A smoke on 24GB does not prove the long-trajectory
A800 configuration fits.

On AutoDL, check container cgroup limits as well as `free -h`: the latter can
report host RAM rather than the instance allocation. The current A800 instance
has a 120 GiB container memory limit and 14 CPU cores of quota.
`/root/NutriMind/data` links to `/root/autodl-tmp/NutriMind/data`; the GRPO
environment and MiMo checkout also reside under `/root/autodl-tmp`.
Save or transfer the data disk separately from a system-disk image; a system
image alone cannot restore these artifacts.

## Evidence required before freezing an image

Run the focused data/token tests, then complete actual optimizer steps with
finite nonzero gradient norms and changed RL adapter weights. Check that the
next rollout reports the updated policy version. Save a checkpoint and resume
it for another real step. Preserve resolved configs, dependency freeze, source
revisions, dataset hashes and logs with the resulting image.

CPU token/harness tests are not GPU training evidence. A Docker recipe or a
dependency freeze alone is not a built or validated image.

## Verified 4090 smoke (2026-10-01 UTC)

Two real optimizer steps completed, with gradient norms 0.359375 and
0.279296875. Step 2 resumed step 1's model, optimizer, scheduler and RNG state.
Its 16 training episodes used policy version 1; two validation episodes used
policy version 2. All 300 RL adapter tensors changed between checkpoints and
remained finite. The GPU was released after completion.

The two validation episodes failed with the 4096-token smoke budget exhausted.
This smoke verifies the training lifecycle, not improvement or generalization.
The A800 long-trajectory profile has not run. The offloaded smoke reported about
112 GiB CPU memory used; budget host RAM as well as VRAM on the next machine.

Focused checks: two data tests passed locally; eight upstream token, native
harness, model metadata and synchronous scheduling tests passed on the server
with the actual SFT tokenizer. The Docker recipe has not been built or tested.

Server evidence lives in `data/rl_runs/grpo_v2_4090/` and
`data/rl_runs/grpo_v2_resume/`; step-2 metrics live in
`data/rl_runs/metrics/nutrimind-v2-grpo/4090-resume.jsonl`.
`scripts/check_grpo_checkpoints.py` reproduces the single-rank adapter check.

To repeat the successful resume without overwriting its output directory:

```bash
bash scripts/run_grpo_v2.sh 4090 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=/root/NutriMind/data/rl_runs/grpo_v2_4090/global_step_1 \
  trainer.default_local_dir=data/rl_runs/grpo_v2_resume_repeat \
  trainer.rollout_data_dir=data/rl_runs/grpo_v2_resume_repeat/rollouts \
  trainer.experiment_name=4090-resume-repeat trainer.test_freq=1
```

The Docker recipe uses the repository root as build context:

```bash
docker build -f infra/grpo/Dockerfile -t nutrimind-grpo:smoke .
docker run --gpus all --ipc=host --rm \
  -v /path/to/nutri-env-pin:/opt/nutri-env-pin:ro \
  -v /path/to/data:/workspace/NutriMind/data \
  nutrimind-grpo:smoke 4090
```

Only label/tag the image as validated after repeating the GPU smoke inside it.
An AutoDL container without a Docker daemon needs a platform image snapshot or
a separate Docker-capable build host; this recipe does not provide a daemon.
