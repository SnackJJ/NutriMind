# ADR-009: GRPO Reward Redesign Against Shortest-Path Collapse

- **Status**: proposed
- **Date**: 2026-04-20
- **Deciders**: zeqing
- **Supersedes**: partially supersedes the reward weights in ADR-008 (hyperparameter tuning did not address the reward-shape problem)

## Context

The GRPO run launched after ADR-008 ran for ~2700 steps (9h 40m, 14% of 19164 total) on a
Qwen3-4B SFT checkpoint using GiGPO on top of TRL's `environment_factory`. Observed
training dynamics indicate that **the model has collapsed into a single-tool-call
"shortcut" policy** and stopped improving.

### Observed symptoms (wandb, steps 0 → 2632)

| Metric | Early (step ~50) | Late (step ~2632) | Interpretation |
|--------|------------------|-------------------|----------------|
| `tools/call_frequency` | ~2–18 (bursty, SFT residuals) | **1.0 locked** | Model gives up on multi-step trajectories |
| `completions/mean_length` | highly variable | 117–285 tokens | Far below T2/T3 target (500+) |
| `rewards/reward_from_env/mean` | 0.65 | 0.80 (slow +15% over 2700 steps) | Capped by the ceiling of the shortcut |
| `rewards/reward_from_env/std` | — | 0.04–0.09 | Low intra-group variance |
| `tools/failure_frequency` | 0 | 0 | Not a tool-failure issue |
| length ↔ reward correlation | — | **shorter → higher reward** (step 2640: 117 tok → 0.72; step 2632: 285 tok → 0.65) | Direct gradient signal against multi-step behaviour |

### GiGPO is structurally unable to mitigate this

`src/training/grpo/gigpo.py` computes anchor states via SHA256 over full
message byte content. Beyond the initial prompt, two rollouts sharing an
anchor requires byte-level identical assistant outputs — a zero-probability
event with sampled generation. Logs confirm:

```
GiGPO found 1 anchor states across 16 rollouts
Old shape: [16] -> New shape: [16, 441]   ← scalar broadcast, not per-token signal
```

With only the initial prompt as an anchor, `step_advantage = 1.0` for step 0
and fallback for step 1+ → `combined_advantage ≡ group_advantage`. GiGPO
silently degrades into GRPO. Fixing GiGPO is tracked separately; it is not
the primary lever here.

### Root cause: reward function does not enforce monotonicity in effort

`reward_v2` in `src/training/grpo/reward.py`:

```python
total = 0.30 * r_format + 0.35 * r_tool + 0.35 * r_outcome
```

Each component saturates at "completed once":

| Component | 1-step shortcut | 3-step correct path | Gap |
|-----------|----------------|--------------------|----- |
| `r_format` (no parse error) | 1.0 | 1.0 | 0 |
| `r_tool_selection` (≥1 valid tool) | 1.0 | 1.0 | 0 |
| `r_outcome` (T2/T3 only checks `len(answer) > 80`) | 0.8 | 0.8 | 0 |
| **`total`** | **0.93** | **0.93** | **0** |

The shortcut has **identical expected reward but lower variance, shorter
context, and smaller accumulated KL** than the multi-step path.
GRPO group-normalization `(reward - mean) / std` therefore actively pushes
policy mass toward the shortcut:

1. **Variance penalty**: the multi-step path occasionally mis-formats or
   gets `max_tokens`-truncated (→ `total *= 0.5`), producing strong negative
   advantages in otherwise-tied groups. The shortcut has near-zero variance.
2. **KL penalty compounds with length**: KL is accumulated per-token.
   At `beta=0.04`, a 500-token trajectory pays ~4× the KL cost of a
   120-token one for the same per-token KL rate.
3. **Importance-sampling clip bias**: `sampling/importance_sampling_ratio/max`
   reaches 2.745 on long completions (vLLM sampling vs transformers
   forward drift accumulates with length). Long trajectories are clipped
   more aggressively → weaker gradient signal than short ones.
4. **SFT prior erosion**: any multi-step behaviour learned in SFT but not
   explicitly reinforced by reward is removed within ~500 GRPO steps
   ("catastrophic reduction"). The early bursty `call_frequency=18` is
   SFT residue being forgotten.

Additional findings:

- `compute_efficiency_score` exists but is **never added to `total`**
  (`r_efficiency=0.0` is hard-coded in the `RewardBreakdown`).
- `task_metadata.expected_tools` is stored in `details` but never used to
  score T2/T3 outcomes.
- The T2/T3 outcome score is purely length-based
  (`reward.py:182-184`): `0.3` if `<30`, `0.6` if `30–80`, `0.8` if `>80`.
  This is a direct invitation to length-based reward hacking, which the
  length↔reward correlation above confirms has happened.

### Why this is not a hyperparameter problem

ADR-008 tuned `G`, `beta`, `lr`, `lr_schedule` to fix the
`frac_reward_zero_std` issue. It succeeded there (`frac_reward_zero_std=0`
in current logs), but hyperparameters cannot make the policy prefer a
multi-step path when **the reward function itself is flat across path
lengths**. The collapse is a reward-shape issue, not an optimization one.

## Decision

Redesign `reward_v2` (introducing `reward_v2.1` in place) so that reward is
**monotonically increasing in task completeness** and the shortcut dominates
no longer. Restart training from the SFT checkpoint (not from the collapsed
2700-step checkpoint, which has memorized the shortcut).

### Reward changes

1. **T2/T3 outcome uses tool-coverage + answer quality** (replaces length-only rule):

    ```python
    if tier in ("T2", "T3"):
        expected = set(task_metadata.expected_tools or [])
        called   = set(trajectory.get_tools_called())
        coverage = len(expected & called) / max(len(expected), 1)
        answer_quality = min(1.0, len(final_answer) / 100)
        return 0.7 * coverage + 0.3 * answer_quality
    ```

    Effect:
    - 1-step shortcut on a 3-tool T2: `coverage=0.33, quality=1.0 → 0.53`
    - Full 3-step correct:            `coverage=1.00, quality=1.0 → 1.00`

    Gap widens from 0 to 0.47 — large enough to survive group normalization.

2. **`compute_efficiency_score` becomes a bidirectional `compute_effort_score`**:

    ```python
    ratio = trajectory.total_tool_calls / max(task_metadata.optimal_steps, 1)
    if 1.0 <= ratio <= 1.3:
        return 1.0
    if ratio < 1.0:
        return ratio                        # 0.33 for 1/3, 0.67 for 2/3
    return max(0.5, 1.0 - (ratio - 1.3))    # mild penalty for excess
    ```

    Replaces the old one-sided "penalize excess only" behaviour, which
    could never push the model off the shortcut.

3. **New weighted total** (includes effort, keeps conditional diagnostic):

    ```python
    total = 0.20 * r_format
          + 0.20 * r_tool_selection
          + 0.30 * r_outcome
          + 0.20 * r_effort
          + 0.10 * r_conditional   # T3 only; neutral 0.5 elsewhere
    ```

4. **Tier-aware hard gate** (bottom bound):

    ```python
    tier_min_steps = {"T1": 1, "T2": max(2, optimal_steps - 1),
                      "T3": max(2, optimal_steps - 1), "T4": 0}
    if successful_tool_calls < tier_min_steps.get(tier[:2], 1):
        total = 0.0
    ```

    Old gate only required ≥1 step for T1/T2/T3; a 1-step shortcut on T2
    passed it trivially. New gate requires near-optimal step count.

### Training restart policy

- Restart from the SFT checkpoint, **not** the 2700-step GRPO checkpoint.
  The GRPO checkpoint has overfit to the old reward's shortcut; changing
  the reward without resetting weights would leave the model in a local
  optimum the new reward is trying to punish.
- Keep ADR-008's hyperparameters (G=8, beta=0.01, lr=1e-5, constant+warmup).
- Smoke-test 50 steps before committing to a full run. Success criteria:
  - `tools/call_frequency` rises from 1.0 to >1.5 within 50 steps
  - `completions/mean_length` rises from ~120 to >250 for T2/T3 samples
  - Reward temporarily dips (shortcut is being closed) then rebounds

### Validation before code change

Before editing `reward.py`, confirm the shortcut hypothesis empirically:

1. Run eval on both the SFT checkpoint and the 2700-step GRPO checkpoint
   against a T2/T3 eval slice.
2. Measure tool-calls-per-query distribution.
3. Expect: SFT → multi-modal around `optimal_steps`; GRPO → sharp peak at 1.
   If observed, the diagnosis is confirmed and the reward change is safe.

## Consequences

### Positive
- Reward now has a strictly monotone relationship with task completion on
  T2/T3. Shortcut policies become strictly dominated.
- `expected_tools` metadata (collected at data-pipeline stage, previously
  unused in scoring) becomes load-bearing.
- Effort score gives a smooth gradient — useful for low-reward-variance
  groups where coverage is tied.

### Negative / risks
- Reward magnitudes will shift. Absolute reward values in wandb are not
  comparable across ADR-008 and ADR-009 runs.
- If `expected_tools` annotation quality in the query pool is poor
  (e.g. T2 items with `expected_tools=[]`), the coverage term degenerates
  to 1.0 and the shortcut comes back on those items. Must audit the
  query pool before training.
- Hard gate is stricter: early-training models that haven't learned
  multi-step behaviour yet may get total=0 on most T2/T3 samples for the
  first few hundred steps. Entropy bonus (`beta=0.01`) may be insufficient
  to prevent collapse during this window. Monitor `frac_reward_zero_std`
  in smoke test — if it spikes above 0.5, soften the gate to
  `≥ ceil(optimal_steps / 2)`.
- T4 logic unchanged (safety boundary) — but the new weights reduce T4's
  relative reward from ~0.95 (old) to ~0.80 (new). Re-balance T4 weight
  if T4 eval performance regresses.

### GiGPO anchor-state matching fix (2026-04-20)

**Previously deferred, now implemented.** The `compute_state_key` function
has been updated to normalize assistant messages before hashing:

1. **Strip `<think>` blocks** — sampling produces unique reasoning text,
   which was causing every post-prompt state to be unique.
2. **Canonicalize tool calls** — extract only `(name, sorted_args)` and
   re-serialize deterministically, ignoring formatting differences.

Implementation in `src/training/grpo/environment.py`:
- `_normalize_assistant_content()` — helper function for normalization
- `compute_state_key()` — updated to call normalizer for assistant messages

Effect: Two rollouts with different `<think>` content but identical tool
calls now hash to the same state key, enabling GiGPO to find anchor states
beyond the initial prompt.

Enhanced logging in `gigpo.py` reports:
- `max_size` — largest anchor state (how many rollouts share it)
- `avg_size` — average rollouts per anchor
- `multi_action_anchors` — anchors where rollouts took different actions
  (these provide the strongest step-level credit assignment signal)

## References

- `src/training/grpo/reward.py` — `reward_v2`, `compute_tool_selection_score`,
  `compute_outcome_score_rule_based`, `compute_efficiency_score`
- `src/training/grpo/gigpo.py` — `_find_anchor_states`, `compute_state_key`
- `src/training/grpo/environment.py:610` — `compute_state_key` definition
- ADR-008 — previous hyperparameter tuning; addressed `frac_reward_zero_std`
  but not the reward-shape problem identified here
- wandb run: GRPO steps 0–2632 (2026-04-17 launch)
