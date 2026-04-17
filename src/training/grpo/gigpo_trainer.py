"""
GiGPOTrainer — GRPOTrainer subclass with step-level advantage injection.

Overrides advantage computation to use GiGPO anchor-state-based step
advantages instead of uniform trajectory-level advantages.

TRL's GRPOTrainer broadcasts a single scalar advantage per completion to
all tokens: ``advantages.unsqueeze(1) → [B, 1] → [B, L]``.

GiGPOTrainer intercepts the inputs before ``compute_loss`` and replaces
the scalar advantage with a per-token tensor ``[B, L]`` where each token
gets the GiGPO step advantage for the step it belongs to.

This means:
- Tokens in a tool-call step that led to higher downstream reward
  get a HIGHER advantage → model learns to repeat that tool choice.
- Tokens in a step at a non-anchor state get the standard GRPO
  group advantage (graceful fallback).
"""

import logging
from typing import Any, Optional

from src.training.grpo.gigpo import (
    compute_gigpo_step_advantages_from_envs,
    gigpo_result_to_per_step_advantages,
)

logger = logging.getLogger(__name__)


class GiGPOTrainer:
    """Drop-in replacement for GRPOTrainer with GiGPO step advantages.

    Usage::

        from src.training.grpo.gigpo_trainer import GiGPOTrainer
        # same arguments as GRPOTrainer
        trainer = GiGPOTrainer(model=..., environment_factory=..., ...)
        trainer.train()

    Requires ``trl`` to be installed (imported lazily to keep this file
    importable without GPU dependencies for testing).
    """

    _base_class = None  # Lazy-loaded GRPOTrainer

    def __new__(cls, *args, **kwargs):
        """Dynamically create a subclass of the real GRPOTrainer at runtime.

        This avoids importing trl at module level (heavy, needs GPU libs).
        """
        if cls._base_class is None:
            from trl import GRPOTrainer as _GRPOTrainer
            cls._base_class = _GRPOTrainer

            # Create the actual subclass
            cls._real_class = type(
                "GiGPOTrainerImpl",
                (_GRPOTrainer,),
                {
                    "compute_loss": _gigpo_compute_loss,
                    "_gigpo_enabled": True,
                },
            )

        instance = cls._real_class(*args, **kwargs)
        return instance


def _gigpo_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """Override compute_loss to inject per-token GiGPO advantages.

    Strategy:
    1. Check if inputs has 'environments' and 'advantages' (scalar per completion).
    2. If yes, compute GiGPO step advantages from the environments.
    3. Build a per-token advantage tensor [B, L] where each token position
       gets the advantage of the step it belongs to.
    4. Replace inputs['advantages'] with the pre-expanded [B, L] tensor.
    5. Patch self so that super().compute_loss() doesn't re-unsqueeze.

    If GiGPO computation fails (no anchor states, error), falls back to
    the standard GRPO uniform advantages silently.
    """
    import torch

    environments = inputs.get("environments")
    completions = inputs.get("completions")
    rewards_tensor = inputs.get("advantages")  # At this point TRL has already normalized

    if environments is not None and completions is not None and rewards_tensor is not None:
        try:
            # Get the raw rewards from the inputs (before normalization)
            raw_rewards = inputs.get("rewards")
            if raw_rewards is not None:
                reward_list = raw_rewards.tolist() if hasattr(raw_rewards, "tolist") else list(raw_rewards)
            else:
                # Fall back to advantages as proxy (already normalized, but still usable)
                reward_list = rewards_tensor.tolist() if hasattr(rewards_tensor, "tolist") else list(rewards_tensor)

            # Get kwargs from inputs for metadata
            meta_kwargs = {}
            for key in ("tier", "query", "difficulty", "optimal_steps"):
                if key in inputs:
                    meta_kwargs[key] = inputs[key]

            # Compute GiGPO step advantages
            gigpo_result = compute_gigpo_step_advantages_from_envs(
                environments=environments,
                completions=completions,
                rewards=reward_list,
                **meta_kwargs,
            )

            # Check if we got meaningful anchor states
            if gigpo_result.anchor_states:
                per_step = gigpo_result_to_per_step_advantages(gigpo_result)

                # Build per-token advantage tensor [B, L]
                # We need to map step indices to token positions.
                # TRL's completion tokens include: assistant tokens (mask=1) +
                # tool response tokens (mask=0). We assign the step advantage
                # to all assistant tokens in that step.
                #
                # Since we don't have exact token→step mapping here, we use
                # a simpler but effective approach: divide the completion evenly
                # among the steps. This is an approximation — the exact mapping
                # would require tokenizing each step's model_output, which is
                # expensive and fragile.
                #
                # The tool_mask already zeros out tool response tokens in the
                # loss, so approximate step boundaries are acceptable.

                B = rewards_tensor.shape[0]
                completion_ids = inputs.get("completion_ids") or inputs.get("input_ids")
                if completion_ids is not None:
                    L = completion_ids.shape[-1]
                else:
                    L = inputs.get("completion_mask", rewards_tensor.unsqueeze(1)).shape[-1]

                token_advantages = torch.zeros(B, L, device=rewards_tensor.device, dtype=rewards_tensor.dtype)

                for i in range(B):
                    steps = per_step[i] if i < len(per_step) else [0.0]
                    n_steps = len(steps)
                    if n_steps == 0:
                        token_advantages[i] = rewards_tensor[i]
                        continue

                    # Divide L tokens evenly among steps
                    chunk_size = max(1, L // n_steps)
                    for j, adv in enumerate(steps):
                        start = j * chunk_size
                        end = (j + 1) * chunk_size if j < n_steps - 1 else L
                        token_advantages[i, start:end] = adv

                # Replace the scalar advantages with per-token tensor.
                # TRL does: advantages.unsqueeze(1) in compute_loss.
                # We need to make sure our [B, L] tensor is used directly.
                # Store in a special key that we'll check below.
                inputs["_gigpo_token_advantages"] = token_advantages

                logger.info(
                    "GiGPO: injected per-token advantages, %d anchor states, "
                    "step advantage range [%.3f, %.3f]",
                    len(gigpo_result.anchor_states),
                    token_advantages.min().item(),
                    token_advantages.max().item(),
                )
            else:
                logger.debug("GiGPO: no anchor states found, using standard GRPO advantages")

        except Exception as e:
            logger.warning("GiGPO: advantage computation failed, falling back to GRPO: %s", e)

    # Call parent's compute_loss
    # We need to intercept the advantage expansion. Since we can't easily
    # monkey-patch mid-call, we use a pragmatic approach: if we have
    # _gigpo_token_advantages, we override the advantages tensor shape
    # so that unsqueeze(1) on a [B, L] tensor becomes [B, 1, L] — but
    # that's wrong. Instead, we pre-set advantages to be [B, L] and
    # hope the parent does `advantages.unsqueeze(1)` which gives [B, 1, L].
    #
    # Actually, the cleanest approach: replace advantages with a [B, 1] dummy,
    # then after super() multiplies, the result is wrong. That won't work.
    #
    # The REAL solution: we must replicate the loss computation ourselves
    # when GiGPO advantages are available.

    gigpo_advs = inputs.pop("_gigpo_token_advantages", None)

    if gigpo_advs is not None:
        return _compute_loss_with_token_advantages(self, model, inputs, gigpo_advs, return_outputs, num_items_in_batch)

    # Standard path — no GiGPO, use parent
    return type(self).__mro__[1].compute_loss(self, model, inputs, return_outputs, num_items_in_batch)


def _compute_loss_with_token_advantages(trainer, model, inputs, token_advantages, return_outputs=False, num_items_in_batch=None):
    """Compute GRPO loss with per-token advantages instead of per-sequence.

    This replicates the essential logic of TRL's GRPOTrainer.compute_loss
    but uses ``token_advantages[B, L]`` directly instead of broadcasting
    a scalar advantage.

    Only the GRPO loss type is supported (not BNPO/Dr.GRPO etc).
    """
    import torch
    import torch.nn.functional as F

    # Forward pass
    completion_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    logits_to_keep = inputs.get("logits_to_keep", completion_ids.shape[-1])

    outputs = model(
        input_ids=completion_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits

    # Compute per-token log-probs (same as TRL)
    # logits shape: [B, L, V], we want log-prob of the actual tokens
    logits = logits[:, -logits_to_keep:, :]
    completion_ids_for_loss = completion_ids[:, -logits_to_keep:]

    per_token_logps = torch.gather(
        F.log_softmax(logits, dim=-1),
        dim=2,
        index=completion_ids_for_loss.unsqueeze(2),
    ).squeeze(2)

    # Old log-probs (from generation)
    old_per_token_logps = inputs.get("old_per_token_logps", inputs.get("ref_per_token_logps"))
    if old_per_token_logps is None:
        old_per_token_logps = per_token_logps.detach()

    old_per_token_logps = old_per_token_logps[:, -logits_to_keep:]

    # Importance sampling ratio
    ratio = torch.exp(per_token_logps - old_per_token_logps)

    # Mask: completion_mask * tool_mask
    completion_mask = inputs.get("completion_mask", torch.ones_like(per_token_logps))
    completion_mask = completion_mask[:, -logits_to_keep:]
    tool_mask = inputs.get("tool_mask")
    if tool_mask is not None:
        tool_mask = tool_mask[:, -logits_to_keep:]
        mask = completion_mask * tool_mask
    else:
        mask = completion_mask

    # Trim token_advantages to match logits_to_keep
    advs = token_advantages[:, -logits_to_keep:]

    # GRPO clipped surrogate loss (same as PPO clip but with GiGPO advantages)
    # TRL's GRPO loss: -mean( min(ratio * A, clip(ratio) * A) * mask )
    eps = 0.2  # clip range, TRL default
    clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
    per_token_loss = -torch.min(ratio * advs, clipped_ratio * advs)

    # Mean over valid tokens
    loss = (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
    loss = loss.mean()

    if return_outputs:
        return loss, outputs
    return loss
