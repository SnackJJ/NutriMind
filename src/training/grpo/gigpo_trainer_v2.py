"""
GiGPOTrainerV2 — Improved GRPOTrainer subclass with true step-level advantage injection.

This version overrides `_generate_and_score_completions` to intercept the computed advantages
and the active `self.environments` before TRL shuffles and drops them. We replace
the scalar advantages with a (B, L) tensor of step-level advantages. TRL's native
`compute_loss` seamlessly handles (B, L) advantages.
"""

import logging
import torch

from src.training.grpo.gigpo import (
    compute_gigpo_step_advantages_from_envs,
    gigpo_result_to_per_step_advantages,
)

logger = logging.getLogger(__name__)


class GiGPOTrainerV2:
    """Drop-in replacement for GRPOTrainer with robust GiGPO step advantages.

    Usage::

        from src.training.grpo.gigpo_trainer_v2 import GiGPOTrainerV2
        trainer = GiGPOTrainerV2(model=..., environment_factory=..., ...)
        trainer.train()
    """

    _base_class = None

    def __new__(cls, *args, **kwargs):
        if cls._base_class is None:
            from trl import GRPOTrainer as _GRPOTrainer
            cls._base_class = _GRPOTrainer

            cls._real_class = type(
                "GiGPOTrainerV2Impl",
                (_GRPOTrainer,),
                {
                    "_generate_and_score_completions": _gigpo_generate_and_score,
                },
            )

        return cls._real_class(*args, **kwargs)


def _gigpo_generate_and_score(self, inputs):
    logger.info("=== GiGPO Intercept ===: Entering _generate_and_score_completions hook.")
    
    # 1. Call parent to do generation, reward calculation, and baseline advantage computation
    outputs = type(self).__mro__[1]._generate_and_score_completions(self, inputs)

    # 2. Grab standard scalar advantages [B]
    rewards_tensor = outputs.get("advantages")
    environments = getattr(self, "environments", None)

    if environments is None:
        logger.warning("=== GiGPO Intercept ===: FAILED! `environments` is None. Falling back to GRPO.")
    elif rewards_tensor is None:
        logger.warning("=== GiGPO Intercept ===: FAILED! `advantages` tensor is None. Falling back to GRPO.")
    else:
        logger.info("=== GiGPO Intercept ===: SUCCESS! Intercepted %d active environments before TRL destroys them.", len(environments))
        try:
            # We need the pre-normalized rewards for GiGPO.
            reward_list = rewards_tensor.tolist() if hasattr(rewards_tensor, "tolist") else list(rewards_tensor)

            # Reconstruct completions
            completion_ids = outputs["completion_ids"]
            contents = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            completions = [[{"role": "assistant", "content": content}] for content in contents]

            meta_kwargs = {}
            for key in ("tier", "query", "difficulty", "optimal_steps"):
                if key in inputs[0]:
                    meta_kwargs[key] = [inp.get(key) for inp in inputs]

            logger.info("=== GiGPO Intercept ===: Computing step-level advantages from environments...")
            gigpo_result = compute_gigpo_step_advantages_from_envs(
                environments=environments,
                completions=completions,
                rewards=reward_list,
                **meta_kwargs,
            )

            if gigpo_result.anchor_states:
                per_step = gigpo_result_to_per_step_advantages(gigpo_result)

                B, L = completion_ids.shape
                token_advantages = torch.zeros(B, L, device=rewards_tensor.device, dtype=rewards_tensor.dtype)

                for i in range(B):
                    steps = per_step[i] if i < len(per_step) else [0.0]
                    n_steps = len(steps)
                    if n_steps == 0:
                        token_advantages[i] = rewards_tensor[i]
                        continue

                    chunk_size = max(1, L // n_steps)
                    for j, adv in enumerate(steps):
                        start = j * chunk_size
                        end = (j + 1) * chunk_size if j < n_steps - 1 else L
                        token_advantages[i, start:end] = adv

                logger.info(
                    "=== GiGPO Intercept ===: Injecting per-token matrix! Old shape: %s -> New shape: %s. "
                    "Found %d anchor states. Advantage range: [%.3f, %.3f]",
                    list(rewards_tensor.shape), list(token_advantages.shape),
                    len(gigpo_result.anchor_states),
                    token_advantages.min().item(),
                    token_advantages.max().item(),
                )
                
                # Replace the advantages in outputs!
                outputs["advantages"] = token_advantages
                
            else:
                logger.info("=== GiGPO Intercept ===: No anchor states found (all paths unique). Using standard scalar GRPO advantages.")

        except Exception as e:
            logger.error("=== GiGPO Intercept ===: CRITICAL ERROR during GiGPO calc: %s. Falling back to standard GRPO.", e, exc_info=True)

    return outputs
