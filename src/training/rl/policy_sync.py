"""Checkpoint → policy_spec endpoint (RL ticket 008).

Not a second seam: ``student_rollout`` still only takes ``policy_spec``.
Mechanism recorded in close notes: colocated vLLM, sync every train step
(veRL ``vllm_rollout`` + weight broadcast). Tests inject a fake server that
reads a weights generation counter.
"""

from __future__ import annotations

from src.training.rl.prompt import prompt_for_package, tokenize_prompt

__all__ = ["SYNC_CADENCE", "VERL_ROLLOUT_ENGINE", "policy_spec_from_checkpoint"]

VERL_ROLLOUT_ENGINE = "verl.workers.rollout.vllm_rollout.vLLMRollout"
SYNC_CADENCE = "every_train_step"


def policy_spec_from_checkpoint(
    checkpoint: dict,
    *,
    complete,
    catalog=None,
) -> dict:
    """Build the D3 ``policy_spec`` from a train-step checkpoint handle."""
    return {
        "url": checkpoint["url"],
        "model": checkpoint["model"],
        "generation": checkpoint.get("generation", 0),
        "complete": complete,
        "catalog": catalog,
        "parallel_tool_calls": False,
        "tokenizer": checkpoint.get("tokenizer"),
        "prompt_fn": prompt_for_package,
        "tokenize_fn": tokenize_prompt,
    }
