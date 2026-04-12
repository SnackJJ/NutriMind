"""
TRL Multi-Turn Rollout via rollout_func for NutriMind Agentic GRPO.

TRL's built-in tool loop hardcodes HF standard tool_calling format, which is
incompatible with our <tool_call> XML format (ADR-001). We use TRL's official
`rollout_func` escape hatch to run our own multi-turn agentic loop.

Architecture:
  1. TRL calls `nutrimind_rollout(prompts, trainer)` instead of its own generation
  2. We call vLLM server to generate → parse <tool_call> → execute tool → inject
     tool_response → continue generating → repeat
  3. We return prompt_ids, completion_ids (multi-turn concatenated), logprobs,
     and env_mask (1 for model tokens, 0 for tool response tokens)
  4. TRL computes policy gradient on model tokens only (masked by env_mask)

Key: env_mask ensures tool response tokens are excluded from the loss, while
the model still sees them in the causal attention for correct conditioning.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

import requests

from src.training.grpo.environment import (
    NutriMindEnv,
    TaskMetadata,
)
from src.training.grpo.reward import (
    _build_trajectory_from_solution,
    reward_v2,
    RewardBreakdown,
)
from src.orchestrator.orchestrator import TOOL_REGISTRY
from src.orchestrator.tool_parser import ToolParser

logger = logging.getLogger(__name__)


def _vllm_generate(
    server_url: str,
    prompt_text: str,
    tokenizer: Any,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop: Optional[List[str]] = None,
    logprobs: bool = True,
) -> dict:
    """
    Call TRL vLLM server's /generate/ endpoint.

    TRL's ``trl vllm-serve`` exposes ``/generate/`` (not OpenAI-compatible
    ``/v1/completions``).  The request takes ``prompts`` (list[str]) and the
    response returns ``completion_ids``, ``prompt_ids``, and ``logprobs``.

    Returns dict with keys: "text", "completion_ids", "token_logprobs".
    """
    payload = {
        "prompts": [prompt_text],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": 1 if logprobs else 0,
    }
    resp = requests.post(f"{server_url}/generate/", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    completion_ids: List[int] = data["completion_ids"][0]

    # Decode completion token IDs back to text
    text = tokenizer.decode(completion_ids, skip_special_tokens=False)

    # Check if generation was cut short by stop token (</tool_call>)
    # TRL server doesn't support stop sequences natively, so we handle
    # it client-side: truncate at the first stop string if found.
    finish_reason = "length"
    if stop:
        for stop_str in stop:
            idx = text.find(stop_str)
            if idx != -1:
                # Truncate text *before* the stop string (same as OpenAI behavior)
                text = text[:idx]
                finish_reason = "stop"
                # Re-encode to get the correct truncated token IDs
                completion_ids = tokenizer.encode(text, add_special_tokens=False)
                break

    # Extract per-token logprobs (shape: [1][seq_len][num_logprobs])
    token_logprobs: List[float] = []
    if data.get("logprobs") and len(data["logprobs"]) > 0:
        # Each position has a list of (logprob, token_id) pairs sorted desc;
        # index [0] is the sampled token's logprob.
        seq_logprobs = data["logprobs"][0]  # first prompt
        for pos_logprobs in seq_logprobs:
            if pos_logprobs:
                token_logprobs.append(pos_logprobs[0])
            else:
                token_logprobs.append(0.0)
        # Truncate to match completion_ids length (in case we truncated at stop)
        token_logprobs = token_logprobs[:len(completion_ids)]

    return {
        "text": text,
        "finish_reason": finish_reason,
        "completion_ids": completion_ids,
        "token_logprobs": token_logprobs,
    }


def _run_single_multiturn_rollout(
    prompt_messages: List[Dict[str, str]],
    env: NutriMindEnv,
    server_url: str,
    tokenizer: Any,
    max_completion_tokens: int = 5120,
    temperature: float = 0.7,
    num_generation: int = 0,
) -> Dict[str, Any]:
    """
    Run one multi-turn rollout for a single prompt.

    Flow:
    1. Format initial prompt with chat template
    2. Generate via vLLM → get first-turn output
    3. Parse with ToolParser:
       - If final_answer → done
       - If tool_call → execute tool, inject response, continue
    4. Repeat until done or max_rounds

    Returns:
        {
            "full_text": str,           # complete multi-turn text (prompt + all turns)
            "completion_text": str,      # completion part only
            "completion_ids": list[int], # token IDs of completion
            "logprobs": list[float],     # per-token logprobs for completion
            "env_mask": list[int],       # 1=model token, 0=tool response token
            "reward": float,             # reward score
            "trajectory": RolloutTrajectory,
        }
    """
    parser = ToolParser(validate_tool_name=True)

    # Initialize env and get initial messages
    messages = env.reset(prompt_messages[-1]["content"])  # Last message is user query

    # Format prompt using tokenizer's chat template
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    all_completion_text = ""
    all_completion_ids: List[int] = []
    all_logprobs: List[float] = []
    all_env_mask: List[int] = []

    tokens_used = 0
    done = False

    while not done and tokens_used < max_completion_tokens:
        remaining_tokens = max_completion_tokens - tokens_used

        # Generate via vLLM
        gen_result = _vllm_generate(
            server_url=server_url,
            prompt_text=prompt_text + all_completion_text,
            tokenizer=tokenizer,
            max_tokens=min(remaining_tokens, 2048),
            temperature=temperature,
            stop=["</tool_call>"],  # Stop at end of tool call for parsing
        )

        model_text = gen_result["text"]

        # If stopped at </tool_call>, add back the stop token
        if gen_result["finish_reason"] == "stop":
            model_text += "</tool_call>"

        # Use token IDs from generate when available, else re-encode
        model_ids = tokenizer.encode(model_text, add_special_tokens=False)
        model_logprobs = gen_result.get("token_logprobs", [0.0] * len(model_ids))

        # Pad logprobs if length mismatch
        if len(model_logprobs) < len(model_ids):
            model_logprobs.extend([0.0] * (len(model_ids) - len(model_logprobs)))
        model_logprobs = model_logprobs[:len(model_ids)]

        # Add model tokens (mask=1, these participate in loss)
        all_completion_text += model_text
        all_completion_ids.extend(model_ids)
        all_logprobs.extend(model_logprobs)
        all_env_mask.extend([1] * len(model_ids))
        tokens_used += len(model_ids)

        # Step the environment
        updated_messages, env_done, info = env.step(model_text)

        if env_done:
            done = True
        else:
            # Tool was called; env injected the response into messages
            # The last message in updated_messages is the tool response
            tool_response_text = updated_messages[-1]["content"]

            # Format tool response as it would appear in the conversation
            # Use the chat template to get proper formatting
            tool_response_formatted = f"\n{tool_response_text}\n"

            # Tokenize tool response
            tool_ids = tokenizer.encode(tool_response_formatted, add_special_tokens=False)

            # Add tool response tokens (mask=0, excluded from loss)
            all_completion_text += tool_response_formatted
            all_completion_ids.extend(tool_ids)
            all_logprobs.extend([0.0] * len(tool_ids))  # Placeholder logprobs
            all_env_mask.extend([0] * len(tool_ids))     # Tool tokens masked out
            tokens_used += len(tool_ids)

    # Get trajectory and compute reward
    trajectory = env.get_trajectory()

    # Build TaskMetadata from env context (will be filled by caller)
    return {
        "prompt_ids": prompt_ids,
        "completion_text": all_completion_text,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_mask": all_env_mask,
        "trajectory": trajectory,
    }


def make_nutrimind_rollout(
    server_url: str = "http://localhost:8000",
    max_tool_rounds: int = 6,
    max_completion_tokens: int = 5120,
    temperature: float = 0.7,
    num_generations: int = 4,
    tool_registry: Optional[Dict[str, Callable]] = None,
):
    """
    Create a rollout_func for TRL GRPOTrainer.

    The returned function runs multi-turn agentic rollouts using vLLM server
    for generation and NutriMindEnv for tool execution.

    Args:
        server_url: vLLM server URL
        max_tool_rounds: Max tool-calling rounds per episode
        max_completion_tokens: Max tokens for completion
        temperature: Sampling temperature
        num_generations: Number of completions per prompt (G in GRPO)
        tool_registry: Tool functions

    Returns:
        A rollout_func with signature (prompts, trainer) -> dict
    """
    registry = tool_registry or TOOL_REGISTRY

    def rollout_func(prompts: list, trainer) -> Dict[str, Any]:
        """
        TRL rollout_func: runs multi-turn agentic rollouts.

        Args:
            prompts: List of prompt message lists (each is [{"role": ..., "content": ...}, ...])
            trainer: GRPOTrainer instance (gives access to tokenizer, model, etc.)

        Returns:
            Dict with keys: prompt_ids, completion_ids, logprobs, env_mask,
            plus extra fields forwarded to reward functions.
        """
        tokenizer = trainer.processing_class

        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []
        all_env_masks = []

        # Extra fields for reward function
        extra_fields: Dict[str, list] = {
            "tier": [],
            "difficulty": [],
            "optimal_steps": [],
            "query": [],
            "env_state": [],
            "branch_condition": [],
            "completion_text": [],
        }

        for prompt_idx, prompt in enumerate(prompts):
            # Extract metadata from the dataset row
            # TRL passes the full dataset row as prompt context
            # We need to retrieve the metadata columns
            row = trainer.train_dataset[prompt_idx % len(trainer.train_dataset)]
            tier = row.get("tier", "T1")
            difficulty = row.get("difficulty", "medium")
            optimal_steps = row.get("optimal_steps", 1)
            query = row.get("query", "")
            env_state_str = row.get("env_state", "{}")
            branch_condition_str = row.get("branch_condition", "")

            try:
                env_state = json.loads(env_state_str) if isinstance(env_state_str, str) else env_state_str
            except (json.JSONDecodeError, TypeError):
                env_state = {}

            # Generate G rollouts for this prompt
            for g in range(num_generations):
                env = NutriMindEnv(
                    tool_registry=registry,
                    max_tool_rounds=max_tool_rounds,
                    user_state_snapshot=env_state if env_state else None,
                )

                result = _run_single_multiturn_rollout(
                    prompt_messages=prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}],
                    env=env,
                    server_url=server_url,
                    tokenizer=tokenizer,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                    num_generation=g,
                )

                all_prompt_ids.append(result["prompt_ids"])
                all_completion_ids.append(result["completion_ids"])
                all_logprobs.append(result["logprobs"])
                all_env_masks.append(result["env_mask"])

                extra_fields["tier"].append(tier)
                extra_fields["difficulty"].append(difficulty)
                extra_fields["optimal_steps"].append(optimal_steps)
                extra_fields["query"].append(query)
                extra_fields["env_state"].append(env_state_str)
                extra_fields["branch_condition"].append(branch_condition_str)
                extra_fields["completion_text"].append(result["completion_text"])

        output = {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_mask": all_env_masks,
            **extra_fields,
        }

        return output

    return rollout_func


def make_multiturn_reward_fn(
    max_tool_rounds: int = 6,
    tool_registry: Optional[Dict[str, Callable]] = None,
):
    """
    Create a TRL-compatible reward function for multi-turn trajectories.

    When used with rollout_func, completions contain the FULL multi-turn
    conversation text (model turns + tool responses). The reward function
    parses this and scores using reward_v2.

    Returns:
        A function with signature (completions: list[str], **kwargs) -> list[float]
    """
    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        tiers = kwargs.get("tier", ["T1"] * len(completions))
        difficulties = kwargs.get("difficulty", ["medium"] * len(completions))
        optimal_steps_list = kwargs.get("optimal_steps", [1] * len(completions))
        queries = kwargs.get("query", [""] * len(completions))
        branch_conditions = kwargs.get("branch_condition", [""] * len(completions))

        # If completion_text is available from rollout_func, use it
        completion_texts = kwargs.get("completion_text", completions)

        scores = []
        for i, completion in enumerate(completion_texts):
            if not completion or not completion.strip():
                scores.append(0.0)
                continue

            bc = None
            bc_str = branch_conditions[i] if i < len(branch_conditions) else ""
            if bc_str:
                try:
                    bc = json.loads(bc_str)
                except (json.JSONDecodeError, TypeError):
                    pass

            task_meta = TaskMetadata(
                query=queries[i] if i < len(queries) else "",
                tier=tiers[i] if i < len(tiers) else "T1",
                difficulty=difficulties[i] if i < len(difficulties) else "medium",
                optimal_steps=optimal_steps_list[i] if i < len(optimal_steps_list) else 1,
                branch_condition=bc,
            )

            trajectory = _build_trajectory_from_solution(completion, task_meta)
            breakdown = reward_v2(trajectory, task_meta)
            scores.append(breakdown.total)

        return scores

    return reward_fn
