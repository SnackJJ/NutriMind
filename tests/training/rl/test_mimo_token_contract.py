"""CPU checks against the pinned upstream code and the actual SFT tokenizer.

Run in the GRPO environment with MIMO_VERL_ROOT on PYTHONPATH.
"""

import json
import os

import pytest

pytest.importorskip("recipes.general.token_trace")

from recipes.general.token_trace import TokenTrace, select_delta_messages
from recipes.general.chat_delta import anchor_ids, render_ids, render_injected_turn
from transformers import AutoTokenizer
from verl.utils.tokenizer.chat_template import initialize_turn_separator
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def test_exact_generated_tokens_and_observation_masks():
    trace = TokenTrace(response_length=100)
    trace.append_prompt([10, 11])
    trace.append_generated([99, 100], [-.2, -.3])
    trace.append_observation([12, 13, 14])
    trace.append_generated([101], [-.4])
    prompt, response, mask, probs = trace.finalize()
    assert prompt == [10, 11]
    assert response == [99, 100, 12, 13, 14, 101]
    assert mask == [1, 1, 0, 0, 0, 1]
    assert probs == [-.2, -.3, 0, 0, 0, -.4]
    delta, cursor = select_delta_messages([
        {"role": "assistant", "content": "must not re-encode"},
        {"role": "tool", "content": "observation"}], 0)
    assert cursor == 2 and [m["role"] for m in delta] == ["tool"]


def test_qwen_tool_turn_keeps_boundary_and_groups_observations():
    tokenizer_path = os.environ.get("GRPO_TOKENIZER")
    if not tokenizer_path:
        pytest.skip("set GRPO_TOKENIZER to actual SFT tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tools = [{"role": "tool", "tool_call_id": "c1", "content": "first"},
             {"role": "tool", "tool_call_id": "c2", "content": "second"}]
    anchor = anchor_ids(tokenizer, enable_thinking=True)
    separator = initialize_turn_separator(tokenizer, enable_thinking=True)
    injected = render_injected_turn(tokenizer, tools, anchor=anchor,
        turn_separator=separator, enable_thinking=True)
    reference = render_ids(tokenizer, [{"role": "user", "content": ""}] + tools,
        add_generation_prompt=True, enable_thinking=True)
    assert injected == separator + reference[len(anchor):]
    decoded = tokenizer.decode(injected)
    assert decoded.count("<|im_start|>user") == 1
    assert "first" in decoded and "second" in decoded
    assert decoded.endswith("<think>\n")


def test_grpo_invalid_rollout_is_not_a_negative_example():
    import numpy as np
    import torch
    from types import SimpleNamespace
    rewards = torch.tensor([[1.], [0.], [-999.], [1.], [1.]])
    mask = torch.ones_like(rewards)
    advantages, _ = compute_grpo_outcome_advantage(rewards, mask,
        np.array(["mixed", "mixed", "mixed", "constant", "constant"]),
        config=SimpleNamespace(invalid_reward_value=-999))
    assert advantages[0] > 0 and advantages[1] < 0
    assert advantages[2:].eq(0).all()
    assert torch.isfinite(advantages).all()


def test_native_harness_bridge_scores_and_masks_real_tool_observations():
    import asyncio
    from types import SimpleNamespace
    from scripts.prepare_grpo_v2 import collect
    from src.training.rl.verl_agent_loop import NutriMindAgentLoop
    from nutrienv.harness.tools_schema import NUTRIENV_TOOLS
    from verl.experimental.agent_loop.tool_parser import ToolParser
    from verl.tools.schemas import OpenAIFunctionToolSchema

    tokenizer_path = os.environ.get("GRPO_TOKENIZER")
    if not tokenizer_path:
        pytest.skip("set GRPO_TOKENIZER")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    package = next(p for p in collect(["data/student/v2-batch1-200/sft/train.jsonl"]).values()
                   if p.family == "log")
    tool_turns = []
    for row in package.oracle.payload["ledger_tail"]:
        values = {k: row[k] for k in ("food_id", "grams", "eaten_at")}
        params = "".join(f"<parameter={k}>{v}</parameter>" for k, v in values.items())
        tool_turns.append(f"recording</think>\n\n<tool_call>\n<function=log_meal>\n{params}\n</function>\n</tool_call><|im_end|>")
    tool_turns.append("done</think>\n\n<tool_call>\n<function=done>\n</function>\n</tool_call><|im_end|>")
    generated = iter([tokenizer.encode(t, add_special_tokens=False) for t in tool_turns])
    requests = []

    class Server:
        async def generate(self, **kwargs):
            requests.append(kwargs)
            ids = next(generated)
            return SimpleNamespace(token_ids=ids, log_probs=[-.1] * len(ids),
                                   extra_fields={"max_global_steps": 0})

    async def exercise():
        bridge = object.__new__(NutriMindAgentLoop)
        bridge.tokenizer = tokenizer
        bridge.loop = asyncio.get_running_loop()
        bridge.apply_chat_template_kwargs = {"enable_thinking": True}
        bridge.rollout_config = SimpleNamespace(prompt_length=8192, response_length=8192)
        bridge.per_turn_max_tokens = 2048
        bridge.turn_separator = initialize_turn_separator(tokenizer, enable_thinking=True)
        bridge.anchor = anchor_ids(tokenizer, enable_thinking=True)
        bridge.parser = ToolParser.get_tool_parser("qwen3_coder", tokenizer)
        bridge.schemas = [OpenAIFunctionToolSchema.model_validate(t) for t in NUTRIENV_TOOLS]
        bridge.server_manager = Server()
        return await bridge.run({"temperature": 1}, extra_info={
            "task_package_json": json.dumps(package.to_dict())})

    output = asyncio.run(exercise())
    assert output.reward_score == 1 and output.extra_fields["status"] == "pass"
    assert 0 in output.response_mask and 1 in output.response_mask
    assert len(output.response_ids) == len(output.response_logprobs) == len(output.response_mask)
    assert output.prompt_ids == requests[0]["prompt_ids"]
    # The next generation receives the previous generated IDs verbatim.
    first_ids = tokenizer.encode(tool_turns[0], add_special_tokens=False)
    assert requests[1]["prompt_ids"][len(output.prompt_ids):][:len(first_ids)] == first_ids
    assert all(p == 0 for p, m in zip(output.response_logprobs, output.response_mask, strict=True) if not m)


def test_text_model_uses_upstream_hybrid_cache_metadata():
    from types import SimpleNamespace
    from nutrimind_vllm_compat_model import Qwen3_5TextForCausalLM
    from vllm.model_executor.models.interfaces import is_hybrid, supports_multimodal
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    assert is_hybrid(Qwen3_5TextForCausalLM)
    assert not supports_multimodal(Qwen3_5TextForCausalLM)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(
            linear_num_key_heads=16, linear_num_value_heads=32,
            linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4)),
        parallel_config=SimpleNamespace(tensor_parallel_size=1), speculative_config=None)
    assert Qwen3_5TextForCausalLM.get_mamba_state_shape_from_config(config) == \
        Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(config)


def test_text_model_exposes_mrope_positions_used_by_vllm_runner():
    import torch
    from nutrimind_vllm_compat_model import Qwen3_5TextForCausalLM
    from vllm.model_executor.models.interfaces import supports_mrope
    assert supports_mrope(Qwen3_5TextForCausalLM)
    model = object.__new__(Qwen3_5TextForCausalLM)
    positions, delta = model.get_mrope_input_positions([12, 13, 14], [])
    assert positions.shape == (3, 3)
    assert positions.eq(torch.arange(3).expand(3, -1)).all()
    assert delta == 0
