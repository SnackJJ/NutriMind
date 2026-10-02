"""NutriEnv native FC harness bridged to MiMo's token-in/token-out AgentLoop.

The pinned MiMo recipes own incremental token bookkeeping and observation
rendering. NutriEnv owns tool execution, episode limits and final-state scoring.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import re
import time
from uuid import uuid4

from recipes.general.chat_delta import anchor_ids, render_ids, render_injected_turn
from recipes.general.token_trace import TokenTrace, select_delta_messages
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema

from nutrienv.bench import Scorer
from nutrienv.harness.tools_schema import NUTRIENV_TOOLS
from nutrienv.world.catalog_store import load_catalog

from src.training.data_factory.concepts import TaskPackage
from src.training.data_factory.rollout_fc import rollout_tool_call
from src.training.rl.rollout import _task_from_package

# Each running episode holds one thread, so the pool size caps concurrent episodes per worker.
_HARNESS_WORKERS = 8
_HARNESS_POOL = None


def _harness_pool():
    global _HARNESS_POOL
    if _HARNESS_POOL is None:
        _HARNESS_POOL = ThreadPoolExecutor(max_workers=_HARNESS_WORKERS, thread_name_prefix="nutrimind-harness")
    return _HARNESS_POOL


class NutriMindAgentLoop(AgentLoopBase):
    def __init__(self, *args, per_turn_max_tokens=2048, harness_workers=8, **kwargs):
        global _HARNESS_WORKERS
        super().__init__(*args, **kwargs)
        _HARNESS_WORKERS = int(harness_workers)
        self.per_turn_max_tokens = int(per_turn_max_tokens)
        self.parser = ToolParser.get_tool_parser("qwen3_coder", self.tokenizer)
        self.schemas = [OpenAIFunctionToolSchema.model_validate(t) for t in NUTRIENV_TOOLS]
        self.anchor = anchor_ids(self.tokenizer, **self.apply_chat_template_kwargs)

    async def run(self, sampling_params, **kwargs):
        package = TaskPackage.from_dict(json.loads(kwargs["extra_info"]["task_package_json"]))
        catalog = load_catalog()
        task = _task_from_package(package, catalog)
        trace = TokenTrace(response_length=self.rollout_config.response_length)
        cursor = 0
        calls = 0
        budget_exhausted = False
        request_id = uuid4().hex
        engine_fields = {}
        started = time.monotonic()

        async def generate(request):
            nonlocal cursor, calls, budget_exhausted
            if request.get("parallel_tool_calls"):
                raise ValueError("parallel tool calls are not allowed")
            delta, cursor = select_delta_messages(request["messages"], cursor)
            if not trace.prompt_len:
                ids = render_ids(self.tokenizer, delta, add_generation_prompt=True,
                                 tools=NUTRIENV_TOOLS, **self.apply_chat_template_kwargs)
                if len(ids) > self.rollout_config.prompt_length:
                    raise ValueError("native FC prompt exceeds configured prompt_length")
                trace.append_prompt(ids)
            elif delta:
                ids = render_injected_turn(self.tokenizer, delta,
                    turn_separator=self.turn_separator, anchor=self.anchor,
                    **self.apply_chat_template_kwargs)
                if len(ids) >= trace.remaining_budget():
                    budget_exhausted = True
                    return {"content": "", "tool_calls": [], "finish_reason": "length"}
                trace.append_observation(ids)
            if budget_exhausted or trace.remaining_budget() < 2:
                budget_exhausted = True
                return {"content": "", "tool_calls": [], "finish_reason": "length"}
            params = {**sampling_params, "max_tokens": min(self.per_turn_max_tokens, trace.remaining_budget())}
            output = await self.server_manager.generate(request_id=request_id,
                prompt_ids=list(trace.token_ids), sampling_params=params)
            if output.log_probs is None or len(output.log_probs) != len(output.token_ids):
                raise ValueError("rollout engine must return aligned token log probabilities")
            trace.append_generated(output.token_ids, output.log_probs)
            calls += 1
            engine_fields.update(output.extra_fields or {})
            content, parsed = await self.parser.extract_tool_calls(output.token_ids, self.schemas)
            # Parsing affects tool execution only; the training trace retains exact generated IDs.
            reasoning = None
            if "</think>" in content:
                reasoning, content = content.split("</think>", 1)
                reasoning = reasoning.removeprefix("<think>").strip()
            content = re.sub(r"<\|im_end\|>\s*$", "", content).strip()
            return {"content": content, "reasoning_content": reasoning,
                    "tool_calls": [{"id": p.tool_call_id or f"call_{calls}_{i}", "type": "function",
                        "function": {"name": p.name, "arguments": p.arguments}}
                        for i, p in enumerate(parsed)],
                    "finish_reason": "tool_calls" if parsed else "stop",
                    "usage": {"prompt_tokens": len(trace.token_ids) - len(output.token_ids),
                              "completion_tokens": len(output.token_ids)}}

        def complete(request):
            return asyncio.run_coroutine_threadsafe(generate(request), self.loop).result()

        episode = await self.loop.run_in_executor(_harness_pool(),
            lambda: rollout_tool_call(task, teacher_complete=complete, catalog=catalog,
                                      parallel_tool_calls=False, model="nutrimind-grpo"))
        if episode.error:
            status, reward, score_tag = "indeterminate", -999.0, None
        else:
            score = Scorer().score(episode.end_state, task.oracle)
            reward = float(score.get("passed") is True)
            status, score_tag = ("pass" if reward else "fail"), score.get("tag")
        prompt, response, mask, logprobs = trace.finalize()
        if not response or not any(mask):
            raise ValueError("empty policy trajectory; cannot train")
        print(json.dumps({"event": "nutrimind_rollout", "task_id": package.task_id,
            "status": status, "reward": reward, "score_tag": score_tag, "calls": calls,
            "policy_tokens": sum(mask), "observation_tokens": len(mask) - sum(mask),
            "budget_exhausted": budget_exhausted, "seconds": round(time.monotonic() - started, 2),
            "policy_version": engine_fields.get("max_global_steps")}), flush=True)
        return AgentLoopOutput(prompt_ids=prompt, response_ids=response, response_mask=mask,
            response_logprobs=logprobs, reward_score=reward, num_turns=calls * 2,
            metrics={"generate_sequences": time.monotonic() - started},
            extra_fields={**engine_fields, "task_id": package.task_id, "status": status,
                          "is_infra": float(episode.error is not None),
                          "score_tag": score_tag or "", "budget_exhausted": budget_exhausted,
                          "policy_tokens": sum(mask), "observation_tokens": len(mask) - sum(mask),
                          "turn_scores": [], "tool_rewards": []})
