"""Teacher rollout via the lab native tool-calling loop (ticket 025).

Reuses ``nutrienv.harness.tool_call.run_episode_tool_call`` with an injected
completion. The loop is not copied. ``parallel_tool_calls`` is false.
Ticket 009's ``TeacherReActHarness`` stays CLOSED; this is the v2 teacher path.

``teacher_complete(request) -> dict`` returns ``reasoning_content`` and
``tool_calls`` (OpenAI-shaped). Real network still requires
``NUTRIMIND_ALLOW_NETWORK=1`` (enforced by the injected complete, not by
letting the lab post).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import time
from collections.abc import Callable

from nutrienv.env import NutriEnv
from nutrienv.harness.runner import FINISH_OPS
from nutrienv.harness.tool_call import ToolCallInfraError, run_episode_tool_call
from nutrienv.harness.tools_schema import NUTRIENV_TOOLS, TOOL_SYSTEM_PROMPT

from src.training.data_factory.concepts import EpisodeResult, TurnMeta

__all__ = [
    "NUTRIENV_TOOLS",
    "TOOL_SYSTEM_PROMPT",
    "ScriptedFCTeacher",
    "rollout_tool_call",
]

Completion = dict


def _obs_text(observation) -> str:
    return json.dumps(observation, default=str)[:6000]


@dataclasses.dataclass
class _StepTel:
    step_index: int
    action: dict
    observation_snippet: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    latency_seconds: float
    is_valid_tool: bool
    error: str | None


@dataclasses.dataclass
class _TaskTel:
    task_id: str
    family: str
    query: str
    persona: str
    passed: bool
    score_tag: str
    n_steps: int
    max_budget: int
    wall_time_seconds: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_reasoning_tokens: int
    total_tokens: int
    tool_counts: dict
    invalid_tool_count: int
    allergen_violated: bool
    steps: list
    is_void: bool
    void_reason: str | None


class _RecordingEnv(NutriEnv):
    """NutriEnv that records reset + each stepped action for EpisodeResult."""

    def reset(self, s0):  # noqa: ANN001
        obs = super().reset(s0)
        self.reset_observation = obs
        self.step_log: list[tuple[dict, object]] = []
        return obs

    def step(self, action):  # noqa: ANN001
        result = super().step(action)
        if result.get("ok") and isinstance(result.get("observation"), dict):
            obs = result["observation"]
        else:
            obs = {"error": result.get("error")}
        self.step_log.append((action, obs))
        return result


@contextlib.contextmanager
def _inject_lab(*, complete_raw, env_cls):
    import nutrienv.harness.tool_call as lab

    orig_raw, orig_env = lab.post_chat_completion_raw, lab.NutriEnv
    lab.post_chat_completion_raw = complete_raw
    lab.NutriEnv = env_cls
    try:
        yield
    finally:
        lab.post_chat_completion_raw = orig_raw
        lab.NutriEnv = orig_env


def _as_openai_body(completion: Completion) -> dict:
    message = {
        "role": "assistant",
        "content": completion.get("content"),
        "reasoning_content": completion.get("reasoning_content"),
        "tool_calls": completion.get("tool_calls") or [],
    }
    usage = dict(completion.get("usage") or {})
    details = {}
    if "reasoning_tokens" in usage:
        details["reasoning_tokens"] = usage.pop("reasoning_tokens")
        usage["completion_tokens_details"] = details
    return {"choices": [{"message": message}], "usage": usage}


def _build_turns(
    completions: list[Completion],
    step_log: list[tuple[dict, object]],
) -> tuple[list[TurnMeta], bool]:
    turns: list[TurnMeta] = []
    stepped = 0
    reached_finish = False
    for completion in completions:
        tool_calls = list(completion.get("tool_calls") or [])
        reasoning = completion.get("reasoning_content")
        usage = completion.get("usage")
        if not tool_calls:
            turns.append(
                TurnMeta(
                    tool_calls=[],
                    tool_call_id=None,
                    executed_op=None,
                    reasoning_content=reasoning,
                    content=completion.get("content"),
                    usage=usage,
                    raw_action_text=None,
                    parse_status=None,
                    fallback_used=False,
                    fallback_reason=None,
                )
            )
            continue
        first = tool_calls[0]
        func = first.get("function") or {}
        name = func.get("name") or ""
        call_id = first.get("id")
        if name in FINISH_OPS:
            reached_finish = True
            turns.append(
                TurnMeta(
                    tool_calls=tool_calls,
                    tool_call_id=call_id,
                    executed_op=None,
                    reasoning_content=reasoning,
                    content=completion.get("content"),
                    usage=usage,
                    observation=_obs_text({"op": "finish", "done": True}),
                    raw_action_text=None,
                    parse_status=None,
                    fallback_used=False,
                    fallback_reason=None,
                )
            )
            break
        action, obs = step_log[stepped]
        stepped += 1
        turns.append(
            TurnMeta(
                tool_calls=tool_calls,
                tool_call_id=call_id,
                executed_op=action,
                reasoning_content=reasoning,
                content=completion.get("content"),
                usage=usage,
                observation=_obs_text(obs),
                raw_action_text=None,
                parse_status=None,
                fallback_used=False,
                fallback_reason=None,
            )
        )
        if name == "submit_plan":
            reached_finish = True
            break
    return turns, reached_finish


def rollout_tool_call(
    task,
    *,
    teacher_complete: Callable[[dict], Completion],
    catalog,
    parallel_tool_calls: bool = False,
    model: str = "scripted-teacher",
) -> EpisodeResult:
    """Drive one episode of ``task`` through the lab FC loop."""
    if parallel_tool_calls:
        raise ValueError("parallel_tool_calls must be false (ADR-014)")

    completions: list[Completion] = []
    env_holder: list[_RecordingEnv] = []

    class _Env(_RecordingEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            env_holder.append(self)

    def complete_raw(_url, payload, _api_key, **_kwargs):
        request = {
            "model": payload.get("model"),
            "messages": payload.get("messages"),
            "tools": payload.get("tools"),
            "temperature": payload.get("temperature"),
            "parallel_tool_calls": payload.get("parallel_tool_calls", False),
        }
        completion = teacher_complete(request)
        completions.append(completion)
        return _as_openai_body(completion)

    harness_spec = {
        "model": model,
        "url": "injected://teacher",
        "api_key": "unused-injected-teacher",
        "timeout": 60.0,
        "retries": 0,
        "parallel_tool_calls": False,
    }
    started = time.monotonic()
    error: str | None = None
    try:
        with _inject_lab(complete_raw=complete_raw, env_cls=_Env):
            run_episode_tool_call(
                task, harness_spec, catalog, _StepTel, _TaskTel
            )
    except ToolCallInfraError as exc:
        error = f"teacher error: {type(exc).__name__}: {exc}"

    env = env_holder[0] if env_holder else _RecordingEnv()
    if not env_holder:
        env.reset(task.s0)
    turns, reached_finish = _build_turns(completions, getattr(env, "step_log", []))
    reset_obs = getattr(env, "reset_observation", None)
    return EpisodeResult(
        end_state=env.state(),
        turns=turns,
        reached_finish=reached_finish,
        error=error,
        task=task,
        latency_s=round(time.monotonic() - started, 6),
        reset_observation=_obs_text(reset_obs) if reset_obs is not None else None,
    )


class ScriptedFCTeacher:
    """Queue of ``(reasoning_content, tool_calls)`` for offline FC tests.

    An exhausted queue returns a no-tool-call completion (not an exception),
    so a no-finish episode can drain the step budget without a teacher error.
    """

    def __init__(self, turns: list[tuple[str | None, list]]):
        self.turns = list(turns)
        self.requests: list[dict] = []
        self._index = 0

    def __call__(self, request: dict) -> Completion:
        self.requests.append(
            {k: v for k, v in request.items() if k != "messages"}
        )
        if self._index >= len(self.turns):
            return {
                "content": None,
                "reasoning_content": "",
                "tool_calls": [],
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "reasoning_tokens": 0,
                },
            }
        reasoning, tool_calls = self.turns[self._index]
        self._index += 1
        return {
            "content": None,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls,
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "reasoning_tokens": 20 if reasoning else 0,
            },
        }



