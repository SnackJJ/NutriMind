"""Teacher rollouts — instrumented ReActHarness subclass + ark/ clients (009).

**(a) ``TeacherReActHarness``** — a ``nutrienv.harness.ReActHarness`` subclass
whose completion comes from an injected ``teacher_complete(request) -> dict``
(the single seam; the production implementation is the ark client below, tests
inject a scripted queue). ``version="v2"``, ``context_limit=None`` (the full
ReAct log — published protocol; the 12-message slide is an ablation). The
base class still owns message assembly, the step-budget lines, the 6000-char
observation cap, and its private ``_parse_action`` **to drive the env**; v2's
own :func:`~src.training.data_factory.verify.parse_action_text` re-parses each
assistant text afterwards to record ``parse_status`` / ``fallback_used`` /
``fallback_reason`` on the shared ``TurnMeta`` (spec §12 legality — never the
private parser's return path).

**(b) ``rollout``** — the episode driver mirroring nutri-env's protocol (env
``reset`` → ``act`` → ``step``, finish ops terminate without an env step,
failed steps feed ``{"error": ...}`` back to the teacher) but WITHOUT the
eval runner's idle-read/submit breaks: a v2 training trajectory must end in
an explicit FINISH op (ADR-011), and the step budget is the only other
terminator. Builds the ticket-003 ``EpisodeResult``.

**(c) ark clients** — thin production ``teacher_complete`` against
``api/plan/v3/chat/completions`` (ADR-011 amended): reads
``message.content`` + ``message.reasoning_content`` SEPARATELY (nutri-env's
``complete_chat`` collapses them), ``usage.completion_tokens_details.
reasoning_tokens`` for usage. Transport retries live here; attempt-level k
retries belong to build (ticket 011). Real calls are guarded by
``NUTRIMIND_ALLOW_NETWORK=1``; the credential comes from the environment and
is never logged, echoed, or cached. The expander client is the same wire
format with ``thinking: {"type": "disabled"}``.

Stage module: imports nutrienv at module level (allowed by spec §18).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from nutrienv.env import NutriEnv
from nutrienv.harness import ReActHarness
from nutrienv.harness.react import context_messages
from nutrienv.harness.runner import (
    DEFAULT_MAX_STEPS,
    FAMILY_MAX_STEPS,
    FINISH_OPS,
)

from src.training.data_factory.concepts import EpisodeResult, TurnMeta
from src.training.data_factory.verify import parse_action_text

__all__ = [
    "TeacherReActHarness",
    "ScriptedTeacher",
    "make_ark_chat_client",
    "make_ark_teacher_client",
    "make_ark_expander_client",
    "rollout",
]

Completion = dict  # {"content", "reasoning_content", "finish_reason", "usage"}


# --------------------------------------------------------------------------- #
# (a) instrumented harness
# --------------------------------------------------------------------------- #


class TeacherReActHarness(ReActHarness):
    """ReActHarness driving the env, completing via the injected teacher.

    ``teacher_complete(request: dict) -> Completion`` receives exactly the
    request the base class would have posted (``model``, context-limited
    ``messages``, ``temperature``, plus ``extra_body`` — build puts the
    ``thinking`` length control and any retry-temperature override in
    ``extra_body``) and returns the completion with ``content`` and
    ``reasoning_content`` kept separate.
    """

    def __init__(
        self,
        *,
        teacher_complete: Callable[[dict], Completion],
        model: str = "scripted-teacher",
        max_steps: int = DEFAULT_MAX_STEPS,
        extra_body: dict | None = None,
        api_key: str | None = None,
    ) -> None:
        # api_key: the base class demands one even though completion is
        # injected — "unused" for scripted teachers; production harnesses pass
        # the real env credential so construction fails fast when it is unset.
        super().__init__(
            model=model,
            api_key=api_key or "unused-injected-teacher",
            version="v2",
            context_limit=None,
            max_steps=max_steps,
            extra_body=extra_body,
        )
        self._teacher_complete = teacher_complete
        self._last_completion: Completion | None = None
        self._pending_turn: TurnMeta | None = None

    # -- completion seam ---------------------------------------------------- #

    def _complete(self) -> str:
        request = {
            "model": self.model,
            "messages": context_messages(self.messages, limit=self.context_limit),
            "temperature": 0.0,
            **self.extra_body,
        }
        completion = self._teacher_complete(request)
        content = completion.get("content")
        if not isinstance(content, str):
            raise RuntimeError("teacher_complete returned no content string")
        self._last_completion = completion
        return content

    # -- per-turn instrumentation ------------------------------------------- #

    def act(self, observation: dict, query: str, history: list) -> dict:
        action = super().act(observation, query, history)
        raw = self._last_completion.get("content") if self._last_completion else None
        parsed, status = parse_action_text(raw)
        if parsed is not None and parsed == action:
            fallback_used, fallback_reason = False, None
        elif parsed is None:
            fallback_used, fallback_reason = True, f"parse:{status}"
        else:
            # the base harness substituted a fallback op (or normalized the
            # action differently) — the executed action is not the text's
            fallback_used, fallback_reason = True, "action-not-text"
        self._pending_turn = TurnMeta(
            raw_action_text=raw,
            executed_op=action,
            parse_status=status,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            content=self._last_completion.get("content"),
            reasoning_content=self._last_completion.get("reasoning_content"),
            finish_reason=self._last_completion.get("finish_reason"),
            usage=self._last_completion.get("usage"),
        )
        return action

    def take_turn_meta(self, observation: str | None = None) -> TurnMeta:
        """Pop the turn recorded by ``act``; the episode driver fills the env
        observation this turn produced (``None`` for the finish turn)."""
        turn, self._pending_turn = self._pending_turn, None
        if turn is None:
            raise RuntimeError("take_turn_meta called with no pending turn")
        turn.observation = observation
        return turn


class ScriptedTeacher:
    """A deterministic ``teacher_complete`` for tests: a queue of
    ``(content, reasoning_content)`` pairs consumed one per completion call.
    An exhausted queue raises (the episode records it as a teacher error)."""

    def __init__(self, turns: list[tuple[str, str | None]]):
        self.turns = list(turns)
        self.requests: list[dict] = []  # every request seen, for assertions
        self._index = 0

    def __call__(self, request: dict) -> Completion:
        self.requests.append({k: v for k, v in request.items() if k != "messages"})
        if self._index >= len(self.turns):
            raise RuntimeError("scripted teacher queue exhausted")
        content, reasoning = self.turns[self._index]
        self._index += 1
        return {
            "content": content,
            "reasoning_content": reasoning,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "reasoning_tokens": 20 if reasoning else 0,
            },
        }


# --------------------------------------------------------------------------- #
# (b) the episode driver
# --------------------------------------------------------------------------- #


def rollout(
    harness: TeacherReActHarness,
    task,
    *,
    max_steps: int | None = None,
) -> EpisodeResult:
    """Drive one episode of ``task`` through ``harness`` (mirrors nutri-env's
    protocol; no eval-runner early breaks — a training trajectory ends in an
    explicit FINISH op or the step budget)."""
    budget = max_steps if max_steps is not None else FAMILY_MAX_STEPS.get(
        task.family, DEFAULT_MAX_STEPS
    )
    started = time.monotonic()
    env = NutriEnv()
    observation = env.reset(task.s0)
    harness.reset()
    turns: list[TurnMeta] = []
    history: list[dict] = []
    error: str | None = None
    reached_finish = False

    for _ in range(budget):
        try:
            action = harness.act(observation, task.query, history)
        except Exception as exc:  # teacher/transport failure after client retries
            error = f"teacher error: {type(exc).__name__}: {exc}"
            break
        op = action.get("op") if isinstance(action, dict) else None
        if op in FINISH_OPS:
            history.append(action)
            turns.append(harness.take_turn_meta(None))
            reached_finish = True
            break
        try:
            result = env.step(action)
        except Exception as exc:
            turns.append(harness.take_turn_meta(None))
            error = f"env error: {type(exc).__name__}: {exc}"
            break
        history.append(action)
        if result.get("ok") and isinstance(result.get("observation"), dict):
            observation = result["observation"]
        else:
            observation = {"error": result.get("error")}
        turns.append(
            harness.take_turn_meta(
                json.dumps(observation, default=str, ensure_ascii=False)
            )
        )

    return EpisodeResult(
        end_state=env.state(),
        turns=turns,
        reached_finish=reached_finish,
        error=error,
        task=task,
        latency_s=round(time.monotonic() - started, 6),
    )


# --------------------------------------------------------------------------- #
# (c) ark clients (production teacher_complete / expander)
# --------------------------------------------------------------------------- #

_TRANSPORT_RETRIES = 3
_RETRY_BACKOFF_S = 1.0
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
_DEFAULT_ARK_BASE = "https://ark.cn-beijing.volces.com/api/plan/v3"


def _endpoint_model(model: str) -> str:
    """The wire model id: the routing prefix (``ark/``) is resolved by
    ``lookup_chat_model`` for the base class, the endpoint wants the bare id."""
    return model.split("/", 1)[1] if "/" in model else model


def _resolve_endpoint(endpoint: str, *, base_env: str = "ARK_BASE_URL") -> str:
    """Resolve the config's endpoint (often a path like ``api/plan/v3/chat/
    completions``) against ``ARK_BASE_URL`` without duplicating segments the
    base already carries (it may be host-only, host + api/plan/v3, or a full
    completions URL)."""
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    base = os.environ.get(base_env, _DEFAULT_ARK_BASE).rstrip("/")
    path = "/" + endpoint.lstrip("/")
    if base.endswith(path):
        return base
    # drop the leading path SEGMENTS the base already carries (the base may be
    # host-only, host + /api/plan/v3, or anything in between)
    segments = path.split("/")
    for i in range(1, len(segments)):
        prefix = "/".join(segments[:i])
        if prefix and base.endswith(prefix):
            return base + "/" + "/".join(segments[i:])
    return base + path


def make_ark_chat_client(
    *,
    endpoint: str,
    model: str,
    credential_env: str,
    thinking: dict,
    timeout_s: float,
) -> Callable[[dict], Completion]:
    """A thin ``teacher_complete``-shaped client for one ark endpoint.

    Keeps ``content`` / ``reasoning_content`` separate; transport-level
    retries (HTTP 408/409/429/5xx, timeouts, connection errors) happen here —
    attempt-level k retries belong to build. The credential is read from the
    environment at call time and never logged or returned.
    """

    def complete(request: dict) -> Completion:
        if os.environ.get("NUTRIMIND_ALLOW_NETWORK") != "1":
            raise RuntimeError(
                "real network disabled: set NUTRIMIND_ALLOW_NETWORK=1 to "
                "call the ark endpoint"
            )
        api_key = os.environ.get(credential_env, "")
        if not api_key:
            raise RuntimeError(f"{credential_env} is not set")

        body = {
            "model": _endpoint_model(request.get("model", model)),
            "messages": request["messages"],
            "thinking": dict(thinking),
        }
        if "temperature" in request:
            body["temperature"] = request["temperature"]
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        url = _resolve_endpoint(endpoint)

        last_error = "unreachable"
        for attempt in range(_TRANSPORT_RETRIES):
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code in _RETRYABLE_STATUS and attempt + 1 < _TRANSPORT_RETRIES:
                    last_error = f"HTTP {exc.code}"
                    time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
                    continue
                # never include the response/request body — it carries no key,
                # but keep error surfaces minimal and predictable
                raise RuntimeError(f"ark request failed: HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt + 1 < _TRANSPORT_RETRIES:
                    last_error = f"{type(exc).__name__}"
                    time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"ark request failed after {_TRANSPORT_RETRIES} attempts "
                    f"({last_error})"
                ) from None

        message = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return {
            "content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content"),
            "finish_reason": (data["choices"][0].get("finish_reason")),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": details.get("reasoning_tokens"),
            },
        }

    return complete


def make_ark_teacher_client(teacher_config) -> Callable[[dict], Completion]:
    """The production ``teacher_complete``: ark/deepseek-v4-flash on
    api/plan/v3 with ``thinking`` as the length control (ADR-011 amended)."""
    return make_ark_chat_client(
        endpoint=teacher_config.endpoint,
        model=teacher_config.model_id,
        credential_env=teacher_config.credential_env,
        thinking=teacher_config.thinking,  # {"type": "enabled"}
        timeout_s=teacher_config.per_turn_timeout_s,
    )


def make_ark_expander_client(expander_config) -> Callable[[dict], Completion]:
    """The expander chat client — same endpoint + credential, one provider,
    ``thinking: {"type": "disabled"}`` (structured {query, foods} JSON)."""
    return make_ark_chat_client(
        endpoint=expander_config.endpoint,
        model=expander_config.model_id,
        credential_env=expander_config.credential_env,
        thinking=expander_config.thinking,  # {"type": "disabled"}
        timeout_s=expander_config.timeout_s,
    )
