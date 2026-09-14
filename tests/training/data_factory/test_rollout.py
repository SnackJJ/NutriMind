"""Ticket 009 — instrumented teacher rollouts (harness subclass + chat clients).

Offline: episodes run through the real NutriEnv with a scripted
``teacher_complete``; the production client is tested against a fake
``urlopen`` (request capture, retry behavior, the network guard) — never a
real endpoint. The live smoke test sits behind ``NUTRIMIND_ALLOW_NETWORK=1``
and ``COMMANDCODE_API_KEY`` and skips otherwise.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.harness.react import react_manual  # noqa: E402

from src.training.data_factory import rollout as ro  # noqa: E402
from src.training.data_factory import verify as vf  # noqa: E402
from src.training.data_factory.config import load_config  # noqa: E402
from src.training.data_factory.materialize import RunContext  # noqa: E402
from src.training.data_factory.rollout import (  # noqa: E402
    ScriptedTeacher,
    TeacherReActHarness,
    make_ark_expander_client,
    make_ark_teacher_client,
    rollout,
)

from tests.training.data_factory import _fixtures as fx  # noqa: E402

CONFIG_PATH = "configs/data_factory.yaml"


def pkg_for(catalog, task, seed=30):
    from src.training.data_factory import materialize as mz

    return mz.materialize(
        task,
        RunContext(
            catalog=catalog,
            catalog_sha=mz.catalog_digest(catalog),
            nutrienv_rev="203d807b",
            nutrimind_rev="d" * 40,
            config_sha="0" * 64,
            seed=seed,
            built_at="2026-09-09T12:00:00+00:00",
        ),
    )


def log_script(task, *, grams_scale=1.0):
    """(content, reasoning) turns for a log episode: the tail rows, then done."""
    turns = []
    for row in task.oracle.ledger_tail:
        action = {
            "op": "log_meal",
            "food_id": row.food_id,
            "grams": round(row.grams * grams_scale, 2),
            "eaten_at": row.eaten_at,
        }
        turns.append((json.dumps(action), "I should log my lunch."))
    turns.append(('{"op": "done"}', "All logged, finishing."))
    return turns


@pytest.fixture(scope="module")
def catalog():
    return fx.gold_catalog()


@pytest.fixture(scope="module")
def log_task(catalog):
    return fx.make_log_task(catalog, fx.first_person(), seed=30)


# --------------------------------------------------------------------------- #
# (a) the harness: protocol reuse + instrumentation
# --------------------------------------------------------------------------- #


def test_harness_is_v2_full_context_manual():
    harness = TeacherReActHarness(teacher_complete=ScriptedTeacher([]))
    assert harness.version == "v2"
    assert harness.context_limit is None  # full ReAct log (spec §2.1)
    assert harness.messages[0]["content"] == react_manual("v2")


def test_scripted_pass_episode_deterministic(catalog, log_task):
    teacher = ScriptedTeacher(log_script(log_task))
    harness = TeacherReActHarness(teacher_complete=teacher)
    first = rollout(harness, log_task)

    assert first.reached_finish is True
    assert first.error is None
    assert len(first.turns) == len(log_task.oracle.ledger_tail) + 1
    assert isinstance(first.end_state, object) and first.end_state is not None
    # every TurnMeta carries the two load-bearing fields (spec §12)
    for turn in first.turns:
        assert turn.raw_action_text and turn.executed_op is not None
        assert turn.parse_status == "ok"
        assert turn.fallback_used is False
    # observation recorded for env-stepped turns, None on the finish turn
    assert all(t.observation for t in first.turns[:-1])
    assert first.turns[-1].observation is None
    # completion payload kept on the turn: reasoning separate from content
    assert first.turns[0].reasoning_content == "I should log my lunch."
    assert first.turns[0].usage["reasoning_tokens"] == 20

    # deterministic: a fresh harness + same script → identical episodes
    # (latency excluded — wall-clock, not protocol)
    second = rollout(
        TeacherReActHarness(teacher_complete=ScriptedTeacher(log_script(log_task))),
        log_task,
    )
    a, b = first.to_dict(), second.to_dict()
    a.pop("latency_s"), b.pop("latency_s")
    assert a == b


def test_scripted_fail_episode(catalog, log_task):
    script = log_script(log_task, grams_scale=1.5)  # beyond the ±15 % tolerance
    episode = rollout(
        TeacherReActHarness(teacher_complete=ScriptedTeacher(script)), log_task
    )
    assert episode.reached_finish is True and episode.error is None
    result = vf.verify(pkg_for(catalog, log_task), episode)
    assert result.status == "fail"
    assert result.failure_codes == ["task_fail", "log_miss"]


def test_scripted_no_finish_episode(catalog, log_task):
    script = [('{"op": "get_profile"}', "reading my profile.")] * 12  # budget cap
    episode = rollout(
        TeacherReActHarness(teacher_complete=ScriptedTeacher(script)), log_task
    )
    assert episode.reached_finish is False
    assert episode.error is None
    result = vf.verify(pkg_for(catalog, log_task), episode)
    assert result.status == "indeterminate"
    assert result.failure_codes == ["teacher_no_finish"]


def test_scripted_invalid_op_episode_records_fallback(catalog, log_task):
    """Garbage assistant text → the base harness substitutes get_profile; the
    v2 metadata must flag it (fallback_used=True) — and verify turns that into
    teacher_invalid_op."""
    script = [
        ("I would maybe log the beans? (no json)", "unsure…"),
        *log_script(log_task)[1:],
    ]
    episode = rollout(
        TeacherReActHarness(teacher_complete=ScriptedTeacher(script)), log_task
    )
    turn0 = episode.turns[0]
    assert turn0.executed_op == {"op": "get_profile"}  # the substituted fallback
    assert turn0.fallback_used is True
    assert turn0.fallback_reason == "parse:no_json"
    assert turn0.parse_status == "no_json"
    result = vf.verify(pkg_for(catalog, log_task), episode)
    assert result.status == "indeterminate"
    assert result.failure_codes == ["teacher_invalid_op"]


def test_teacher_exception_is_episode_error(catalog, log_task):
    def exploding(request):
        raise RuntimeError("api down")

    episode = rollout(
        TeacherReActHarness(teacher_complete=exploding), log_task
    )
    assert episode.reached_finish is False
    assert episode.error and episode.error.startswith("teacher error")
    result = vf.verify(pkg_for(catalog, log_task), episode)
    assert result.status == "indeterminate"
    assert result.failure_codes == ["teacher_error"]


def test_exhausted_script_is_teacher_error(catalog, log_task):
    episode = rollout(
        TeacherReActHarness(teacher_complete=ScriptedTeacher([])), log_task
    )
    assert episode.error and episode.error.startswith("teacher error")


def test_harness_reuses_base_protocol(catalog, log_task):
    """Message assembly, the step-budget line, and the 6000-char observation
    cap come from the base class (v2 only instruments)."""
    teacher = ScriptedTeacher(log_script(log_task))
    harness = TeacherReActHarness(teacher_complete=teacher)
    rollout(harness, log_task)
    messages = harness.messages
    assert messages[0]["role"] == "system"
    assert any("Task:\n" in m["content"] for m in messages if m["role"] == "user")
    assert any("Step budget:" in m["content"] for m in messages if m["role"] == "user")
    # context_limit=None: the full log is sent, nothing slides
    assert teacher.requests, "no requests captured"


def test_extra_body_flows_into_request(log_task):
    """The thinking length control (and any retry-temperature override) ride
    the base extra_body into the request."""
    teacher = ScriptedTeacher(log_script(log_task))
    harness = TeacherReActHarness(
        teacher_complete=teacher,
        extra_body={"thinking": {"type": "enabled"}, "temperature": 0.7},
    )
    rollout(harness, log_task)
    assert all(
        r["thinking"] == {"type": "enabled"} and r["temperature"] == 0.7
        for r in teacher.requests
    )


# --------------------------------------------------------------------------- #
# (c) the ark clients
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture()
def captured(monkeypatch):
    box = {"calls": 0, "requests": []}

    def fake_urlopen(req, timeout=None):
        box["calls"] += 1
        box["requests"].append(
            {
                "url": req.full_url,
                "headers": dict(req.headers),
                "body": json.loads(req.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"op": "done"}',
                            "reasoning_content": "the user finished, I am done",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 123,
                    "completion_tokens": 45,
                    "completion_tokens_details": {"reasoning_tokens": 67},
                },
            }
        )

    monkeypatch.setattr(ro.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("NUTRIMIND_ALLOW_NETWORK", "1")
    monkeypatch.setenv("COMMANDCODE_API_KEY", "sk-test-key-never-appear")
    monkeypatch.setenv("ARK_API_KEY", "sk-test-key-never-appear")
    monkeypatch.setenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")
    return box


@pytest.fixture()
def config():
    return load_config(CONFIG_PATH)


def test_teacher_client_keeps_content_and_reasoning_separate(captured, config):
    client = make_ark_teacher_client(config.teacher)
    completion = client(
        {
            "model": config.teacher.model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.0,
        }
    )
    # separate fields, not concatenated (nutri-env's complete_chat collapses)
    assert completion["content"] == '{"op": "done"}'
    assert completion["reasoning_content"] == "the user finished, I am done"
    assert completion["finish_reason"] == "stop"
    assert completion["usage"] == {
        "prompt_tokens": 123,
        "completion_tokens": 45,
        "reasoning_tokens": 67,
    }


def test_client_request_shape_from_config(captured, config):
    client = make_ark_teacher_client(config.teacher)
    client(
        {
            "model": config.teacher.model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.0,
        }
    )
    (request,) = captured["requests"]
    assert request["url"] == config.teacher.endpoint
    assert request["body"]["model"] == "deepseek/deepseek-v4.1-flash"
    assert request["body"]["thinking"] == {"type": "enabled"}  # length control
    assert request["body"]["temperature"] == 0.0
    assert request["timeout"] == config.teacher.per_turn_timeout_s
    headers = {k.lower(): v for k, v in request["headers"].items()}
    assert headers["authorization"] == "Bearer sk-test-key-never-appear"
    assert headers["user-agent"] == ro._HTTP_USER_AGENT


def test_client_maps_reasoning_field_and_forwards_tools(monkeypatch, config):
    box = {"body": None}

    def fake_urlopen(req, timeout=None):
        box["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning": "search oats",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "search_foods",
                                        "arguments": '{"q": "oats"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "completion_tokens_details": {"reasoning_tokens": 4},
                },
            }
        )

    monkeypatch.setattr(ro.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("NUTRIMIND_ALLOW_NETWORK", "1")
    monkeypatch.setenv(config.teacher.credential_env, "sk-x")
    client = make_ark_teacher_client(config.teacher)
    tools = [{"type": "function", "function": {"name": "search_foods"}}]
    completion = client(
        {
            "model": config.teacher.model_id,
            "messages": [{"role": "user", "content": "search oats"}],
            "tools": tools,
            "parallel_tool_calls": False,
        }
    )
    assert box["body"]["model"] == "deepseek/deepseek-v4.1-flash"
    assert box["body"]["tools"] == tools
    assert box["body"]["parallel_tool_calls"] is False
    assert completion["content"] == ""
    assert completion["reasoning_content"] == "search oats"
    assert completion["tool_calls"][0]["function"]["name"] == "search_foods"
    assert completion["finish_reason"] == "tool_calls"


def test_endpoint_model_strips_ark_prefix_only():
    assert ro._endpoint_model("ark/deepseek-v4-flash") == "deepseek-v4-flash"
    assert ro._endpoint_model("deepseek/deepseek-v4.1-flash") == (
        "deepseek/deepseek-v4.1-flash"
    )
    assert ro._endpoint_model("deepseek-v4.1-flash") == "deepseek-v4.1-flash"


def test_endpoint_resolution_variants(monkeypatch):
    resolve = ro._resolve_endpoint
    path = "api/plan/v3/chat/completions"
    expected = "https://ark.cn-beijing.volces.com/api/plan/v3/chat/completions"

    # base carries host + api/plan/v3 → no duplicated segments
    monkeypatch.setenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")
    assert resolve(path) == expected
    # host-only base → the full path is appended
    monkeypatch.setenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com")
    assert resolve(path) == expected
    # base already the full completions URL → unchanged
    monkeypatch.setenv("ARK_BASE_URL", expected)
    assert resolve(path) == expected
    # unset base → the documented default plan base
    monkeypatch.delenv("ARK_BASE_URL", raising=False)
    assert resolve(path) == expected
    # absolute endpoint wins outright
    assert resolve(expected) == expected


def test_expander_client_sends_thinking_disabled(captured, config):
    client = make_ark_expander_client(config.expander)
    client(
        {
            "model": config.expander.model_id,
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    (request,) = captured["requests"]
    assert request["url"] == config.expander.endpoint
    assert request["body"]["model"] == "deepseek/deepseek-v4.1-flash"
    assert request["body"]["thinking"] == {"type": "disabled"}
    assert request["timeout"] == config.expander.timeout_s


def test_client_retries_transport_errors(captured, config, monkeypatch):
    attempts = {"n": 0}

    def flaky(req, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.HTTPError(
                req.full_url, 429, "Too Many Requests", {}, io.BytesIO(b"{}")
            )
        return _FakeResponse(
            {"choices": [{"message": {"content": "ok", "reasoning_content": "r"}}],
             "usage": {}}
        )

    monkeypatch.setattr(ro.urllib.request, "urlopen", flaky)
    client = make_ark_teacher_client(config.teacher)
    completion = client(
        {"model": config.teacher.model_id,
         "messages": [{"role": "user", "content": "hi"}], "temperature": 0.0}
    )
    assert attempts["n"] == 3
    assert completion["content"] == "ok"


def test_client_non_retryable_status_raises(captured, config, monkeypatch):
    def forbidden(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"{}")
        )

    monkeypatch.setattr(ro.urllib.request, "urlopen", forbidden)
    client = make_ark_teacher_client(config.teacher)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        client(
            {"model": config.teacher.model_id,
             "messages": [{"role": "user", "content": "hi"}]}
        )


def test_network_disabled_without_guard(monkeypatch, config):
    monkeypatch.delenv("NUTRIMIND_ALLOW_NETWORK", raising=False)
    monkeypatch.setenv(config.teacher.credential_env, "sk-x")

    def must_not_call(req, timeout=None):
        raise AssertionError("real network call attempted without the guard")

    monkeypatch.setattr(ro.urllib.request, "urlopen", must_not_call)
    client = make_ark_teacher_client(config.teacher)
    with pytest.raises(RuntimeError, match="NUTRIMIND_ALLOW_NETWORK"):
        client(
            {"model": config.teacher.model_id,
             "messages": [{"role": "user", "content": "hi"}]}
        )


def test_missing_credential_raises(captured, monkeypatch, config):
    monkeypatch.delenv(config.teacher.credential_env, raising=False)
    client = make_ark_teacher_client(config.teacher)
    with pytest.raises(RuntimeError, match=f"{config.teacher.credential_env} is not set"):
        client(
            {"model": config.teacher.model_id,
             "messages": [{"role": "user", "content": "hi"}]}
        )


def test_api_key_never_in_episodes_or_cache(captured, config, log_task):
    """A full production-shaped rollout: the credential must never surface in
    the episode, its turns, or the serialized episode cache."""
    client = make_ark_teacher_client(config.teacher)
    teacher = ScriptedTeacher(log_script(log_task))  # script drives; client unused
    harness = TeacherReActHarness(teacher_complete=teacher)
    episode = rollout(harness, log_task)
    blob = json.dumps(episode.to_dict(), default=str)
    assert "sk-test-key-never-appear" not in blob
    # the client itself returns no credential either
    completion = client(
        {"model": config.teacher.model_id,
         "messages": [{"role": "user", "content": "hi"}], "temperature": 0.0}
    )
    assert "sk-test-key-never-appear" not in json.dumps(completion)


# --------------------------------------------------------------------------- #
# live smoke (local only, behind the guard)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    __import__("os").environ.get("NUTRIMIND_ALLOW_NETWORK") != "1"
    or not __import__("os").environ.get("COMMANDCODE_API_KEY"),
    reason="live call: set NUTRIMIND_ALLOW_NETWORK=1 and COMMANDCODE_API_KEY",
)
def test_live_teacher_smoke_reasoning_content():
    config = load_config(CONFIG_PATH)
    client = make_ark_teacher_client(config.teacher)
    completion = client(
        {
            "model": config.teacher.model_id,
            "messages": [
                {"role": "user", "content": "Reply with the single word: hello"}
            ],
            "temperature": 0.0,
        }
    )
    assert completion["content"]
    assert completion["reasoning_content"]  # thinking enabled → non-empty
    assert completion["usage"]["reasoning_tokens"] is not None
