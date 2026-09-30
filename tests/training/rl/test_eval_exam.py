"""B2 exam eval — scripted-policy end to end, report math, vLLM client shape.

No network, no GPU. The gate runs for real, against the installed lab HEAD.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench import EXAM_SPLIT_PATH, load_split  # noqa: E402

import src.training.data_factory.rollout_fc as rollout_fc  # noqa: E402
from src.training.rl import eval_exam  # noqa: E402
from src.training.rl.exam_gate import ExamGateError, _git_output, _lab_root  # noqa: E402


class _AnyTel:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture(autouse=True)
def _lab_compatible_step_telemetry(monkeypatch):
    """The installed lab (47367d9) passes telemetry fields the pin-era
    ``rollout_fc._StepTel`` / ``_TaskTel`` lack; accept any fields so the FC
    loop runs on either lab. At the pin this only widens the classes."""
    monkeypatch.setattr(rollout_fc, "_StepTel", _AnyTel)
    monkeypatch.setattr(rollout_fc, "_TaskTel", _AnyTel)


@pytest.fixture(scope="module")
def lab_head() -> str:
    return _git_output(["git", "rev-parse", "HEAD"], cwd=_lab_root())


@pytest.fixture(scope="module")
def by_family():
    picked = {}
    for task in load_split(EXAM_SPLIT_PATH):
        picked.setdefault(task.family, task)
    return picked


def _call(name: str, args: dict, call_id: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _completion(tool_calls) -> dict:
    return {
        "content": None,
        "reasoning_content": "plan" if tool_calls else "",
        "tool_calls": tool_calls,
        "finish_reason": "tool_calls" if tool_calls else "stop",
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "reasoning_tokens": 0},
    }


class ExamScript:
    """Stateless scripted policy: script chosen by the task query, step by the
    number of assistant turns already in the request. Picklable (fork pool)."""

    def __init__(self, scripts: dict):
        self.scripts = scripts  # query -> list[tool_calls] | "silent" | "raise"

    def __call__(self, request: dict) -> dict:
        assert request["parallel_tool_calls"] is False
        query = next(m["content"] for m in request["messages"] if m["role"] == "user")
        script = self.scripts.get(query.removeprefix("Task:\n"), [])
        if script == "raise":
            raise RuntimeError(f"{eval_exam._TRANSPORT} connection refused")
        if script == "silent":
            return _completion([])
        step = sum(1 for m in request["messages"] if m["role"] == "assistant")
        if step < len(script):
            return _completion(script[step])
        return _completion([_call("finish", {}, f"f{step}")])


class Factory:
    def __init__(self, scripts: dict):
        self.scripts = scripts
        self.runs: list[int] = []

    def __call__(self, run: int) -> ExamScript:
        self.runs.append(run)
        return ExamScript(self.scripts)


def _scripts(by_family, *, raise_family: str | None = None) -> dict:
    log = by_family["log"]
    scripts = {
        log.query: [
            [_call("log_meal", {"food_id": r.food_id, "grams": r.grams, "eaten_at": r.eaten_at}, f"c{i}")]
            for i, r in enumerate(log.oracle.ledger_tail)
        ],
        by_family["update"].query: "silent",
        by_family["composite"].query: [[_call("get_profile", {}, "p0")]],
    }
    if raise_family:
        scripts[by_family[raise_family].query] = "raise"
    return scripts


def _run(out_dir, lab_head, by_family, scripts, **kw):
    params = dict(
        out_dir=out_dir,
        exam_path=None,
        expected_rev=lab_head,
        complete_factory=Factory(scripts),
        model="scripted-student",
        runs=2,
        sampling=eval_exam.sampling_preset(False),
        seed_base=0,
        concurrency=1,
        task_ids=[by_family[f].id for f in ("log", "update", "composite")],
    )
    params.update(kw)
    return eval_exam.run_eval(**params)


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #


def test_end_to_end_scripted_policy(tmp_path, lab_head, by_family):
    report = _run(tmp_path, lab_head, by_family, _scripts(by_family))
    rows = {(r["run"], r["family"]): r for r in eval_exam.load_rows(tmp_path)}
    assert len(rows) == 6

    for run in (1, 2):
        log, upd, comp = rows[(run, "log")], rows[(run, "update")], rows[(run, "composite")]
        assert log["status"] == "pass" and log["finished"] and log["execution"] == "ok"
        assert log["n_legal_tool_calls"] == log["n_steps"] == len(by_family["log"].oracle.ledger_tail) + 1
        assert upd["finished"] is False and upd["execution"] == "no_finish"
        assert upd["n_no_tool_call"] == upd["n_steps"] == 6  # update budget
        # No finish is still scored on the end state (the lab rule), not voided.
        assert upd["status"] in ("pass", "fail") and upd["error"] is None
        assert comp["status"] == "fail" and comp["score_tag"] and comp["finished"]
        assert log["tools_only_status"] == log["status"]

    expected = [sum(rows[(r, f)]["status"] == "pass" for f in ("log", "update", "composite")) / 3 for r in (1, 2)]
    assert report["exam"]["lab_head"] == lab_head
    assert report["exam"]["subset"] is True
    assert report["pass_at_1"]["per_run"] == pytest.approx(expected)
    assert report["pass_at_k"] == pytest.approx(expected[0])  # scripted: runs agree
    assert report["by_family"]["log"]["pass_at_1_mean"] == 1.0
    assert report["ticket_010_run1"]["with_reasoning"]["pass_at_1"] == pytest.approx(expected[0])
    health = report["protocol_health"]
    assert health["finish_rate"] == pytest.approx(4 / 6)
    assert health["no_tool_call_turns"] == 12
    assert (tmp_path / "report.md").read_text().startswith("# Exam eval")
    assert json.loads((tmp_path / "report.json").read_text())["k"] == 2


def test_process_pool_matches_sequential(tmp_path, lab_head, by_family):
    seq = _run(tmp_path / "seq", lab_head, by_family, _scripts(by_family))
    par = _run(tmp_path / "par", lab_head, by_family, _scripts(by_family), concurrency=2)
    assert par["per_task_pass_rate"] == seq["per_task_pass_rate"]
    assert par["status_counts"] == seq["status_counts"]


def test_gate_refuses_before_any_rollout(tmp_path, by_family):
    factory = Factory(_scripts(by_family))
    with pytest.raises(ExamGateError, match="eval aborts before any rollout"):
        _run(tmp_path, "0" * 40, by_family, {}, complete_factory=factory)
    assert factory.runs == []
    assert not (tmp_path / eval_exam.EPISODES).exists()


def test_transport_error_is_indeterminate_and_resume_reruns_it(tmp_path, lab_head, by_family):
    report = _run(tmp_path, lab_head, by_family, _scripts(by_family, raise_family="log"), runs=1)
    (log_row,) = [r for r in eval_exam.load_rows(tmp_path) if r["family"] == "log"]
    assert log_row["status"] == "indeterminate" and log_row["error_kind"] == "transport"
    first = eval_exam.load_rows(tmp_path)
    n_pass = sum(r["status"] == "pass" for r in first)
    assert report["pass_at_1"]["per_run"] == [pytest.approx(n_pass / 3)]  # indeterminate = not pass
    assert report["pass_at_1"]["pass_rate_excl_indeterminate"] == pytest.approx(n_pass / 2)

    with pytest.raises(SystemExit, match="--resume"):
        _run(tmp_path, lab_head, by_family, _scripts(by_family), runs=1)
    with pytest.raises(SystemExit, match="sampling"):
        _run(tmp_path, lab_head, by_family, _scripts(by_family), runs=1, resume=True,
             sampling=eval_exam.sampling_preset(True))

    factory = Factory(_scripts(by_family))
    report = _run(tmp_path, lab_head, by_family, {}, runs=1, resume=True, complete_factory=factory)
    assert factory.runs == [1]  # only the transport-failed episode re-ran
    assert len(eval_exam.load_rows(tmp_path)) == 4
    assert report["pass_at_1"]["per_run"] == [pytest.approx((n_pass + 1) / 3)]  # log now passes


# --------------------------------------------------------------------------- #
# report math (pure, synthetic rows)
# --------------------------------------------------------------------------- #


def _row(task_id, family, run, status, **kw):
    row = {
        "task_id": task_id, "family": family, "run": run, "status": status,
        "tools_only_status": status, "score_tag": None if status == "pass" else "WrongPlan",
        "execution": "ok", "finished": True, "error": None, "error_kind": None,
        "n_steps": 4, "n_legal_tool_calls": 4, "n_no_tool_call": 0, "n_unknown_tool": 0,
        "n_bad_args": 0, "n_multi_call": 0, "n_env_error": 0, "n_length_truncated": 0,
        "prompt_tokens": 1000, "completion_tokens": 100, "reasoning_tokens": 0,
        "reasoning_chars": 0, "latency_s": 2.0,
    }
    row.update(kw)
    return row


# task -> statuses over 3 runs
_GRID = {
    "a": ("log", ["pass", "pass", "pass"]),
    "b": ("log", ["pass", "fail", "pass"]),
    "c": ("composite", ["fail", "fail", "pass"]),
    "d": ("composite", ["fail", "fail", "fail"]),
}


def _manifest(grid, runs=3):
    return {"runs": runs, "task_ids": list(grid), "families": {t: f for t, (f, _) in grid.items()}}


def _rows(grid):
    return [_row(t, f, r, s) for t, (f, statuses) in grid.items() for r, s in enumerate(statuses, start=1)]


def test_aggregate_pass_mean_range_ci():
    out = eval_exam.aggregate(_manifest(_GRID), _rows(_GRID))
    p1 = out["pass_at_1"]
    assert p1["per_run"] == [0.5, 0.25, 0.75]
    assert p1["mean"] == pytest.approx(0.5)
    assert (p1["min"], p1["max"]) == (0.25, 0.75)
    assert out["pass_at_k"] == 0.75
    task_rates = [1.0, 2 / 3, 1 / 3, 0.0]
    assert out["per_task_pass_rate"] == pytest.approx(dict(zip("abcd", task_rates)))
    rng = np.random.default_rng(eval_exam.BOOTSTRAP_SEED)
    means = np.asarray(task_rates)[rng.integers(0, 4, size=(eval_exam.BOOTSTRAP_N, 4))].mean(axis=1)
    assert p1["ci95_bootstrap_tasks"] == pytest.approx(list(np.quantile(means, [0.025, 0.975])))
    assert p1["ci95_bootstrap_tasks"][0] <= p1["mean"] <= p1["ci95_bootstrap_tasks"][1]
    assert out["by_family"]["log"]["pass_at_1_mean"] == pytest.approx(5 / 6)
    assert out["by_family"]["composite"]["pass_at_1_per_run"] == [0.0, 0.0, 0.5]
    assert out["fail_tags"] == {"WrongPlan": 6}
    assert out["protocol_health"]["go_no_go_005"]["mixed_reward_group_rate"] == 0.5


def test_bootstrap_is_deterministic_and_degenerate_on_constant():
    assert eval_exam.bootstrap_mean_ci([0.2, 0.9, 0.4]) == eval_exam.bootstrap_mean_ci([0.2, 0.9, 0.4])
    assert eval_exam.bootstrap_mean_ci([1.0] * 5) == (1.0, 1.0)


def test_aggregate_refuses_incomplete_matrix():
    rows = _rows(_GRID)[:-1]
    with pytest.raises(SystemExit, match="incomplete"):
        eval_exam.aggregate(_manifest(_GRID), rows)


def test_indeterminate_counts_as_not_pass():
    grid = {"a": ("log", ["pass"]), "b": ("log", ["indeterminate"])}
    out = eval_exam.aggregate(_manifest(grid, runs=1), _rows(grid))
    assert out["pass_at_1"]["mean"] == 0.5
    assert out["pass_at_1"]["pass_rate_excl_indeterminate"] == 1.0


# --------------------------------------------------------------------------- #
# vLLM client (no network)
# --------------------------------------------------------------------------- #


def _client(**kw):
    return eval_exam.VLLMChatClient(
        base_url="http://localhost:8000/v1/", model="qwen", sampling=eval_exam.sampling_preset(False, **kw), seed=7
    )


def test_client_payload_is_serial_fc_with_seed_and_thinking_flag():
    request = {
        "messages": [
            {"role": "user", "content": "Task:\nx"},
            {"role": "assistant", "content": None, "reasoning_content": "why", "tool_calls": [{"id": "c"}]},
        ],
        "tools": [{"type": "function"}],
        "temperature": 0.0,  # the lab's default; the client's sampling wins
        "parallel_tool_calls": False,
    }
    payload = _client(temperature=0.6).payload(request)
    assert payload["parallel_tool_calls"] is False
    assert payload["seed"] == 7 and payload["temperature"] == 0.6
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["messages"][1]["reasoning"] == "why"
    assert "reasoning_content" not in payload["messages"][1]
    assert _client().url == "http://localhost:8000/v1/chat/completions"


def test_client_requires_network_flag(monkeypatch):
    monkeypatch.delenv("NUTRIMIND_ALLOW_NETWORK", raising=False)
    with pytest.raises(RuntimeError, match="NUTRIMIND_ALLOW_NETWORK=1"):
        _client()({"messages": [], "tools": []})


def test_completion_from_vllm_body_reads_reasoning_field():
    body = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {"content": None, "reasoning": "think", "tool_calls": [{"id": "c1"}]},
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "completion_tokens_details": None},
    }
    completion = eval_exam.completion_from_body(body)
    assert completion["reasoning_content"] == "think"
    assert completion["tool_calls"] == [{"id": "c1"}]
    assert completion["finish_reason"] == "tool_calls"
    assert completion["usage"] == {"prompt_tokens": 12, "completion_tokens": 3, "reasoning_tokens": 0}


def test_torn_last_line_is_ignored(tmp_path: pathlib.Path):
    (tmp_path / eval_exam.EPISODES).write_text('{"run": 1, "task_id": "a"}\n{"run": 1, "ta')
    assert eval_exam.load_rows(tmp_path) == [{"run": 1, "task_id": "a"}]


def test_client_http_roundtrip_against_loopback_stub(monkeypatch):
    """The real urllib path against an in-process stub (loopback only)."""
    import http.server
    import threading

    seen = {}

    class Stub(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # /v1/models
            self._send({"data": [{"id": "qwen"}]})

        def do_POST(self):
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send({
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "content": None, "reasoning": None,
                    "tool_calls": [{"id": "c", "type": "function",
                                    "function": {"name": "finish", "arguments": "{}"}}]}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            })

        def _send(self, payload):
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("NUTRIMIND_ALLOW_NETWORK", "1")
        base = f"http://127.0.0.1:{server.server_port}/v1"
        eval_exam.check_server(base, "qwen")
        with pytest.raises(SystemExit, match="not served"):
            eval_exam.check_server(base, "other")
        client = eval_exam.VLLMChatClient(
            base_url=base, model="qwen", sampling=eval_exam.sampling_preset(True), seed=3
        )
        completion = client({"messages": [{"role": "user", "content": "Task:\nx"}], "tools": [],
                             "parallel_tool_calls": False})
    finally:
        server.shutdown()
    assert completion["tool_calls"][0]["function"]["name"] == "finish"
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert seen["body"]["top_p"] == 0.95 and seen["body"]["seed"] == 3
