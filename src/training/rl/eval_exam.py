"""Zero-shot / checkpoint exam eval over k independent runs (B2 / B5 eval half).

``python -m src.training.rl.eval_exam --base-url http://localhost:8000/v1 --model <served>``

Every episode goes through the factory FC loop (``rollout_tool_call`` → lab
``run_episode_tool_call``, ``parallel_tool_calls`` false, ADR-014) with a
vLLM OpenAI-compatible completion injected. The exam gate runs before any
rollout and again when the report is built (ticket 005 / 010).

Scoring is the lab's: ``Scorer().score(end_state, task.oracle)`` on every
episode that did not die on an error; an errored episode is ``indeterminate``.
Pass@1 counts ``indeterminate`` as not-pass (denominator = every task).

Network (including a localhost vLLM) requires ``NUTRIMIND_ALLOW_NETWORK=1``.

Output (``--out-dir``): ``manifest.json``, ``episodes.jsonl`` (one row per
(run, task), appended as each finishes; ``--resume`` skips done rows and
re-runs rows that died on a transport error), ``report.json``, ``report.md``.

This is a stage module: nutrienv / the FC loop are imported at function level,
after the gate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence

import numpy as np

__all__ = [
    "VLLMChatClient",
    "bootstrap_mean_ci",
    "build_report",
    "episode_row",
    "load_rows",
    "main",
    "run_eval",
    "sampling_preset",
]

EPISODES = "episodes.jsonl"
MANIFEST = "manifest.json"
BOOTSTRAP_SEED = 0
BOOTSTRAP_N = 10_000

# Qwen3.5 model-card presets for text tasks (non-thinking is the 2B default).
_PRESETS = {
    False: {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "presence_penalty": 2.0, "max_tokens": 2048},
    True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 8192},
}
# Manifest keys that must match for --resume to append to an existing run.
_RESUME_KEYS = ("model", "sampling", "runs", "seed_base", "exam")
_TRANSPORT = "transport:"


# --------------------------------------------------------------------------- #
# policy client
# --------------------------------------------------------------------------- #


def sampling_preset(enable_thinking: bool, **overrides) -> dict:
    """Model-card preset for the thinking mode; non-None overrides win."""
    sampling = dict(_PRESETS[bool(enable_thinking)])
    sampling.update({k: v for k, v in overrides.items() if v is not None})
    sampling["enable_thinking"] = bool(enable_thinking)
    return sampling


class VLLMChatClient:
    """Minimal OpenAI-compatible chat client for a vLLM server.

    ``__call__(request) -> completion`` in the shape ``rollout_tool_call``
    expects (content, reasoning_content, tool_calls, finish_reason, usage).
    Only ``messages`` and ``tools`` come from the lab request; sampling is the
    client's. Picklable (worker processes each hold one).
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        sampling: dict,
        seed: int,
        timeout_s: float = 300.0,
        retries: int = 3,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.sampling = dict(sampling)
        self.seed = seed
        self.timeout_s = timeout_s
        self.retries = retries

    def payload(self, request: dict) -> dict:
        s = self.sampling
        return {
            "model": self.model,
            "messages": [_history_message(m) for m in request["messages"]],
            "tools": request["tools"],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": s["temperature"],
            "top_p": s["top_p"],
            "top_k": s["top_k"],
            "presence_penalty": s["presence_penalty"],
            "max_tokens": s["max_tokens"],
            "seed": self.seed,
            "chat_template_kwargs": {"enable_thinking": s["enable_thinking"]},
        }

    def __call__(self, request: dict) -> dict:
        if os.environ.get("NUTRIMIND_ALLOW_NETWORK") != "1":
            raise RuntimeError(
                "real network disabled: set NUTRIMIND_ALLOW_NETWORK=1 to call the "
                "policy server (a localhost vLLM counts)"
            )
        if request.get("parallel_tool_calls"):
            raise ValueError("parallel_tool_calls must be false (ADR-014)")
        data = _post_json(self.url, self.payload(request), self.timeout_s, self.retries)
        return completion_from_body(data)


def _history_message(message: dict) -> dict:
    """vLLM reads past-turn thinking from ``reasoning``; the lab loop stores it
    as ``reasoning_content``. Forward it so the Qwen chat template renders the
    history the way SFT's ``apply_chat_template`` does."""
    if message.get("role") != "assistant":
        return message
    reasoning = message.get("reasoning_content")
    out = {k: v for k, v in message.items() if k != "reasoning_content"}
    if isinstance(reasoning, str) and reasoning and "reasoning" not in out:
        out["reasoning"] = reasoning
    return out


def _post_json(url: str, body: dict, timeout_s: float, retries: int) -> dict:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last = "unreachable"
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                detail = exc.read().decode("utf-8", "replace")[:300]
                raise RuntimeError(f"http_{exc.code}: {detail}") from None
            last = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = type(exc).__name__
        if attempt < retries:
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{_TRANSPORT} policy server failed after {retries + 1} attempts ({last})")


def completion_from_body(data: dict) -> dict:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = message.get("reasoning")
    return {
        "content": message.get("content"),
        "reasoning_content": reasoning,
        "tool_calls": message.get("tool_calls") or [],
        "finish_reason": choice.get("finish_reason"),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens") or 0,
            "completion_tokens": usage.get("completion_tokens") or 0,
            "reasoning_tokens": details.get("reasoning_tokens") or 0,
        },
    }


def check_server(base_url: str, model: str) -> None:
    """Fail fast (before any rollout) if the server is down or lacks ``model``."""
    if os.environ.get("NUTRIMIND_ALLOW_NETWORK") != "1":
        raise SystemExit("set NUTRIMIND_ALLOW_NETWORK=1 (a localhost vLLM counts as network)")
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            served = [m.get("id") for m in json.loads(response.read()).get("data", [])]
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"policy server {url} unreachable: {type(exc).__name__}") from None
    if model not in served:
        raise SystemExit(f"model {model!r} not served at {base_url}; served: {served}")


# --------------------------------------------------------------------------- #
# one episode → one row
# --------------------------------------------------------------------------- #


def _error_kind(error: str | None) -> str | None:
    if error is None:
        return None
    if _TRANSPORT in error:
        return "transport"
    if "http_4" in error:
        return "http_4xx"
    return "other"


def episode_row(task, complete: Callable[[dict], dict], *, catalog, model: str, run: int) -> dict:
    """Run one FC episode of ``task`` and score it the lab's way."""
    from nutrienv.bench import Scorer
    from nutrienv.harness.runner import FINISH_OPS
    from nutrienv.harness.tools_schema import NUTRIENV_TOOLS

    from src.training.data_factory.rollout_fc import rollout_tool_call
    from src.training.data_factory.verify import derive_execution, observation_error_code
    from src.training.rl.eval_report import tools_only_episode

    finish_reasons: list = []

    def recording(request: dict) -> dict:
        completion = complete(request)
        finish_reasons.append(completion.get("finish_reason"))
        return completion

    episode = rollout_tool_call(task, teacher_complete=recording, catalog=catalog, model=model)

    if episode.error is not None:
        status = tools_only_status = "indeterminate"
        score_tag = None
    else:
        score = Scorer().score(episode.end_state, task.oracle)
        status = "pass" if score.get("passed") is True else "fail"
        score_tag = str(score.get("tag", "UNKNOWN"))
        # Ticket 010: reasoning stripped, tool_calls kept. Scoring reads the
        # end state only, so this equals ``status`` by construction; the
        # reasoning contrast that can move Pass is --enable-thinking vs not.
        stripped = tools_only_episode(episode)
        tools_only_status = (
            "pass" if Scorer().score(stripped.end_state, task.oracle).get("passed") is True else "fail"
        )

    tool_names = {tool["function"]["name"] for tool in NUTRIENV_TOOLS} | set(FINISH_OPS)
    counts = Counter()
    usage_totals = Counter()
    for turn in episode.turns:
        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
            usage_totals[key] += int((turn.usage or {}).get(key) or 0)
        calls = turn.tool_calls or []
        if not calls:
            counts["no_tool_call"] += 1
            continue
        if len(calls) > 1:
            counts["multi_call"] += 1
        func = calls[0].get("function") or {}
        raw = func.get("arguments")
        try:
            args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
        except (TypeError, ValueError):
            args = None
        name_ok = func.get("name") in tool_names
        args_ok = isinstance(args, dict)
        if not name_ok:
            counts["unknown_tool"] += 1
        if not args_ok:
            counts["bad_args"] += 1
        if name_ok and args_ok:
            counts["legal"] += 1
        if observation_error_code(turn.observation) is not None:
            counts["env_error"] += 1

    n_turns = len(episode.turns)
    return {
        "task_id": task.id,
        "family": task.family,
        "run": run,
        "status": status,
        "tools_only_status": tools_only_status,
        "score_tag": score_tag,
        "execution": derive_execution(episode),
        "finished": bool(episode.reached_finish),
        "error": episode.error,
        "error_kind": _error_kind(episode.error),
        "n_steps": n_turns,
        "n_legal_tool_calls": counts["legal"],
        "n_no_tool_call": counts["no_tool_call"],
        "n_unknown_tool": counts["unknown_tool"],
        "n_bad_args": counts["bad_args"],
        "n_multi_call": counts["multi_call"],
        "n_env_error": counts["env_error"],
        "n_length_truncated": sum(1 for r in finish_reasons if r == "length"),
        "prompt_tokens": usage_totals["prompt_tokens"],
        "completion_tokens": usage_totals["completion_tokens"],
        "reasoning_tokens": usage_totals["reasoning_tokens"],
        "reasoning_chars": sum(len(t.reasoning_content or "") for t in episode.turns),
        "latency_s": episode.latency_s,
    }


# --------------------------------------------------------------------------- #
# run loop (resumable, append-only JSONL)
# --------------------------------------------------------------------------- #

_WORKER: dict = {}


def _init_worker(exam_path: str, task_ids: list[str], complete_factory, model: str) -> None:
    from nutrienv.bench import load_split

    tasks = {t.id: t for t in load_split(exam_path)}
    _WORKER.update(
        tasks=tasks,
        catalog=next(iter(tasks.values())).s0.catalog,
        complete_factory=complete_factory,
        model=model,
    )
    missing = set(task_ids) - set(tasks)
    if missing:
        raise RuntimeError(f"worker exam lacks task ids: {sorted(missing)[:5]}")


def _worker_episode(run: int, task_id: str) -> dict:
    w = _WORKER
    return episode_row(
        w["tasks"][task_id],
        w["complete_factory"](run),
        catalog=w["catalog"],
        model=w["model"],
        run=run,
    )


def load_rows(out_dir: pathlib.Path) -> list[dict]:
    """Rows of ``episodes.jsonl``; a torn last line (crash mid-write) is skipped."""
    path = pathlib.Path(out_dir) / EPISODES
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _latest(rows: Sequence[dict]) -> dict[tuple[int, str], dict]:
    return {(row["run"], row["task_id"]): row for row in rows}


def _append_row(path: pathlib.Path, row: dict) -> None:
    line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _write_atomic(path: pathlib.Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _exam_identity(exam_path: pathlib.Path, expected_rev: str | None, task_ids: list[str]) -> dict:
    from src.training.rl.exam_gate import assert_lab_at_rev, exam_file_blob

    try:
        from nutrienv.harness.runner import LOOP_VERSION
    except ImportError:
        LOOP_VERSION = None
    return {
        "exam_path": str(exam_path),
        "exam_blob": exam_file_blob(exam_path),
        "lab_head": assert_lab_at_rev(expected_rev),
        "loop_version": LOOP_VERSION,
        "task_ids_sha256": hashlib.sha256("\n".join(sorted(task_ids)).encode()).hexdigest(),
    }


def run_eval(
    *,
    out_dir: pathlib.Path,
    exam_path: pathlib.Path | str | None,
    expected_rev: str | None,
    complete_factory: Callable[[int], Callable[[dict], dict]],
    model: str,
    runs: int,
    sampling: dict,
    seed_base: int,
    concurrency: int = 1,
    resume: bool = False,
    limit: int | None = None,
    task_ids: Sequence[str] | None = None,
    extra_manifest: dict | None = None,
) -> dict:
    """Gate, then run ``runs`` × tasks episodes, then build the report."""
    from src.training.rl.exam_gate import assert_exam_byte_identical

    # Ticket 005: nothing runs on a dirty exam or a lab off the expected rev.
    assert_exam_byte_identical(exam_path, expected_rev=expected_rev)

    from nutrienv.bench import EXAM_SPLIT_PATH, load_split

    exam_path = pathlib.Path(exam_path) if exam_path is not None else pathlib.Path(EXAM_SPLIT_PATH)
    tasks = load_split(exam_path)
    subset = task_ids is not None or limit is not None
    if task_ids is not None:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t.id in wanted]
    if limit is not None:
        tasks = tasks[:limit]
    task_ids = [t.id for t in tasks]

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": model,
        "sampling": sampling,
        "runs": runs,
        "seed_base": seed_base,
        "seeds": {str(r): seed_base + r for r in range(1, runs + 1)},
        "exam": {
            **_exam_identity(exam_path, expected_rev, task_ids),
            "limit": limit,
            "subset": subset,
        },
        "task_ids": task_ids,
        "families": {t.id: t.family for t in tasks},
        **(extra_manifest or {}),
    }
    manifest_path = out_dir / MANIFEST
    episodes_path = out_dir / EPISODES
    if manifest_path.exists() or episodes_path.exists():
        if not resume:
            raise SystemExit(f"{out_dir} already holds an eval; pass --resume or pick a new --out-dir")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        diff = [k for k in _RESUME_KEYS if old.get(k) != manifest.get(k)]
        if diff or old.get("task_ids") != task_ids:
            raise SystemExit(f"--resume refused: manifest differs on {diff or ['task_ids']}")
    _write_atomic(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    done = {
        key
        for key, row in _latest(load_rows(out_dir)).items()
        if row.get("error_kind") != "transport"
    }
    todo = [(r, tid) for r in range(1, runs + 1) for tid in task_ids if (r, tid) not in done]
    print(f"[eval_exam] {len(task_ids)} tasks x {runs} runs; {len(todo)} episodes to run", flush=True)

    started = time.monotonic()
    if concurrency <= 1:
        _init_worker(str(exam_path), task_ids, complete_factory, model)
        for index, (run, tid) in enumerate(todo, start=1):
            row = _worker_episode(run, tid)
            _append_row(episodes_path, row)
            _progress(index, len(todo), row, started)
    else:
        # Processes, not threads: rollout_tool_call patches lab module globals.
        ctx = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=concurrency,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(str(exam_path), task_ids, complete_factory, model),
        ) as pool:
            futures = [pool.submit(_worker_episode, run, tid) for run, tid in todo]
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                row = future.result()
                _append_row(episodes_path, row)
                _progress(index, len(todo), row, started)

    return build_report(out_dir, expected_rev=expected_rev)


def _progress(index: int, total: int, row: dict, started: float) -> None:
    elapsed = time.monotonic() - started
    print(
        f"[eval_exam] {index}/{total} run={row['run']} {row['task_id']} "
        f"{row['status']} ({row['score_tag']}) steps={row['n_steps']} {elapsed:.0f}s",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling ``values`` (tasks)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def _rate(num: float, den: float) -> float | None:
    return num / den if den else None


def per_task_matrix(manifest: dict, rows: Sequence[dict]) -> dict[str, list[dict]]:
    """task_id → rows ordered by run 1..k. Raises if any (run, task) is missing."""
    latest = _latest(rows)
    runs = manifest["runs"]
    missing = [
        (r, tid) for r in range(1, runs + 1) for tid in manifest["task_ids"] if (r, tid) not in latest
    ]
    if missing:
        raise SystemExit(
            f"eval incomplete: {len(missing)} of {runs * len(manifest['task_ids'])} episodes missing "
            f"(e.g. run {missing[0][0]} {missing[0][1]}); re-run with --resume"
        )
    return {
        tid: [latest[(r, tid)] for r in range(1, runs + 1)] for tid in manifest["task_ids"]
    }


def aggregate(manifest: dict, rows: Sequence[dict]) -> dict:
    """All report numbers from the manifest + episode rows (pure; no gate)."""
    from src.training.rl.eval_report import pass_at_k
    from src.training.sft.go_no_go import cold_start_go_no_go

    matrix = per_task_matrix(manifest, rows)
    k = manifest["runs"]
    task_ids = manifest["task_ids"]
    n = len(task_ids)
    flat = [row for tid in task_ids for row in matrix[tid]]

    per_run = [
        sum(matrix[tid][r]["status"] == "pass" for tid in task_ids) / n for r in range(k)
    ]
    per_task = {tid: sum(row["status"] == "pass" for row in matrix[tid]) / k for tid in task_ids}
    task_rates = [per_task[tid] for tid in task_ids]
    ci = bootstrap_mean_ci(task_rates)

    families: dict[str, list[str]] = defaultdict(list)
    for tid in task_ids:
        families[manifest["families"][tid]].append(tid)
    by_family = {
        fam: {
            "n_tasks": len(tids),
            "pass_at_1_mean": sum(per_task[t] for t in tids) / len(tids),
            "pass_at_1_per_run": [
                sum(matrix[t][r]["status"] == "pass" for t in tids) / len(tids) for r in range(k)
            ],
            "pass_at_k": sum(pass_at_k([row["status"] for row in matrix[t]], k) for t in tids)
            / len(tids),
        }
        for fam, tids in sorted(families.items())
    }

    def _passk(key: str) -> dict:
        at1 = [sum(matrix[t][r][key] == "pass" for t in task_ids) / n for r in range(k)]
        return {
            "pass_at_1_mean": sum(at1) / k,
            "pass_at_1_per_run": at1,
            "pass_at_k": sum(pass_at_k([row[key] for row in matrix[t]], k) for t in task_ids) / n,
        }

    total_turns = sum(row["n_steps"] for row in flat)
    health_rows = [
        {
            "family": row["family"],
            "schema_valid": row["n_steps"] > 0 and row["n_legal_tool_calls"] == row["n_steps"],
            "finished": row["finished"],
            "execution": row["execution"],
            "oracle_exec": "ok" if row["error"] is None else "error",
            "status": row["status"],
            "group_id": row["task_id"],
        }
        for row in flat
    ]
    statuses = Counter(row["status"] for row in flat)
    return {
        "n_tasks": n,
        "k": k,
        "n_episodes": len(flat),
        "status_counts": dict(statuses),
        "pass_at_1": {
            "per_run": per_run,
            "mean": sum(per_run) / k,
            "min": min(per_run),
            "max": max(per_run),
            "ci95_bootstrap_tasks": list(ci),
            "ci_method": f"percentile bootstrap over tasks of per-task pass rate, B={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}",
            "pass_rate_excl_indeterminate": _rate(statuses["pass"], statuses["pass"] + statuses["fail"]),
        },
        "pass_at_k": sum(pass_at_k([row["status"] for row in matrix[t]], k) for t in task_ids) / n,
        "by_family": by_family,
        "with_reasoning": _passk("status"),
        "tools_only": _passk("tools_only_status"),
        "protocol_health": {
            "legal_tool_call_rate": _rate(sum(r["n_legal_tool_calls"] for r in flat), total_turns),
            "finish_rate": _rate(sum(r["finished"] for r in flat), len(flat)),
            "no_finish_episodes": sum(not r["finished"] for r in flat),
            "no_tool_call_turns": sum(r["n_no_tool_call"] for r in flat),
            "unknown_tool_turns": sum(r["n_unknown_tool"] for r in flat),
            "bad_args_turns": sum(r["n_bad_args"] for r in flat),
            "multi_call_turns": sum(r["n_multi_call"] for r in flat),
            "env_error_turns": sum(r["n_env_error"] for r in flat),
            "length_truncated_turns": sum(r["n_length_truncated"] for r in flat),
            "execution_counts": dict(Counter(r["execution"] for r in flat)),
            "error_kinds": dict(Counter(r["error_kind"] for r in flat if r["error_kind"])),
            "mean_steps": total_turns / len(flat),
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in flat) / len(flat),
            "mean_completion_tokens": sum(r["completion_tokens"] for r in flat) / len(flat),
            "mean_latency_s": sum(r["latency_s"] or 0.0 for r in flat) / len(flat),
            "go_no_go_005": cold_start_go_no_go(health_rows, group_size=k),
        },
        "fail_tags": dict(Counter(r["score_tag"] for r in flat if r["status"] == "fail").most_common()),
        "per_task_pass_rate": per_task,
    }


def build_report(out_dir: pathlib.Path, *, expected_rev: str | None = None) -> dict:
    """Gate (ticket 010: a dirty exam never produces a number), then write the report."""
    from src.training.rl.eval_report import exam_report

    out_dir = pathlib.Path(out_dir)
    manifest = json.loads((out_dir / MANIFEST).read_text(encoding="utf-8"))
    exam = manifest["exam"]
    rows = load_rows(out_dir)
    report = aggregate(manifest, rows)
    matrix = per_task_matrix(manifest, rows)
    ticket_010 = exam_report(
        task_results=[
            {
                "statuses": [row["status"] for row in matrix[tid]],
                "tools_only_statuses": [row["tools_only_status"] for row in matrix[tid]],
            }
            for tid in manifest["task_ids"]
        ],
        k=manifest["runs"],
        exam_path=exam["exam_path"],
        expected_rev=expected_rev or exam["lab_head"],
    )
    report = {
        "exam": exam,
        "model": manifest["model"],
        "sampling": manifest["sampling"],
        "seeds": manifest["seeds"],
        "adapter": manifest.get("adapter"),
        **report,
        "ticket_010_run1": ticket_010,
    }
    _write_atomic(out_dir / "report.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_atomic(out_dir / "report.md", render_markdown(report))
    return report


def _pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def render_markdown(report: dict) -> str:
    exam, p1, health = report["exam"], report["pass_at_1"], report["protocol_health"]
    s = report["sampling"]
    lines = [
        f"# Exam eval — {report['model']}",
        "",
        f"- exam: `{exam['exam_path']}` blob `{exam['exam_blob']}`; lab HEAD `{exam['lab_head']}`; "
        f"loop `{exam['loop_version']}`" + ("; **task subset (smoke, not a result)**" if exam.get("subset") else ""),
        f"- tasks {report['n_tasks']} × runs {report['k']} = {report['n_episodes']} episodes; "
        f"status {report['status_counts']}",
        f"- sampling: thinking={s['enable_thinking']} T={s['temperature']} top_p={s['top_p']} "
        f"top_k={s['top_k']} presence={s['presence_penalty']} max_tokens={s['max_tokens']}; "
        f"seeds {report['seeds']}",
        "",
        "## Pass",
        "",
        "| metric | value |",
        "|---|---|",
        *[f"| Pass@1 run {i} | {_pct(v)} |" for i, v in enumerate(p1["per_run"], start=1)],
        f"| Pass@1 mean (min–max) | {_pct(p1['mean'])} ({_pct(p1['min'])}–{_pct(p1['max'])}) |",
        f"| 95% CI (bootstrap over tasks) | {_pct(p1['ci95_bootstrap_tasks'][0])}–{_pct(p1['ci95_bootstrap_tasks'][1])} |",
        f"| pass@{report['k']} | {_pct(report['pass_at_k'])} |",
        f"| Pass excl. indeterminate | {_pct(p1['pass_rate_excl_indeterminate'])} |",
        f"| with-reasoning Pass@1 mean / pass@k | {_pct(report['with_reasoning']['pass_at_1_mean'])} / {_pct(report['with_reasoning']['pass_at_k'])} |",
        f"| tools-only Pass@1 mean / pass@k | {_pct(report['tools_only']['pass_at_1_mean'])} / {_pct(report['tools_only']['pass_at_k'])} |",
        "",
        "Indeterminate episodes count as not-pass. Tools-only equals with-reasoning under end-state scoring.",
        "",
        "## By family",
        "",
        "| family | n | Pass@1 mean | per run | pass@k |",
        "|---|---|---|---|---|",
        *[
            f"| {fam} | {row['n_tasks']} | {_pct(row['pass_at_1_mean'])} | "
            f"{' / '.join(_pct(v) for v in row['pass_at_1_per_run'])} | {_pct(row['pass_at_k'])} |"
            for fam, row in report["by_family"].items()
        ],
        "",
        "## Protocol health",
        "",
        "| metric | value |",
        "|---|---|",
        f"| legal tool-call rate (per turn) | {_pct(health['legal_tool_call_rate'])} |",
        f"| finish rate | {_pct(health['finish_rate'])} |",
        f"| no-finish episodes | {health['no_finish_episodes']} |",
        f"| no-tool-call / unknown-tool / bad-args / multi-call turns | "
        f"{health['no_tool_call_turns']} / {health['unknown_tool_turns']} / {health['bad_args_turns']} / {health['multi_call_turns']} |",
        f"| env-refused turns | {health['env_error_turns']} |",
        f"| length-truncated turns | {health['length_truncated_turns']} |",
        f"| execution axis | {health['execution_counts']} |",
        f"| errors | {health['error_kinds']} |",
        f"| mean steps / prompt tok / completion tok / latency | {health['mean_steps']:.1f} / "
        f"{health['mean_prompt_tokens']:.0f} / {health['mean_completion_tokens']:.0f} / {health['mean_latency_s']:.1f}s |",
        f"| mixed-reward group rate (005) | {_pct(health['go_no_go_005']['mixed_reward_group_rate'])} |",
        "",
        "## Fail tags",
        "",
        "| tag | episodes |",
        "|---|---|",
        *[f"| {tag} | {count} |" for tag, count in report["fail_tags"].items()],
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


class _ClientFactory:
    """run → VLLMChatClient with that run's seed (picklable for workers)."""

    def __init__(self, *, seed_base: int, **client_kwargs):
        self.seed_base = seed_base
        self.client_kwargs = client_kwargs

    def __call__(self, run: int) -> VLLMChatClient:
        return VLLMChatClient(**self.client_kwargs, seed=self.seed_base + run)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--exam-path", default=None, help="default: nutrienv EXAM_SPLIT_PATH of the importable lab")
    p.add_argument("--expected-rev", default=None, help="full 40-hex lab SHA; default: config nutrienv_rev")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True, help="served model name (vLLM --served-model-name or LoRA name)")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--presence-penalty", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    thinking = p.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", dest="enable_thinking", action="store_true",
                          help="Qwen chat_template_kwargs.enable_thinking=true")
    thinking.add_argument("--no-thinking", dest="enable_thinking", action="store_false",
                          help="enable_thinking=false (default; Qwen3.5-2B's own default)")
    p.set_defaults(enable_thinking=False)
    p.add_argument("--seed-base", type=int, default=0, help="run r uses seed seed_base + r")
    p.add_argument("--concurrency", type=int, default=16, help="parallel episodes (worker processes)")
    p.add_argument("--request-timeout", type=float, default=300.0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="first N exam tasks (smoke only)")
    p.add_argument("--adapter", default=None, help="recorded in the manifest (LoRA path / merged dir)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    sampling = sampling_preset(
        args.enable_thinking,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_tokens,
    )
    check_server(args.base_url, args.model)
    factory = _ClientFactory(
        base_url=args.base_url,
        model=args.model,
        sampling=sampling,
        seed_base=args.seed_base,
        timeout_s=args.request_timeout,
    )
    report = run_eval(
        out_dir=pathlib.Path(args.out_dir),
        exam_path=args.exam_path,
        expected_rev=args.expected_rev,
        complete_factory=factory,
        model=args.model,
        runs=args.runs,
        sampling=sampling,
        seed_base=args.seed_base,
        concurrency=args.concurrency,
        resume=args.resume,
        limit=args.limit,
        extra_manifest={"adapter": args.adapter, "base_url": args.base_url},
    )
    p1 = report["pass_at_1"]
    print(
        f"[eval_exam] Pass@1 mean {p1['mean']:.3f} (runs {p1['per_run']}), "
        f"CI95 {p1['ci95_bootstrap_tasks']}, pass@{report['k']} {report['pass_at_k']:.3f}"
    )
    print(f"[eval_exam] report: {pathlib.Path(args.out_dir) / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
