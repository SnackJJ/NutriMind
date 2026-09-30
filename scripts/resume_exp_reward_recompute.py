"""Offline recompute of the phase-1 reward on multi-step teacher trajectories.

Loads ``reward_v1`` / ``reward_v2`` from the commit immediately before ADR-009
(``ec41782``, parent of ``c678b24``) and scores each stored multi-step
trajectory against a one-tool shortcut built from the same record.

Does not modify ``src/training/grpo/reward.py``. The current tree's T2/T3
outcome reads ``expected_tools``; these records do not have that field, and
the pre-ADR-009 function is the one ADR-009's 0.93 figure refers to.
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
REWARD_COMMIT = "ec41782cfb2c1f9222643335766cc6db3e2fe7a6"
TRAJECTORY_PATH = REPO / "data/trajectories/sft_train_trajectory.jsonl"
OUT_PATH = REPO / "docs/research/resume-exp-20260928-reward-recompute.json"
PLACEHOLDER_ANSWER = "x" * 81

sys.path.insert(0, str(REPO))

from src.orchestrator.tool_parser import ToolParser  # noqa: E402
from src.training.grpo.environment import (  # noqa: E402
    RolloutStep,
    RolloutTrajectory,
    TaskMetadata,
    ToolExecutionResult,
)


def _load_historical_reward():
    source = subprocess.check_output(
        ["git", "show", f"{REWARD_COMMIT}:src/training/grpo/reward.py"],
        cwd=REPO,
    )
    module_name = "reward_hist_ec41782"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = f"{REWARD_COMMIT}:src/training/grpo/reward.py"
    sys.modules[module_name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _parse_tool_result(raw: str | None) -> tuple[dict | None, bool, str]:
    """Return (result, success, parse_status).

    Success matches ``NutriMindEnv._execute_tool``: ``status != "error"``.
    A missing tool message is not an execution; the caller records that
    separately as a format error.
    """
    if raw is None:
        return None, False, "missing"
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "unparsed", "raw_prefix": raw[:120]}, True, "unparsed"
    if not isinstance(result, dict):
        return {"status": "unparsed", "value_type": type(result).__name__}, True, "non_dict"
    status = result.get("status")
    return result, status != "error", "ok"


def _steps_from_messages(messages: list[dict], parser: ToolParser) -> tuple[list[dict], list[str]]:
    """Walk assistant turns. A tool turn consumes the following tool message."""
    notes: list[str] = []
    idx = 0
    while idx < len(messages) and messages[idx].get("role") != "user":
        idx += 1
    idx += 1
    steps: list[dict] = []
    while idx < len(messages):
        message = messages[idx]
        role = message.get("role")
        if role != "assistant":
            notes.append(f"skipped_{role}")
            idx += 1
            continue
        content = message.get("content") or ""
        parsed = parser.parse(content)
        if parsed.type == "tool_call":
            tool_raw = None
            if idx + 1 < len(messages) and messages[idx + 1].get("role") == "tool":
                tool_raw = messages[idx + 1].get("content")
                if not isinstance(tool_raw, str):
                    tool_raw = json.dumps(tool_raw, ensure_ascii=False)
                idx += 2
            else:
                notes.append("tool_call_without_tool_message")
                idx += 1
            result, success, parse_status = _parse_tool_result(tool_raw)
            steps.append(
                {
                    "kind": "tool_call",
                    "content": content,
                    "parsed": parsed,
                    "result": result,
                    "success": success,
                    "result_parse": parse_status,
                }
            )
        else:
            if parsed.type == "final_answer" and steps and any(
                s["kind"] == "tool_call" for s in steps
            ):
                # A non-tool assistant turn before the end would have
                # terminated a live rollout. Count it; still keep the turn.
                if idx + 1 < len(messages):
                    notes.append("intermediate_final_answer")
            steps.append(
                {
                    "kind": parsed.type,
                    "content": content,
                    "parsed": parsed,
                    "result": None,
                    "success": False,
                    "result_parse": None,
                }
            )
            idx += 1
    return steps, notes


def _to_trajectory(query: str, steps: list[dict], final_answer: str) -> RolloutTrajectory:
    rollout_steps: list[RolloutStep] = []
    tool_calls = 0
    for step_idx, step in enumerate(steps):
        parsed = step["parsed"]
        rollout = RolloutStep(
            step_idx=step_idx,
            model_output=step["content"],
            think_content=parsed.think,
            action_type=parsed.type,
        )
        if parsed.type == "tool_call" and step["result"] is not None and parsed.tool_call is not None:
            rollout.tool_execution = ToolExecutionResult(
                tool_name=parsed.tool_call.name,
                tool_args=parsed.tool_call.arguments,
                result=step["result"],
                success=step["success"],
            )
            tool_calls += 1
        rollout_steps.append(rollout)
    return RolloutTrajectory(
        prompt=query,
        steps=rollout_steps,
        final_answer=final_answer,
        terminated=True,
        termination_reason="final_answer",
        total_tool_calls=tool_calls,
    )


def _final_answer_text(steps: list[dict]) -> str | None:
    for step in reversed(steps):
        if step["kind"] == "final_answer":
            return step["parsed"].content or ""
    return None


def _shortcut_steps(steps: list[dict], answer_text: str) -> list[dict]:
    first = next(step for step in steps if step["kind"] == "tool_call")
    final = {
        "kind": "final_answer",
        "content": answer_text,
        "parsed": _Final.parse(answer_text),
        "result": None,
        "success": False,
        "result_parse": None,
    }
    return [first, final]


class _Final:
    """Minimal stand-in so a synthetic final step has ``.think`` and ``.type``."""

    def __init__(self, content: str):
        self.type = "final_answer"
        self.think = None
        self.content = content
        self.tool_call = None

    @staticmethod
    def parse(content: str) -> "_Final":
        return _Final(content)


def _score(reward_fn, trajectory: RolloutTrajectory, meta: TaskMetadata) -> float:
    return float(reward_fn(trajectory, meta).total)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _dist(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    def pct(q: float) -> float:
        pos = (len(ordered) - 1) * q
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)

    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "mean_ci95_normal": [
            mean - 1.96 * std / math.sqrt(len(values)),
            mean + 1.96 * std / math.sqrt(len(values)),
        ],
        "min": ordered[0],
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "max": ordered[-1],
    }


def _pair_summary(rows: list[dict], full_key: str, short_key: str) -> dict[str, Any]:
    n = len(rows)
    ge = sum(1 for row in rows if row[short_key] >= row[full_key])
    gt = sum(1 for row in rows if row[short_key] > row[full_key])
    eq = sum(1 for row in rows if row[short_key] == row[full_key])
    lt = n - ge
    deltas = [row[short_key] - row[full_key] for row in rows]
    both_093 = sum(
        1
        for row in rows
        if math.isclose(row[full_key], 0.93, abs_tol=1e-9)
        and math.isclose(row[short_key], 0.93, abs_tol=1e-9)
    )
    lo, hi = _wilson(ge, n)
    return {
        "n": n,
        "shortcut_ge_full": ge,
        "shortcut_ge_full_rate": ge / n,
        "shortcut_ge_full_wilson95": [lo, hi],
        "shortcut_gt_full": gt,
        "shortcut_eq_full": eq,
        "shortcut_lt_full": lt,
        "both_equal_0_93": both_093,
        "delta_shortcut_minus_full": _dist(deltas),
        "full_score": _dist([row[full_key] for row in rows]),
        "shortcut_score": _dist([row[short_key] for row in rows]),
    }


def _by_tier(rows: list[dict], full_key: str, short_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["tier"], []).append(row)
    return {tier: _pair_summary(group, full_key, short_key) for tier, group in sorted(grouped.items())}


def _assert_formula(reward_mod) -> float:
    """A long answer plus one valid tool call must hit the ADR-009 0.93 figure."""
    parser = ToolParser(validate_tool_name=False)
    call = (
        '<tool_call>\n{"name": "retrieve_knowledge", "arguments": {"query": "x"}}\n</tool_call>'
    )
    parsed = parser.parse(call)
    answer = "y" * 120
    step = {
        "kind": "tool_call",
        "content": call,
        "parsed": parsed,
        "result": {"status": "success", "data": {}},
        "success": True,
        "result_parse": "ok",
    }
    final = {
        "kind": "final_answer",
        "content": answer,
        "parsed": _Final(answer),
        "result": None,
        "success": False,
        "result_parse": None,
    }
    trajectory = _to_trajectory("synthetic", [step, final], answer)
    meta = TaskMetadata(query="synthetic", tier="T2", expected_tools=[], optimal_steps=3)
    score = _score(reward_mod.reward_v1, trajectory, meta)
    if not math.isclose(score, 0.93, abs_tol=1e-9):
        raise SystemExit(f"historical reward_v1 did not reproduce 0.93, got {score!r}")
    return score


def main() -> None:
    reward_mod = _load_historical_reward()
    formula_check = _assert_formula(reward_mod)
    parser = ToolParser(validate_tool_name=False)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()

    rows: list[dict] = []
    skipped = Counter()
    notes = Counter()
    result_parse = Counter()
    tool_names_first = Counter()
    n_read = 0
    v2_fields = Counter()

    with TRAJECTORY_PATH.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            record = json.loads(line)
            n_read += 1
            for field in ("oracle", "end_state", "expected_tools", "reward_semantics", "task_package"):
                if field in record:
                    v2_fields[field] += 1
            steps, step_notes = _steps_from_messages(record["messages"], parser)
            for note in step_notes:
                notes[note] += 1
            tool_steps = [step for step in steps if step["kind"] == "tool_call"]
            if len(tool_steps) < 2:
                skipped["fewer_than_2_tool_calls"] += 1
                continue
            answer = _final_answer_text(steps)
            if answer is None:
                skipped["no_final_answer"] += 1
                continue
            for step in tool_steps:
                result_parse[step["result_parse"]] += 1

            meta = TaskMetadata(
                query=record["query"],
                tier=str(record.get("tier") or ""),
                expected_tools=[],
                optimal_steps=len(tool_steps),
            )
            full = _to_trajectory(record["query"], steps, answer)
            shortcut = _to_trajectory(record["query"], _shortcut_steps(steps, answer), answer)
            placeholder = _to_trajectory(
                record["query"],
                _shortcut_steps(steps, PLACEHOLDER_ANSWER),
                PLACEHOLDER_ANSWER,
            )
            full_v1 = _score(reward_mod.reward_v1, full, meta)
            short_v1 = _score(reward_mod.reward_v1, shortcut, meta)
            hold_v1 = _score(reward_mod.reward_v1, placeholder, meta)
            full_v2 = _score(reward_mod.reward_v2, full, meta)
            short_v2 = _score(reward_mod.reward_v2, shortcut, meta)

            first = tool_steps[0]
            tool_names_first[first["parsed"].tool_call.name] += 1
            full_names = [step["parsed"].tool_call.name for step in tool_steps]
            invalid = [name for name in full_names if name not in {
                "get_food_nutrition", "log_meal", "get_today_summary",
                "get_history", "retrieve_knowledge", "set_goal",
            }]
            reason = "tie"
            if short_v1 != full_v1 or short_v2 != full_v2:
                if any(step["kind"] == "parse_error" for step in steps):
                    reason = "full_path_has_parse_error"
                elif invalid and first["parsed"].tool_call.name not in invalid:
                    reason = "later_tool_name_invalid"
                elif not first["success"] and any(step["success"] for step in tool_steps[1:]):
                    reason = "first_tool_failed_later_succeeded"
                else:
                    reason = "other"

            metadata = record.get("metadata") or {}
            rows.append(
                {
                    "line": line_no,
                    "tier": meta.tier,
                    "n_tool_calls": len(tool_steps),
                    "answer_chars": len(answer),
                    "first_tool": first["parsed"].tool_call.name,
                    "first_tool_success": bool(first["success"]),
                    "real_tool_executed": bool(metadata.get("real_tool_executed")),
                    "mock_tools_used": list(metadata.get("mock_tools_used") or []),
                    "full_v1": full_v1,
                    "shortcut_v1": short_v1,
                    "placeholder_v1": hold_v1,
                    "full_v2": full_v2,
                    "shortcut_v2": short_v2,
                    "reason_if_untied": reason,
                    "notes": step_notes,
                }
            )

    untied = [row for row in rows if row["reason_if_untied"] != "tie"]
    payload = {
        "git_head": head,
        "reward_source_commit": REWARD_COMMIT,
        "reward_source_note": (
            "Parent of c678b24 (ADR-009). reward_v1 total = "
            "0.30*r_format + 0.35*r_tool_selection + 0.35*r_outcome. "
            "T2/T3 r_outcome is 0.8 if len(final_answer)>80, 0.6 if 30-80, "
            "0.3 if <30. reward_v2 total equals reward_v1 unless the T1/T2/T3 "
            "hard gate fires (zero successful tool calls) or termination_reason "
            "is max_tokens. r_efficiency is hard-coded 0 and r_conditional is "
            "not added to the total."
        ),
        "formula_check_reward_v1_on_synthetic_long_answer": formula_check,
        "runs": 1,
        "deterministic": True,
        "trajectory_file": str(TRAJECTORY_PATH.relative_to(REPO)),
        "n_records_in_file": n_read,
        "n_multistep_scored": len(rows),
        "skipped": dict(skipped),
        "construction": {
            "multistep": "assistant turns parsed as tool_call >= 2, and a final answer exists",
            "shortcut": (
                "Keep the first tool_call turn and its stored tool message, "
                "then the original final answer (think blocks stripped by ToolParser). "
                "Later tool calls are dropped. The answer is not regenerated."
            ),
            "placeholder_sensitivity": (
                "Same first tool call, but final_answer is 81 'x' characters. "
                "Not the headline construction."
            ),
            "parser": "src.orchestrator.tool_parser.ToolParser(validate_tool_name=False), same as NutriMindEnv",
            "success": "tool JSON status != 'error', matching NutriMindEnv._execute_tool",
        },
        "v2_terminal_reward": {
            "definition": (
                "ADR-015 / data_factory RewardSemantics v2-r1: "
                "Scorer.score(end_state, oracle)['passed'] maps to 1.0 / 0.0, "
                "and missing execution or oracle yields null (indeterminate). "
                "See src/training/data_factory/concepts.py RewardSemantics and verify()."
            ),
            "records_with_end_state_or_oracle_or_expected_tools": dict(v2_fields),
            "n_scored": 0,
            "n_coverage": len(rows),
            "reason": (
                "None of the scored records carry end_state, oracle, "
                "reward_semantics, or expected_tools. They are phase-1 chat "
                "transcripts (get_food_nutrition / retrieve_knowledge / ...), "
                "not NutriEnv episodes. data/student/run_manifest.json is a "
                "dry run with accepted_traces=0, so there is no sibling set of "
                "Pass-filtered episodes for these same queries."
            ),
        },
        "structure_notes": dict(notes),
        "tool_result_parse_counts": dict(result_parse),
        "first_tool_name_counts": dict(tool_names_first),
        "reward_v1": _pair_summary(rows, "full_v1", "shortcut_v1"),
        "reward_v1_by_tier": _by_tier(rows, "full_v1", "shortcut_v1"),
        "reward_v1_placeholder_answer": _pair_summary(rows, "full_v1", "placeholder_v1"),
        "reward_v2_historical": _pair_summary(rows, "full_v2", "shortcut_v2"),
        "reward_v2_historical_by_tier": _by_tier(rows, "full_v2", "shortcut_v2"),
        "untied_reason_counts": dict(Counter(row["reason_if_untied"] for row in rows)),
        "untied_examples": [
            {
                "line": row["line"],
                "tier": row["tier"],
                "n_tool_calls": row["n_tool_calls"],
                "first_tool": row["first_tool"],
                "first_tool_success": row["first_tool_success"],
                "full_v1": row["full_v1"],
                "shortcut_v1": row["shortcut_v1"],
                "full_v2": row["full_v2"],
                "shortcut_v2": row["shortcut_v2"],
                "reason": row["reason_if_untied"],
                "notes": row["notes"],
                "answer_chars": row["answer_chars"],
            }
            for row in untied[:30]
        ],
        "answer_chars": _dist([row["answer_chars"] for row in rows]),
        "n_answer_le_80": sum(1 for row in rows if row["answer_chars"] <= 80),
        "n_answer_lt_30": sum(1 for row in rows if row["answer_chars"] < 30),
        "n_real_tool_executed": sum(1 for row in rows if row["real_tool_executed"]),
        "n_with_mock_tools": sum(1 for row in rows if row["mock_tools_used"]),
        "tool_call_count_hist": dict(Counter(row["n_tool_calls"] for row in rows)),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    v1 = payload["reward_v1"]
    print(
        f"n={v1['n']} reward_v1 shortcut>=full "
        f"{v1['shortcut_ge_full']}/{v1['n']} = {v1['shortcut_ge_full_rate']:.6f} "
        f"wilson95={v1['shortcut_ge_full_wilson95']}"
    )
    print("untied", payload["untied_reason_counts"])
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
