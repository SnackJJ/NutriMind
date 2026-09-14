"""RL ticket 010 — tools-only strip, pass@1/pass@k, dirty exam produces no number."""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench import EXAM_SPLIT_PATH  # noqa: E402

from src.training.data_factory.concepts import EpisodeResult, TurnMeta  # noqa: E402
from src.training.rl.exam_gate import ExamGateError, pinned_exam_blob  # noqa: E402
from src.training.rl.eval_report import exam_report, pass_at_k, tools_only_episode  # noqa: E402


def test_tools_only_strips_reasoning_keeps_tool_calls():
    calls = [{"id": "c1", "function": {"name": "log_meal", "arguments": "{}"}}]
    episode = EpisodeResult(
        turns=[
            TurnMeta(reasoning_content="plan text", tool_calls=calls, observation="{}"),
            TurnMeta(reasoning_content="done", tool_calls=[{"id": "c2"}], observation=None),
        ]
    )
    stripped = tools_only_episode(episode)
    assert stripped.turns[0].reasoning_content is None
    assert stripped.turns[1].reasoning_content is None
    assert stripped.turns[0].tool_calls == calls
    assert stripped.turns[1].tool_calls == [{"id": "c2"}]
    assert episode.turns[0].reasoning_content == "plan text"


def test_pass_at_1_and_pass_at_k():
    statuses = ["fail", "pass", "fail"]
    assert pass_at_k(statuses, 1) == 0.0
    assert pass_at_k(statuses, 2) == 1.0
    assert pass_at_k(statuses, 3) == 1.0


def test_report_carries_exam_revision_and_both_numbers():
    report = exam_report(
        task_results=[
            {"statuses": ["pass", "fail"], "tools_only_statuses": ["fail", "fail"]},
            {"statuses": ["fail", "pass"], "tools_only_statuses": ["fail", "pass"]},
        ],
        k=2,
    )
    assert report["exam_revision"] == pinned_exam_blob()
    assert "pass_at_1" in report["with_reasoning"]
    assert "pass_at_k" in report["with_reasoning"]
    assert "pass_at_1" in report["tools_only"]
    assert "pass_at_k" in report["tools_only"]
    assert report["with_reasoning"]["pass_at_1"] == 0.5
    assert report["with_reasoning"]["pass_at_k"] == 1.0
    assert report["tools_only"]["pass_at_1"] == 0.0
    assert report["tools_only"]["pass_at_k"] == 0.5


def test_dirty_exam_never_produces_a_number(tmp_path: pathlib.Path):
    mutated = tmp_path / "exam.json"
    mutated.write_bytes(pathlib.Path(EXAM_SPLIT_PATH).read_bytes() + b"\n")
    with pytest.raises(ExamGateError, match="eval aborts before any rollout"):
        exam_report(
            task_results=[{"statuses": ["pass"]}],
            k=1,
            exam_path=mutated,
        )
