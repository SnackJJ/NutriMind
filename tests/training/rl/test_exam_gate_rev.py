"""Exam gate checks the pin HEAD and its clean tree before any blob check."""

from __future__ import annotations

import pathlib

import pytest

from src.training.data_factory.config import load_config
from src.training.rl import exam_gate
from src.training.rl.exam_gate import (
    ExamGateError,
    assert_exam_byte_identical,
    assert_lab_at_rev,
    before_eval_rollout,
)

PIN = "0" * 40
DRIFT = "4" * 40
LAB = pathlib.Path("/fake/nutri-env-lab")


@pytest.fixture
def fake_lab(monkeypatch):
    """Fake pin at HEAD ``state['head']`` with ``state['dirty']`` output; records blob calls."""
    state = {"head": PIN, "dirty": "", "blob_calls": []}
    monkeypatch.setattr(exam_gate, "_lab_root", lambda: LAB)

    def git_output(args, *, cwd=None):
        assert cwd == LAB
        if args == ["git", "rev-parse", "HEAD"]:
            return state["head"]
        if args == ["git", "status", "--porcelain", "-uno"]:
            return state["dirty"]
        raise AssertionError(f"unexpected git call {args}")

    monkeypatch.setattr(exam_gate, "_git_output", git_output)
    monkeypatch.setattr(
        exam_gate, "exam_file_blob", lambda path: state["blob_calls"].append(path) or "b"
    )
    monkeypatch.setattr(exam_gate, "pinned_exam_blob", lambda: "b")
    return state


def test_head_at_rev_passes(fake_lab):
    assert assert_lab_at_rev(PIN) == PIN
    assert_exam_byte_identical("exam.json", expected_rev=PIN)
    assert fake_lab["blob_calls"] == [pathlib.Path("exam.json")]


def test_head_off_rev_raises_before_blob_check_and_rollout(fake_lab):
    fake_lab["head"] = DRIFT
    called: list[bool] = []
    with pytest.raises(ExamGateError, match=f"HEAD {DRIFT}, expected rev {PIN}") as err:
        before_eval_rollout(lambda: called.append(True), exam_path="exam.json", expected_rev=PIN)
    assert str(LAB) in str(err.value)
    assert fake_lab["blob_calls"] == []
    assert called == []


def test_dirty_tree_raises_before_blob_check_and_rollout(fake_lab):
    fake_lab["dirty"] = " M src/nutrienv/bench/scorer.py"
    called: list[bool] = []
    with pytest.raises(ExamGateError, match="uncommitted tracked changes") as err:
        before_eval_rollout(lambda: called.append(True), exam_path="exam.json", expected_rev=PIN)
    assert "scorer.py" in str(err.value)
    assert fake_lab["blob_calls"] == []
    assert called == []


def test_default_expected_rev_comes_from_config(fake_lab):
    configured = load_config(exam_gate._CONFIG).nutrienv_rev
    fake_lab["head"] = configured
    assert assert_lab_at_rev() == configured
    fake_lab["head"] = DRIFT
    with pytest.raises(ExamGateError, match=f"expected rev {configured}"):
        assert_lab_at_rev()


def test_short_rev_rejected(fake_lab):
    with pytest.raises(ExamGateError, match="not a full 40-hex SHA"):
        assert_lab_at_rev(PIN[:7])


def test_exam_report_aborts_on_rev_mismatch(fake_lab):
    from src.training.rl.eval_report import exam_report

    fake_lab["head"] = DRIFT
    with pytest.raises(ExamGateError, match="expected rev"):
        exam_report(task_results=[{"statuses": ["pass"]}], k=1, expected_rev=PIN)
    assert fake_lab["blob_calls"] == []
