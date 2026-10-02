"""B5 eval half — paired exam comparison on hand-built eval dirs (no lab needed)."""

from __future__ import annotations

import json
import pathlib

import pytest

from src.training.rl import compare_exam
from src.training.rl.compare_exam import CompareRefused, compare_dirs, mcnemar_exact
from src.training.rl.eval_exam import EPISODES, MANIFEST

_EXAM = {
    "exam_path": "/lab/data/splits/nutrienv-v1.0.json",
    "exam_blob": "b" * 40,
    "lab_head": "0ee68eaa6c246e8079915761c95fc986c53d4979",
    "loop_version": None,
    "task_ids_sha256": "s" * 64,
}
_TASKS = [f"t{i}" for i in range(8)]
_FAMILIES = {t: ("log" if i < 4 else "composite") for i, t in enumerate(_TASKS)}


def _write_dir(path: pathlib.Path, passing: dict[str, list[bool]], *, exam=None, tasks=None, model="m"):
    tasks = tasks or _TASKS
    path.mkdir(parents=True)
    runs = len(next(iter(passing.values()), [False] * 3))
    manifest = {
        "model": model,
        "runs": runs,
        "task_ids": tasks,
        "families": {t: _FAMILIES.get(t, "log") for t in tasks},
        "exam": exam or dict(_EXAM),
    }
    (path / MANIFEST).write_text(json.dumps(manifest))
    lines = [
        json.dumps({"task_id": t, "run": r + 1, "status": "pass" if ok else "fail"})
        for t in tasks
        for r, ok in enumerate(passing.get(t, [False] * runs))
    ]
    (path / EPISODES).write_text("\n".join(lines) + "\n")
    return path


def test_mcnemar_exact_hand_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 5) == pytest.approx(14 / 64)  # 2 * (C(6,0)+C(6,1)) / 2^6
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_exact(5, 1) == mcnemar_exact(1, 5)
    assert mcnemar_exact(3, 3) == 1.0


def test_paired_compare_on_hand_example(tmp_path):
    base = _write_dir(tmp_path / "base", {"t0": [True, True, False]})
    cand = _write_dir(
        tmp_path / "cand",
        {"t1": [True] * 3, "t2": [True] * 3, "t3": [True, True, False],
         "t4": [True] * 3, "t5": [True] * 3, "t6": [True, False, False]},
    )
    result = compare_dirs(base, cand)

    assert result["mcnemar_majority"] == {
        "baseline_only": 1, "candidate_only": 5, "p_exact": pytest.approx(14 / 64),
    }
    assert result["flipped_to_pass"] == ["t1", "t2", "t3", "t4", "t5"]
    assert result["flipped_to_fail"] == ["t0"]
    # run 1: base passes t0; cand passes t1..t6  -> b=1, c=6
    # run 3: base none; cand t1,t2,t4,t5       -> b=0, c=4
    per_run = {r["run"]: (r["baseline_only"], r["candidate_only"]) for r in result["mcnemar_per_run"]}
    assert per_run == {1: (1, 6), 2: (1, 5), 3: (0, 4)}

    pr = result["pass_rate"]
    assert pr["baseline"] == pytest.approx((2 / 3) / 8)
    assert pr["candidate"] == pytest.approx((3 + 3 + 2 + 3 + 3 + 1) / 3 / 8)
    assert pr["mean_delta"] == pytest.approx(pr["candidate"] - pr["baseline"])
    lo, hi = pr["delta_ci95_paired_bootstrap"]
    assert lo <= pr["mean_delta"] <= hi
    assert result["by_family"]["log"]["delta"] == pytest.approx(((1 + 1 + 2 / 3) - 2 / 3) / 4)


def test_refuses_on_exam_revision_mismatch(tmp_path):
    base = _write_dir(tmp_path / "base", {})
    cand = _write_dir(tmp_path / "cand", {}, exam={**_EXAM, "exam_blob": "c" * 40})
    with pytest.raises(CompareRefused, match="exam_blob"):
        compare_dirs(base, cand)
    lab = _write_dir(tmp_path / "lab", {}, exam={**_EXAM, "lab_head": "4" * 40})
    with pytest.raises(CompareRefused, match="lab_head"):
        compare_dirs(base, lab)


def test_refuses_on_task_set_mismatch(tmp_path):
    base = _write_dir(tmp_path / "base", {})
    cand = _write_dir(tmp_path / "cand", {}, tasks=_TASKS[:-1])
    with pytest.raises(CompareRefused, match="task sets differ"):
        compare_dirs(base, cand)


def test_cli_writes_compare_files(tmp_path, capsys):
    base = _write_dir(tmp_path / "base", {"t0": [True] * 3})
    cand = _write_dir(tmp_path / "cand", {"t1": [True] * 3})
    assert compare_exam.main(["--baseline", str(base), "--candidate", str(cand)]) == 0
    out = cand / "compare_vs_base"
    assert json.loads((out / "compare.json").read_text())["mcnemar_majority"]["p_exact"] == 1.0
    assert (out / "compare.md").read_text().startswith("# Exam compare")
