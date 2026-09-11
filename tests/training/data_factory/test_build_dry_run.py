"""Ticket 015 — --dry-run projection + run_manifest health / §20 metrics."""

from __future__ import annotations

import json
import pathlib

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from src.training.data_factory import build as build_mod  # noqa: E402
from src.training.data_factory.build import build  # noqa: E402
from src.training.data_factory.concepts import GateResult  # noqa: E402
from src.training.data_factory.rollout_fc import ScriptedFCTeacher  # noqa: E402
from src.training.data_factory.synthetic import synth_expander  # noqa: E402

from tests.training.data_factory.test_build import tiny_config  # noqa: E402
from tests.training.data_factory.test_build_sft import (  # noqa: E402
    author_all,
    sft_config,
    teacher_script,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def expander():
    from nutrienv.world.catalog_store import load_catalog

    return synth_expander(load_catalog())


def test_dry_run_zero_teacher_calls_and_writes_report(tmp_path, expander):
    calls: list = []

    def teacher(_request):
        calls.append(1)
        raise AssertionError("teacher must not run on --dry-run")

    out = tmp_path / "out"
    manifest = build(
        tiny_config(out),
        expander=expander,
        teacher_complete=teacher,
        dry_run=True,
        output_dir=out,
    )
    assert calls == []
    assert manifest["dry_run"] is True
    assert manifest["counts"]["accepted"] == 0
    assert manifest["counts"]["materialized"] == 0
    report = json.loads((out / "dry_run_report.json").read_text(encoding="utf-8"))
    assert report["projected_accepts"]["total"] == manifest["counts"]["gate_kept"]
    assert "reject_histogram" in report
    assert report["gate_kept"] == manifest["counts"]["gate_kept"]
    assert not (out / "sft" / "train.jsonl").exists()


def test_manifest_status_split_versions_and_health_hand_count(tmp_path, expander):
    out = tmp_path / "out"
    config = sft_config(out, teacher_k=1, target_n=1, over_generate_x=1.0)
    tasks = author_all(config, expander)
    script = teacher_script(tasks, pass_at_attempt=1, teacher_k=1)
    manifest = build(
        config,
        expander=expander,
        teacher_complete=ScriptedFCTeacher(script),
        output_dir=out,
    )
    by_status = manifest["counts"]["by_status"]
    assert by_status["accepted"] == manifest["counts"]["accepted"]
    assert "fail" in by_status and "indeterminate" in by_status
    assert "task_fail" not in by_status  # indeterminate is not mixed into fail
    versions = manifest["versions"]
    records = [
        json.loads(line)
        for line in (out / "sft" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records
    meta = records[0]["meta"]
    for key in (
        "oracle_version", "rubric_version", "reward_version",
        "environment_version", "task_schema_version",
    ):
        assert versions[key] == meta[key]
    health = manifest["health"]
    assert health["catalog_sha_match"] is True
    accepted = by_status["accepted"]
    assert health["serialization_success_rate"] == pytest.approx(accepted / accepted)
    assert health["teacher_completion_rate"] == pytest.approx(1.0)
    assert health["teacher_pass_rate"] == pytest.approx(1.0)
    assert health["indeterminate_rate"] is None
    assert "attempted_task_ids=" in health["indeterminate_rate_note"]
    assert "log" in manifest["family_mix"]
    assert manifest["family_mix"]["log"]["actual"] == accepted
    assert manifest["family_mix"]["log"]["target"] == 1
    assert "est_usd" in manifest["cost"]
    assert manifest["cost"]["budget_usd"] == config.usd_budget


def test_reject_histogram_flags_inflated_draft_invalid(tmp_path, expander, monkeypatch):
    out = tmp_path / "out"

    def fake_run(task, ctx):
        return GateResult(
            keep=False,
            failure_code="gate.draft_invalid",
            reason_detail="forced",
        )

    monkeypatch.setattr(build_mod.gates_mod, "run", fake_run)
    manifest = build(
        tiny_config(out), expander=expander, dry_run=True, output_dir=out
    )
    hist = manifest["counts"]["by_failure_code"]
    assert hist.get("gate.draft_invalid", 0) >= 1
    assert manifest["health"]["reject_histogram_ok"] is False
    report = json.loads((out / "dry_run_report.json").read_text(encoding="utf-8"))
    assert report["reject_histogram"]["gate.draft_invalid"] == hist["gate.draft_invalid"]
