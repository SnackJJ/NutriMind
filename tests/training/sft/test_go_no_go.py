"""Pilot ticket 005 — SFT cold-start go/no-go (no SFT loss)."""

from __future__ import annotations

import inspect

from src.training.sft.go_no_go import cold_start_go_no_go


def _row(**kwargs):
    base = {
        "family": "log",
        "schema_valid": True,
        "finished": True,
        "execution": "ok",
        "oracle_exec": "ok",
        "status": "pass",
        "recovery_positive": False,
        "group_id": "g0",
    }
    base.update(kwargs)
    return base


def test_signature_has_no_sft_loss():
    sig = inspect.signature(cold_start_go_no_go)
    assert "sft_loss" not in sig.parameters
    assert "loss" not in sig.parameters
    assert list(sig.parameters) == ["rows", "group_size"]


def test_all_invalid_is_insufficient():
    rows = [
        _row(schema_valid=False, finished=False, status="indeterminate", group_id="g0"),
        _row(
            family="recommend",
            schema_valid=False,
            finished=False,
            status="indeterminate",
            group_id="g1",
        ),
    ]
    report = cold_start_go_no_go(rows, group_size=2)
    assert report["verdict"] == "insufficient"
    assert "schema_tool_call_validity" in report
    assert "finish_rate" in report


def test_all_no_finish_is_insufficient():
    rows = [
        _row(finished=False, execution="no_finish", status="indeterminate", group_id="g0"),
        _row(
            family="evaluate",
            finished=False,
            execution="no_finish",
            status="indeterminate",
            group_id="g1",
        ),
    ]
    assert cold_start_go_no_go(rows, group_size=2)["verdict"] == "insufficient"


def test_all_zero_groups_insufficient():
    rows = [
        _row(status="fail", group_id="g0"),
        _row(status="fail", group_id="g0"),
        _row(family="recommend", status="fail", group_id="g1"),
        _row(family="recommend", status="fail", group_id="g1"),
    ]
    report = cold_start_go_no_go(rows, group_size=2)
    assert report["verdict"] == "insufficient"
    assert report["mixed_reward_group_rate"] == 0.0


def test_mixed_pass_fail_two_families_usable():
    rows = [
        _row(status="pass", group_id="g0"),
        _row(status="fail", group_id="g0"),
        _row(family="recommend", status="pass", group_id="g1"),
        _row(family="recommend", status="fail", group_id="g1"),
    ]
    report = cold_start_go_no_go(rows, group_size=2)
    assert report["verdict"] == "usable"
    assert report["pass_at_1_by_family"]["log"] == 0.5
    assert report["pass_at_1_by_family"]["recommend"] == 0.5
    assert report["mixed_pass_fail_families"] == 2
    assert report["mixed_reward_group_rate"] == 1.0
    assert report["execution_health"] == 1.0
    assert report["oracle_reconstruction_health"] == 1.0
    assert report["planned_g"] == 2
    assert "recovery_positive_rate" in report
    assert "families_covered" in report
    assert "composite_types_covered" in report
    assert "no_finish_rate" in report
