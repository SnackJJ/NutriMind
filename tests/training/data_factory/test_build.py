"""Ticket 010 — build skeleton: author → gate → materialize, --stop-after,
reject routing, failure isolation, resume (spec §4.1/§6/§8).

Offline: the expander is the ticket-003 synthetic one (or a test double); the
teacher path is ticket 011 — here build runs through materialize only. CLI
behavior (exit codes) is exercised through subprocess runs of the real entry
point.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil
import sqlite3
import subprocess
import sys

import pytest
import yaml

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.world.catalog_store import GOLD_CATALOG_PATH, load_catalog  # noqa: E402

from src.training.data_factory import build as build_mod  # noqa: E402
from src.training.data_factory.build import BuildError, build, enumerate_intents  # noqa: E402
from src.training.data_factory.config import load_config  # noqa: E402
from src.training.data_factory.synthetic import synth_expander  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "data_factory.yaml"


def tiny_config(output_dir, **overrides):
    """The real config, shrunk to a 3-intent log-only run (target 2 × 1.5)."""
    config = load_config(CONFIG_PATH)
    log_cfg = dataclasses.replace(
        config.families["log"], target_n=overrides.pop("target_n", 2),
        over_generate_x=overrides.pop("over_generate_x", 1.5),
    )
    return dataclasses.replace(
        config, families={"log": log_cfg}, output_dir=str(output_dir), **overrides
    )


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.fixture(scope="module")
def expander(catalog):
    return synth_expander(catalog)


# --------------------------------------------------------------------------- #
# enumeration + staged artifacts
# --------------------------------------------------------------------------- #


def test_enumerate_intents_deterministic_and_sorted(tmp_path):
    config = tiny_config(tmp_path)
    intents = enumerate_intents(config)
    assert len(intents) == 3  # ceil(2 * 1.5)
    assert [i["task_id"] for i in intents] == sorted(i["task_id"] for i in intents)
    assert all(i["schema_version"] == "nutrimind-v2-intent/1" for i in intents)
    first = intents[0]
    assert first["task_key"] == "log--log--train-alba"
    assert first["task_id"] == "log--log--train-alba--000000"
    assert first["steps"] == ["log"] and first["tier"] == ""
    # pure function of the config: a second call is identical
    assert enumerate_intents(config) == intents


def test_stop_after_gate_writes_staged_artifacts(tmp_path, expander):
    out = tmp_path / "out"
    manifest = build(
        tiny_config(out), expander=expander, stop_after="gate", output_dir=out
    )
    assert manifest["status"] == "complete"
    counts = manifest["counts"]
    assert counts["intents"] == 3
    assert counts["authored"] == 3
    assert counts["gate_kept"] == 3
    assert counts["materialized"] == 3
    assert counts["rejected"] == {"author": 0, "gate": 0, "indeterminate": 0}

    intents_blob = (out / "intents" / "log.jsonl").read_text(encoding="utf-8")
    intent_ids = [json.loads(line)["task_id"] for line in intents_blob.splitlines()]
    assert intent_ids == sorted(intent_ids)

    task_lines = [
        json.loads(line) for line in (out / "tasks" / "log.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert all(line["gated"] is True for line in task_lines)
    assert {line["task_id"] for line in task_lines} == set(intent_ids)

    packages = sorted((out / "task_packages").glob("*.json"))
    assert {p.stem for p in packages} == set(intent_ids)
    for path, intent_id in zip(packages, sorted(intent_ids)):
        package = json.loads(path.read_text(encoding="utf-8"))
        assert package["task_id"] == intent_id  # author/materialize consistency
        assert package["schema_version"] == "nutrimind-v2-taskpackage/1"

    written = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert written["status"] == "complete"
    assert written["catalog_sha"] == manifest["catalog_sha"]
    assert written["nutrienv_rev"] == manifest["nutrienv_rev"]


def test_stop_after_author_writes_no_packages(tmp_path, expander):
    out = tmp_path / "out"
    manifest = build(
        tiny_config(out), expander=expander, stop_after="author", output_dir=out
    )
    assert manifest["status"] == "complete"
    assert manifest["counts"]["authored"] == 3
    assert manifest["counts"]["materialized"] == 0
    assert (out / "intents" / "log.jsonl").is_file()
    task_lines = [
        json.loads(line)
        for line in (out / "tasks" / "log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(line["gated"] is False for line in task_lines)
    assert not (out / "task_packages").exists()


def test_intents_byte_identical_across_runs(tmp_path, expander):
    first, second = tmp_path / "a", tmp_path / "b"
    build(tiny_config(first), expander=expander, stop_after="gate", output_dir=first)
    build(tiny_config(second), expander=expander, stop_after="gate", output_dir=second)
    assert (first / "intents" / "log.jsonl").read_bytes() == (
        second / "intents" / "log.jsonl"
    ).read_bytes()


# --------------------------------------------------------------------------- #
# whole-run aborts (before any task is authored)
# --------------------------------------------------------------------------- #


def _tampered_catalog(tmp_path) -> str:
    copy = tmp_path / "tampered.sqlite"
    shutil.copy(GOLD_CATALOG_PATH, copy)
    con = sqlite3.connect(copy)
    con.execute(
        "UPDATE foods SET name = 'Milk, NFS (tampered copy)' WHERE food_id = '2705384'"
    )
    con.commit()
    con.close()
    return str(copy)


def test_catalog_sha_mismatch_aborts_before_authoring(tmp_path, expander):
    out = tmp_path / "out"
    config = tiny_config(out, catalog_path=_tampered_catalog(tmp_path))
    with pytest.raises(BuildError, match="catalog SHA mismatch"):
        build(config, expander=expander, stop_after="gate", output_dir=out)
    # partial manifest written; nothing else touched
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "aborted"
    assert "catalog SHA mismatch" in manifest["abort_reason"]
    assert not (out / "intents").exists()
    assert not (out / "task_packages").exists()


def test_nutrienv_rev_mismatch_aborts_before_authoring(tmp_path, expander):
    out = tmp_path / "out"
    config = tiny_config(out)
    pin = dataclasses.replace(config.nutrienv, rev="0" * 40)
    config = dataclasses.replace(config, nutrienv=pin)
    with pytest.raises(BuildError, match="nutrienv rev mismatch"):
        build(config, expander=expander, stop_after="gate", output_dir=out)
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "aborted"
    assert not (out / "intents").exists()


# --------------------------------------------------------------------------- #
# single-task failures (run continues)
# --------------------------------------------------------------------------- #


def _flaky_expander(inner, *, fail_when):
    def expander(pool, *, persona, family, amount_path=None):
        if fail_when(amount_path):
            return {"query": "", "foods": []}
        return inner(pool, persona=persona, family=family, amount_path=amount_path)

    return expander


def test_one_unauthorable_intent_among_good(tmp_path, catalog, expander):
    out = tmp_path / "out"
    flaky = _flaky_expander(expander, fail_when=lambda ap: ap == "explicit_grams")
    manifest = build(
        tiny_config(out), expander=flaky, stop_after="gate", output_dir=out
    )
    assert manifest["status"] == "complete"  # failure isolation
    assert manifest["counts"]["materialized"] == 2
    assert manifest["counts"]["rejected"]["author"] == 1

    reject_lines = [
        json.loads(line)
        for line in (out / "rejects" / "author.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    (reject,) = reject_lines
    assert reject["stage"] == "author"
    assert reject["status"] == "dropped"
    assert reject["failure_codes"][0].startswith("author.")
    assert reject["intent"]["task_id"] == reject["task_id"]
    assert reject["intent"]["amount_path"] == "explicit_grams"
    # the good ones still materialized
    assert len(list((out / "task_packages").glob("*.json"))) == 2


def test_gate_drop_routing(tmp_path, expander, monkeypatch):
    """A gate drop → rejects/gate.jsonl (routed by rejects_record status);
    unachievable → rejects/indeterminate.jsonl. The gate itself is ticket
    005's tested territory — here build only routes."""
    out = tmp_path / "out"
    config = tiny_config(out)
    calls = {"n": 0}

    def fake_run(task, ctx):
        calls["n"] += 1
        code = "gate.verbatim" if calls["n"] == 1 else "gate.unachievable"
        return build_mod.gates_mod.GateResult(
            keep=False, failure_code=code, reason_detail=f"forced {code}"
        )

    monkeypatch.setattr(build_mod.gates_mod, "run", fake_run)
    manifest = build(config, expander=expander, stop_after="gate", output_dir=out)
    assert manifest["status"] == "complete"
    assert manifest["counts"]["materialized"] == 0
    assert manifest["counts"]["rejected"] == {
        "author": 0, "gate": 1, "indeterminate": 2
    }
    gate_lines = [
        json.loads(line)
        for line in (out / "rejects" / "gate.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert gate_lines[0]["failure_codes"] == ["gate.verbatim"]
    assert gate_lines[0]["status"] == "dropped"
    indeterminate_lines = [
        json.loads(line)
        for line in (out / "rejects" / "indeterminate.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert indeterminate_lines[0]["failure_codes"] == ["gate.unachievable"]
    assert indeterminate_lines[0]["status"] == "indeterminate"
    assert not (out / "task_packages").exists()


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #


def test_rerun_skips_terminal_task_ids(tmp_path, expander):
    out = tmp_path / "out"
    first = build(
        tiny_config(out), expander=expander, stop_after="gate", output_dir=out
    )
    assert first["counts"]["materialized"] == 3
    before = {p.name: p.read_bytes() for p in (out / "task_packages").glob("*.json")}

    second = build(
        tiny_config(out), expander=expander, stop_after="gate", output_dir=out
    )
    assert second["counts"]["skipped_terminal"] == 3
    assert second["counts"]["materialized"] == 0
    after = {p.name: p.read_bytes() for p in (out / "task_packages").glob("*.json")}
    assert after == before  # untouched, not rewritten
    # tasks/log.jsonl not appended twice
    assert len((out / "tasks" / "log.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_force_reruns_terminal_tasks(tmp_path, expander):
    out = tmp_path / "out"
    build(tiny_config(out), expander=expander, stop_after="gate", output_dir=out)
    second = build(
        tiny_config(out), expander=expander, stop_after="gate", output_dir=out,
        force=True,
    )
    assert second["counts"]["skipped_terminal"] == 0
    assert second["counts"]["materialized"] == 3
    # write_package stays idempotent: still exactly 3 package files
    assert len(list((out / "task_packages").glob("*.json"))) == 3


# --------------------------------------------------------------------------- #
# the CLI (exit codes, composition root)
# --------------------------------------------------------------------------- #


def _cli_config(tmp_path, **overrides) -> pathlib.Path:
    config = tiny_config(tmp_path / "out", **overrides)
    payload = config.to_dict()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path


def _run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.training.data_factory.build", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )


def test_cli_stop_after_gate_exits_zero(tmp_path):
    config_path = _cli_config(tmp_path)
    result = _run_cli(
        "--config", str(config_path), "--stop-after", "gate",
        "--expander", "synthetic",
    )
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    assert (out / "intents" / "log.jsonl").is_file()
    assert (out / "run_manifest.json").is_file()
    assert len(list((out / "task_packages").glob("*.json"))) == 3


def test_cli_config_error_exits_nonzero(tmp_path):
    payload = load_config(CONFIG_PATH).to_dict()
    del payload["teacher"]["temperature_first"]  # missing key
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    result = _run_cli(
        "--config", str(path), "--stop-after", "gate", "--expander", "synthetic"
    )
    assert result.returncode == 1
    assert "config error" in result.stderr


def test_cli_refuses_implicit_expander(tmp_path):
    config_path = _cli_config(tmp_path)
    result = _run_cli("--config", str(config_path), "--stop-after", "gate")
    assert result.returncode == 1
    assert "--expander" in result.stderr


def test_cli_rev_mismatch_exits_nonzero(tmp_path):
    config_path = _cli_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["nutrienv"]["rev"] = "0" * 40
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    result = _run_cli(
        "--config", str(config_path), "--stop-after", "gate",
        "--expander", "synthetic",
    )
    assert result.returncode == 1
    assert "build aborted" in result.stderr
    manifest = json.loads(
        (tmp_path / "out" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "aborted"
