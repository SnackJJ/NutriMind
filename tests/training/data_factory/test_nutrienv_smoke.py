"""Ticket 001 — NutriEnv public-API smoke test.

Proves `nutrienv` is importable and the public surface the v2 data factory borrows
(ADR-012, spec §18) actually works. No v2 business code is exercised here.

NutriEnv must be installed strict-editable (`scripts/setup_nutrienv.sh`); a default
wheel / git+ install drops `nutrienv/env/` and every assertion below fails at import.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

# Pinned rev — keep in sync with configs/data_factory.yaml : nutrienv.rev
NUTRIENV_PIN = "203d807b19953a86b5486303ba6f7dd3b9cf7bb6"

nutrienv = pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

_CONFIG = pathlib.Path(__file__).resolve().parents[3] / "configs" / "data_factory.yaml"


def _src_root() -> pathlib.Path:
    # .../nutri-env/src/nutrienv/__init__.py -> .../nutri-env
    return pathlib.Path(nutrienv.__file__).resolve().parents[2]


def _config_rev() -> str:
    import re

    text = _CONFIG.read_text()
    m = re.search(r"^\s*rev:\s*[\"']?([0-9a-f]{40})\b", text, re.MULTILINE)
    assert m, f"no 40-hex rev on a 'rev:' line in {_CONFIG}"
    return m.group(1)


def test_pin_is_single_sourced():
    # configs/data_factory.yaml is the source of truth; the test constant and the
    # installed source tree must both agree with it.
    cfg = _config_rev()
    assert cfg == NUTRIENV_PIN, f"config rev {cfg} != test constant {NUTRIENV_PIN}"
    head = subprocess.check_output(
        ["git", "-C", str(_src_root()), "rev-parse", "HEAD"], text=True
    ).strip()
    assert head == cfg, f"installed nutri-env HEAD {head} != config rev {cfg}"


def test_version_and_editable_source():
    assert nutrienv.__version__ == "1.0.0"
    # strict-editable install points __file__ at the real source tree, not a copied wheel
    assert (_src_root() / "src" / "nutrienv" / "env" / "nutri_env.py").is_file()


def test_installed_rev_matches_pin():
    head = subprocess.check_output(
        ["git", "-C", str(_src_root()), "rev-parse", "HEAD"], text=True
    ).strip()
    assert head == NUTRIENV_PIN, f"nutri-env HEAD {head} != pinned {NUTRIENV_PIN}"


# Public-API import coverage now lives in the ticket-016 guard: test_imports.py, test_borrowed_api_signatures.py, test_two_class_rule.py.


def test_catalog_digest_matches_exam_split():
    from nutrienv.bench import EXAM_SPLIT_PATH
    from nutrienv.bench.pipeline.types import catalog_digest
    from nutrienv.world.catalog_store import GOLD_CATALOG_PATH, load_catalog

    digest = catalog_digest(load_catalog(GOLD_CATALOG_PATH))
    declared = json.loads(pathlib.Path(EXAM_SPLIT_PATH).read_text())["catalog_sha256"]
    assert digest == declared


def test_load_exam_is_63_tasks():
    from nutrienv.bench import load_exam

    exam = load_exam()
    assert len(exam) == 63


def test_reset_score_and_achievable_wire_up():
    from nutrienv.bench import Scorer, check_achievable, load_exam
    from nutrienv.env import NutriEnv

    task = load_exam()[0]
    env = NutriEnv()
    obs = env.reset(task.s0)
    assert isinstance(obs, dict)

    result = Scorer().score(env.state(), task.oracle)
    assert "passed" in result  # shape only; not asserting Pass on the raw s0

    report = check_achievable([task])
    assert task.id not in report.unreachable


def test_generate_one_update_needs_no_expander():
    from nutrienv.bench.pipeline.freezer import task_to_item
    from nutrienv.bench.pipeline.generate_one import generate_one
    from nutrienv.world.catalog_store import GOLD_CATALOG_PATH, load_catalog

    result = generate_one(
        catalog=load_catalog(GOLD_CATALOG_PATH),
        family="update",
        seed=7,
        shell="upd-weight",
        slots={"n": "70"},
    )
    assert result.accepted is not None, f"generate_one rejected: {result.rejected}"
    item = task_to_item(result.accepted)
    assert isinstance(item, dict) and "id" in item


def test_runner_constants():
    from nutrienv.harness.runner import FAMILY_MAX_STEPS, FINISH_OPS

    assert FAMILY_MAX_STEPS["composite"] == 30
    assert "finish" in FINISH_OPS
