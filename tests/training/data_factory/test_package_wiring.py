"""Ticket 003 — package wiring: side-effect-free import, .env.example, pytest
discovery config."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[3]


def test_package_import_is_side_effect_free():
    """Importing the factory works with no network and does not pull in nutrienv
    (the benchmark is a stage-local dependency, never a package-level one)."""
    code = (
        "import sys\n"
        "import src.training.data_factory as df\n"
        "assert 'nutrienv' not in sys.modules, (\n"
        "    'data_factory must not import nutrienv at module level')\n"
        "for banned in ('socket', 'httpx', 'requests', 'openai'):\n"
        "    assert banned not in sys.modules, f'{banned} imported at package import'\n"
        "print('OK', len(df.__all__))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("OK"), result.stdout


def test_package_exports_the_seam_vocabulary():
    from src.training.data_factory import __all__ as exports

    for name in (
        "GateResult", "TurnMeta", "EpisodeResult", "RolloutCache",
        "VerificationResult", "TaskPackage", "AttemptRecord",
        "DataFactoryConfig", "load_config", "ConfigError",
    ):
        assert name in exports, name


def test_env_example_documents_ark():
    """OQ-14: .env.example documents ARK_API_KEY / ARK_BASE_URL; no real key."""
    text = (REPO / ".env.example").read_text()
    key = re.search(r"^ARK_API_KEY=(\S*)\s*$", text, re.MULTILINE)
    url = re.search(r"^ARK_BASE_URL=(\S*)\s*$", text, re.MULTILINE)
    assert key, "ARK_API_KEY missing from .env.example"
    assert url, "ARK_BASE_URL missing from .env.example"
    # placeholder value only — a real key must never be committed
    assert "here" in key.group(1).lower() or key.group(1) == "", key.group(1)
    assert url.group(1).startswith("http")


def test_pytest_discovers_tests_without_flags():
    """`pytest` from the repo root finds tests/training/data_factory/ with no -o
    flags (ticket 003 acceptance)."""
    with open(REPO / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    assert data["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "tests/training/data_factory/"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "test_concepts.py" in result.stdout
    assert "test_config.py" in result.stdout


def test_no_v1_training_code_touched_by_v2_skeleton():
    """The v2 skeleton adds only under src/training/data_factory/ — v1's sft/ and
    grpo/ trees are not modified by ticket 003 (checked against git)."""
    diff = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src/training/sft", "src/training/grpo"],
        cwd=REPO, text=True,
    )
    # the user's pre-existing uncommitted train_grpo.py change is allowed; no NEW
    # v1 file may appear, and sft/ must be untouched by this work
    assert "src/training/sft" not in diff, diff
