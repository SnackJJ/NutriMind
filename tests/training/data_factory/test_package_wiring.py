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
    """OQ-14: .env.example documents credentials as placeholders, no real key."""
    text = (REPO / ".env.example").read_text()
    key = re.search(r"^ARK_API_KEY=(\S*)\s*$", text, re.MULTILINE)
    url = re.search(r"^ARK_BASE_URL=(\S*)\s*$", text, re.MULTILINE)
    cc_key = re.search(r"^COMMANDCODE_API_KEY=(\S*)\s*$", text, re.MULTILINE)
    cc_url = re.search(r"^COMMANDCODE_BASE_URL=(\S*)\s*$", text, re.MULTILINE)
    assert key, "ARK_API_KEY missing from .env.example"
    assert url, "ARK_BASE_URL missing from .env.example"
    assert cc_key, "COMMANDCODE_API_KEY missing from .env.example"
    assert cc_url, "COMMANDCODE_BASE_URL missing from .env.example"
    # placeholder value only — a real key must never be committed
    for match in (key, cc_key):
        assert "here" in match.group(1).lower() or match.group(1) == "", match.group(1)
    assert url.group(1).startswith("http")
    assert cc_url.group(1).startswith("http")


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
    """No **already-committed** file under v1's sft/ or grpo/ trees is modified.

    The v2 line adds its own files to those trees (`data_factory` is the only v2
    package, but the pilot's go/no-go seam and ticket 028's loader live beside the
    v1 modules). Adding a file is not touching v1; only a modification of a
    tracked v1 file is. An untracked file is therefore fine here — the staged work
    in this repo is committed in feature batches, so a status-based check would
    fail mid-batch for a reason that has nothing to do with v1.
    """
    diff = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src/training/sft", "src/training/grpo"],
        cwd=REPO, text=True,
    )
    for line in diff.splitlines():
        state, path = line[:2], line[3:].strip()
        # `??` untracked (not in HEAD), `A ` newly added — both are v2 additions.
        if state == "??" or state == "A ":
            continue
        if path.startswith("src/training/sft") or path.startswith("src/training/grpo"):
            raise AssertionError(f"committed v1 training file modified: {line}")
