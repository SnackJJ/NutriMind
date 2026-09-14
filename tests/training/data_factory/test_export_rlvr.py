"""Ticket 027 — RLVR export native-FC prompt from a TaskPackage."""

from __future__ import annotations

import inspect
import json

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench import load_split  # noqa: E402
from nutrienv.env import NutriEnv  # noqa: E402
from nutrienv.harness.tools_schema import NUTRIENV_TOOLS, TOOL_SYSTEM_PROMPT  # noqa: E402
from nutrienv.world.catalog_store import load_catalog  # noqa: E402

from src.training.data_factory.build import build  # noqa: E402
from src.training.data_factory.export_rlvr import export_rlvr  # noqa: E402
from src.training.data_factory.materialize import RunContext, catalog_digest, materialize  # noqa: E402
from src.training.data_factory.synthetic import synth_expander  # noqa: E402

from tests.training.data_factory import _fixtures as fx  # noqa: E402
from tests.training.data_factory.test_build import tiny_config  # noqa: E402


def test_export_takes_taskpackage_never_sft_record():
    sig = inspect.signature(export_rlvr)
    param = next(iter(sig.parameters.values()))
    assert param.name == "package"
    with pytest.raises(TypeError, match="TaskPackage"):
        export_rlvr({"messages": [], "schema_version": "nutrimind-v2-sft/1"})


def test_export_prompt_is_lab_schema(catalog=None):
    catalog = load_catalog()
    task = fx.make_log_task(catalog, fx.first_person(), seed=30)
    package = materialize(
        task,
        RunContext(
            catalog=catalog,
            catalog_sha=catalog_digest(catalog),
            nutrienv_rev="0ee68eaa6c246e8079915761c95fc986c53d4979",
            nutrimind_rev="d" * 40,
            config_sha="0" * 64,
            seed=30,
            built_at="2026-09-11T12:00:00+00:00",
        ),
    )
    export = export_rlvr(package)
    assert export["prompt"]["system"] is TOOL_SYSTEM_PROMPT
    assert export["prompt"]["tools"] is NUTRIENV_TOOLS
    assert export["prompt"]["task"] == f"Task:\n{package.query}"
    blob = json.dumps(export)
    assert "react_manual" not in blob
    assert export["reward"]["map"] == {"pass": 1.0, "fail": 0.0, "indeterminate": None}
    assert export["termination"]["finish_ops"] == package.termination.finish_ops
    assert export["termination"]["max_steps"] == package.termination.max_steps

    # environment reconstructs to a runnable NutriEnv
    item = {
        "id": package.task_id,
        "family": package.family,
        "persona": "everyday",
        "situations": [],
        "query": package.query,
        "s0": export["environment"]["s0"],
        "oracle": package.oracle.payload,
    }
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        path = pathlib.Path(scratch) / "item.json"
        path.write_text(json.dumps({"items": [item]}) + "\n")
        (rebuilt,) = load_split(path, catalog=catalog)
    env = NutriEnv()
    env.reset(rebuilt.s0)
    assert env.state().profile.user_id == rebuilt.s0.profile.user_id


def test_build_target_rlvr_zero_teacher(tmp_path):
    calls = []

    def teacher(_request):
        calls.append(1)
        raise AssertionError("rlvr must not call the teacher")

    out = tmp_path / "out"
    config = tiny_config(out, target="rlvr")
    expander = synth_expander(load_catalog())
    manifest = build(
        config, expander=expander, teacher_complete=teacher, output_dir=out
    )
    assert calls == []
    assert manifest["counts"]["rlvr_exported"] >= 1
    files = list((out / "rlvr").glob("*.json"))
    assert files
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["prompt"]["system"] == TOOL_SYSTEM_PROMPT
    assert payload["prompt"]["tools"] == NUTRIENV_TOOLS
    assert "messages" not in payload
