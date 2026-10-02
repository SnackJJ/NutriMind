"""Training data must retain environment/oracle but never teacher answers."""

import json

import pytest

from scripts.prepare_grpo_v2 import clean_holdouts, collect, training_row
from src.training.data_factory.materialize import RunContext, catalog_digest, materialize
from nutrienv.world.catalog_store import load_catalog
from nutrienv.harness.tools_schema import TOOL_SYSTEM_PROMPT
from tests.training.data_factory import _fixtures as fx


def test_row_round_trips_package_without_teacher():
    catalog = load_catalog()
    task = fx.make_log_task(catalog, fx.first_person(), seed=30)
    package = materialize(task, RunContext(catalog=catalog,
        catalog_sha=catalog_digest(catalog), nutrienv_rev="a" * 40,
        nutrimind_rev="b" * 40, config_sha="c" * 64, seed=30,
        built_at="2026-10-01T00:00:00Z"))
    row = training_row(package)
    assert row["prompt"] == [{"role": "system", "content": TOOL_SYSTEM_PROMPT},
                              {"role": "user", "content": f"Task:\n{package.query}"}]
    assert json.loads(row["extra_info"]["task_package_json"]) == package.to_dict()
    assert row["agent_name"] == "nutrimind_v2"
    assert "messages" not in row and "teacher" not in row
    from dataclasses import replace
    alias = replace(package, task_id="different-id-same-problem")
    clean, audit = clean_holdouts([package], [alias])
    assert clean == []
    assert audit["sft_seen_task_ids"] == [alias.task_id]
    clean, audit = clean_holdouts([], [package, alias])
    assert clean == [package]
    assert audit["duplicate_holdout_task_ids"] == [alias.task_id]


def test_duplicate_tasks_are_rejected(tmp_path):
    source = "data/student/v2-batch1-200/sft/train.jsonl"
    from pathlib import Path
    if not Path(source).exists():
        pytest.skip("local SFT artifacts not present")
    with pytest.raises(ValueError, match="duplicate task"):
        collect([source, source])
