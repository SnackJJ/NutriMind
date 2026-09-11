"""Ticket 017 — build --freeze-mini → sft/val_mini.json (30 TRAIN_ROSTER tasks)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench import check_achievable, load_split  # noqa: E402
from nutrienv.env import NutriEnv  # noqa: E402
from nutrienv.world.catalog_store import load_catalog  # noqa: E402

from src.training.data_factory.build import (  # noqa: E402
    MINI_EXAM_N,
    MINI_EXAM_SEED_BASE,
    build,
    enumerate_intents,
    mini_exam_intents,
)
from src.training.data_factory.synthetic import synth_expander  # noqa: E402

from tests.training.data_factory.test_build import tiny_config  # noqa: E402


@pytest.fixture(scope="module")
def expander():
    return synth_expander(load_catalog())


def test_freeze_mini_writes_30_without_teacher(tmp_path, expander):
    calls: list = []

    def teacher(_request):
        calls.append(1)
        raise AssertionError("teacher must not run on --freeze-mini")

    out = tmp_path / "out"
    manifest = build(
        tiny_config(out),
        expander=expander,
        teacher_complete=teacher,
        freeze_mini=True,
        output_dir=out,
    )
    assert calls == []
    path = out / "sft" / "val_mini.json"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["items"]) == MINI_EXAM_N
    assert manifest["counts"]["eval_frozen"] == MINI_EXAM_N
    assert not (out / "sft" / "train.jsonl").exists()


def test_freeze_mini_achievable_gated_and_reconstructable(tmp_path, expander):
    out = tmp_path / "out"
    build(tiny_config(out), expander=expander, freeze_mini=True, output_dir=out)
    catalog = load_catalog()
    tasks = load_split(out / "sft" / "val_mini.json", catalog=catalog)
    assert len(tasks) == MINI_EXAM_N
    report = check_achievable(tasks)
    assert all(task.id not in report.unreachable for task in tasks)
    env = NutriEnv()
    for task in tasks:
        obs = env.reset(task.s0)
        assert isinstance(obs, dict)
        assert task.s0.profile.user_id.startswith("train-")


def test_freeze_mini_byte_identical_and_seed_disjoint(tmp_path, expander):
    a, b = tmp_path / "a", tmp_path / "b"
    build(tiny_config(a), expander=expander, freeze_mini=True, output_dir=a)
    build(tiny_config(b), expander=expander, freeze_mini=True, output_dir=b)
    left = (a / "sft" / "val_mini.json").read_bytes()
    right = (b / "sft" / "val_mini.json").read_bytes()
    assert left == right
    payload = json.loads(left)
    ids = [item["id"] for item in payload["items"]]
    assert len(ids) == len(set(ids)) == MINI_EXAM_N
    seeds = [int(task_id.rsplit("--", 1)[1]) for task_id in ids]
    assert all(seed >= MINI_EXAM_SEED_BASE for seed in seeds)
    batch1 = {intent["task_id"] for intent in enumerate_intents(tiny_config(tmp_path / "cfg"))}
    assert set(ids).isdisjoint(batch1)
    assert {intent["seed"] for intent in mini_exam_intents()}.isdisjoint(
        {intent["seed"] for intent in enumerate_intents(tiny_config(tmp_path / "cfg2"))}
    )
