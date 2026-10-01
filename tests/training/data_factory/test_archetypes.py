"""Batch-2 ADR 0029 archetype authors and the exam-leak gates."""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench import EXAM_SPLIT_PATH, load_split  # noqa: E402

from src.training.data_factory import author as A  # noqa: E402
from src.training.data_factory import build as B  # noqa: E402
from src.training.data_factory import gates as G  # noqa: E402
from src.training.data_factory.archetypes import ARCHETYPE_STRATEGIES  # noqa: E402
from src.training.data_factory.config import load_config  # noqa: E402
from tests.training.data_factory import _fixtures as fx  # noqa: E402

ARCHETYPES = ("evaluate_hypo", *ARCHETYPE_STRATEGIES)


@pytest.fixture(scope="module")
def catalog():
    return fx.gold_catalog()


@pytest.fixture(scope="module")
def exam(catalog):
    return load_split(EXAM_SPLIT_PATH, catalog=catalog)


@pytest.fixture(scope="module")
def ctx(exam):
    return G.GateContext.from_exam(exam)


@pytest.fixture(scope="module")
def config():
    config = load_config("configs/data_factory.yaml")
    families = dict(config.families)
    for family in ARCHETYPES:
        families.setdefault(family, families["recommend"])
    return dataclasses.replace(config, families=families)


def _author(config, catalog, family, index):
    intent = B.intent_for(config, family, index)
    return A.author_task(intent, catalog=catalog, expander=None)


@pytest.mark.parametrize("family", ARCHETYPE_STRATEGIES)
def test_archetypes_author_gate_passing_tasks(config, catalog, ctx, family):
    kept = 0
    for index in range(6):
        task, reject = _author(config, catalog, family, index)
        if task is None:
            continue
        result = G.run(task, ctx)
        assert result.keep, (family, task.query, result)
        kept += 1
    assert kept >= 4


def test_inventory_and_menu_are_closed(config, catalog):
    for family in ("recommend_inventory", "recommend_menu"):
        task, _ = _author(config, catalog, family, 0)
        allowed = task.s0.allowed_food_ids
        assert allowed and task.oracle.allowed_food_ids == allowed
        assert task.oracle.plan_must_fit_windows and task.oracle.last_plan == []


def test_amend_corrects_one_row_in_place(config, catalog):
    task, _ = _author(config, catalog, "composite_amend_recommend", 1)
    amend, recommend = task.oracle.sub_oracles
    before, after = list(task.s0.ledger), list(amend.ledger)
    assert len(before) == len(after)
    assert sum(a != b for a, b in zip(before, after)) == 1
    assert list(recommend.ledger) == after
    assert "for dinner" in task.query


def test_refuse_keeps_the_profile(config, catalog):
    task, _ = _author(config, catalog, "composite_refuse_recommend", 2)
    hold, _recommend = task.oracle.sub_oracles
    assert hold.profile == task.s0.profile and hold.ledger == ()


def test_held_out_archetypes_are_never_authored():
    """Grocery allocation and recipe deconstruction are the transfer probe."""
    names = " ".join(B.FAMILY_SPECS)
    assert "buy" not in names and "dish" not in names and "grocery" not in names


def test_archetype_authors_never_read_the_exam():
    source = pathlib.Path("src/training/data_factory/archetypes.py").read_text()
    for needle in ("EXAM_SPLIT_PATH", "load_split", "nutrienv-v1", "splits/"):
        assert needle not in source


def test_near_duplicate_query_gate(exam, ctx, catalog, config):
    task, _ = _author(config, catalog, "recommend_inventory", 0)
    copied = dataclasses.replace(task, query=exam[0].query + " Thanks!")
    result = G.run(copied, ctx)
    assert not result.keep and result.failure_code == G.NEAR_DUP


def test_food_set_gate(exam, ctx, catalog, config):
    menu = next(t for t in exam if len(G.task_food_set(t)) >= 5)
    task, _ = _author(config, catalog, "recommend_menu", 0)
    leaked = dataclasses.replace(
        task, s0=dataclasses.replace(task.s0, allowed_food_ids=G.task_food_set(menu)))
    result = G.run(leaked, ctx)
    assert not result.keep and result.failure_code == G.FOOD_SET
