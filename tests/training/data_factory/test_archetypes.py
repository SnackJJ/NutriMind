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


@pytest.fixture(scope="module")
def expander(catalog):
    from src.training.data_factory.synthetic import synth_expander

    return synth_expander(catalog)


def _author(config, catalog, family, index, expander=None):
    intent = B.intent_for(config, family, index)
    return A.author_task(intent, catalog=catalog, expander=expander)


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


# --------------------------------------------------------------------------- #
# end to end: a reference episode through the real lab loop must verify Pass
# (catches verifier assumptions a new shape breaks before any teacher spend)
# --------------------------------------------------------------------------- #

def _call(index, name, args):
    import json

    return {"id": f"call_{index}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _reference_actions(task):
    from nutrienv.bench.realize import scored_oracles
    from nutrienv.bench.validator import fitting_plan

    subs = scored_oracles(task.oracle)
    actions = []
    if task.family == "evaluate":
        oracle = subs[0]
        if oracle.last_verdict == "accept":
            actions.append(("submit_plan", {"items": list(oracle.evaluated_plan),
                                            "verdict": "accept"}))
        else:
            actions.append(("submit_plan", {"items": [], "verdict": "reject",
                                            "reasons": sorted(oracle.last_reasons)}))
        return actions
    before = list(task.s0.ledger)
    for sub in subs:
        if sub.ledger and len(sub.ledger) == len(before) and list(sub.ledger) != before:
            for index, (old, new) in enumerate(zip(before, sub.ledger)):
                if old != new:
                    actions.append(("amend_meal", {"index": index, "food_id": new.food_id,
                                                   "grams": new.grams,
                                                   "eaten_at": new.eaten_at}))
    recommend = subs[-1]
    plan = fitting_plan(task.s0.catalog, dict(recommend.plan_windows),
                        task.s0.profile.allergies,
                        allowed_food_ids=recommend.allowed_food_ids)
    actions.append(("submit_plan", {"items": plan}))
    return actions


@pytest.mark.parametrize("family", ARCHETYPES)
def test_reference_episode_verifies_pass(config, catalog, ctx, expander, family):
    from src.training.data_factory import materialize as mz
    from src.training.data_factory import verify as V
    from src.training.data_factory.rollout_fc import ScriptedFCTeacher, rollout_tool_call
    from tests.training.data_factory.test_materialize import make_ctx

    checked = 0
    for index in range(8):
        intent = B.intent_for(config, family, index)
        task, _ = A.author_task(intent, catalog=catalog, expander=expander)
        if task is None or not G.run(task, ctx).keep:
            continue
        package = mz.materialize(task, make_ctx(catalog, steps=tuple(intent["steps"]),
                                                seed=intent["seed"]))
        teacher = ScriptedFCTeacher([
            ("reference", [_call(i, name, args)])
            for i, (name, args) in enumerate(_reference_actions(task))
        ])
        episode = rollout_tool_call(task, teacher_complete=teacher, catalog=catalog)
        result = V.verify(package, episode)
        assert result.status == "pass", (family, task.query, result)
        checked += 1
        if checked == 2:
            break
    assert checked == 2


def test_update_recommend_family_authors_on_its_own_quota(catalog, ctx, expander):
    config = load_config("configs/data_factory.yaml")
    families = dict(config.families)
    families["composite_update_recommend"] = families["composite"]
    config = dataclasses.replace(config, families=families)
    kept = 0
    for index in range(6):
        intent = B.intent_for(config, "composite_update_recommend", index)
        assert tuple(intent["steps"]) == ("update", "recommend") and intent["shell"]
        task, _ = A.author_task(intent, catalog=catalog, expander=expander)
        kept += task is not None and G.run(task, ctx).keep
    assert kept >= 3
