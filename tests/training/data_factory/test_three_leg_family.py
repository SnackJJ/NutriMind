"""Ticket 013 — 3-leg composite family authoring, ladder, sizing, public-only."""

from __future__ import annotations

import dataclasses
import math

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench import Scorer, check_achievable  # noqa: E402
from nutrienv.bench.pipeline.review_harness import stage_a_code_gate  # noqa: E402
from nutrienv.bench.realize import scored_oracles  # noqa: E402
from nutrienv.bench.validator import fitting_plan, semantic_key, validate_draft  # noqa: E402
from nutrienv.env import NutriEnv  # noqa: E402
from nutrienv.world.catalog_store import load_catalog  # noqa: E402

from src.training.data_factory import author as author_mod  # noqa: E402
from src.training.data_factory import build as build_mod  # noqa: E402
from src.training.data_factory.build import build, candidate_count, enumerate_intents  # noqa: E402
from src.training.data_factory.config import load_config  # noqa: E402
from src.training.data_factory.roster_train import TRAIN_ROSTER  # noqa: E402
from src.training.data_factory.synthetic import (  # noqa: E402
    QWEN_MAX_MODEL,
    qwen_max_fallback_expander,
    synth_expander,
)

from tests.training.data_factory import _fixtures as fx  # noqa: E402

_KNOWN_DRAFT_FP = ["update oracle ledger is missing"]


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def test_three_leg_strategy_imports_no_private_helpers():
    fn = author_mod._author_three_leg
    used = set(fn.__code__.co_names)
    for const in fn.__code__.co_consts:
        if hasattr(const, "co_names"):
            used |= set(const.co_names)
    assert "_update_from_template" not in used
    assert "_bind_log_foods" not in used


def test_named_measure_not_force_forced():
    base = load_config("configs/data_factory.yaml")
    three = base.families["composite_update_log_recommend"]
    config = dataclasses.replace(
        base,
        families={"composite_update_log_recommend": three},
        max_intents=2000,
    )
    intents = enumerate_intents(config)
    paths = {intent["amount_path"] for intent in intents}
    assert paths != {"named_measure"}
    assert paths <= {"named_measure", "explicit_grams", "unspecified"}


def test_candidate_count_formula_and_manifest_shape():
    assert candidate_count(0.5, target_n=40) == 120  # max(120, ceil(40/0.5*1.5)=120)
    assert candidate_count(0.2, target_n=40) == 300  # ceil(40/0.2*1.5)=300
    assert candidate_count(0.0, max_candidate_limit=50) == 50
    p = 1.0 / 3.0
    n = candidate_count(p, target_n=40)
    assert n == min(2000, max(120, math.ceil(40 / p * 1.5)))


def _three_leg_intent(person, seed=5, amount_path="explicit_grams"):
    return {
        "task_id": f"composite--update+log+recommend--{person.user_id}--{seed:06d}",
        "family": "composite_update_log_recommend",
        "user_id": person.user_id,
        "seed": seed,
        "occasion": "lunch",
        "scene": "empty",
        "amount_path": amount_path,
        "shell": None,
        "slots": {"allergen": "fish"},
        "tier": "",
        "recovery_trap": None,
    }


def test_assembled_three_leg_gates_and_wrong_tags(catalog):
    person = next(p for p in TRAIN_ROSTER if not p.allergies)
    expander = synth_expander(catalog)
    task, reject = None, None
    for seed in range(5, 40):
        for amount_path in ("explicit_grams", "named_measure"):
            intent = _three_leg_intent(person, seed=seed, amount_path=amount_path)
            task, reject = author_mod.author_task(
                intent, catalog=catalog, expander=expander,
                gram_anchor=author_mod.portion_table_gram_anchor(catalog),
            )
            if task is not None:
                break
        if task is not None:
            break
    assert task is not None, reject
    stage_a = stage_a_code_gate(task)
    draft = validate_draft(task)
    assert stage_a == []
    assert draft in ([], _KNOWN_DRAFT_FP)
    assert task.id not in check_achievable([task]).unreachable

    actions = fx.replay_actions(task)
    env = NutriEnv()
    env.reset(task.s0)
    for action in actions:
        if action.get("op") == "done":
            continue
        env.step(action)
    score = Scorer().score(env.state(), task.oracle)
    assert score["passed"] is True
    assert score.get("sub_tags") == ("pass", "pass", "pass")

    subs = scored_oracles(task.oracle)
    env = NutriEnv()
    env.reset(task.s0)
    env.step({"op": "update_profile", "patch": {"allergies": list(subs[0].profile.allergies)}})
    plan = fitting_plan(task.s0.catalog, subs[2].plan_windows, subs[0].profile.allergies)
    if plan:
        env.step({"op": "submit_plan", "items": plan})
    assert Scorer().score(env.state(), task.oracle)["tag"] == "log_miss"

    env = NutriEnv()
    env.reset(task.s0)
    env.step({"op": "update_profile", "patch": {"allergies": list(subs[0].profile.allergies)}})
    for row in subs[1].ledger_tail:
        env.step({"op": "log_meal", "food_id": row.food_id, "grams": row.grams, "eaten_at": row.eaten_at})
    if plan:
        env.step({"op": "submit_plan", "items": [{"food_id": it["food_id"], "grams": it["grams"] * 8} for it in plan]})
    tag = Scorer().score(env.state(), task.oracle)["tag"]
    assert tag in ("window", "wrong_goal")

    env = NutriEnv()
    env.reset(task.s0)
    for row in subs[1].ledger_tail:
        env.step({"op": "log_meal", "food_id": row.food_id, "grams": row.grams, "eaten_at": row.eaten_at})
    if plan:
        env.step({"op": "submit_plan", "items": plan})
    assert Scorer().score(env.state(), task.oracle)["tag"] == "update_miss"


def test_semantic_key_dedup_keeps_lowest_seed(catalog):
    person = next(p for p in TRAIN_ROSTER if not p.allergies)
    expander = synth_expander(catalog)
    tasks = []
    for seed in (8, 9):
        task, _ = author_mod.author_task(
            _three_leg_intent(person, seed=seed),
            catalog=catalog,
            expander=expander,
            gram_anchor=author_mod.portion_table_gram_anchor(catalog),
        )
        if task is not None:
            tasks.append(task)
    if len(tasks) < 2:
        pytest.skip("could not author two 3-leg tasks")
    kept: dict = {}
    for task in sorted(tasks, key=lambda item: item.id):
        kept.setdefault(semantic_key(task), task)
    for key, task in kept.items():
        twins = [item.id for item in tasks if semantic_key(item) == key]
        assert task.id == min(twins)


def test_build_qwen_max_fallback_used_when_primary_bind_fails(tmp_path, catalog, monkeypatch):
    calls = {"n": 0}
    real = qwen_max_fallback_expander

    def factory(cat, *, complete=None):
        inner = real(cat, complete=complete)
        assert inner.model_id == QWEN_MAX_MODEL

        def wrapped(pool, *, persona, family, amount_path=None):
            calls["n"] += 1
            return inner(pool, persona=persona, family=family, amount_path=amount_path)

        wrapped.model_id = inner.model_id
        return wrapped

    monkeypatch.setattr(build_mod, "qwen_max_fallback_expander", factory)

    def primary_fail(pool, *, persona, family, amount_path=None):
        return {"query": "", "foods": []}

    base = load_config("configs/data_factory.yaml")
    three = dataclasses.replace(
        base.families["composite_update_log_recommend"],
        target_n=1,
        over_generate_x=1.0,
        gram_anchor=True,
    )
    config = dataclasses.replace(
        base,
        families={"composite_update_log_recommend": three},
        max_intents=2,
        output_dir=str(tmp_path / "out"),
    )
    build(
        config,
        expander=primary_fail,
        stop_after="author",
        output_dir=tmp_path / "out",
    )
    assert calls["n"] >= 1
    assert QWEN_MAX_MODEL == "qwen3.8-max"
