"""2-leg composite family: task-key spec, author, gate/replay, brief expander."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench import Scorer, check_achievable, load_exam  # noqa: E402
from nutrienv.bench.pipeline.review_harness import stage_a_code_gate  # noqa: E402
from nutrienv.bench.realize import scored_oracles  # noqa: E402
from nutrienv.bench.validator import fitting_plan, semantic_key, validate_draft  # noqa: E402
from nutrienv.env import NutriEnv  # noqa: E402
from nutrienv.world.catalog_store import load_catalog  # noqa: E402

from src.training.data_factory import author as author_mod  # noqa: E402
from src.training.data_factory import gates as gates_mod  # noqa: E402
from src.training.data_factory.author import TWO_LEG_COMPOSITE_STEPS, author_task  # noqa: E402
from src.training.data_factory.build import FAMILY_SPECS, enumerate_intents  # noqa: E402
from src.training.data_factory.consistency import foods_from_task  # noqa: E402
from src.training.data_factory.config import load_config  # noqa: E402
from src.training.data_factory.roster_train import TRAIN_ROSTER  # noqa: E402
from src.training.data_factory.speech import (  # noqa: E402
    bind_speech_context,
    build_semantic_brief,
    make_brief_expander,
    render_semantic_brief,
)
from src.training.data_factory.synthetic import synth_expander  # noqa: E402

from tests.training.data_factory.test_build import tiny_config  # noqa: E402

_KNOWN_DRAFT_FP = ["update oracle ledger is missing"]
_LOG_REC = ("log", "recommend")
_UPD_REC = ("update", "recommend")


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.fixture(scope="module")
def expander(catalog):
    return synth_expander(catalog)


@pytest.fixture(scope="module")
def exam_ctx():
    return gates_mod.GateContext.from_exam(load_exam())


def _no_allergy_person():
    return next(person for person in TRAIN_ROSTER if not person.allergies)


def _two_leg_intent(person, *, steps, seed=5, amount_path="explicit_grams"):
    task_key = f"composite--{'+'.join(steps)}--{person.user_id}"
    update = steps == _UPD_REC
    return {
        "schema_version": "nutrimind-v2-intent/1",
        "task_id": f"{task_key}--{seed:06d}",
        "task_key": task_key,
        "family": "composite",
        "task_family": "composite",
        "steps": list(steps),
        "user_id": person.user_id,
        "seed": seed,
        "occasion": "lunch",
        "scene": "empty",
        "shell": "upd-add-allergy-short" if update else None,
        "slots": {"allergen": "fish"} if update else None,
        "amount_path": amount_path,
        "ounce_phrasing": False,
        "knife": None,
        "tier": "",
        "recovery_trap": None,
        "gram_anchor": False,
    }


def _author_first(catalog, expander, *, steps, person=None):
    person = person or _no_allergy_person()
    last_reject = None
    for seed in range(5, 40):
        for amount_path in ("explicit_grams", "named_measure"):
            intent = _two_leg_intent(
                person, steps=steps, seed=seed, amount_path=amount_path
            )
            task, reject = author_task(
                intent,
                catalog=catalog,
                expander=expander,
                gram_anchor=author_mod.portion_table_gram_anchor(catalog),
            )
            if task is not None:
                return task, intent
            last_reject = reject
    raise AssertionError(f"could not author 2-leg {steps}: {last_reject}")


def _replay_two_leg(task):
    subs = scored_oracles(task.oracle)
    env = NutriEnv()
    env.reset(task.s0)
    s0 = task.s0.profile
    logged = False
    for sub in subs:
        if sub.profile is not None:
            patch = {}
            if tuple(sub.profile.allergies) != tuple(s0.allergies):
                patch["allergies"] = list(sub.profile.allergies)
            if sub.profile.weight_kg != s0.weight_kg:
                patch["weight_kg"] = sub.profile.weight_kg
            if sub.profile.phase != s0.phase:
                patch["phase"] = sub.profile.phase
            if patch:
                env.step({"op": "update_profile", "patch": patch})
        # recommend child copies the log tail; apply it once
        if sub.ledger_tail and not logged:
            for row in sub.ledger_tail:
                env.step(
                    {
                        "op": "log_meal",
                        "food_id": row.food_id,
                        "grams": row.grams,
                        "eaten_at": row.eaten_at,
                    }
                )
            logged = True
        if sub.plan_windows:
            allergies = (sub.profile or s0).allergies
            plan = fitting_plan(task.s0.catalog, sub.plan_windows, allergies)
            if plan:
                env.step({"op": "submit_plan", "items": plan})
    return Scorer().score(env.state(), task.oracle)


def test_shipped_config_has_composite_task_key_spec():
    shipped = load_config("configs/data_factory.yaml")
    assert "composite" in FAMILY_SPECS
    assert set(shipped.families) <= set(FAMILY_SPECS)
    task_family, default_steps = FAMILY_SPECS["composite"]
    assert task_family == "composite"
    assert default_steps == _LOG_REC
    assert TWO_LEG_COMPOSITE_STEPS == (_LOG_REC, _UPD_REC)


def test_enumerate_two_leg_task_key_and_id_and_mix(tmp_path):
    base = load_config("configs/data_factory.yaml")
    family = dataclasses.replace(
        base.families["composite"], target_n=10, over_generate_x=1.0
    )
    config = dataclasses.replace(
        base,
        families={"composite": family},
        max_intents=20,
        output_dir=str(tmp_path / "out"),
    )
    intents = enumerate_intents(config)
    assert len(intents) == 10
    assert [row["task_id"] for row in intents] == sorted(row["task_id"] for row in intents)
    by_steps = Counter(tuple(row["steps"]) for row in intents)
    assert by_steps[_LOG_REC] == 5
    assert by_steps[_UPD_REC] == 5
    for row in intents:
        steps = tuple(row["steps"])
        assert row["task_family"] == "composite"
        assert row["family"] == "composite"
        assert row["tier"] == ""
        expected_key = f"composite--{'+'.join(steps)}--{row['user_id']}"
        assert row["task_key"] == expected_key
        assert row["task_id"] == f"{expected_key}--{row['seed']:06d}"
        if steps == _UPD_REC:
            assert row["shell"]
        else:
            assert row["shell"] is None


def test_enumerate_shipped_config_no_longer_aborts_on_composite():
    shipped = load_config("configs/data_factory.yaml")
    intents = enumerate_intents(shipped)
    families = {row["family"] for row in intents}
    assert "composite" in families
    assert "composite_update_log_recommend" in families
    two_leg = [row for row in intents if row["family"] == "composite"]
    assert two_leg
    assert {tuple(row["steps"]) for row in two_leg} == set(TWO_LEG_COMPOSITE_STEPS)


@pytest.mark.parametrize("steps", [_LOG_REC, _UPD_REC], ids=["log_rec", "upd_rec"])
def test_author_two_leg_gates_and_replays(catalog, expander, exam_ctx, steps):
    task, intent = _author_first(catalog, expander, steps=steps)
    assert task.family == "composite"
    assert task.id == intent["task_id"]
    assert task.tier == ""
    assert task.oracle.sub_oracles is not None
    assert len(task.oracle.sub_oracles) == 2
    update_sub = task.oracle.sub_oracles[0]
    if steps == _UPD_REC:
        assert update_sub.ledger is None
        assert update_sub.ledger_tail is None
    else:
        foods = foods_from_task(task)
        assert foods
        entry = catalog.get(foods[0]) or {}
        handles = [str(entry.get("name") or "").split(",")[0].strip().lower()]
        handles.extend(str(alias).lower() for alias in entry.get("aliases") or [])
        blob = task.query.lower()
        assert any(handle and handle in blob for handle in handles)

    stage_a = stage_a_code_gate(task)
    draft = validate_draft(task)
    assert stage_a == []
    assert draft in ([], _KNOWN_DRAFT_FP)
    assert task.id not in check_achievable([task]).unreachable

    result = gates_mod.run(task, exam_ctx)
    assert result.keep is True, result

    score = _replay_two_leg(task)
    assert score["passed"] is True, score
    assert score.get("sub_tags") == ("pass", "pass")


def test_gate_still_flags_real_draft_invalid(catalog, expander, exam_ctx):
    task, _ = _author_first(catalog, expander, steps=_LOG_REC)
    bad = dataclasses.replace(task, query="Please log food_id 99999 then recommend dinner.")
    result = gates_mod.run(bad, exam_ctx)
    assert result.keep is False
    assert result.failure_code == "gate.draft_invalid"


def test_semantic_key_dedup_keeps_lowest_seed(catalog, expander):
    person = _no_allergy_person()
    tasks = []
    for seed in (8, 9):
        task, reject = author_task(
            _two_leg_intent(person, steps=_LOG_REC, seed=seed),
            catalog=catalog,
            expander=expander,
            gram_anchor=author_mod.portion_table_gram_anchor(catalog),
        )
        if task is not None:
            tasks.append(task)
    if len(tasks) < 2:
        pytest.skip("could not author two 2-leg log→recommend tasks")
    kept: dict = {}
    for task in sorted(tasks, key=lambda item: item.id):
        kept.setdefault(semantic_key(task), task)
    for key, task in kept.items():
        twins = [item.id for item in tasks if semantic_key(item) == key]
        assert task.id == min(twins)


def test_illegal_pair_rejects_without_raising(catalog, expander):
    person = _no_allergy_person()
    intent = _two_leg_intent(person, steps=("evaluate", "recommend"), seed=1)
    task, reject = author_task(intent, catalog=catalog, expander=expander)
    assert task is None
    assert reject["failure_codes"] == ["author.illegal_pair"]
    assert reject["status"] == "dropped"


def test_author_failure_is_isolated_in_build(tmp_path, catalog):
    from src.training.data_factory.build import build

    def fail_expander(pool, *, persona, family, amount_path=None):
        return {"query": "", "foods": []}

    base = load_config("configs/data_factory.yaml")
    family = dataclasses.replace(
        base.families["composite"], target_n=2, over_generate_x=1.0
    )
    log_family = dataclasses.replace(
        base.families["log"], target_n=1, over_generate_x=1.0
    )
    config = dataclasses.replace(
        base,
        families={"composite": family, "log": log_family},
        max_intents=10,
        output_dir=str(tmp_path / "out"),
    )
    manifest = build(
        config,
        expander=fail_expander,
        stop_after="author",
        output_dir=tmp_path / "out",
    )
    assert manifest["status"] == "complete"
    assert manifest["counts"]["rejected"]["author"] >= 1


def test_brief_expander_composite_uses_semantic_brief_and_overrides_foods(catalog):
    captured: list = []

    def complete(_model_id, messages):
        captured.append(messages)
        return json.dumps(
            {
                "query": "For lunch I had a bowl of the dish. What's for dinner?",
                "foods": ["HACKED"],
            }
        )

    expander = make_brief_expander(complete=complete, catalog=catalog)
    from nutrienv.bench.pipeline.sampler import sample_pools

    pool = sample_pools(catalog, seed=1, family="log", n_pools=1, pool_size=8)[0]
    bound = bind_speech_context(expander, {"occasion": "lunch", "scene": "empty"})
    out = bound(pool, persona="everyday", family="composite", amount_path="named_measure")
    assert captured
    blob = "\n".join(message["content"] for message in captured[0]).lower()
    assert "food_id" not in blob
    assert "allergen_tags" not in blob
    assert "what's for dinner" in blob or "what to eat next" in blob
    assert "logging" in blob or "log" in blob
    brief = build_semantic_brief(
        pool,
        catalog=catalog,
        persona="everyday",
        family="composite",
        amount_path="named_measure",
        occasion="lunch",
        scene="empty",
    )
    assert brief is not None
    assert "what's for" in render_semantic_brief(brief).lower()
    assert out["foods"] == [brief.food_id]
    assert "HACKED" not in out["foods"]


def test_tiny_log_config_still_enumerates(tmp_path):
    intents = enumerate_intents(tiny_config(tmp_path))
    assert all(row["family"] == "log" for row in intents)
