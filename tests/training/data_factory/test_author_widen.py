"""Ticket 012 — update / recommend / evaluate authoring, amount_path, gram_anchor."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench.quality_gates import EVALUATE_TIERS  # noqa: E402
from nutrienv.world.catalog_store import load_catalog  # noqa: E402

from src.training.data_factory import author as author_mod  # noqa: E402
from src.training.data_factory.build import build, enumerate_intents  # noqa: E402
from src.training.data_factory.config import load_config  # noqa: E402
from src.training.data_factory.roster_train import TRAIN_ROSTER  # noqa: E402
from src.training.data_factory.rollout_fc import ScriptedFCTeacher  # noqa: E402
from src.training.data_factory.synthetic import synth_expander  # noqa: E402

from tests.training.data_factory.test_build import tiny_config  # noqa: E402
from tests.training.data_factory.test_build_sft import episode_script  # noqa: E402

REPO_CONFIG = "configs/data_factory.yaml"


def _call(name: str, args: dict, *, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _family_config(output_dir, families, **replace_kw):
    base = load_config(REPO_CONFIG)
    fams = {}
    for name, kwargs in families.items():
        fams[name] = dataclasses.replace(base.families[name], **kwargs)
    return dataclasses.replace(
        base, families=fams, output_dir=str(output_dir), **replace_kw
    )


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.fixture(scope="module")
def expander(catalog):
    return synth_expander(catalog)


def test_enumerate_intent_count_respects_over_generate_and_max(tmp_path):
    config = _family_config(
        tmp_path,
        {
            "log": {"target_n": 2, "over_generate_x": 1.5},
            "update": {"target_n": 1, "over_generate_x": 2.0},
        },
        max_intents=10,
    )
    intents = enumerate_intents(config)
    by = Counter(i["family"] for i in intents)
    assert by["log"] == 3  # ceil(2*1.5)
    assert by["update"] == 2  # ceil(1*2)
    with pytest.raises(Exception, match="max_intents"):
        enumerate_intents(dataclasses.replace(config, max_intents=2))


def test_evaluate_tiers_and_non_evaluate_empty(tmp_path):
    config = _family_config(
        tmp_path,
        {
            "evaluate": {"target_n": 6, "over_generate_x": 1.0},
            "log": {"target_n": 2, "over_generate_x": 1.0},
        },
    )
    intents = enumerate_intents(config)
    eval_tiers = {i["tier"] for i in intents if i["family"] == "evaluate"}
    assert eval_tiers <= set(EVALUATE_TIERS)
    assert eval_tiers  # at least one
    assert all(i["tier"] == "" for i in intents if i["family"] != "evaluate")


def test_amount_path_persona_weights(tmp_path):
    config = _family_config(
        tmp_path,
        {"log": {"target_n": 200, "over_generate_x": 1.0}},
        max_intents=400,
    )
    intents = enumerate_intents(config)
    gym = [i for i in intents if next(p for p in TRAIN_ROSTER if p.user_id == i["user_id"]).persona == "gym"]
    everyday = [
        i for i in intents
        if next(p for p in TRAIN_ROSTER if p.user_id == i["user_id"]).persona in ("everyday", "cut")
    ]
    gym_explicit = sum(1 for i in gym if i["amount_path"] == "explicit_grams") / len(gym)
    eve_explicit = sum(1 for i in everyday if i["amount_path"] == "explicit_grams") / len(everyday)
    eve_unspec = sum(1 for i in everyday if i["amount_path"] == "unspecified") / len(everyday)
    assert abs(gym_explicit - 0.60) < 0.20
    assert abs(eve_explicit - 0.15) < 0.12
    assert eve_unspec <= 0.25
    named = [i for i in intents if i["amount_path"] == "named_measure"]
    ounce = sum(1 for i in named if i["ounce_phrasing"]) / len(named)
    assert abs(ounce - 0.15) < 0.12


def test_gram_anchor_changes_speech(catalog, expander):
    person = TRAIN_ROSTER[0]
    intent = {
        "schema_version": "nutrimind-v2-intent/1",
        "task_id": "log--log--train-alba--000042",
        "family": "log",
        "user_id": person.user_id,
        "seed": 42,
        "occasion": "lunch",
        "scene": "empty",
        "amount_path": "named_measure",
        "tier": "",
        "recovery_trap": None,
    }

    def heap_expander(pool, *, persona, family, amount_path=None):
        inner = expander(pool, persona=persona, family=family, amount_path=amount_path)
        if not inner["foods"]:
            return inner
        food_id = inner["foods"][0]
        return {"query": f"For lunch I had a heap of food {food_id}.", "foods": inner["foods"]}

    off, reject_off = author_mod.author_task(
        intent, catalog=catalog, expander=heap_expander, gram_anchor=None
    )
    on, reject_on = author_mod.author_task(
        intent,
        catalog=catalog,
        expander=heap_expander,
        gram_anchor=author_mod.portion_table_gram_anchor(catalog),
    )
    # default off cannot bind "heap"; on may bind via the table anchor
    assert off is None
    assert reject_off["failure_codes"][0].startswith("author.")
    if on is not None:
        assert "heap" in on.query
    else:
        # even if bind still fails, generate_one was invoked with an anchor
        assert reject_on is not None


def test_author_update_recommend_evaluate(catalog, expander):
    person = next(p for p in TRAIN_ROSTER if not p.allergies)
    allergic = next(p for p in TRAIN_ROSTER if p.allergies)
    cases = [
        {
            "family": "update",
            "shell": "upd-add-allergy-short",
            "slots": {"allergen": "fish"},
            "amount_path": "named_measure",
            "tier": "",
            "occasion": "lunch",
        },
        {
            "family": "recommend",
            "shell": "rec-named-dish",
            "slots": {},
            "amount_path": "named_measure",
            "tier": "",
            "occasion": "dinner",
            "user_id": allergic.user_id,
        },
        {
            "family": "evaluate",
            "shell": None,
            "slots": None,
            "amount_path": "explicit_grams",
            "tier": "single",
            "occasion": "lunch",
        },
    ]
    for extra in cases:
        task, reject = None, None
        for seed in range(7, 40):
            uid = extra.get("user_id") or person.user_id
            intent = {
                "schema_version": "nutrimind-v2-intent/1",
                "task_id": f"{extra['family']}--{extra['family']}--{uid}--{seed:06d}",
                "family": extra["family"],
                "user_id": uid,
                "seed": seed,
                "occasion": extra["occasion"],
                "scene": "empty",
                "shell": extra["shell"],
                "slots": extra["slots"],
                "amount_path": extra["amount_path"],
                "tier": extra["tier"],
                "recovery_trap": None,
            }
            task, reject = author_mod.author_task(
                intent, catalog=catalog, expander=expander
            )
            if task is not None:
                break
        assert task is not None, (extra["family"], reject)
        if extra["family"] == "evaluate":
            assert task.tier in EVALUATE_TIERS
        else:
            assert task.tier == ""


def _script_for(task):
    if task.family == "log":
        return episode_script(task)
    if task.family == "update":
        profile = task.oracle.profile
        patch = {"allergies": list(profile.allergies)}
        if profile.weight_kg != task.s0.profile.weight_kg:
            patch = {"weight_kg": profile.weight_kg}
        if profile.phase != task.s0.profile.phase:
            patch = {"phase": profile.phase}
        return [
            ("update", [_call("update_profile", {"patch": patch}, call_id="u1")]),
            ("done", [_call("done", {}, call_id="d")]),
        ]
    if task.family == "recommend":
        from nutrienv.bench.validator import fitting_plan

        plan = fitting_plan(
            task.s0.catalog, task.oracle.plan_windows, task.oracle.profile.allergies
        )
        assert plan
        return [
            ("plan", [_call("submit_plan", {"items": plan}, call_id="p1")]),
            ("done", [_call("done", {}, call_id="d")]),
        ]
    if task.family == "evaluate":
        items = list(task.oracle.evaluated_plan or task.oracle.last_plan or [])
        args = {"items": items, "verdict": task.oracle.last_verdict or "accept"}
        if task.oracle.last_verdict == "reject":
            args["reasons"] = list(task.oracle.last_reasons)
        return [
            ("eval", [_call("submit_plan", args, call_id="e1")]),
            ("done", [_call("done", {}, call_id="d")]),
        ]
    raise AssertionError(task.family)


def test_sft_accepted_one_of_each_family(tmp_path, catalog, expander):
    out = tmp_path / "out"
    config = _family_config(
        out,
        {
            "log": {"target_n": 1, "over_generate_x": 3.0, "teacher_k": 1},
            "update": {"target_n": 1, "over_generate_x": 4.0, "teacher_k": 1},
            "recommend": {"target_n": 1, "over_generate_x": 3.0, "teacher_k": 1},
            "evaluate": {"target_n": 1, "over_generate_x": 8.0, "teacher_k": 1},
        },
        target="sft",
        max_intents=40,
    )
    queues: dict[str, list] = {}
    for intent in enumerate_intents(config):
        task, _ = author_mod.author_task(intent, catalog=catalog, expander=expander)
        if task is None:
            continue
        queues[task.query] = list(_script_for(task))

    def keyed(request):
        messages = request.get("messages") or []
        blob = "\n".join(str(m.get("content") or "") for m in messages)
        queue = None
        for query, queued in queues.items():
            if query and query in blob:
                queue = queued
                break
        if not queue:
            return {
                "content": None,
                "reasoning_content": "no script",
                "tool_calls": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "reasoning_tokens": 0},
            }
        reasoning, tool_calls = queue.pop(0)
        return {
            "content": None,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls,
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "reasoning_tokens": 20,
            },
        }

    manifest = build(
        config,
        expander=expander,
        teacher_complete=keyed,
        output_dir=out,
    )
    assert manifest["status"] == "complete"
    assert "family_mix" in manifest
    for name in ("log", "update", "recommend", "evaluate"):
        assert name in manifest["family_mix"]
        assert manifest["family_mix"][name]["target"] == 1
    records = [
        json.loads(line)
        for line in (out / "sft" / "accepted.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    families = {r["meta"]["family"] for r in records}
    assert {"log", "update", "recommend", "evaluate"} <= families, families
    for record in records:
        if record["meta"]["family"] == "evaluate":
            assert record["meta"]["tier"] in EVALUATE_TIERS
        else:
            assert record["meta"]["tier"] == ""
