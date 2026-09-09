"""Ticket 002 Part B / spec OQ-7 — assemble `update+log->recommend` (3-leg) from
PUBLIC nutrienv symbols only.

Reference: the frozen v1.0 exam item ``adr24-comp-8255`` has the same 3-sub-oracle
shape (update: profile only; log: ledger + ledger_tail + profile; recommend:
last_plan=[] + plan_windows + profile). It *also* trips
``validate_draft`` with ``"update oracle ledger is missing"`` — so for this shape
that single issue is a known false-positive; the authoritative gates are
``stage_a_code_gate`` + ``check_achievable`` + a correct-replay ``Scorer`` Pass.

Only these public symbols are used:
``generate_one``, ``plan_windows_for_meal``, ``ledger_totals``, ``Oracle``,
``Task``, ``compose_oracles``, ``recommend_query``, ``speakable_tracer_food``.
No ``_update_from_template`` / ``_bind_log_foods``.
"""

from __future__ import annotations

import copy
import dataclasses

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench import Scorer, check_achievable  # noqa: E402
from nutrienv.bench.pipeline.generate_one import generate_one  # noqa: E402
from nutrienv.bench.pipeline.review_harness import stage_a_code_gate  # noqa: E402
from nutrienv.bench.pipeline.roster import ROSTER  # noqa: E402
from nutrienv.bench.pipeline.sampler import speakable_tracer_food  # noqa: E402
from nutrienv.bench.pipeline.templates import recommend_query  # noqa: E402
from nutrienv.bench.realize import Oracle, Task, compose_oracles, scored_oracles  # noqa: E402
from nutrienv.bench.validator import fitting_plan, validate_draft  # noqa: E402
from nutrienv.env import NutriEnv  # noqa: E402
from nutrienv.world.catalog_store import GOLD_CATALOG_PATH, load_catalog  # noqa: E402
from nutrienv.world.daily_windows import plan_windows_for_meal  # noqa: E402
from nutrienv.world.types import ledger_totals  # noqa: E402

_KNOWN_DRAFT_FP = ["update oracle ledger is missing"]  # v1.0 adr24-comp-8255 has it too
_ALLERGENS = ["soy", "milk", "egg", "peanut", "wheat", "fish"]


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(GOLD_CATALOG_PATH)


def _synth_expander(catalog):
    def expander(pool, *, persona, family, amount_path=None):
        picked = speakable_tracer_food(pool, catalog, amount_path=amount_path or "named_measure")
        if picked is None:
            return {"query": "", "foods": []}
        food, phrase, spoken = picked
        return {"query": f"For lunch I had {phrase} of {spoken}.", "foods": [food.food_id]}

    return expander


def _assemble_three_leg(catalog, seed):
    """update+log->recommend, PUBLIC symbols only. Returns (task, None) or (None, reason)."""
    no_allergy = [p for p in ROSTER if not p.allergies]
    person = no_allergy[seed % len(no_allergy)]
    allergen = _ALLERGENS[seed % len(_ALLERGENS)]

    upd = generate_one(
        catalog=catalog, family="update", person=person, seed=seed,
        shell="upd-add-allergy-short", slots={"allergen": allergen},
    )
    if upd.accepted is None:
        return None, f"author.update.{getattr(upd.rejected, 'reason', '?')}"

    log = None
    for amount_path in ("named_measure", "explicit_grams"):
        log = generate_one(
            catalog=catalog, family="log", person=person, seed=seed,
            occasion="lunch", amount_path=amount_path, expander=_synth_expander(catalog),
        )
        if log.accepted is not None:
            break
    if log.accepted is None:
        return None, f"author.log.{getattr(log.rejected, 'reason', '?')}"

    update_task, log_task = upd.accepted, log.accepted
    expected = update_task.oracle.profile  # for upd-add-allergy-short == s0.profile + allergen
    s0 = update_task.s0
    tail = list(log_task.oracle.ledger_tail)
    final_ledger = (*s0.ledger, *tail)
    plan_windows = plan_windows_for_meal(
        expected.windows, ledger_totals(list(final_ledger), catalog), "dinner"
    )
    if plan_windows is None:
        return None, "recwin.empty_windows"

    partial = copy.deepcopy(expected)
    update_sub = dataclasses.replace(update_task.oracle, ledger=None, ledger_tail=None)
    log_sub = Oracle(ledger_tail=list(tail), ledger=final_ledger, profile=copy.deepcopy(partial))
    rec_sub = Oracle(
        profile=copy.deepcopy(partial), last_plan=[], ledger=final_ledger,
        plan_must_be_safe=True, plan_must_fit_windows=True, plan_windows=plan_windows,
    )
    query = (
        f"{update_task.query} {log_task.query} "
        f"{recommend_query('rec-dinner', {'occasion': 'dinner'})}"
    )
    task = Task(
        f"3leg--update+log+recommend--{person.user_id}--{seed:06d}",
        "composite", query, s0,
        compose_oracles(update_sub, log_sub, rec_sub),
        ("multi_item_log",), person.persona, tier="",
    )
    return task, None


def _replay(task, *, skip_update=False, skip_log=False, over_window=False):
    subs = scored_oracles(task.oracle)
    exp = subs[0].profile
    env = NutriEnv()
    env.reset(task.s0)
    if not skip_update:
        env.step({"op": "update_profile", "patch": {"allergies": list(exp.allergies)}})
    if not skip_log:
        for row in subs[1].ledger_tail:
            env.step({"op": "log_meal", "food_id": row.food_id, "grams": row.grams,
                      "eaten_at": row.eaten_at})
    plan = fitting_plan(task.s0.catalog, subs[2].plan_windows, exp.allergies)
    if plan and over_window:
        plan = [{"food_id": it["food_id"], "grams": it["grams"] * 4.0} for it in plan]
    if plan:
        env.step({"op": "submit_plan", "items": plan})
    return env.state()


def _gate_ok(task):
    stage_a = stage_a_code_gate(task)
    draft = validate_draft(task)
    achievable = task.id not in check_achievable([task]).unreachable
    return (not stage_a) and (draft in ([], _KNOWN_DRAFT_FP)) and achievable


def test_public_only_no_private_helper_use():
    import importlib

    gen_mod = importlib.import_module("nutrienv.bench.pipeline.generate_one")
    # sanity: the private helpers exist in nutrienv — we deliberately do not use them
    assert hasattr(gen_mod, "_update_from_template")
    assert hasattr(gen_mod, "_bind_log_foods")

    # the assembly function references only public names; no private nutrienv helper
    used = set(_assemble_three_leg.__code__.co_names)
    for const in _assemble_three_leg.__code__.co_consts:
        if hasattr(const, "co_names"):  # nested code objects (comprehensions)
            used |= set(const.co_names)
    assert "_update_from_template" not in used
    assert "_bind_log_foods" not in used


def test_three_leg_assembles_gates_and_scores(catalog):
    task, reason = _assemble_three_leg(catalog, seed=101)
    assert task is not None, f"assembly failed: {reason}"
    assert len(scored_oracles(task.oracle)) == 3

    assert not stage_a_code_gate(task)
    assert validate_draft(task) in ([], _KNOWN_DRAFT_FP)
    assert task.id not in check_achievable([task]).unreachable
    assert _gate_ok(task)

    good = Scorer().score(_replay(task), task.oracle)
    assert good["passed"] is True
    assert good["sub_tags"] == ("pass", "pass", "pass")


@pytest.mark.parametrize(
    "kw,expected_tag",
    [
        ({"skip_log": True}, "log_miss"),
        ({"over_window": True}, "window"),
        ({"skip_update": True}, "update_miss"),
    ],
)
def test_wrong_end_states_get_the_right_tag(catalog, kw, expected_tag):
    task, reason = _assemble_three_leg(catalog, seed=101)
    assert task is not None, reason
    result = Scorer().score(_replay(task, **kw), task.oracle)
    assert result["passed"] is False
    assert result["tag"] == expected_tag


def test_small_yield_sweep(catalog):
    """~20 seeds: every assembled+gated task must correct-replay to a Pass.

    The absolute yield here is with a crude synthetic tracer expander and is only a
    Part-C sizing input, not an acceptance bar.
    """
    seeds = range(1000, 1020)
    assembled = gated = passed = 0
    for seed in seeds:
        task, _ = _assemble_three_leg(catalog, seed)
        if task is None:
            continue
        assembled += 1
        if not _gate_ok(task):
            continue
        gated += 1
        if Scorer().score(_replay(task), task.oracle)["passed"]:
            passed += 1

    assert gated >= 8, f"only {gated}/20 assembled+gated"
    assert passed == gated, f"{passed}/{gated} gated tasks correct-replayed to Pass"
