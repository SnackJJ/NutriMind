"""Shared offline fixtures for the data-factory seam tests.

Synthetic expander + deterministic ``generate_one`` task factories + the public
3-leg assembly (spec §22.9). No network, no LLM; every helper is seed-pinned.
Adapted from ticket 002's spike tests (``test_three_leg_public_assembly.py`` /
``test_env_reconstruction.py``).
"""

from __future__ import annotations

import copy
import dataclasses

from nutrienv.bench.pipeline.generate_one import generate_one
from nutrienv.bench.pipeline.roster import ROSTER
from nutrienv.bench.pipeline.sampler import speakable_tracer_food
from nutrienv.bench.pipeline.templates import recommend_query
from nutrienv.bench.realize import Oracle, Task, compose_oracles
from nutrienv.bench.validator import fitting_plan
from nutrienv.env import NutriEnv
from nutrienv.world.catalog_store import GOLD_CATALOG_PATH, load_catalog
from nutrienv.world.daily_windows import plan_windows_for_meal
from nutrienv.world.types import ledger_totals

# allergen values that are NOT exam update-slot values (egg/milk/peanut/
# shellfish/tree_nut at rev 203d807) — keeps gate-3 quiet in 3-leg fixtures
SAFE_ALLERGENS = ["fish", "soy", "wheat"]


def synth_expander(catalog):
    """A deterministic stand-in for the speech LLM: picks one pool food and
    speaks a fixed {query, foods} pair (nutri-env's expander contract)."""

    def expander(pool, *, persona, family, amount_path=None):
        picked = speakable_tracer_food(pool, catalog, amount_path=amount_path or "named_measure")
        if picked is None:
            return {"query": "", "foods": []}
        food, phrase, spoken = picked
        return {"query": f"For lunch I had {phrase} of {spoken}.", "foods": [food.food_id]}

    return expander


def first_person(*, require_no_allergies: bool = True):
    people = [p for p in ROSTER if not p.allergies] if require_no_allergies else list(ROSTER)
    return people[0]


def make_log_task(catalog, person, seed: int):
    """First accepted log task over (seed, amount_path) — deterministic."""
    expander = synth_expander(catalog)
    for amount_path in ("named_measure", "explicit_grams"):
        result = generate_one(
            catalog=catalog, family="log", person=person, seed=seed,
            occasion="lunch", amount_path=amount_path, expander=expander,
        )
        if result.accepted is not None:
            return result.accepted
    raise AssertionError(f"no accepted log task for seed={seed} ({result.rejected})")


def make_update_task(catalog, person, seed: int, *, shell: str, slots: dict):
    result = generate_one(
        catalog=catalog, family="update", person=person, seed=seed,
        shell=shell, slots=slots,
    )
    assert result.accepted is not None, f"update rejected: {result.rejected}"
    return result.accepted


def make_recommend_task(catalog, person, seed: int):
    result = generate_one(
        catalog=catalog, family="recommend", person=person, seed=seed,
        occasion="dinner", shell="rec-dinner",
    )
    assert result.accepted is not None, f"recommend rejected: {result.rejected}"
    return result.accepted


def assemble_three_leg(catalog, seed: int, *, allergen: str = "fish"):
    """update+log→recommend from PUBLIC symbols only (spec §22.9).

    Returns (task, None) or (None, reason). Mirrors ticket 002 Part B.
    """
    person = first_person(require_no_allergies=True)

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
            occasion="lunch", amount_path=amount_path,
            expander=synth_expander(catalog),
        )
        if log.accepted is not None:
            break
    if log.accepted is None:
        return None, f"author.log.{getattr(log.rejected, 'reason', '?')}"

    update_task, log_task = upd.accepted, log.accepted
    expected = update_task.oracle.profile
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
    log_sub = Oracle(
        ledger_tail=list(tail), ledger=final_ledger, profile=copy.deepcopy(partial)
    )
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


def replay_three_leg(task, *, skip_update=False, skip_log=False):
    """A correct (or deliberately broken) replay of the 3-leg — returns the
    end state for Scorer assertions."""
    from nutrienv.bench.realize import scored_oracles

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
    if plan:
        env.step({"op": "submit_plan", "items": plan})
    return env.state()


def gold_catalog():
    return load_catalog(GOLD_CATALOG_PATH)
