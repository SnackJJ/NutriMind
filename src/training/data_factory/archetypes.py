"""Training authors for NutriEnv ADR 0029 query archetypes (batch 2).

The v1.1 exam mixes in archetypes the batch-1 mill never authored. These
strategies author *training* tasks of four of them from lab primitives only
(``WorldState.allowed_food_ids``, ``Oracle``, ``compose_oracles``,
``plan_windows_for_meal``, ``fitting_plan``) — the same derivations the lab's
own minting uses, never its items:

* ``recommend_inventory`` — ADR 0029 §3.1: a closed home inventory, plan a meal
  from it only (an allergen food may sit in the inventory as a trap).
* ``recommend_menu`` — §4 item 6: a closed shop/canteen menu, same mechanics.
* ``composite_amend_recommend`` — §3.4: an earlier ledger row is wrong; amend
  it in place, then recommend on the corrected remainder.
* ``composite_refuse_recommend`` — §3.6: the user asks for an unsafe daily
  calorie target; the profile must stay as it is, plus a safe meal.

Counterfactual Evaluate (§3.5) is the ``evaluate_hypo`` family in
``author.py``. Grocery allocation (§3.2) and recipe deconstruction (§3.3) are
held out on purpose: they measure whether SFT transfers to archetypes it never
saw.

Isolation from the exam: people come from ``TRAIN_ROSTER``, foods from the
seeded mill pools, wording from the templates below (written from the ADR's
prose, not from exam items). ``gates.py`` still rejects any task whose query
or food set lands near an exam item.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping

from nutrienv.bench.pipeline.roster import profile_for
from nutrienv.bench.pipeline.sampler import sample_pools, spoken_display_name
from nutrienv.bench.realize import Oracle, Task, compose_oracles
from nutrienv.bench.validator import fitting_plan
from nutrienv.world.daily_windows import plan_windows_for_meal
from nutrienv.world.types import LedgerRow, WorldState, ledger_totals

__all__ = ["ARCHETYPE_STRATEGIES"]

_POOL_SIZE = 80
_SUBSET_TRIES = 24

_INVENTORY_TEMPLATES = (
    "Here's what's in my kitchen right now: {foods}. Can you put together a {occasion} "
    "from just those?",
    "I only have {foods} at home and I'm not going shopping. What {occasion} can I make "
    "out of that?",
    "Pantry check: {foods}. Build me a {occasion} that uses nothing else.",
)
_MENU_TEMPLATES = (
    "I'm stuck at a gas station shop that only sells {foods}. What should I buy for "
    "{occasion}?",
    "The work canteen has {foods} today and that's all. Pick my {occasion} off that menu.",
    "Airport kiosk options: {foods}. Choose a {occasion} for me from those items only.",
)
# The recommend occasion is read from the first "for <meal>" (nutrienv
# bench.occasions), so the earlier slot is never spoken as "for <slot>".
_AMEND_TEMPLATES = (
    "I made a mistake earlier: my {slot} entry says {wrong}, but I actually had {right}. "
    "Please correct it, then suggest what I should eat for {occasion}.",
    "Fix my {slot} log — it should be {right}, not {wrong}. After that, what's good "
    "for {occasion}?",
    "At {slot} I had {right} rather than {wrong}, so that row is wrong. Update it and "
    "then plan something for {occasion}.",
)
_REFUSE_TEMPLATES = (
    "I overate yesterday, so set my daily calories to {kcal} for the rest of the week. "
    "Also, what should I have for {occasion}?",
    "Drop my calorie target to {kcal} a day — I want fast results. And give me an "
    "idea for {occasion}.",
    "Let's go extreme: {kcal} calories per day from now on. What do I eat for {occasion}?",
)
_UNSAFE_KCAL = (400, 450, 500, 600, 650, 700)
# Below this the lab's own refuse items are not minted (a person whose real
# floor is near the asked target makes "refuse" ambiguous).
_SAFE_FLOOR_KCAL = 1200
_MULTIPLES = (0.5, 1.0, 1.5, 2.0)
_EARLIER_SLOTS = ("breakfast", "lunch")


def _reject(intent: Mapping, code: str, detail: str):
    from src.training.data_factory.author import _reject as author_reject

    return None, author_reject(intent, code, detail)


def _person(intent: Mapping):
    from src.training.data_factory.author import person_for_intent

    return person_for_intent(intent)


# WWEIA categories nobody keeps in a fridge or buys off a canteen counter:
# infant formula / baby food (90xx-94xx), alcohol (75xx), uncategorized (9999).
_SKIP_CATEGORY_PREFIXES = ("90", "91", "92", "93", "94", "75", "9999")
_JUNK_WORDS = {"nfs", "ns", "or", "for", "use", "made", "from", "including", "excluding"}


def _speakable(catalog, food) -> bool:
    """A food a person would name in a sentence: an ordinary category and a
    short display name without catalog qualifiers ("nfs", "ns as to", "or")."""
    entry = catalog.get(food.food_id) or {}
    category = str(entry.get("category") or "")
    if category.startswith(_SKIP_CATEGORY_PREFIXES):
        return False
    words = spoken_display_name(catalog, food.food_id).split()
    return (len(words) <= 4 and not set(words) & _JUNK_WORDS
            and not any(ch.isdigit() or ch in "/;(" for ch in "".join(words)))


def _pool(catalog, seed: int, *, with_allergen: str | None = None):
    pools = sample_pools(
        catalog, seed=seed, family="recommend", n_pools=1, pool_size=_POOL_SIZE,
        spoken_only=True, with_allergen=with_allergen,
    )
    return tuple(food for food in (pools[0].foods if pools else ())
                 if _speakable(catalog, food))


def _speak_list(catalog, food_ids) -> str:
    names = [spoken_display_name(catalog, food_id) for food_id in food_ids]
    return ", ".join(names[:-1]) + ", and " + names[-1] if len(names) > 1 else names[0]


def _windows(profile, eaten, occasion: str):
    return plan_windows_for_meal(profile.windows, eaten, occasion)


def _recommend_oracle(profile, windows, ledger, allowed):
    return Oracle(
        profile=copy.deepcopy(profile),
        last_plan=[],
        plan_must_be_safe=True,
        plan_must_fit_windows=True,
        plan_windows=windows,
        ledger=tuple(ledger),
        allowed_food_ids=allowed,
    )


def _closed_inventory(intent: Mapping, *, catalog, templates, size_range):
    """Recommend from a closed inventory: a seeded pool subset that admits a plan."""
    person = _person(intent)
    profile = profile_for(person)
    seed = int(intent["seed"])
    rng = random.Random(f"inventory:{seed}")
    occasion = intent["occasion"] if intent["occasion"] != "snack" else "lunch"
    windows = _windows(profile, {}, occasion)
    if windows is None:
        return _reject(intent, "author.empty_windows", f"no {occasion} windows")
    trap = rng.choice(sorted(profile.allergies)) if profile.allergies and rng.random() < 0.4 else None
    foods = _pool(catalog, seed, with_allergen=trap)
    if not foods:
        return _reject(intent, "author.empty_pool", "no pool")
    trap_ids = [food.food_id for food in foods if trap and trap in food.allergen_tags]
    safe_ids = [food.food_id for food in foods if food.food_id not in trap_ids]
    for _ in range(_SUBSET_TRIES):
        size = rng.randint(*size_range)
        chosen = rng.sample(safe_ids, min(size - bool(trap_ids), len(safe_ids)))
        if trap_ids:
            chosen.append(rng.choice(trap_ids))
        rng.shuffle(chosen)
        allowed = frozenset(chosen)
        if fitting_plan(catalog, dict(windows), profile.allergies, allowed_food_ids=allowed):
            break
    else:
        return _reject(intent, "author.unachievable_inventory",
                       f"no fitting plan in {_SUBSET_TRIES} inventories")
    query = templates[seed % len(templates)].format(
        foods=_speak_list(catalog, chosen), occasion=occasion)
    s0 = WorldState(profile=profile, ledger=[], catalog=catalog, allowed_food_ids=allowed)
    oracle = _recommend_oracle(profile, dict(windows), (), allowed)
    return Task(f"arch-inv-{seed:06d}", "recommend", query, s0, oracle, (),
                person.persona), None


def _author_inventory(intent, *, catalog, expander=None, gram_anchor=None, **_):
    return _closed_inventory(intent, catalog=catalog, templates=_INVENTORY_TEMPLATES,
                             size_range=(5, 8))


def _author_menu(intent, *, catalog, expander=None, gram_anchor=None, **_):
    return _closed_inventory(intent, catalog=catalog, templates=_MENU_TEMPLATES,
                             size_range=(6, 10))


def _portion_rows(catalog, food_id: str):
    """(grams, spoken phrase) for each clean household amount of ``food_id``."""
    from src.training.data_factory.author import _format_evaluate_food_phrase

    rows = []
    portions = (catalog.get(food_id) or {}).get("portions") or {}
    for unit, base in sorted(portions.items()):
        try:
            base = float(base)
        except (TypeError, ValueError):
            continue
        if base <= 0 or unit == "qns":
            continue
        for multiple in _MULTIPLES:
            grams = round(base * multiple, 2)
            phrase = _format_evaluate_food_phrase(food_id, grams, "named_measure", catalog)
            if not phrase.endswith(f"g of {catalog[food_id]['name']}"):  # a household amount
                rows.append((grams, phrase.replace(catalog[food_id]["name"],
                                                   spoken_display_name(catalog, food_id))))
    return rows


def _author_amend(intent, *, catalog, expander=None, gram_anchor=None, **_):
    """One earlier row is wrong (food or amount); amend it, then recommend."""
    person = _person(intent)
    profile = profile_for(person)
    seed = int(intent["seed"])
    rng = random.Random(f"amend:{seed}")
    occasion = "dinner"
    slots = _EARLIER_SLOTS[: rng.randint(1, 2)]
    foods = [food for food in _pool(catalog, seed)
             if not set(food.allergen_tags) & set(profile.allergies)]
    priced = [(food.food_id, _portion_rows(catalog, food.food_id)) for food in foods]
    priced = [(food_id, rows) for food_id, rows in priced if rows]
    if len(priced) < len(slots) + 1:
        return _reject(intent, "author.empty_pool", "too few portioned foods")
    rng.shuffle(priced)
    ledger = []
    for slot, (food_id, rows) in zip(slots, priced):
        grams, _phrase = rng.choice(rows)
        ledger.append(LedgerRow(food_id, grams, f"today-{slot}"))
    index = rng.randrange(len(ledger))
    wrong = ledger[index]
    wrong_rows = dict(priced)[wrong.food_id]
    wrong_phrase = next(phrase for grams, phrase in wrong_rows if grams == wrong.grams)
    other_amounts = [(g, p) for g, p in wrong_rows if abs(g - wrong.grams) > 1e-6]
    if other_amounts and rng.random() < 0.5:   # amount was wrong
        grams, right_phrase = rng.choice(other_amounts)
        fixed = LedgerRow(wrong.food_id, grams, wrong.eaten_at)
    else:                                      # food was wrong
        food_id, rows = priced[len(slots)]
        grams, right_phrase = rng.choice(rows)
        fixed = LedgerRow(food_id, grams, wrong.eaten_at)
    corrected = list(ledger)
    corrected[index] = fixed
    windows = _windows(profile, ledger_totals(corrected, catalog), occasion)
    if windows is None or not fitting_plan(catalog, dict(windows), profile.allergies):
        return _reject(intent, "author.unachievable_inventory", "no dinner after amend")
    slot = wrong.eaten_at.removeprefix("today-")
    query = _AMEND_TEMPLATES[seed % len(_AMEND_TEMPLATES)].format(
        slot=slot, wrong=wrong_phrase, right=right_phrase, occasion=occasion)
    s0 = WorldState(profile=profile, ledger=list(ledger), catalog=catalog)
    amend = Oracle(profile=copy.deepcopy(profile), ledger=tuple(corrected))
    recommend = _recommend_oracle(profile, dict(windows), corrected, None)
    return Task(f"arch-amend-{seed:06d}", "composite", query, s0,
                compose_oracles(amend, recommend), (), person.persona), None


def _author_refuse(intent, *, catalog, expander=None, gram_anchor=None, **_):
    """An unsafe calorie target is asked for: keep the profile, recommend a meal."""
    person = _person(intent)
    profile = profile_for(person)
    seed = int(intent["seed"])
    rng = random.Random(f"refuse:{seed}")
    if profile.windows["kcal"][0] < _SAFE_FLOOR_KCAL:
        return _reject(intent, "author.unsafe_floor",
                       f"daily kcal floor {profile.windows['kcal'][0]} < {_SAFE_FLOOR_KCAL}")
    occasion = intent["occasion"] if intent["occasion"] != "snack" else "dinner"
    windows = _windows(profile, {}, occasion)
    if windows is None or not fitting_plan(catalog, dict(windows), profile.allergies):
        return _reject(intent, "author.unachievable_inventory", f"no {occasion} plan")
    query = _REFUSE_TEMPLATES[seed % len(_REFUSE_TEMPLATES)].format(
        kcal=rng.choice(_UNSAFE_KCAL), occasion=occasion)
    s0 = WorldState(profile=profile, ledger=[], catalog=catalog)
    hold = Oracle(profile=copy.deepcopy(profile), ledger=())
    recommend = _recommend_oracle(profile, dict(windows), (), None)
    return Task(f"arch-refuse-{seed:06d}", "composite", query, s0,
                compose_oracles(hold, recommend), (), person.persona), None


ARCHETYPE_STRATEGIES = {
    "recommend_inventory": _author_inventory,
    "recommend_menu": _author_menu,
    "composite_amend_recommend": _author_amend,
    "composite_refuse_recommend": _author_refuse,
}
