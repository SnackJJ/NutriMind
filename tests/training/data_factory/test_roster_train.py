"""Ticket 004 — TRAIN_ROSTER: structure, personas, and the train/exam isolation.

The isolation guarantee under test (spec §2, US-16): for every body-derived
nutrient, no TRAIN_ROSTER person's ``derive_profile_windows`` tuple equals any
nutri-env ``ROSTER`` person's tuple — so a template-family oracle authored on
TRAIN_ROSTER cannot collide with the frozen v1.0 exam. The ``train-*`` /
``roster-*`` ``user_id`` prefix split is the other half of that isolation.

Only public nutrienv symbols are used (spec §18): ``ROSTER``, ``RosterPerson``,
``profile_for``, ``derive_profile_windows``, ``generate_one``,
``speakable_tracer_food``, ``load_catalog``, ``Task``.
"""

from __future__ import annotations

from collections import Counter

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench import Task  # noqa: E402
from nutrienv.bench.pipeline.generate_one import generate_one  # noqa: E402
from nutrienv.bench.pipeline.roster import ROSTER, RosterPerson, profile_for  # noqa: E402
from nutrienv.bench.pipeline.sampler import speakable_tracer_food  # noqa: E402
from nutrienv.world.catalog_store import GOLD_CATALOG_PATH, load_catalog  # noqa: E402
from nutrienv.world.daily_windows import derive_profile_windows  # noqa: E402

from src.training.data_factory.roster_train import TRAIN_ROSTER  # noqa: E402

# The five body-derived windows (all distinct per person in both rosters).
# sodium_mg is deliberately NOT in this contract: derive_profile_windows
# returns the constant (0.0, 2300.0) for every person in BOTH rosters, so it
# cannot separate train from exam — a disjointness assertion on it would be
# unsatisfiable and meaningless. See the isolation test, which also proves
# that constancy.
_BODY_DERIVED_KEYS = ("kcal", "protein_g", "carb_g", "fat_g", "fiber_g")

_PERSONAS = ("everyday", "gym", "cut")


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(GOLD_CATALOG_PATH)


def _synth_expander(catalog):
    """Same synthetic tracer pattern as test_three_leg_public_assembly."""

    def expander(pool, *, persona, family, amount_path=None):
        picked = speakable_tracer_food(
            pool, catalog, amount_path=amount_path or "named_measure"
        )
        if picked is None:
            return {"query": "", "foods": []}
        food, phrase, spoken = picked
        return {
            "query": f"For lunch I had {phrase} of {spoken}.",
            "foods": [food.food_id],
        }

    return expander


def _windows_by_id(people):
    """derive_profile_windows(profile_for(p)) for each person, keyed by id."""
    out = {}
    for person in people:
        windows = derive_profile_windows(profile_for(person))
        assert windows is not None, f"{person.user_id}: body facts did not resolve"
        out[person.user_id] = windows
    return out


# --------------------------------------------------------------------------- #
# ticket 004 checkbox 1 — importable TRAIN_ROSTER, train-* ids, RosterPerson
# --------------------------------------------------------------------------- #


def test_train_roster_entries_are_train_prefixed_roster_people():
    assert len(TRAIN_ROSTER) == 20
    ids = [person.user_id for person in TRAIN_ROSTER]
    # the prefix split is the train/exam isolation point: exam ids are roster-*
    assert all(user_id.startswith("train-") for user_id in ids), ids
    assert len(set(ids)) == len(ids), "duplicate TRAIN_ROSTER user_id"
    assert all(isinstance(person, RosterPerson) for person in TRAIN_ROSTER)
    # no id is shared with the exam roster (implied by the prefixes; asserted
    # so a future prefix change cannot silently bridge the two rosters)
    assert not set(ids) & {p.user_id for p in ROSTER}


# --------------------------------------------------------------------------- #
# ticket 004 checkbox 3 — profile_for / persona lookups resolve
# --------------------------------------------------------------------------- #


def test_profile_for_resolves_windows_and_personas():
    for person in TRAIN_ROSTER:
        profile = profile_for(person)
        assert profile.user_id == person.user_id
        windows = derive_profile_windows(profile)
        assert windows is not None, person.user_id
        assert set(windows) == {*_BODY_DERIVED_KEYS, "sodium_mg"}
        # profile_for carries exactly the derived windows (never invented ones)
        assert profile.windows == windows
    assert {p.persona for p in TRAIN_ROSTER} <= set(_PERSONAS)
    # persona -> body design (mirrors the exam roster's convention): gym
    # people lift (phase muscle = the 1.6 g/kg protein floor, so rec-post-gym
    # speakers have plausible post-gym profiles), cut people deficit, everyday
    # people maintain.
    assert all(p.phase == "muscle" for p in TRAIN_ROSTER if p.persona == "gym")
    assert all(p.phase == "cut" for p in TRAIN_ROSTER if p.persona == "cut")
    assert all(p.phase == "maintain" for p in TRAIN_ROSTER if p.persona == "everyday")
    assert {p.activity for p in TRAIN_ROSTER if p.persona == "gym"} <= {
        "active",
        "very_active",
    }


# --------------------------------------------------------------------------- #
# ticket 004 checkbox 4 — provisional 65/20/15 persona mix
# --------------------------------------------------------------------------- #


def test_persona_counts_match_provisional_65_20_15_mix():
    # 13/4/3 of n=20 is the provisional 65/20/15 everyday/gym/cut mix, exact
    # at this roster size (spec §23 OQ-9 — the persona-ratio citation is
    # deferred and non-blocking; it affects the roster, not the pipeline).
    counts = Counter(p.persona for p in TRAIN_ROSTER)
    assert counts == Counter({"everyday": 13, "gym": 4, "cut": 3})


# --------------------------------------------------------------------------- #
# ticket 004 checkbox 2 — the train/exam window isolation proof
# --------------------------------------------------------------------------- #


def test_train_exam_window_isolation():
    train_windows = _windows_by_id(TRAIN_ROSTER)
    exam_windows = _windows_by_id(ROSTER)

    # Per-nutrient disjointness over the five body-derived nutrients: no
    # TRAIN window tuple equals ANY exam window tuple, exhaustively.
    for key in _BODY_DERIVED_KEYS:
        train_tuples = {w[key] for w in train_windows.values()}
        exam_tuples = {w[key] for w in exam_windows.values()}
        assert not train_tuples & exam_tuples, (
            f"{key}: TRAIN/exam window tuples collide — "
            "train/exam isolation broken"
        )

    # Full-window dicts: mutually distinct within TRAIN_ROSTER …
    train_full = {frozenset(w.items()) for w in train_windows.values()}
    assert len(train_full) == len(TRAIN_ROSTER), (
        "two TRAIN_ROSTER people share a full window dict"
    )
    # … and disjoint from the exam full-window dicts.
    exam_full = {frozenset(w.items()) for w in exam_windows.values()}
    assert not train_full & exam_full

    # Documented exclusion, proved: sodium_mg is the constant (0.0, 2300.0)
    # for every person in BOTH rosters (see _BODY_DERIVED_KEYS), which is
    # exactly why it cannot be part of the disjointness contract.
    assert {w["sodium_mg"] for w in train_windows.values()} == {(0.0, 2300.0)}
    assert {w["sodium_mg"] for w in exam_windows.values()} == {(0.0, 2300.0)}


# --------------------------------------------------------------------------- #
# roster design requirements — allergies
# --------------------------------------------------------------------------- #


def test_allergies_use_supported_tags_and_allergy_free_people_exist(catalog):
    # generate_one resolves allergens only through catalog allergen_tags, and
    # that vocabulary is EXACTLY the exam roster's nine tags — no supported
    # allergy value exists outside the exam set (verified live below). The
    # pre-spec draft's "sesame" is unsupported: the catalog's "Sesame *" foods
    # carry no allergen_tags, so the public update shell rejects it.
    exam_tags = {tag for p in ROSTER for tag in p.allergies}
    vocab = {
        tag
        for entry in catalog.values()
        for tag in (entry.get("allergen_tags") or [])
    }
    assert vocab == exam_tags, "a catalog tag outside the exam set appeared"

    for person in TRAIN_ROSTER:
        assert set(person.allergies) <= vocab, (
            f"{person.user_id}: allergy {set(person.allergies) - vocab} is not "
            "a catalog allergen tag — generate_one cannot resolve it"
        )

    # sesame concretely refuses through the public shell (the documented
    # reason TRAIN_ROSTER reuses the supported vocabulary)
    free = next(p for p in TRAIN_ROSTER if not p.allergies)
    refused = generate_one(
        catalog=catalog,
        family="update",
        person=free,
        seed=5,
        shell="upd-add-allergy-short",
        slots={"allergen": "sesame"},
    )
    assert refused.accepted is None
    assert refused.rejected.reason == "no_allergen_food"

    # several allergy-free people in each persona: the upd-add-allergy-short
    # composite needs speakers who start with no allergies
    for persona in _PERSONAS:
        allergy_free = [
            p
            for p in TRAIN_ROSTER
            if p.persona == persona and not p.allergies
        ]
        assert len(allergy_free) >= 2, f"{persona}: too few allergy-free people"


@pytest.mark.parametrize("persona", _PERSONAS)
def test_upd_add_allergy_short_accepts_allergy_free_train_person(catalog, persona):
    person = next(
        p for p in TRAIN_ROSTER if p.persona == persona and not p.allergies
    )
    result = generate_one(
        catalog=catalog,
        family="update",
        person=person,
        seed=5,
        shell="upd-add-allergy-short",
        slots={"allergen": "fish"},
    )
    assert result.accepted is not None, (
        f"{persona}: {result.rejected and result.rejected.reason}"
    )
    task = result.accepted
    # the update lands on the TRAIN person's own world
    assert task.s0.profile.user_id == person.user_id
    assert task.persona == persona
    assert "fish" in task.oracle.profile.allergies


# --------------------------------------------------------------------------- #
# ticket 004 checkbox 5 — generate_one accepts a log Task per persona
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("persona", _PERSONAS)
def test_generate_one_log_accepts_train_task_per_persona(catalog, persona):
    expander = _synth_expander(catalog)
    accepted = None
    for person in (p for p in TRAIN_ROSTER if p.persona == persona):
        for amount_path in ("named_measure", "explicit_grams"):
            result = generate_one(
                catalog=catalog,
                family="log",
                person=person,
                seed=7,
                occasion="lunch",
                amount_path=amount_path,
                expander=expander,
            )
            if result.accepted is not None:
                accepted = (person, amount_path, result.accepted)
                break
        if accepted:
            break
    assert accepted is not None, f"no accepted log Task for persona {persona!r}"
    person, _amount_path, task = accepted
    assert isinstance(task, Task)
    # the accepted task runs in the TRAIN person's world, not an exam one
    assert task.s0.profile.user_id == person.user_id
    assert task.persona == persona


# --------------------------------------------------------------------------- #
# roster design requirement — rec-post-gym needs gym people
# --------------------------------------------------------------------------- #


def test_rec_post_gym_recommend_shell_needs_train_gym_people(catalog):
    gym_people = [p for p in TRAIN_ROSTER if p.persona == "gym"]
    assert gym_people, "rec-post-gym hard-requires gym people"

    result = generate_one(
        catalog=catalog,
        family="recommend",
        person=gym_people[0],
        seed=3,
        occasion="lunch",
        shell="rec-post-gym",
    )
    assert result.accepted is not None, result.rejected
    task = result.accepted
    assert task.persona == "gym"
    assert task.s0.profile.user_id == gym_people[0].user_id

    # the shell hard-requires the gym persona — an everyday TRAIN person is
    # refused, so the persona split is load-bearing
    everyday = next(p for p in TRAIN_ROSTER if p.persona == "everyday")
    refused = generate_one(
        catalog=catalog,
        family="recommend",
        person=everyday,
        seed=3,
        occasion="lunch",
        shell="rec-post-gym",
    )
    assert refused.accepted is None
    assert refused.rejected.reason == "not_gym_persona"
