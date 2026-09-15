"""Pool suitability and spoken naming (2-leg composite round 2).

Two lab-side wordings fight the factory, and both are fixed on the NutriMind side:

- the pool spans FNDDS, so an adult meal's pool can offer infant formula;
- `spoken_display_name` reverses a comma name's qualifiers into a noun phrase the
  binder does not match, while natural re-orderings do.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench.pipeline.generate_one import _local_clause  # noqa: E402
from nutrienv.bench.pipeline.resolver import spoken_grams_from_query  # noqa: E402
from nutrienv.bench.pipeline.sampler import sample_pools  # noqa: E402
from nutrienv.world.catalog_store import load_catalog  # noqa: E402
from nutrienv.world.portions import resolve_portion  # noqa: E402

from src.training.data_factory.pool_filter import (  # noqa: E402
    filter_pool,
    is_suitable_meal_food,
    speakable_additions,
    spoken_identity,
)
from src.training.data_factory.consistency import query_entity_consistency  # noqa: E402
from src.training.data_factory.search_gate import (  # noqa: E402
    qualifier_complement,
    search_locatability,
)
from src.training.data_factory.speech import pin_speech_portion  # noqa: E402


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


# --------------------------------------------------------------------------- #
# spoken_identity: a record name becomes a noun phrase a person would say
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Pastrami, made from any kind of meat, reduced fat", "reduced-fat pastrami"),
        ("Cereal, oat squares", "oat-squares cereal"),
        ("Crackers, butter, reduced sodium", "butter reduced-sodium crackers"),
        ("Chicken drumstick, fried, coated, skin / coating not eaten, from pre-cooked",
         "fried coated chicken drumstick"),
        ("Buttermilk", "buttermilk"),
        ("Soup, chicken noodle", "chicken-noodle soup"),
    ],
)
def test_spoken_identity_reads_as_speech(name, expected):
    assert spoken_identity(name) == expected


def test_clause_qualifiers_are_not_hyphenated_into_nonsense():
    """"made from any kind of meat" is a clause, not an adjective."""
    got = spoken_identity("Pastrami, made from any kind of meat, reduced fat")
    assert "made-from" not in got
    assert "any-kind" not in got


def test_measurements_and_brands_stay_out_of_the_name():
    assert "%" not in spoken_identity("Rice, wild, 100%, cooked, fat added")
    assert "stage" not in spoken_identity("Infant formula, Gerber Good Start, Stage 2")


def test_aliases_win_when_present():
    assert spoken_identity("Cereal, oat squares", aliases=["oaty loops"]) == "oaty loops"


# --------------------------------------------------------------------------- #
# speakable_additions: what a speaker could add when the handle is not enough
# --------------------------------------------------------------------------- #


def test_additions_are_taken_from_the_record_and_offered_whole_first():
    """A segment is the dish's name; its last word alone is not what a speaker says."""
    cands = speakable_additions("Soup, New England clam chowder")
    assert cands[0] == "new england clam chowder"
    assert "chowder" in cands
    # the head is not an addition: it is already in every handle
    assert all("soup" != cand for cand in cands)


def test_additions_skip_boilerplate_and_never_negate():
    """Nobody says "NFS", and "not" inverts what it was meant to narrow."""
    assert speakable_additions("Cornmeal mush, NS as to fat") == []
    cands = speakable_additions("Chicken leg, drumstick and thigh, sauteed, skin not eaten")
    assert "not" not in " ".join(cands).split()
    assert "skin not eaten" not in cands


def test_additions_do_not_begin_or_end_on_a_function_word():
    """A run like "and gravy" or "skin not" reads as a phrase but is not speech."""
    cands = speakable_additions("Rice, white, with vegetables and gravy, no added fat")
    for cand in cands:
        first, last = cand.split()[0], cand.split()[-1]
        assert first not in {"and", "or", "with", "of", "to", "as", "no"}
        assert last not in {"and", "or", "with", "of", "to", "as", "no"}
    assert not any(cand.startswith("and ") for cand in cands)


# --------------------------------------------------------------------------- #
# the planner's names still bind — that is the whole point
# --------------------------------------------------------------------------- #


def test_planner_names_bind_grams_across_pools(catalog):
    """Every pinned phrase + planner handle must locate its clause and resolve."""
    checked = 0
    for seed in (0, 1, 7, 42, 101):
        pool = sample_pools(catalog, seed=seed, family="log", n_pools=1)[0]
        for amount_path in ("explicit_grams", "named_measure", "unspecified"):
            food, handle, pin = pin_speech_portion(
                pool, amount_path=amount_path, catalog=catalog
            )
            if pin is None:
                continue
            query = f"I had {pin.phrase} of {handle} for lunch."
            clause = _local_clause(query, food.food_id, catalog)
            assert clause, f"binder cannot locate {handle!r} in {query!r}"
            grams = spoken_grams_from_query(clause, food.food_id, catalog)
            if grams is None:
                grams = resolve_portion(food.food_id, clause, catalog)
            assert grams, f"{handle!r} did not resolve grams in {clause!r}"
            checked += 1
    assert checked >= 10, f"only {checked} combinations were checkable"


def test_consistency_accepts_the_planner_form(catalog):
    """The check used to demand the scrambled display form and reject this."""
    from nutrienv.bench.pipeline.sampler import spoken_display_name

    food_id = "2706184"  # Pastrami, made from any kind of meat, reduced fat
    query = "I had a slice of reduced-fat pastrami from the cafeteria for lunch."
    assert "reduced-fat pastrami" in query
    assert spoken_display_name(catalog, food_id) not in query
    assert (
        query_entity_consistency(
            query, [food_id], intent={"family": "composite"}, catalog=catalog
        )
        is None
    )


# --------------------------------------------------------------------------- #
# pin selection judges locatability, not just speakability
# --------------------------------------------------------------------------- #


def _pool_like(pool, foods):
    """A pool holding exactly ``foods``, spelled the way the planner spells it."""
    return type(pool)(pool_id=pool.pool_id, family=pool.family, foods=tuple(foods))


def _lab_accepts(pool, food, amount_path, catalog):
    """The lab's own half of pin selection, spelled out without the new gate.

    Re-derived here so the gate has something to be measured against: a test that
    only called `pin_speech_portion` could not tell a skipped food from one the lab
    refused for its own reasons.
    """
    from nutrienv.bench.pipeline.sampler import speakable_tracer_food

    from src.training.data_factory.speech import _pin_for, _speech_amount_path

    if not is_suitable_meal_food((catalog.get(food.food_id) or {}).get("name")):
        return False
    if _pin_for(food, amount_path) is None:
        return False
    picked = speakable_tracer_food(
        _pool_like(pool, (food,)), catalog, amount_path=amount_path
    )
    return picked is not None and _speech_amount_path(picked[1]) == amount_path


def test_pin_selection_skips_a_food_the_search_cannot_locate(catalog):
    """A pin whose utterance buries it is not a task: the search moves on.

    The test finds a pool where the lab accepts a food that no natural phrase locates
    (its record has nothing a speaker can add), and asserts the pin does not land on
    it — and that whatever it does land on is located by the phrase it returns.
    """
    for seed in range(6):
        pool = sample_pools(catalog, seed=seed, family="log", n_pools=1)[0]
        for amount_path in ("explicit_grams", "named_measure", "unspecified"):
            accepted = [
                food
                for food in pool.foods
                if _lab_accepts(pool, food, amount_path, catalog)
            ]
            if len(accepted) < 2:
                continue
            first = accepted[0]
            if qualifier_complement(str(first.food_id), catalog=catalog) is not None:
                continue  # this pool skips nothing; try the next draw
            food, handle, pin = pin_speech_portion(
                pool, amount_path=amount_path, catalog=catalog
            )
            assert food is not None, "the pool holds a locatable food"
            assert str(food.food_id) != str(first.food_id)
            assert pin is not None
            assert (
                search_locatability(
                    str(food.food_id),
                    (catalog.get(food.food_id) or {}).get("name"),
                    catalog=catalog,
                    spoken=handle,
                ).status
                == "unique"
            )
            return
    pytest.skip("no sampled pool offered a food the gate has to skip")


def test_pin_selection_returns_a_form_the_search_locates(catalog):
    """Sampled over the lab's own pools: whatever is pinned must be findable."""
    checked = 0
    for seed in (0, 1, 7, 42, 101):
        pool = sample_pools(catalog, seed=seed, family="log", n_pools=1)[0]
        for amount_path in ("explicit_grams", "named_measure", "unspecified"):
            food, handle, pin = pin_speech_portion(
                pool, amount_path=amount_path, catalog=catalog
            )
            if pin is None:
                continue
            verdict = search_locatability(
                str(food.food_id),
                (catalog.get(food.food_id) or {}).get("name"),
                catalog=catalog,
                spoken=handle,
            )
            assert verdict.status == "unique", (food.food_id, handle, verdict.status)
            checked += 1
    assert checked >= 10, f"only {checked} combinations were checkable"


# --------------------------------------------------------------------------- #
# suitability filter
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "Infant formula, Gerber Good Start Gentle, Stage 2",
        "Baby Toddler yogurt, plain",
        "Baby Toddler food, NFS",
    ],
)
def test_unsuitable_foods_are_excluded(name):
    assert not is_suitable_meal_food(name)


@pytest.mark.parametrize(
    "name",
    [
        "Carrots, baby",
        "Cereal, oat squares",
        "Soup, chicken noodle",
        "Buttermilk",
        "",
        None,
    ],
)
def test_normal_foods_are_kept(name):
    """"Carrots, baby" is not baby food — the marker matches the head only."""
    assert is_suitable_meal_food(name)


def test_filter_pool_removes_only_unsuitable(catalog):
    pool = sample_pools(catalog, seed=0, family="log", n_pools=1)[0]
    filtered = filter_pool(pool, catalog)
    assert len(filtered.foods) <= len(pool.foods)
    for food in filtered.foods:
        assert is_suitable_meal_food((catalog.get(food.food_id) or {}).get("name"))


def test_filter_pool_never_empties_a_pool(catalog):
    """An empty pool is a hard failure downstream; a long shot beats no shot."""
    pool = sample_pools(catalog, seed=0, family="log", n_pools=1)[0]
    only_unsuitable = type(pool)(
        pool_id=pool.pool_id,
        family=pool.family,
        foods=tuple(
            food
            for food in pool.foods
            if not is_suitable_meal_food((catalog.get(food.food_id) or {}).get("name"))
        ),
    )
    if not only_unsuitable.foods:
        pytest.skip("this pool happens to be all-suitable")
    assert filter_pool(only_unsuitable, catalog).foods == only_unsuitable.foods


# --------------------------------------------------------------------------- #
# composite occasion: the eaten meal is in the log span, not the ask
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("query", "food_id"),
    [
        ("I had a cup of oat-squares cereal. What's for dinner?", "2708466"),
        ("I had a cup of oat-squares cereal for lunch. What's for dinner?", "2708466"),
        (
            "I already ate lunch at the cafeteria — a can of minestrone soup. "
            "What's for dinner?",
            "2710114",  # Soup, minestrone
        ),
    ],
)
def test_lunch_intent_accepts_a_dinner_ask(catalog, query, food_id):
    """"What's for dinner?" on a lunch intent is correct: lunch asks about dinner.

    The check used to scan the whole sentence, find "dinner", and reject with
    `author.intent_conflict` — the ask names the *next* meal by construction.
    """
    assert (
        query_entity_consistency(
            query,
            [food_id],
            intent={"family": "composite", "occasion": "lunch",
                    "amount_path": "named_measure"},
            catalog=catalog,
        )
        is None
    )


@pytest.mark.parametrize(
    "query",
    [
        "For breakfast I had a cup of oat-squares cereal. What's for lunch?",
        "For dinner I had a cup of oat-squares cereal. What's for lunch?",
    ],
)
def test_wrong_meal_in_the_log_span_is_still_rejected(catalog, query):
    """Reading only the log span must not stop the check from working."""
    assert (
        query_entity_consistency(
            query,
            ["2708466"],
            intent={"family": "composite", "occasion": "lunch",
                    "amount_path": "named_measure"},
            catalog=catalog,
        )
        == "author.intent_conflict"
    )


def test_grams_in_the_ask_do_not_trip_a_named_measure_intent(catalog):
    """"1 fl oz (2 tablespoons)" is named measure; grams in the ask are irrelevant."""
    ok = "I had 1 fl oz of cranberry juice for breakfast. What's for lunch?"
    assert (
        query_entity_consistency(
            ok, ["2709324"],  # Cranberry juice, 100%, not a blend
            intent={"family": "composite", "occasion": "breakfast",
                    "amount_path": "named_measure"},
            catalog=catalog,
        )
        is None
    )
