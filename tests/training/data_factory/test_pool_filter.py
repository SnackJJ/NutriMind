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
    spoken_identity,
)
from src.training.data_factory.consistency import query_entity_consistency  # noqa: E402
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
