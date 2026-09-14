"""Search locatability: the environment's own search as the disambiguation judge.

The student acts through `search_foods` (FTS5 BM25, AND semantics, SEARCH_LIMIT rows).
A task is only well-defined if that search reaches the pinned food; otherwise the agent
picks a neighbour and the Scorer compares a different food's end state.

Every assertion here was calibrated against the pinned catalog, and several exist
because an earlier version of this module was wrong in a way that looked plausible: it
dropped words like "wing", "frozen", "100" and "with"/"without" from the form it
searched. Each of those is indexed text, so dropping one could only widen the query —
it manufactured every `unreachable` verdict of one version, inflated the ambiguity rate
of the next, and called FNDDS's "grilled with sauce" / "grilled without sauce" pair a
pair of twins. The judge now searches the lab's own tokens and nothing else, and the
form it reports is the one the utterance carries, not the catalog record.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.world.catalog import SEARCH_LIMIT  # noqa: E402
from nutrienv.world.catalog_store import load_catalog  # noqa: E402

from src.training.data_factory.search_gate import (  # noqa: E402
    judge_food,
    search_locatability,
    search_words,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "data_factory.yaml"


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


# --------------------------------------------------------------------------- #
# word extraction: a form is searched as the lab searches it, nothing dropped
# --------------------------------------------------------------------------- #


def test_identity_words_survive():
    """"wing", "frozen", "bar" are indexed identity terms, not quantity noise.

    Measured in the pinned catalog: "wing" retrieves 16 foods, "bar" 25, and
    "frozen" is the difference between "Frozen dinner" and "dinner".
    """
    assert search_words("Chicken wing, stewed") == ["chicken", "wing", "stewed"]
    assert search_words("Frozen yogurt bar, vanilla") == [
        "frozen",
        "yogurt",
        "bar",
        "vanilla",
    ]


def test_multi_digit_numbers_are_real_terms():
    """"100" narrows: "pineapple juice 100" returns 1 row, "pineapple juice" 2."""
    assert search_words("Pineapple juice, 100%") == ["pineapple", "juice", "100"]


def test_single_characters_are_dropped_like_the_lab_does():
    """The lab's own `_tokens` keeps `len(tok) >= 2`; matching it avoids dead terms."""
    assert search_words("Cereal, O's, NFS") == ["cereal", "nfs"]
    assert search_words("A B c") == []


def test_no_other_word_is_dropped():
    """Every remaining word is in the indexed text, so dropping one only widens.

    `with` / `without` are the sharpest case: FNDDS carries both variants of a food,
    and a stopword list that drops them reports each as its neighbour's twin.
    """
    assert search_words("Milk, NFS") == ["milk", "nfs"]
    assert search_words("Rice, white, cooked, NS as to fat") == [
        "rice",
        "white",
        "cooked",
        "ns",
        "as",
        "to",
        "fat",
    ]
    assert search_words("Chicken breast, grilled with sauce") != search_words(
        "Chicken breast, grilled without sauce"
    )


def test_the_with_variants_do_not_merge(catalog):
    """Measured: the lab separates them, and so must the judge.

    `2705968`/`2705970` are the FNDDS pair "grilled without sauce" / "grilled with
    sauce". Their spoken handles keep the preposition, so both are uniquely
    locatable — an earlier version dropped `with`/`without` and called both
    `ambiguous`.
    """
    out = search_locatability(
        "2705968", "Chicken breast, grilled without sauce, skin not eaten", catalog=catalog
    )
    inn = search_locatability(
        "2705970", "Chicken breast, grilled with sauce, skin not eaten", catalog=catalog
    )
    assert out.status == "unique" and inn.status == "unique"
    assert judge_food("2705968", catalog=catalog).status == "unique"
    assert judge_food("2705970", catalog=catalog).status == "unique"


def test_negation_words_survive_because_the_catalog_uses_them():
    """'fat added' and 'no added fat' are two different foods in FNDDS."""
    assert search_words("Sweet potato, baked, no added fat") == [
        "sweet",
        "potato",
        "baked",
        "no",
        "added",
        "fat",
    ]


def test_hyphens_are_separators_not_tokens(catalog):
    """`pre-cooked` and `pre cooked` are the same query to the lab.

    `_tokens` is `[a-z0-9]+` and the FTS table tokenises with `unicode61`, so the
    hyphen never survives. Verified by behaviour, not by string form.
    """
    hyphened = catalog.search("chicken drumstick pre-cooked", limit=SEARCH_LIMIT)
    spaced = catalog.search("chicken drumstick pre cooked", limit=SEARCH_LIMIT)
    assert [hit["food_id"] for hit in hyphened] == [hit["food_id"] for hit in spaced]
    assert "pre-cooked" not in search_words("Chicken drumstick, from pre-cooked")


# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #


def test_unique_when_the_name_reaches_exactly_the_pin(catalog):
    v = search_locatability("2708466", "Cereal, oat squares", catalog=catalog)
    assert v.status == "unique" and v.usable
    assert v.hit_ids == ("2708466",)
    assert v.ordinal == 0


def test_ambiguous_when_a_neighbour_shares_the_words(catalog):
    """fat-added vs no-added-fat: the pin is searchable, but not alone."""
    v = search_locatability(
        "2709700", "Sweet potato, baked, fat added", catalog=catalog
    )
    assert v.status == "ambiguous"
    assert "2709700" in v.hit_ids and len(v.hit_ids) > 1


def test_unique_when_the_negation_disinherits_the_neighbour(catalog):
    v = search_locatability(
        "2709699", "Sweet potato, baked, no added fat", catalog=catalog
    )
    assert v.status == "unique"


def test_reduced_fat_milk_is_ambiguous_not_unreachable(catalog):
    """The calibration case, with the right food id.

    `2705386` is "Milk, reduced fat (2%)" and its own words return it among 14
    neighbours. Its numeric qualifier cannot separate it, so a separating word has to
    come from somewhere else — but the food IS reachable, which is what the agent
    needs. (`2708558`, used in an earlier version of this test, is a burrito bowl.)
    """
    v = search_locatability("2705386", "Milk, reduced fat (2%)", catalog=catalog)
    assert v.status == "ambiguous"
    assert "2705386" in v.hit_ids
    assert v.terms == ("milk", "reduced", "fat")


def test_unreachable_when_only_a_generic_head_is_left(catalog):
    """"Cereal, O's, NFS" is spoken as "o's cereal" — that is, "cereal", among 25.

    The record's own words keep `nfs` and stay `ambiguous`; the handle the brief
    commits the utterance to collapses to the head, which buries the pin.
    """
    v = judge_food("2708475", catalog=catalog)
    assert v.status == "unreachable"
    assert v.terms == ("cereal",)
    assert len(v.hit_ids) >= SEARCH_LIMIT
    assert "2708475" not in v.hit_ids


def test_unmatched_when_the_terms_do_not_retrieve_the_food(catalog):
    """A wrong id or a tokeniser bug must not read as "hopeless food".

    These words never retrieve `2708558` and the result set is short of the limit, so
    the query simply does not match the food — a defect, not a property of the
    catalog. Distinguishing this from `unreachable` is what stops a wrong id from
    passing as a verdict.
    """
    v = search_locatability("2708558", "Milk, reduced fat (2%)", catalog=catalog)
    assert v.status == "unmatched"
    assert "2708558" not in v.hit_ids
    assert len(v.hit_ids) < SEARCH_LIMIT


def test_no_terms_for_a_form_without_searchable_words(catalog):
    """Single characters are all the lab's tokeniser keeps out, so this is the one
    way a form can carry nothing searchable at all."""
    v = search_locatability("x", "A B c", catalog=catalog)
    assert v.status == "no_terms" and v.terms == () and not v.usable


# --------------------------------------------------------------------------- #
# the spoken form is the one the agent reads
# --------------------------------------------------------------------------- #


def test_the_utterance_handle_is_judged_as_its_own_form(catalog):
    """The record name is not the query; the brief's handle is.

    `2705386` is "Milk, reduced fat (2%)", whose own words return it among fourteen
    neighbours. The brief asks the speaker to say "milk" — and "milk" returns 25 rows
    without the pin, so the agent never reaches it. Judging only the record says
    `ambiguous`; the utterance says `unreachable`.
    """
    record = search_locatability("2705386", "Milk, reduced fat (2%)", catalog=catalog)
    spoken = search_locatability(
        "2705386", "Milk, reduced fat (2%)", catalog=catalog, spoken="milk"
    )
    assert record.status == "ambiguous" and "2705386" in record.hit_ids
    assert spoken.status == "unreachable"
    assert spoken.terms == ("milk",)
    assert spoken.ordinal > record.ordinal, "the weaker form must decide"


def test_judge_food_derives_the_handle_the_brief_speaks(catalog):
    """A food id alone must be enough: the caller may hold no name at all."""
    from src.training.data_factory.pool_filter import spoken_identity

    entry = catalog.get("2705386") or {}
    aliases = tuple(entry.get("aliases") or ())
    handle = spoken_identity(entry.get("name"), aliases=aliases)
    assert handle == "milk"
    explicit = search_locatability(
        "2705386", entry.get("name"), catalog=catalog, aliases=aliases, spoken=handle
    )
    assert judge_food("2705386", catalog=catalog) == explicit
    assert judge_food("2705386", catalog=catalog).status == "unreachable"


def test_judge_food_flags_an_id_the_catalog_does_not_hold(catalog):
    """A wrong id is a defect, not a property of a food (the burrito-bowl lesson)."""
    v = judge_food("9999999", catalog=catalog)
    assert v.status == "unmatched"
    assert v.hit_ids == ()


# --------------------------------------------------------------------------- #
# aliases are judged on their own
# --------------------------------------------------------------------------- #


def test_aliases_are_not_glued_into_the_name_query(catalog):
    """Combining name + alias fakes uniqueness.

    `Peanut butter`'s own words return 25 rows; adding its "pb" alias to the same AND
    bag returns exactly 1 — a query no speaker would ever say. The alias must be
    judged as its own spoken form, and the food's verdict must then be the weaker of
    the two.
    """
    food_id = "2707537"  # Peanut butter, aliases ("pb", "peanut spread", ...)
    entry = catalog.get(food_id) or {}
    aliases = tuple(entry.get("aliases") or ())
    if not aliases:
        pytest.skip("pinned catalog carries no alias for this food")
    name_terms = search_words(entry["name"])
    glued = " ".join(name_terms + search_words(aliases[0]))
    assert len(catalog.search(glued, limit=SEARCH_LIMIT)) == 1, (
        "the alias no longer narrows this food; pick another case"
    )
    verdict = search_locatability(
        food_id, entry.get("name"), catalog=catalog, aliases=aliases
    )
    assert " ".join(verdict.terms) != glued, "alias terms must be judged separately"
    name_only = search_locatability(food_id, entry.get("name"), catalog=catalog)
    assert verdict.ordinal >= name_only.ordinal, "the weakest form decides"


def test_the_weakest_form_decides(catalog):
    """A food is only as findable as its worst spoken form."""
    food_id = "2705386"
    entry = catalog.get(food_id) or {}
    strong = search_locatability(food_id, entry.get("name"), catalog=catalog)
    weak = search_locatability(
        food_id, entry.get("name"), catalog=catalog, aliases=("milk",)
    )
    assert weak.ordinal >= strong.ordinal


# --------------------------------------------------------------------------- #
# the manifest observable built on this verdict
# --------------------------------------------------------------------------- #


def _materialized(catalog):
    """A real (Task, TaskPackage) for a log intent, built offline."""
    from src.training.data_factory import author as author_mod
    from src.training.data_factory.build import enumerate_intents
    from src.training.data_factory.config import load_config
    from src.training.data_factory.materialize import (
        RunContext,
        catalog_digest,
        materialize,
    )
    from src.training.data_factory.synthetic import synth_expander

    config = load_config(CONFIG_PATH)
    expander = synth_expander(catalog)
    intent = next(i for i in enumerate_intents(config) if i["family"] == "log")
    task, reject = author_mod.author_task(intent, catalog=catalog, expander=expander)
    assert task is not None, reject
    package = materialize(
        task,
        RunContext(
            catalog=catalog,
            catalog_sha=catalog_digest(catalog),
            nutrienv_rev=config.nutrienv_rev,
            nutrimind_rev="d" * 40,
            config_sha="0" * 64,
            steps=tuple(intent["steps"]),
            seed=intent["seed"],
            built_at="2026-09-14T12:00:00+00:00",
            intent_ref=f"intents/{intent['family']}.jsonl#{intent['seed']:06d}",
        ),
    )
    return task, package


def test_locatability_status_is_the_weakest_of_the_tasks_foods(catalog):
    from src.training.data_factory.build import _locatability_status, _oracle_food_ids

    task, package = _materialized(catalog)
    ids = _oracle_food_ids(task.oracle)
    assert ids, "a log task must expose its bound foods"
    expected = max(
        (judge_food(food_id, catalog=catalog) for food_id in ids),
        key=lambda verdict: verdict.ordinal,
    ).status
    assert _locatability_status(catalog, package) == expected
    # The caller that holds the live task must not rebuild it out of the package and
    # get a different answer (the rebuild is a temp-dir file round-trip).
    assert _locatability_status(catalog, package, task) == expected


def test_locatability_status_reports_every_pinned_food(catalog):
    """A task is only as findable as its least findable food, not its first.

    The order is flipped between the two cases on purpose: an implementation that
    returns the first pinned food's verdict passes one and fails the other.
    """
    from src.training.data_factory.build import _locatability_status

    class _Oracle:
        def __init__(self, food_ids):
            self.ledger_tail = [{"food_id": food_id} for food_id in food_ids]

    class _Task:
        def __init__(self, food_ids):
            self.oracle = _Oracle(food_ids)

    findable, buried = "2708466", "2708475"  # "Cereal, oat squares" / "Cereal, O's, NFS"
    assert judge_food(findable, catalog=catalog).status == "unique"
    assert judge_food(buried, catalog=catalog).status == "unreachable"
    for food_ids in ([findable, buried], [buried, findable]):
        assert _locatability_status(catalog, None, _Task(food_ids)) == "unreachable"


def test_locatability_status_logs_an_unrebuildable_package(catalog, caplog):
    """An unbuildable package is a defect, not a verdict about a food.

    `unavailable` means "the package would not rebuild", which is not a property of
    the pinned food — it must be logged (never silent) and excluded from the rate's
    denominator, which `test_finalize_observability_*` pins.
    """
    import dataclasses

    from src.training.data_factory.build import _locatability_status

    _task, package = _materialized(catalog)
    broken = dataclasses.replace(
        package,
        oracle=dataclasses.replace(package.oracle, payload="not-an-oracle-object"),
    )
    with caplog.at_level("WARNING"):
        status = _locatability_status(catalog, broken)
    assert status == "unavailable"
    assert package.task_id in caplog.text


def test_finalize_observability_reports_the_locatability_block():
    """The metric lands in metrics with a rate, and is None-safe when empty."""
    from collections import Counter

    from src.training.data_factory.build import _finalize_observability
    from src.training.data_factory.config import load_config

    def finalize(manifest, **kwargs):
        base = dict(
            config=load_config(CONFIG_PATH),
            catalog_sha="x",
            reject_histogram=Counter(),
            accepted_by_family=Counter(),
            teacher_completed=0,
            teacher_error=0,
            teacher_no_finish=0,
            pass_count=0,
            serialized=0,
            attempted_task_ids=0,
            indeterminate_task_ids=0,
            accepted_records=[],
            tokens=0,
        )
        base.update(kwargs)
        _finalize_observability(manifest, **base)

    manifest = {"counts": {"rejected": {"indeterminate": 0, "gate": 0}}}
    finalize(manifest, search_locatability_counts=Counter({"unique": 3, "ambiguous": 1}))
    assert manifest["metrics"]["search_locatability"] == {"ambiguous": 1, "unique": 3}
    assert manifest["metrics"]["search_locatability_usable_rate"] == pytest.approx(0.75)

    # Defect statuses stay in the block but out of the denominator: a package that will
    # not rebuild, and an oracle that exposes no food, are not verdicts about a food.
    defects = {"counts": {"rejected": {"indeterminate": 0, "gate": 0}}}
    finalize(
        defects,
        search_locatability_counts=Counter(
            {"unique": 3, "ambiguous": 1, "unavailable": 5, "no_foods": 2}
        ),
    )
    assert defects["metrics"]["search_locatability"]["no_foods"] == 2
    assert defects["metrics"]["search_locatability_usable_rate"] == pytest.approx(0.75)

    # A run whose every pin failed to rebuild has no rate at all, not a 0% one.
    all_broken = {"counts": {"rejected": {"indeterminate": 0, "gate": 0}}}
    finalize(all_broken, search_locatability_counts=Counter({"unavailable": 4}))
    assert all_broken["metrics"]["search_locatability_usable_rate"] is None

    empty = {"counts": {"rejected": {"indeterminate": 0, "gate": 0}}}
    finalize(empty)
    assert empty["metrics"]["search_locatability"] == {}
    assert empty["metrics"]["search_locatability_usable_rate"] is None


# --------------------------------------------------------------------------- #
# the committed script that produced the distribution
# --------------------------------------------------------------------------- #


def test_the_measurement_script_reruns_the_number():
    """A distribution whose script was never committed cannot be reproduced.

    Runs the real entry point on a two-seed pool draw: it must exit clean and report
    both columns, the record-name one and the spoken one the metric is built on.
    """
    import subprocess
    import sys

    script = REPO_ROOT / "scripts" / "measure_search_locatability.py"
    done = subprocess.run(
        [
            sys.executable, str(script),
            "--source", "pools", "--family", "log", "--seeds", "2", "--show-worst", "0",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=600,
    )
    assert done.returncode == 0, done.stderr
    assert "record name+alias" in done.stdout
    assert "spoken forms" in done.stdout
    assert "source=pools" in done.stdout
