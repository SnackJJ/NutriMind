"""Ticket nutrimind-pilot/002 — query/entity consistency after speech."""

from __future__ import annotations

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.world.catalog_store import load_catalog  # noqa: E402

from src.training.data_factory.author import author_task  # noqa: E402
from src.training.data_factory.consistency import (  # noqa: E402
    CONSISTENCY_CODES,
    query_entity_consistency,
)
from src.training.data_factory.roster_train import TRAIN_ROSTER  # noqa: E402
from src.training.data_factory.synthetic import synth_expander  # noqa: E402

_TINY = {
    "a1": {"name": "Tomato, raw", "aliases": ["tomato"]},
    "a2": {"name": "Tomato soup, canned", "aliases": ["tomato soup"]},
    "b1": {"name": "Egg, whole", "aliases": ["egg"]},
}

_INTENT = {
    "occasion": "dinner",
    "amount_path": "named_measure",
    "family": "log",
    "scene": "empty",
}


@pytest.mark.parametrize(
    ("query", "foods", "intent", "expected"),
    [
        (
            "I had tomato soup at a restaurant for breakfast.",
            ["zzz"],
            _INTENT,
            "author.foods_outside_binding",
        ),
        (
            "I had tomato soup for dinner.",
            ["a1"],
            _INTENT,
            "author.unselected_variant",
        ),
        (
            "I had tomato for dinner.",
            ["a1"],
            _INTENT,
            "author.missing_disambiguation",
        ),
        (
            "I had tomato for breakfast.",
            ["a1"],
            _INTENT,
            "author.intent_conflict",
        ),
        (
            "I had egg for dinner.",
            ["a1"],
            _INTENT,
            "author.query_foods_mismatch",
        ),
        (
            "I had tomato and tomato soup for dinner.",
            ["a1", "a2"],
            _INTENT,
            "author.ambiguous_entity",
        ),
    ],
    ids=[
        "foods_outside_binding",
        "unselected_variant",
        "missing_disambiguation",
        "intent_conflict",
        "query_foods_mismatch",
        "ambiguous_entity",
    ],
)
def test_each_consistency_reason_has_a_code(query, foods, intent, expected):
    assert expected in CONSISTENCY_CODES
    assert query_entity_consistency(
        query, foods, intent=intent, catalog=_TINY, allowed_ids={"a1", "a2", "b1"}
    ) == expected


def test_uniquely_binding_matching_foods_is_kept():
    assert (
        query_entity_consistency(
            "I bought ingredients at the market and had tomato soup for dinner.",
            ["a2"],
            intent=_INTENT,
            catalog=_TINY,
            allowed_ids={"a1", "a2", "b1"},
        )
        is None
    )


def _log_intent(**overrides):
    person = TRAIN_ROSTER[0]
    intent = {
        "schema_version": "nutrimind-v2-intent/1",
        "task_id": "log--log--train-alba--000001",
        "task_key": f"log--log--{person.user_id}",
        "family": "log",
        "task_family": "log",
        "steps": ["log"],
        "user_id": person.user_id,
        "seed": 1,
        "occasion": "lunch",
        "scene": "empty",
        "shell": None,
        "slots": None,
        "amount_path": "named_measure",
        "ounce_phrasing": False,
        "knife": None,
        "tier": "",
        "recovery_trap": None,
        "gram_anchor": False,
    }
    intent.update(overrides)
    return intent


def test_author_keeps_unique_bind_and_rejects_intent_conflict(catalog=None):
    catalog = load_catalog()
    expander = synth_expander(catalog)
    task, reject = author_task(
        _log_intent(occasion="lunch", recovery_trap=None),
        catalog=catalog,
        expander=expander,
        parse_retries=0,
    )
    assert reject is None, reject
    assert task is not None

    inner = synth_expander(catalog)

    def conflicting(pool, *, persona, family, amount_path=None):
        out = inner(pool, persona=persona, family=family, amount_path=amount_path)
        return {
            "query": out["query"].replace("lunch", "breakfast").replace("dinner", "breakfast"),
            "foods": out["foods"],
        }

    task, reject = author_task(
        _log_intent(occasion="lunch", recovery_trap=None),
        catalog=catalog,
        expander=conflicting,
        parse_retries=0,
    )
    assert task is None
    assert reject["failure_codes"] == ["author.intent_conflict"]
    assert "unresolvable" not in reject["failure_codes"][0]


def test_regenerate_then_reject_does_not_loop():
    catalog = load_catalog()
    inner = synth_expander(catalog)
    calls = {"n": 0}

    def flaky(pool, *, persona, family, amount_path=None):
        calls["n"] += 1
        out = inner(pool, persona=persona, family=family, amount_path=amount_path)
        return {
            "query": out["query"].replace("lunch", "breakfast").replace("dinner", "breakfast"),
            "foods": out["foods"],
        }

    task, reject = author_task(
        _log_intent(occasion="lunch", recovery_trap=None),
        catalog=catalog,
        expander=flaky,
        parse_retries=1,
    )
    assert task is None
    assert reject["failure_codes"] == ["author.intent_conflict"]
    assert calls["n"] == 2

    calls["n"] = 0

    def once_then_ok(pool, *, persona, family, amount_path=None):
        calls["n"] += 1
        out = inner(pool, persona=persona, family=family, amount_path=amount_path)
        if calls["n"] == 1:
            return {
                "query": out["query"].replace("lunch", "breakfast"),
                "foods": out["foods"],
            }
        return out

    task, reject = author_task(
        _log_intent(occasion="lunch", recovery_trap=None),
        catalog=catalog,
        expander=once_then_ok,
        parse_retries=1,
    )
    assert reject is None, reject
    assert task is not None
    assert calls["n"] == 2
