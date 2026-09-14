"""Ticket nutrimind-pilot/001 — semantic brief expander (single-query speech)."""

from __future__ import annotations

import inspect
import json
import os
import pathlib

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench.pipeline.generate_one import generate_one  # noqa: E402
from nutrienv.bench.pipeline.sampler import sample_pools  # noqa: E402
from nutrienv.world.catalog_store import load_catalog  # noqa: E402

from src.training.data_factory.author import author_task  # noqa: E402
from src.training.data_factory.roster_train import TRAIN_ROSTER  # noqa: E402
from src.training.data_factory.speech import (  # noqa: E402
    bind_speech_context,
    build_semantic_brief,
    complete_from_chat_client,
    make_brief_expander,
    render_semantic_brief,
)
from src.training.data_factory.synthetic import synth_expander  # noqa: E402

_DUMP_NEEDLES = (
    "id=",
    "variant=",
    "speakable portions",
    "food_id",
    "allergen_tags",
)


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.fixture(scope="module")
def pool(catalog):
    pools = sample_pools(catalog, seed=1, family="log", n_pools=1, pool_size=8)
    assert pools and pools[0].foods
    return pools[0]


def _log_intent(**overrides):
    person = TRAIN_ROSTER[0]
    intent = {
        "schema_version": "nutrimind-v2-intent/1",
        "task_id": "log--log--train-alba--000000",
        "task_key": f"log--log--{person.user_id}",
        "family": "log",
        "task_family": "log",
        "steps": ["log"],
        "user_id": person.user_id,
        "seed": 0,
        "occasion": "dinner",
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


def test_semantic_brief_is_not_a_catalog_dump(catalog, pool):
    brief = build_semantic_brief(
        pool,
        catalog=catalog,
        persona="everyday",
        family="log",
        amount_path="named_measure",
        occasion="dinner",
        scene="empty",
    )
    assert brief is not None
    text = render_semantic_brief(brief)
    lowered = text.lower()
    for needle in _DUMP_NEEDLES:
        assert needle not in lowered, needle
    assert brief.food_id not in text
    assert "everyday" in lowered
    assert "dinner" in lowered
    assert "log" in lowered or "logging" in lowered
    assert brief.entity_handle
    assert brief.entity_handle in text
    assert brief.source
    assert brief.situation


def test_llm_facing_payload_is_brief_not_pool_table(catalog, pool):
    captured: list = []

    def complete(model_id, messages):
        captured.append(messages)
        return json.dumps({"query": "I stopped by the market this morning."})

    expander = make_brief_expander(complete=complete, catalog=catalog)
    bound = bind_speech_context(
        expander, {"occasion": "dinner", "scene": "empty"}
    )
    bound(pool, persona="everyday", family="log", amount_path="named_measure")
    assert captured
    blob = "\n".join(message["content"] for message in captured[0]).lower()
    for needle in _DUMP_NEEDLES:
        assert needle not in blob, needle
    assert "dinner" in blob
    assert "everyday" in blob
    assert "logging" in blob or "log" in blob
    assert "one natural" in blob or "single" in blob


def test_complete_from_chat_client_uses_content():
    seen = {}

    def client(request):
        seen["messages"] = request["messages"]
        return {"content": '{"query": "I had oats for dinner."}', "reasoning_content": "skip"}

    complete = complete_from_chat_client(client)
    text = complete("brief-expander", ({"role": "user", "content": "brief"},))
    assert text == '{"query": "I had oats for dinner."}'
    assert seen["messages"][0]["content"] == "brief"


def test_brief_expander_via_chat_client(catalog, pool):
    def client(request):
        return {"content": json.dumps({"query": "After the market I cooked dinner."})}

    expander = make_brief_expander(
        complete=complete_from_chat_client(client), catalog=catalog
    )
    out = expander(pool, persona="everyday", family="log", amount_path="named_measure")
    assert out["query"] == "After the market I cooked dinner."
    assert out["foods"] and isinstance(out["foods"][0], str)


def test_brief_expander_matches_generate_one_contract(catalog, pool):
    def complete(model_id, messages):
        return json.dumps({"query": "For dinner I had a bowl of the dish.", "foods": ["HACKED"]})

    expander = make_brief_expander(complete=complete, catalog=catalog)
    sig = inspect.signature(expander)
    for name in ("persona", "family", "amount_path"):
        assert name in sig.parameters
    out = expander(pool, persona="everyday", family="log", amount_path="named_measure")
    assert set(out) == {"query", "foods"}
    assert isinstance(out["query"], str) and out["query"].strip()
    assert out["foods"] == [
        build_semantic_brief(
            pool,
            catalog=catalog,
            persona="everyday",
            family="log",
            amount_path="named_measure",
            occasion="lunch",
            scene="empty",
        ).food_id
    ]
    assert "HACKED" not in out["foods"]
    assert "\nUser:" not in out["query"]
    assert "Assistant:" not in out["query"]


def test_brief_preserves_amount_path_instruction(catalog, pool):
    brief = build_semantic_brief(
        pool,
        catalog=catalog,
        persona="gym",
        family="log",
        amount_path="explicit_grams",
        occasion="lunch",
        scene="empty",
    )
    text = render_semantic_brief(brief)
    assert "gram" in text.lower()
    assert "=" not in text
    named = render_semantic_brief(
        build_semantic_brief(
            pool,
            catalog=catalog,
            persona="everyday",
            family="log",
            amount_path="named_measure",
            occasion="lunch",
            scene="empty",
        )
    ).lower()
    assert "do not mention grams" in named


def test_synth_expander_still_bindable_through_generate_one(catalog):
    person = TRAIN_ROSTER[0]
    expander = synth_expander(catalog)
    result = generate_one(
        catalog=catalog,
        family="log",
        person=person,
        seed=0,
        occasion="lunch",
        amount_path="named_measure",
        expander=expander,
        enable_semantic_vote=False,
    )
    assert result.accepted is not None, result.rejected
    query = result.accepted.query
    assert query.strip()
    assert "User:" not in query
    assert query.count("\n\n") == 0


def test_author_task_single_query_with_synth(catalog):
    task, reject = author_task(
        _log_intent(recovery_trap=None),
        catalog=catalog,
        expander=synth_expander(catalog),
    )
    assert reject is None, reject
    assert task is not None
    assert "User:" not in task.query
    assert "Assistant:" not in task.query


@pytest.mark.skipif(
    os.environ.get("NUTRIMIND_ALLOW_NETWORK") != "1"
    or not os.environ.get("COMMANDCODE_API_KEY"),
    reason="live call: set NUTRIMIND_ALLOW_NETWORK=1 and COMMANDCODE_API_KEY",
)
def test_live_brief_expander_smoke(catalog, pool):
    from src.training.data_factory.config import load_config
    from src.training.data_factory.rollout import make_ark_expander_client

    config = load_config(
        pathlib.Path(__file__).resolve().parents[3] / "configs" / "data_factory.yaml"
    )
    expander = make_brief_expander(
        complete=complete_from_chat_client(
            make_ark_expander_client(config.expander)
        ),
        catalog=catalog,
        parse_retries=config.expander.parse_retries,
    )
    out = expander(pool, persona="everyday", family="log", amount_path="named_measure")
    assert out["query"].strip()
    assert out["foods"] and isinstance(out["foods"][0], str)
