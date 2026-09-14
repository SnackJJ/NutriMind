"""Portion pinning: the amount class is a code commitment, not a parse.

The binder classifies the uttered amount word and requires it to equal the
intent's ``amount_path`` exactly (``_bind_log_foods``). A live expander left to
itself says "a bowl" for a ``named_measure`` intent — "bowl" resolves to the
``serving``/QNS class — and the draft dies as ``author.amount_path``. These tests
pin that the brief hands over a portion phrase that already classifies correctly.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench.pipeline.generate_one import _speech_amount_path  # noqa: E402
from nutrienv.bench.pipeline.sampler import sample_pools  # noqa: E402
from nutrienv.world.catalog_store import load_catalog  # noqa: E402

from src.training.data_factory.author import _retryable_author_reject  # noqa: E402
from src.training.data_factory.speech import (  # noqa: E402
    build_semantic_brief,
    pin_speech_portion,
    render_semantic_brief,
    revision_hint,
)

_AMOUNT_PATHS = ("explicit_grams", "named_measure", "unspecified")
_SEEDS = (0, 1, 7, 42, 101)


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def _pool(catalog, seed, family="log"):
    return sample_pools(catalog, seed=seed, family=family, n_pools=1)[0]


# --------------------------------------------------------------------------- #
# the pin classifies as the requested amount path
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("amount_path", _AMOUNT_PATHS)
@pytest.mark.parametrize("seed", _SEEDS)
def test_pin_phrase_classifies_as_its_amount_path(catalog, amount_path, seed):
    _food, _handle, pin = pin_speech_portion(
        _pool(catalog, seed), amount_path=amount_path, catalog=catalog
    )
    assert pin is not None, f"no pinnable food for {amount_path} at seed {seed}"
    assert pin.klass == amount_path
    assert _speech_amount_path(pin.phrase) == amount_path, pin.phrase


def test_named_measure_pin_is_never_a_qns_word(catalog):
    """"a bowl" is the failure this pin exists to prevent."""
    for seed in _SEEDS:
        _food, _handle, pin = pin_speech_portion(
            _pool(catalog, seed), amount_path="named_measure", catalog=catalog
        )
        assert pin is not None
        assert "bowl" not in pin.phrase
        assert "serving" not in pin.phrase


def test_unspecified_pin_is_natural_speech(catalog):
    """"a serving" is the catalog's phrase; nobody says it. QNS words are said."""
    _food, _handle, pin = pin_speech_portion(
        _pool(catalog, 0), amount_path="unspecified", catalog=catalog
    )
    assert pin is not None
    assert pin.phrase == "a bowl"
    assert _speech_amount_path(pin.phrase) == "unspecified"


def test_explicit_grams_pin_carries_the_number(catalog):
    _food, _handle, pin = pin_speech_portion(
        _pool(catalog, 0), amount_path="explicit_grams", catalog=catalog
    )
    assert pin is not None
    assert pin.phrase.endswith(" g")
    assert float(pin.phrase.split()[0]) > 0


# --------------------------------------------------------------------------- #
# the brief renders the pin instead of the generic class cue
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("amount_path", _AMOUNT_PATHS)
def test_brief_names_the_pinned_phrase(catalog, amount_path):
    brief = build_semantic_brief(
        _pool(catalog, 0),
        catalog=catalog,
        persona="everyday",
        family="log",
        amount_path=amount_path,
        occasion="lunch",
    )
    assert brief is not None
    assert brief.portion
    text = render_semantic_brief(brief)
    assert f'"{brief.portion}"' in text
    assert brief.amount_cue in text


def test_brief_without_a_pin_keeps_the_plain_cue(catalog):
    brief = build_semantic_brief(
        _pool(catalog, 0),
        catalog=catalog,
        persona="everyday",
        family="log",
        amount_path="named_measure",
        occasion="lunch",
    )
    stripped = render_semantic_brief(brief)
    assert "is fixed; do not substitute" in stripped


# --------------------------------------------------------------------------- #
# revision feedback: no placeholder ever reaches the model
# --------------------------------------------------------------------------- #


def test_amount_hint_names_the_pinned_phrase():
    hint = revision_hint("amount_path", portion="a cup")
    assert hint is not None
    assert '"a cup"' in hint


def test_amount_hint_is_dropped_without_a_portion():
    """A placeholder would be echoed back as the answer."""
    assert revision_hint("amount_path", portion="") is None
    assert revision_hint("unresolvable", portion="") is None
    assert revision_hint("query_foods_mismatch", portion="") is None


def test_steps_hint_targets_the_verb_form():
    hint = revision_hint("steps", query="I am logging a cup of cereal.")
    assert hint is not None
    assert "logging" in hint


def test_unknown_reason_yields_no_hint():
    assert revision_hint("no_ledger", portion="a cup") is None


def test_structural_rejects_are_not_retried():
    assert not _retryable_author_reject(
        {"failure_codes": ["author.unknown_shell"]}
    )
    assert not _retryable_author_reject({"failure_codes": ["author.illegal_pair"]})
    assert _retryable_author_reject({"failure_codes": ["author.amount_path"]})
    assert _retryable_author_reject(
        {"failure_codes": ["author.query_foods_mismatch"]}
    )


def test_author_task_retries_a_rewritable_reject(catalog, monkeypatch):
    """The retry must reach the strategy, and carry the pin into the feedback."""
    calls = {"n": 0, "feedback": []}

    def fake_strategy(intent, *, catalog, expander, gram_anchor=None, **kwargs):
        calls["n"] += 1
        calls["feedback"].append(getattr(expander, "_seen_feedback", ""))
        if calls["n"] == 1:
            from src.training.data_factory.author import _reject

            return None, _reject(intent, "author.amount_path", "generate_one rejected: amount_path ('a bowl of x')")
        return None, None

    class _Expander:
        last_portion = "a cup"

        def bind_feedback(self, text):
            self._seen_feedback = text

    monkeypatch.setitem(
        __import__(
            "src.training.data_factory.author", fromlist=["AUTHOR_STRATEGIES"]
        ).AUTHOR_STRATEGIES,
        "log",
        fake_strategy,
    )
    intent = {
        "task_id": "log--log--train-alba--000000",
        "family": "log",
        "user_id": "train-alba",
        "seed": 0,
        "occasion": "lunch",
        "amount_path": "named_measure",
    }
    from src.training.data_factory.author import author_task

    task, reject = author_task(intent, catalog=catalog, expander=_Expander(), parse_retries=1)
    assert calls["n"] == 2, "a rewritable reject must be retried"
    assert "a cup" in (calls["feedback"][1] or ""), calls["feedback"]
