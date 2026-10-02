"""synthetic — the offline deterministic expander (spec §7, US-21).

A stand-in for the speech LLM used by tests, offline runs, and the CLI's
``--expander synthetic`` mode: picks one pool food via
``speakable_tracer_food`` and speaks a fixed ``{query, foods}`` pair — the
exact ``generate_one`` expander contract
(``callable(pool, *, persona, family, amount_path) -> {"query", "foods"}``).

No network, no LLM, fully seed-pinned (the seed lives in the intent).
Production runs use ``--expander commandcode`` (brief expander, ADR-011).

This is a stage module: it imports nutrienv at module level (allowed by spec
§18); the package ``__init__`` stays nutrienv-free.
"""

from __future__ import annotations

from nutrienv.bench.pipeline.sampler import speakable_tracer_food

from src.training.data_factory.speech import NEXT_RECOMMEND_OCCASION

__all__ = ["QWEN_MAX_MODEL", "qwen_max_fallback_expander", "synth_expander"]

QWEN_MAX_MODEL = "qwen3.8-max"


def synth_expander(catalog):
    """Wrap the gold catalog into a deterministic speech expander."""

    def _speak(pool, *, persona, family, amount_path=None, occasion="lunch"):
        picked = speakable_tracer_food(
            pool, catalog, amount_path=amount_path or "named_measure"
        )
        if picked is None:
            return {"query": "", "foods": []}
        food, phrase, spoken = picked
        meal = occasion if occasion in ("breakfast", "lunch", "dinner", "snack") else "lunch"
        query = f"For {meal} I had {phrase} of {spoken}."
        if family == "composite":
            nxt = NEXT_RECOMMEND_OCCASION.get(meal, "dinner")
            query = f"{query} What's for {nxt}?"
        return {
            "query": query,
            "foods": [food.food_id],
        }

    def _call(pool, *, persona, family, amount_path=None, occasion="lunch"):
        spoken = _speak(
            pool,
            persona=persona,
            family=family,
            amount_path=amount_path,
            occasion=occasion,
        )
        return spoken, {food.food_id for food in pool.foods}

    def bind_intent(intent):
        occasion = intent.get("occasion") or "lunch"

        def expander(pool, *, persona, family, amount_path=None):
            spoken, pool_ids = _call(
                pool,
                persona=persona,
                family=family,
                amount_path=amount_path,
                occasion=occasion,
            )
            expander.last_pool_ids = pool_ids
            return spoken

        expander.last_pool_ids = None
        return expander

    def expander(pool, *, persona, family, amount_path=None):
        spoken, pool_ids = _call(
            pool, persona=persona, family=family, amount_path=amount_path
        )
        expander.last_pool_ids = pool_ids
        return spoken

    expander.last_pool_ids = None
    expander.bind_intent = bind_intent
    return expander


def qwen_max_fallback_expander(catalog, *, complete=None):
    """§6 ladder last rung (``qwen3.8-max``). Offline: same bindable speech as
    ``synth_expander`` so tests need no network; ``complete`` is the live hook."""

    inner = synth_expander(catalog)

    def expander(pool, *, persona, family, amount_path=None):
        expander.calls += 1
        if complete is not None:
            result = complete(
                pool, persona=persona, family=family, amount_path=amount_path
            )
        else:
            result = inner(pool, persona=persona, family=family, amount_path=amount_path)
        expander.last_pool_ids = getattr(inner, "last_pool_ids", None)
        return result

    expander.model_id = QWEN_MAX_MODEL
    expander.calls = 0
    expander.last_pool_ids = None
    expander.bind_intent = inner.bind_intent
    return expander
