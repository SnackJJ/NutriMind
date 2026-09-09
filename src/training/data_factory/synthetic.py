"""synthetic — the offline deterministic expander (spec §7, US-21).

A stand-in for the speech LLM used by tests, offline runs, and the CLI's
``--expander synthetic`` mode: picks one pool food via
``speakable_tracer_food`` and speaks a fixed ``{query, foods}`` pair — the
exact ``generate_one`` expander contract
(``callable(pool, *, persona, family, amount_path) -> {"query", "foods"}``).

No network, no LLM, fully seed-pinned (the seed lives in the intent).
Production runs use the ark expander wrapper (ticket 012).

This is a stage module: it imports nutrienv at module level (allowed by spec
§18); the package ``__init__`` stays nutrienv-free.
"""

from __future__ import annotations

from nutrienv.bench.pipeline.sampler import speakable_tracer_food

__all__ = ["synth_expander"]


def synth_expander(catalog):
    """Wrap the gold catalog into a deterministic speech expander."""

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
