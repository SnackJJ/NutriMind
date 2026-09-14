"""Semantic brief expander — single-query speech (nutrimind-pilot/001).

Canonical entity is chosen in code. The live model sees a short brief
(situation, time/meal, source, persona, family intent, natural handle),
never a dumped catalog row. The callable shape is the ``generate_one``
expander contract. The synthetic expander is unchanged and may ignore briefs.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from nutrienv.bench.pipeline.generate_one import parse_query_foods_payload
from nutrienv.bench.pipeline.sampler import speakable_tracer_food, spoken_display_name

__all__ = [
    "BRIEF_SYSTEM",
    "NEXT_RECOMMEND_OCCASION",
    "SemanticBrief",
    "bind_speech_context",
    "build_semantic_brief",
    "complete_from_chat_client",
    "make_brief_expander",
    "render_semantic_brief",
]

BRIEF_SYSTEM = (
    "Write one natural user query, not a database record. "
    "Use the situation, time/meal, source, persona, and intent in the brief. "
    "Mention enough of a cue to distinguish home cooking from restaurant food "
    "when the brief asks for that. "
    "Do not list internal IDs, catalog fields, or every available attribute. "
    "Do not invent foods, quantities, preparation methods, or venues. "
    "Preserve the requested amount path and meal semantics. "
    "Return a JSON object only: {\"query\": \"<one utterance>\"}."
)

_FAMILY_INTENT = {
    "log": "logging a meal they already ate",
    "update": "correcting a profile fact",
    "recommend": "asking for a next-meal recommendation",
    "evaluate": "asking whether a planned or eaten meal is okay",
    "composite": "logging a meal and asking what to eat next",
}

_AMOUNT_CUE = {
    "explicit_grams": "Speak the amount in household grams if an amount is needed.",
    "named_measure": "Speak a named household measure. Do not mention grams.",
    "unspecified": "Do not name a precise quantity.",
}

_OCCASION_SITUATION = {
    "breakfast": ("home kitchen", "The user is making breakfast at home."),
    "lunch": ("cafeteria", "The user is logging lunch they already ate."),
    "dinner": (
        "market",
        "The user bought ingredients at a market and is cooking dinner at home.",
    ),
    "snack": ("home kitchen", "The user is having a snack at home."),
}

# Matches nutrienv generate_one `_NEXT_OCCASION` for log→recommend speech.
NEXT_RECOMMEND_OCCASION = {
    "breakfast": "lunch",
    "lunch": "dinner",
    "dinner": "dinner",
    "snack": "dinner",
}


@dataclass(frozen=True)
class SemanticBrief:
    """Facts the expander may see. ``food_id`` is code-side only — not rendered."""

    family: str
    persona: str
    amount_path: str
    occasion: str
    source: str
    situation: str
    entity_handle: str
    amount_cue: str
    food_id: str
    intent_line: str


def build_semantic_brief(
    pool,
    *,
    catalog: Mapping,
    persona: str,
    family: str,
    amount_path: str,
    occasion: str = "lunch",
    scene: str = "empty",
) -> SemanticBrief | None:
    """Pick the canonical entity in code and return a brief, or None if none bind."""
    picked = speakable_tracer_food(pool, catalog, amount_path=amount_path)
    if picked is None:
        return None
    food, _phrase, spoken = picked
    handle = spoken or spoken_display_name(catalog, food.food_id)
    source, situation = _OCCASION_SITUATION.get(
        occasion, ("home kitchen", f"The user is talking about {occasion}.")
    )
    if scene and scene != "empty":
        situation = f"{situation} Leftovers from an earlier meal are in play."
    return SemanticBrief(
        family=family,
        persona=persona,
        amount_path=amount_path,
        occasion=occasion,
        source=source,
        situation=situation,
        entity_handle=handle,
        amount_cue=_AMOUNT_CUE.get(
            amount_path, "Preserve the requested amount path."
        ),
        food_id=food.food_id,
        intent_line=_FAMILY_INTENT.get(family, family),
    )


def render_semantic_brief(brief: SemanticBrief) -> str:
    """Prose brief. No catalog field list, no internal ids."""
    rec_ask = ""
    if brief.family == "composite":
        nxt = NEXT_RECOMMEND_OCCASION.get(brief.occasion, "dinner")
        rec_ask = (
            f" After logging, ask what to eat next with a phrase like "
            f"\"What's for {nxt}?\"."
        )
    return (
        f"{brief.situation} "
        f"Persona: {brief.persona}. "
        f"Intent: {brief.intent_line}. "
        f"Meal: {brief.occasion}. Source: {brief.source}. "
        f"Express the selected {brief.entity_handle} naturally in one utterance. "
        f"{brief.amount_cue} "
        "Do not mention catalog fields or internal IDs. "
        "Include only a natural cue that distinguishes home preparation from "
        "restaurant food when that contrast matters."
        f"{rec_ask}"
    )


def _speech_payload(raw: object, food_id: str) -> dict[str, object] | None:
    parsed = parse_query_foods_payload(raw)
    if parsed is not None:
        parsed["foods"] = [food_id]
        return parsed
    data = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(data, Mapping):
        return None
    query = data.get("query")
    if isinstance(query, str) and query.strip():
        return {"query": query.strip(), "foods": [food_id]}
    return None


def complete_from_chat_client(
    client: Callable[[dict], Mapping],
) -> Callable[[str, Sequence[Mapping[str, str]]], str]:
    """Adapt ``teacher_complete``-shaped clients to ``make_brief_expander``."""

    def complete(_tag: str, messages: Sequence[Mapping[str, str]]) -> str:
        completion = client({"messages": list(messages)})
        return completion.get("content") or ""

    return complete


def make_brief_expander(
    *,
    complete: Callable[[str, Sequence[Mapping[str, str]]], str],
    catalog: Mapping,
    parse_retries: int = 1,
):
    """Live expander: brief in, ``{query, foods}`` out. ``foods`` is the code pick."""

    retries = max(0, int(parse_retries))

    def bind_intent(intent: Mapping):
        occasion = intent.get("occasion") or "lunch"
        scene = intent.get("scene") or "empty"

        def expander(pool, *, persona, family, amount_path=None):
            expander.last_pool_ids = {food.food_id for food in pool.foods}
            path = amount_path or "named_measure"
            brief = build_semantic_brief(
                pool,
                catalog=catalog,
                persona=persona,
                family=family,
                amount_path=path,
                occasion=occasion,
                scene=scene,
            )
            if brief is None:
                return {"query": "", "foods": []}
            messages = (
                {"role": "system", "content": BRIEF_SYSTEM},
                {"role": "user", "content": render_semantic_brief(brief)},
            )
            last: dict[str, object] = {"query": "", "foods": []}
            for _ in range(1 + retries):
                parsed = _speech_payload(
                    complete("brief-expander", messages), brief.food_id
                )
                if parsed is not None:
                    return parsed
            return last

        expander.last_pool_ids = None
        return expander

    def expander(pool, *, persona, family, amount_path=None):
        return bind_intent({})(pool, persona=persona, family=family, amount_path=amount_path)

    expander.bind_intent = bind_intent  # type: ignore[attr-defined]
    expander.last_pool_ids = None
    return expander


def bind_speech_context(expander, intent: Mapping):
    """Close over intent occasion/scene when the expander speaks from a brief."""
    bind = getattr(expander, "bind_intent", None)
    if bind is None:
        return expander
    return bind(intent)
