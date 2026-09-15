"""Semantic brief expander — single-query speech (nutrimind-pilot/001).

Canonical entity is chosen in code. The live model sees a short brief
(situation, time/meal, source, persona, family intent, natural handle),
never a dumped catalog row. The callable shape is the ``generate_one``
expander contract. The synthetic expander is unchanged and may ignore briefs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from nutrienv.bench.pipeline.generate_one import (
    GRAM_UNITS,
    OUNCE_UNITS,
    UNIT_SYNONYMS,
    _WORD,
    _speech_amount_path,
    parse_query_foods_payload,
)
from nutrienv.bench.pipeline.sampler import (
    speakable_tracer_food,
    spoken_display_name,
    unit_naturalness_rank,
)

from src.training.data_factory.pool_filter import (
    filter_pool,
    is_suitable_meal_food,
    spoken_identity,
)
from src.training.data_factory.search_gate import qualifier_complement

__all__ = [
    "BRIEF_SYSTEM",
    "NEXT_RECOMMEND_OCCASION",
    "SemanticBrief",
    "SpeechPin",
    "bind_speech_context",
    "build_semantic_brief",
    "complete_from_chat_client",
    "make_brief_expander",
    "pin_speech_portion",
    "render_semantic_brief",
    "revision_hint",
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

# Revision instructions for a rejected first attempt, keyed by the reject's
# reason vocabulary. Zero-LLM: the rejected reason already names the fault.
_REVISION_HINTS = {
    "amount_path": (
        "Rejected: the amount word you used is not allowed. You must write "
        "\"{portion}\" — use that exact measure word for the logged food."
    ),
    "unresolvable": (
        "Rejected: the amount did not parse against the food. Write the amount as "
        "\"{portion}\", directly in front of the food name."
    ),
    "steps": (
        "Rejected: the logged-meal clause must contain one of the words \"log\", "
        "\"ate\", \"eaten\", \"had\" — past tense, not \"logging\" or \"logged\"."
    ),
    "query_foods_mismatch": (
        "Rejected: the sentence did not name the logged food closely enough. Write "
        "the food's own name in the sentence, with its amount as \"{portion}\"."
    ),
    "intent_conflict": (
        "Rejected: the sentence conflicts with the requested meal or amount. Keep "
        "the same meal, and state the amount as \"{portion}\"."
    ),
    "ambiguous": "Name each food with its own distinct words; do not merge them.",
    "omitted_food": "Mention every food you log, in the sentence.",
}


def revision_hint(reason: str, *, portion: str = "", query: str = "") -> str | None:
    """Instruction for a second attempt after ``reason`` rejected the first.

    ``portion`` is the pinned phrase from the rejected brief and ``query`` the
    rejected utterance. A hint never states a placeholder in place of a real
    phrase: an amount hint without a pinned portion is dropped rather than told to
    the model, which would otherwise echo the placeholder back as its answer.
    Returns None when the reason is not one a rewrite can fix, so the caller keeps
    the plain retry behaviour.
    """
    if reason == "steps" and "?" not in query and "what" not in query.lower():
        return (
            "Rejected: state plainly what you already ate, in the past tense — the "
            "word \"had\" or \"ate\". Do not write \"logging\" or \"logged\"."
        )
    template = _REVISION_HINTS.get(reason)
    if template is None:
        return None
    if "{portion}" in template and not portion:
        return None
    return template.format(portion=portion)

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

# Where the recommend ask starts in a composite utterance — the same pattern the
# lab uses to cut the log span (`_composite_speech_spans`). A composite names two
# meals, so anything that reads one meal out of the sentence must read this span.
_REC_ASK = re.compile(
    r"what(?:'s| is) for|what should i (?:eat|have)|should i have|recommend",
    re.I,
)


@dataclass(frozen=True)
class SpeechPin:
    """A portion commitment made in code, before any LLM call.

    ``phrase`` is literal speech ("a cup", "150 g", "a bowl"); ``unit`` is the unit
    word inside it. The brief hands the expander ``phrase`` so the uttered amount
    cannot land outside the intent's amount path, which is checked as an exactly
    equal class by the binder.
    """

    phrase: str
    unit: str
    klass: str


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
    portion: str = ""
    feedback: str = ""


def _alternative_rank(alt):
    return (unit_naturalness_rank(alt.key), alt.grams, alt.phrase)


def _pin_for(food, amount_path: str) -> SpeechPin | None:
    """The portion this food speaks under ``amount_path``, or None if it cannot.

    Phrase choice mirrors the milli's tracer (top-ranked quantity-1.0 alternative),
    plus one requirement the tracer does not make: the chosen phrase must classify
    as ``amount_path``. That check is what stops a QNS ``serving`` word ("a bowl")
    from being handed to a ``named_measure`` intent, which the binder rejects.
    """
    candidates = [alt for alt in (food.alternatives or ()) if alt.quantity == 1.0]
    if not candidates:
        return None
    if amount_path == "explicit_grams":
        ranked = sorted((a for a in candidates if a.key != "qns"), key=_alternative_rank)
        for alt in ranked:
            phrase = f"{alt.grams:g} g"
            if _speech_amount_path(phrase) == amount_path:
                return SpeechPin(phrase=phrase, unit="g", klass=amount_path)
        return None
    ranked = sorted((a for a in candidates if a.key != "qns"), key=_alternative_rank)
    for alt in ranked:
        if _speech_amount_path(alt.phrase) == amount_path:
            return SpeechPin(
                phrase=alt.phrase, unit=_unit_word(alt.phrase), klass=amount_path
            )
    if amount_path == "unspecified":
        for alt in candidates:
            if alt.key == "qns":
                return SpeechPin(
                    phrase=_QNS_SPEECH,
                    unit=_unit_word(_QNS_SPEECH),
                    klass=amount_path,
                )
    return None


def _unit_word(phrase: str) -> str:
    """The unit word a phrase is parsed by ("half a cup" -> "cup")."""
    for token in _WORD.findall(phrase.lower())[::-1]:
        if token in UNIT_SYNONYMS or token in GRAM_UNITS or token in OUNCE_UNITS:
            return token
    return phrase


# FNDDS QNS words people actually say. The catalog's own phrase for the qns key is
# "a serving", which no speaker uses; the amount path is still the same class, so
# the natural word is the one worth asking for.
_QNS_SPEECH = "a bowl"


def pin_speech_portion(pool, *, amount_path: str, catalog: Mapping) -> tuple:
    """First pool food that can speak ``amount_path`` **and** be found by the agent.

    Returns ``(food, handle, pin)`` or ``(None, None, None)``. Scoring is the
    milli's own ``speakable_tracer_food`` (collision-free, gram-resolvable); the
    pin is rejected on top of that when its phrase would classify as a different
    amount path, and the search moves to the next food. Foods a roster adult would
    not log (infant formula, baby food) are skipped.

    A food also has to be locatable: something a speaker can say returns it from the
    environment's own search, and nothing else (`search_gate.qualifier_complement`).
    A pin whose handle buries it is not a task — the agent logs a neighbour and the
    Scorer compares a different food — so the search moves on. The handle returned is
    the form that was measured to locate it, which is the brief's phrase; a pool with
    no such food authors nothing, which cost 0.4% of intents when measured.
    """
    for food in pool.foods:
        entry = catalog.get(food.food_id) or {}
        if not is_suitable_meal_food(entry.get("name")):
            continue
        pin = _pin_for(food, amount_path)
        if pin is None:
            continue
        picked = speakable_tracer_food(
            _pool_with(pool, (food,)), catalog, amount_path=amount_path
        )
        if picked is None:
            continue
        _food, phrase, spoken = picked
        if _speech_amount_path(phrase) != amount_path:
            continue
        # The lab's own phrase is a last resort for a record whose name carries no
        # usable segments; the derived handle is the normal path.
        aliases = tuple(entry.get("aliases") or ())
        fallback = None
        if not spoken_identity(entry.get("name"), aliases=aliases):
            fallback = spoken or spoken_display_name(catalog, food.food_id)
        fix = qualifier_complement(
            str(food.food_id), catalog=catalog, extra_form=fallback
        )
        if fix is None:
            continue
        return food, fix.phrase, pin
    return None, None, None


def _pool_with(pool, foods):
    """``pool`` narrowed to ``foods`` — keeps the collision check single-food."""
    return type(pool)(
        pool_id=pool.pool_id, family=pool.family, foods=tuple(foods)
    )


def build_semantic_brief(
    pool,
    *,
    catalog: Mapping,
    persona: str,
    family: str,
    amount_path: str,
    occasion: str = "lunch",
    scene: str = "empty",
    feedback: str = "",
) -> SemanticBrief | None:
    """Pick the canonical entity in code and return a brief, or None if none bind."""
    food, handle, pin = pin_speech_portion(pool, amount_path=amount_path, catalog=catalog)
    if food is None:
        return None
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
        portion=pin.phrase,
        feedback=feedback,
    )


def render_semantic_brief(brief: SemanticBrief) -> str:
    """Prose brief. No catalog field list, no internal ids.

    When the code committed a portion, the amount line names that literal phrase
    instead of the generic class cue: the utterance may still be written freely,
    but it cannot land outside the intent's amount path.
    """
    rec_ask = ""
    if brief.family == "composite":
        nxt = NEXT_RECOMMEND_OCCASION.get(brief.occasion, "dinner")
        rec_ask = (
            f" After logging, ask what to eat next with a phrase like "
            f"\"What's for {nxt}?\"."
        )
    if brief.portion:
        amount_line = (
            f"State the amount for that food exactly as \"{brief.portion}\" "
            f"(this meal's amount is fixed; do not substitute another quantity "
            f"word). {brief.amount_cue}"
        )
    else:
        amount_line = brief.amount_cue
    name_line = (
        f"Name the food exactly \"{brief.entity_handle}\", word for word — do not "
        f"shorten, tidy, or reorder it."
    ) if brief.entity_handle else ""
    revision = f" {brief.feedback}" if brief.feedback else ""
    return (
        f"{brief.situation} "
        f"Persona: {brief.persona}. "
        f"Intent: {brief.intent_line}. "
        f"Meal: {brief.occasion}. Source: {brief.source}. "
        f"Express the selected {brief.entity_handle} naturally in one utterance. "
        f"{name_line} "
        f"{amount_line} "
        "Do not mention catalog fields or internal IDs. "
        "Include only a natural cue that distinguishes home preparation from "
        "restaurant food when that contrast matters."
        f"{rec_ask}"
        f"{revision}"
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
        feedback = ""

        def expander(pool, *, persona, family, amount_path=None):
            pool = filter_pool(pool, catalog)
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
                feedback=feedback,
            )
            if brief is None:
                expander.last_portion = ""
                return {"query": "", "foods": []}
            expander.last_portion = brief.portion
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

        def bind_feedback(text: str | None) -> None:
            """Carry a rejected attempt's reason into the next brief."""
            nonlocal feedback
            feedback = text or ""

        expander.bind_feedback = bind_feedback  # type: ignore[attr-defined]
        expander.last_pool_ids = None
        expander.last_portion = ""
        return expander

    def expander(pool, *, persona, family, amount_path=None):
        return bind_intent({})(pool, persona=persona, family=family, amount_path=amount_path)

    expander.bind_intent = bind_intent  # type: ignore[attr-defined]
    expander.last_pool_ids = None
    expander.last_portion = ""
    return expander


def bind_speech_context(expander, intent: Mapping):
    """Close over intent occasion/scene when the expander speaks from a brief."""
    bind = getattr(expander, "bind_intent", None)
    if bind is None:
        return expander
    return bind(intent)
