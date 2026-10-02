"""Query/entity consistency after speech (nutrimind-pilot/002).

Canonical binding is code-side. If the utterance is still ambiguous or
disagrees with the intent / ``foods``, regenerate or author-reject. The
Scorer never guesses among catalog variants.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from nutrienv.bench.pipeline.sampler import spoken_display_name

from src.training.data_factory.speech import _REC_ASK
from src.training.data_factory.search_gate import search_words

__all__ = [
    "CONSISTENCY_CODES",
    "foods_from_task",
    "query_entity_consistency",
]

CONSISTENCY_CODES = frozenset(
    {
        "author.foods_outside_binding",
        "author.unselected_variant",
        "author.missing_disambiguation",
        "author.intent_conflict",
        "author.query_foods_mismatch",
        "author.ambiguous_entity",
        "author.missing_identifying_words",
    }
)

_MEALS = ("breakfast", "brunch", "lunch", "dinner", "snack")
_GRAMS = re.compile(r"\b\d+(?:\.\d+)?\s*g(?:rams?)?\b", re.I)


def _log_span(query: str, family: str) -> str:
    """The part of the utterance that describes the logged meal.

    A composite names two meals: the one just eaten and the one being asked about.
    The ask is derived from the eaten meal (breakfast asks about lunch), so on a
    lunch intent "What's for dinner?" is *correct* — checking the whole sentence
    against the intent's occasion rejects it. The lab splits the same way
    (`_composite_speech_spans`) before binding; do the same here.
    """
    if not family.startswith("composite"):
        return query
    rec = _REC_ASK.search(query)
    return query[: rec.start()] if rec else query


def _handles(catalog: Mapping, food_id: str) -> list[str]:
    """Surface forms the binder itself accepts for this food, longest first-ish.

    `nutrienv`'s `_local_clause` locates a food by matching the catalog name, its
    **first-comma head**, and the aliases against a comma-split clause. The check
    here has to accept the same set: a scrambled `spoken_display_name`
    ("made from any kind of meat reduced fat pastrami") is a *different* noun
    phrase, and an utterance that says "reduced-fat pastrami" — which binds grams
    correctly — used to be rejected for not containing the scrambled form.
    """
    entry = catalog.get(food_id) or {}
    aliases = [str(alias).strip().lower() for alias in (entry.get("aliases") or [])]
    name = str(entry.get("name") or food_id).strip().lower()
    out: list[str] = []
    for item in aliases + [name, name.split(",", 1)[0].strip()]:
        if item and item not in out:
            out.append(item)
    return out


def _head(catalog: Mapping, food_id: str) -> str:
    """The comma head — the food's identity as the binder resolves it."""
    entry = catalog.get(food_id) or {}
    name = str(entry.get("name") or "").strip()
    if name:
        return name.split(",", 1)[0].strip().lower()
    return str(food_id).lower()


def _primary(catalog: Mapping, food_id: str) -> str:
    handles = _handles(catalog, food_id)
    return handles[0] if handles else food_id.lower()


def _oracle_food_ids(oracle) -> list[str]:
    ids: list[str] = []
    for row in oracle.ledger_tail or oracle.ledger or ():
        food_id = getattr(row, "food_id", None)
        if food_id is None and isinstance(row, Mapping):
            food_id = row.get("food_id")
        if food_id:
            ids.append(str(food_id))
    for item in oracle.evaluated_plan or ():
        food_id = item.get("food_id") if isinstance(item, Mapping) else None
        if food_id:
            ids.append(str(food_id))
    return ids


def foods_from_task(task) -> list[str]:
    """Bound food ids on a ``Task`` (ledger / evaluated plan / composite legs)."""
    oracle = getattr(task, "oracle", None)
    if oracle is None:
        return []
    sources = list(oracle.sub_oracles) if oracle.sub_oracles else [oracle]
    ids: list[str] = []
    for source in sources:
        ids.extend(_oracle_food_ids(source))
    seen: set[str] = set()
    unique: list[str] = []
    for food_id in ids:
        if food_id not in seen:
            seen.add(food_id)
            unique.append(food_id)
    return unique


def query_entity_consistency(
    query: str,
    foods: Sequence[str],
    *,
    intent: Mapping,
    catalog: Mapping,
    allowed_ids: set[str] | None = None,
    required_words: Sequence[str] = (),
) -> str | None:
    """Return an ``author.*`` consistency code, or None if the utterance binds.

    ``required_words`` are the words the agent's own search needs in order to reach the
    pinned food (`search_gate.identifying_words`). They are checked **as tokens**, not
    as substrings: a required `icing` is not satisfied by `icing`'s letters sitting
    inside a longer word. Naming the food by a form the binder accepts is not enough on
    its own — the utterance has to be *findable*, which is the whole reason the words
    exist. Callers that pass none keep the binder-only contract.
    """
    blob = (query or "").lower()
    food_ids = [str(food_id) for food_id in foods]
    family = str(intent.get("family") or "")
    # The archetype families bind every food in code (archetypes.py).
    if family in ("update", "recommend", "evaluate", "evaluate_hypo",
                  "recommend_inventory", "recommend_menu",
                  "composite_amend_recommend", "composite_refuse_recommend"):
        return None

    for food_id in food_ids:
        if food_id not in catalog or (
            allowed_ids is not None and food_id not in allowed_ids
        ):
            return "author.foods_outside_binding"

    occasion = str(intent.get("occasion") or "").lower()
    # the occasion must be the one *eaten*, so read only the log span
    log_blob = _log_span(query or "", family).lower()
    meals = [meal for meal in _MEALS if re.search(rf"\b{re.escape(meal)}\b", log_blob)]
    if occasion and meals and occasion not in meals:
        return "author.intent_conflict"
    if (intent.get("amount_path") or "") == "named_measure" and _GRAMS.search(log_blob):
        return "author.intent_conflict"

    bound_heads = [_head(catalog, food_id) for food_id in food_ids]
    bound_handles = [_primary(catalog, food_id) for food_id in food_ids]
    # Any surface form the binder accepts counts as naming the food. Findability is a
    # separate requirement, checked in the same pass because both are about the same
    # thing: whether this utterance can be resolved to this food at all.
    present = set(search_words(query))
    for food_id in food_ids:
        forms = _handles(catalog, food_id)
        if any(form and form in blob for form in forms):
            continue
        return "author.query_foods_mismatch"
    missing = [word for word in required_words if word not in present]
    if missing:
        return "author.missing_identifying_words"

    if str(intent.get("family") or "").startswith("composite"):
        return None

    neighbors = allowed_ids if allowed_ids is not None else catalog
    bound_displays = [
        spoken_display_name(catalog, food_id).lower() for food_id in food_ids
    ]
    for food_id in neighbors:
        if food_id in food_ids:
            continue
        other = spoken_display_name(catalog, food_id).lower()
        if len(other) < 6 or other not in blob:
            continue
        if any(
            other == bound or (other in bound and len(other) < len(bound))
            for bound in bound_displays
        ):
            continue
        return "author.unselected_variant"

    if len(food_ids) >= 2:
        heads = [handle.split()[0] for handle in bound_handles if handle]
        if heads and len(set(heads)) < len(heads):
            return "author.ambiguous_entity"
        for index, handle in enumerate(bound_handles):
            for other in bound_handles[index + 1 :]:
                if handle.startswith(other + " ") or other.startswith(handle + " "):
                    return "author.ambiguous_entity"

    for food_id, head in zip(food_ids, bound_heads):
        if not head:
            continue
        for other_id in neighbors:
            if other_id == food_id:
                continue
            other = _head(catalog, other_id)
            if other.startswith(head + " ") and other not in blob:
                return "author.missing_disambiguation"
    return None
