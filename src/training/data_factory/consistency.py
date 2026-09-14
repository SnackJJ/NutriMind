"""Query/entity consistency after speech (nutrimind-pilot/002).

Canonical binding is code-side. If the utterance is still ambiguous or
disagrees with the intent / ``foods``, regenerate or author-reject. The
Scorer never guesses among catalog variants.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from nutrienv.bench.pipeline.sampler import spoken_display_name

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
    }
)

_MEALS = ("breakfast", "brunch", "lunch", "dinner", "snack")
_GRAMS = re.compile(r"\b\d+(?:\.\d+)?\s*g(?:rams?)?\b", re.I)


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
) -> str | None:
    """Return an ``author.*`` consistency code, or None if the utterance binds."""
    blob = (query or "").lower()
    food_ids = [str(food_id) for food_id in foods]
    family = str(intent.get("family") or "")
    if family in ("update", "recommend", "evaluate"):
        return None

    for food_id in food_ids:
        if food_id not in catalog or (
            allowed_ids is not None and food_id not in allowed_ids
        ):
            return "author.foods_outside_binding"

    occasion = str(intent.get("occasion") or "").lower()
    meals = [meal for meal in _MEALS if re.search(rf"\b{re.escape(meal)}\b", blob)]
    if occasion and meals and occasion not in meals:
        return "author.intent_conflict"
    if (intent.get("amount_path") or "") == "named_measure" and _GRAMS.search(blob):
        return "author.intent_conflict"

    bound_heads = [_head(catalog, food_id) for food_id in food_ids]
    bound_handles = [_primary(catalog, food_id) for food_id in food_ids]
    for food_id in food_ids:
        # any surface form the binder accepts counts as naming the food
        forms = _handles(catalog, food_id)
        if any(form and form in blob for form in forms):
            continue
        return "author.query_foods_mismatch"

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
