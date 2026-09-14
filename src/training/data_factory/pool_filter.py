"""Pool suitability and spoken naming for the brief expander.

Two lab-side wordings fight the factory, and both are fixed here rather than in
`../nutri-env-lab` (read-only, ADR-012):

**Unsuitable foods.** The catalog spans the whole FNDDS range, so a pool drawn for
an adult meal can contain infant formula, baby food, or "strained" jars. A
RosterPerson never eats those, and a live expander happily logs them. No lab helper
excludes them, so the decision is ours; it is name-based rather than id-based so it
survives a catalog re-pin.

**Spoken names.** `nutrienv`'s `spoken_display_name` reverses a comma name's
trailing qualifiers, which reads as a different noun phrase:

    "Pastrami, made from any kind of meat, reduced fat"
      -> display  "made from any kind of meat reduced fat pastrami"
      -> spoken   "reduced fat pastrami"

The binder locates a food in the sentence by matching `_spoken_names` — the
catalog name, its first-comma head, and the aliases — split on commas. The
scrambled form matches none of them, while re-ordered qualifiers do: offline, all
of "pastrami", "pastrami, reduced fat", "reduced fat pastrami" and
"reduced-fat pastrami" resolve to the same 28.0 g. The head is what carries
identity; the qualifiers may move.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

__all__ = [
    "EXCLUDED_MARKERS",
    "filter_pool",
    "is_suitable_meal_food",
    "spoken_identity",
]

# Matched against the first comma segment of a catalog name, as a whole phrase.
# Names of the shape "<head>, <qualifier>, ..." keep their head as the key, so
# "Baby Toddler yogurt, plain" is caught by "baby toddler" and "Carrots, baby"
# is not caught by anything — baby carrots are a normal adult food.
EXCLUDED_MARKERS: tuple[str, ...] = (
    "infant formula",
    "baby toddler",
    "baby food",
    "junior food",
    "toddler formula",
    "strained",
    "junior",
)

_QUALIFIER_TAIL = (
    r"(?:,\s*)?(?:added|as to|with|without|made|prepared|reduced|low|high|"
    r"fat|lean|not specified|nfs|ns as to|ready-to-feed|powder|from|and|or)\b"
)


def _head(name: str) -> str:
    return str(name).split(",", 1)[0].strip()


def is_suitable_meal_food(name: object) -> bool:
    """False for foods a TRAIN_ROSTER adult would not log as a meal."""
    head = _head(name or "").lower()
    if not head:
        return True
    return not any(
        head == marker or head.startswith(marker) for marker in EXCLUDED_MARKERS
    )


def filter_pool(pool, catalog: Mapping):
    """``pool`` without foods whose catalog name marks them unsuitable.

    Empties the *foods* only when the pool has nothing left; a pool that would
    become empty is returned unchanged, because an empty pool is a hard failure
    downstream and a long shot beats no shot.
    """
    foods = tuple(
        food
        for food in pool.foods
        if is_suitable_meal_food((catalog.get(food.food_id) or {}).get("name"))
    )
    if not foods:
        return pool
    return type(pool)(pool_id=pool.pool_id, family=pool.family, foods=foods)


def _segments(name: str) -> list[str]:
    parts = [part.strip() for part in str(name).split(",")]
    if len(parts) > 1 and parts[-1].lower() in {"", "nfs"}:
        parts = parts[:-1]
    return [part for part in parts if part]


def _natural_case(text: str) -> str:
    """Title Case is a catalog artifact; a sentence says "reduced-fat pastrami"."""
    return text[0].lower() + text[1:] if text else text


def _qualifier(part: str) -> str | None:
    """A qualifier that reads as an adjective phrase, or None if it does not.

    "reduced fat" -> "reduced-fat". "made from any kind of meat" -> None: it is a
    clause, not a modifier, and hyphenating it produces the nonsense the display
    name is already guilty of. Measured and branded words are dropped too — "100%",
    "stage 2" and "NFS" do not belong in a spoken food name.
    """
    text = part.strip().lower()
    if not text or "," in text:
        return None
    if re.search(r"\bmade (?:with|from)\b|\bns as to\b|\bnfs\b", text):
        return None
    if re.search(r"[%]|\d", text):
        return None
    words = text.split()
    if not words or len(words) > 3:
        return None
    if words[0] in {"and", "or", "with", "without", "as", "to", "of"}:
        return None
    return "-".join(words)


def spoken_identity(
    name: str | None, *, aliases: Sequence[str] = ()
) -> str:
    """A natural noun phrase that still binds: head + the qualifiers that fit.

    "Pastrami, made from any kind of meat, reduced fat" -> "reduced-fat pastrami";
    "Oatmeal, instant, plain, made with non-dairy milk, fat added" ->
    "instant plain oatmeal". Falls back to the head alone, which is always a
    candidate the binder accepts, and to an alias only when there is no name.
    """
    for alias in aliases:
        if str(alias).strip():
            return _natural_case(str(alias).strip())
    parts = _segments(name or "")
    if not parts:
        return ""
    head = parts[0]
    qualifiers: list[str] = []
    for index, part in enumerate(parts[1:], start=1):
        qualifier = _qualifier(part)
        if qualifier is None:
            continue
        qualifiers.append(qualifier)
        if index >= 3 or len(qualifiers) >= 2:
            break
    if not qualifiers:
        return _natural_case(head)
    return " ".join([*qualifiers, _natural_case(head)])
