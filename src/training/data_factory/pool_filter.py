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
    "speakable_additions",
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


# Words a spoken addition may not begin or end on: the FNDDS boilerplate and the
# function words a qualifier phrase is built around. Without this, a run of the name
# like "and gravy" reads as a phrase and gets offered as speech.
_ADDITION_EDGE_WORDS = frozenset(
    {
        "nfs", "ns", "as", "to", "of", "and", "or", "with", "without", "the", "a",
        "an", "in", "on", "at", "for", "from", "made", "prepared", "ready", "other",
        "than", "including", "excluding", "kind", "form", "specified", "unspecified",
        "unclassified", "unknown",
    }
)

# Negation never reads as an addition: "not skin eaten" inverts what it was meant to
# narrow. The variants it distinguishes stay unreachable on purpose (see search_gate).
_ADDITION_BANNED = frozenset({"not", "no", "none", "never"})

_ADDITION_WORD = re.compile(r"[a-z][a-z'-]*")
_PARENTHETICAL = re.compile(r"\([^)]*\)")
_BOILERPLATE_SEGMENT = re.compile(
    r"\bnfs\b|\bns as to\b|^unspecified|^unclassified|^not specified|^specified\b|"
    r"^kind\b|^form\b|^other\b|^unknown\b"
)

# Longest addition offered, in words. Measured on the planner's own pins: a single
# additive phrase recovers what is recoverable, and a second one recovers nothing.
ADDITION_MAX_WORDS = 4


def speakable_additions(name: str | None) -> list[str]:
    """Phrases from a record that a speaker could add to a handle, most natural first.

    Adding a word the catalog does not carry would zero an AND query, so the only
    usable additions are the record's own words. Each phrase here is a run of at most
    ``ADDITION_MAX_WORDS`` whole words from one comma segment, with FNDDS boilerplate,
    negation and edge function words excluded.

    A segment is offered whole first — `New England clam chowder` and `sopa de fideo
    aguada` are the dish names, while their last word alone ("chowder", "aguada") is
    not what a speaker reaches for. Then the segment's tails shortest-first, which is
    what is left when the whole segment is too long or starts on a function word, then
    its interior runs. The order is presentation only: `search_gate` accepts a
    candidate because the search returns this food and nothing else.
    """
    ordered: list[str] = []
    for part in _segments(name or "")[1:]:
        text = _PARENTHETICAL.sub(" ", part).strip()
        if not text or _BOILERPLATE_SEGMENT.search(text.lower()):
            continue
        words = [
            word
            for word in _ADDITION_WORD.findall(text.lower())
            if len(word) >= 2 and "/" not in word
        ]
        if not words:
            continue
        whole = " ".join(words)
        if len(words) <= ADDITION_MAX_WORDS and _addition_ok(words):
            ordered.append(whole)
        tails: list[str] = []
        interior: list[str] = []
        for size in range(1, min(ADDITION_MAX_WORDS, len(words)) + 1):
            for start in range(len(words) - size + 1):
                run = words[start : start + size]
                if not _addition_ok(run):
                    continue
                phrase = " ".join(run)
                if phrase == whole:
                    continue
                target = tails if start + size == len(words) else interior
                if phrase not in target:
                    target.append(phrase)
        for phrase in (*tails, *interior):
            if phrase not in ordered:
                ordered.append(phrase)
    return ordered


def _addition_ok(run: list[str]) -> bool:
    if run[0] in _ADDITION_EDGE_WORDS or run[-1] in _ADDITION_EDGE_WORDS:
        return False
    return not any(word in _ADDITION_BANNED for word in run)
