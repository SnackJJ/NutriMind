"""Search locatability — can the agent actually find the pinned food?

The student acts through NutriEnv's `search_foods` tool: SQLite FTS5 BM25 over the
catalog's name/alias text, **AND** semantics, `SEARCH_LIMIT` rows returned. A task is
only well-defined if that search reaches the pinned food. If the words a speaker
would use return several foods, the agent picks one of them and the Scorer compares a
different food's end state — a false negative nothing inside our own pipeline sees.

So the judge is the environment's own search, not a string test. These facts about that
search shape the implementation, all measured against the pinned catalog:

- **AND semantics.** A term that matches nothing zeroes the whole query. This matters
  for a term the *speaker* adds ("from the cafeteria" is in no food name); it does not
  license dropping words from the food's own name, where every term is indexed by
  construction.
- **A form is searched as the lab would search it, and nothing else.** `_tokens` is
  `[a-z0-9]+` plus a `len(tok) >= 2` filter, and the FTS table tokenises with
  `unicode61`: hyphens and `%` are separators, single characters are dropped, and
  multi-digit numbers are real terms (`pineapple juice 100` returns 1 row where
  `pineapple juice` returns 2). No other word may be dropped: every one of them is in
  the indexed text, so removing one can only widen the query. Measured, dropping
  "with"/"without" merges `Chicken breast, grilled with sauce` with its
  `without sauce` twin and reports both `ambiguous`; an earlier version dropped
  "frozen", "wing" and "100" and manufactured every one of its verdicts.
- **A name can be ambiguous without being unreachable.** `Milk, reduced fat (2%)` is
  returned by its own words, next to thirteen neighbours: a qualifier can still
  separate them. That is a different problem from a food the search never surfaces.
- **The agent reads the utterance, not the record.** A catalog name is full of FNDDS
  furniture ("NS as to fat", "made from any kind of meat") that no speaker says, so
  judging the name measures a query nobody issues. What the utterance carries is the
  brief's handle — the phrase `speech.py` tells the author to say word for word — and
  it is much weaker: on the planner's own pins, `Bread, wheat or cracked wheat,
  toasted` is spoken as "toasted bread", which returns 25 rows without the pin. The
  record's own words stay in as a diagnostic (they are what detects a wrong id), but
  they never decide a verdict: measured over the catalog, the name form is never worse
  than the spoken forms, so the report is the spoken one.

This module only *judges*. Which pins to reject, and which qualifier a speaker should
be asked for, is a separate decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from nutrienv.world.catalog import SEARCH_LIMIT

from src.training.data_factory.pool_filter import spoken_identity

__all__ = [
    "Locatability",
    "judge_food",
    "search_locatability",
    "search_words",
]

# `ambiguous` and `unreachable` are different problems, and both differ from a name
# whose terms simply do not match the food at all:
#   unique       the search returns this food and nothing else
#   ambiguous    the food is returned, among neighbours — a qualifier can separate
#   unreachable  its terms match but BM25 ranks it past the rows the tool returns
#   unmatched    the terms do not retrieve the food at all (a tokenisation bug or a
#                wrong food id), not a property of the food
#   no_terms     the name carries nothing searchable
Locatability = Literal[
    "unique", "ambiguous", "unreachable", "unmatched", "no_terms"
]

# The lab's own tokeniser: `[a-z0-9]+`. Keeping hyphens would look meaningful and
# change nothing — the lab splits the needle before MATCH (measured identical).
_WORD = re.compile(r"[a-z0-9]+")


def search_words(text: object, *, extra: tuple[str, ...] = ()) -> list[str]:
    """The searchable terms of ``text``, in order, deduplicated.

    The lab's tokens and nothing else: `[a-z0-9]+`, single characters dropped (its
    `len(tok) >= 2` filter), the rest kept. Every remaining word is in the indexed text
    by construction, so dropping one could only widen the query and merge this food
    with a neighbour that shares the rest of its words.
    """
    out: list[str] = []
    for raw in [*(str(text).split() if text else []), *extra]:
        for token in _WORD.findall(str(raw).lower()):
            if len(token) < 2:
                continue
            if token not in out:
                out.append(token)
    return out


@dataclass(frozen=True)
class LocatabilityVerdict:
    """What the environment's search does with one spoken form of a food."""

    status: Locatability
    terms: tuple[str, ...] = ()
    hit_ids: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return self.status == "unique"

    @property
    def ordinal(self) -> int:
        """Worst-first ordering, for reporting the weakest form of a food."""
        return _ORDINAL[self.status]


_ORDINAL = {
    "unique": 0,
    "ambiguous": 1,
    "unreachable": 2,
    "unmatched": 3,
    "no_terms": 4,
}


def _probe(terms: list[str], food_id: str, catalog) -> LocatabilityVerdict:
    if not terms:
        return LocatabilityVerdict(status="no_terms")
    hits = list(catalog.search(" ".join(terms), limit=SEARCH_LIMIT))
    ids = tuple(dict.fromkeys(str(hit["food_id"]) for hit in hits))
    if ids == (str(food_id),):
        return LocatabilityVerdict("unique", tuple(terms), ids)
    if str(food_id) in ids:
        return LocatabilityVerdict("ambiguous", tuple(terms), ids)
    # The food's own words did not retrieve it. Past the agent's slice is a
    # different diagnosis from a query that never matched, and telling them apart
    # is what stops a wrong food id from reading as a hopeless one.
    status: Locatability = "unreachable" if len(ids) >= SEARCH_LIMIT else "unmatched"
    return LocatabilityVerdict(status, tuple(terms), ids)


def search_locatability(
    food_id: str,
    name: object,
    *,
    catalog,
    aliases: tuple[str, ...] = (),
    spoken: object = None,
) -> LocatabilityVerdict:
    """How far the agent's own search gets from a spoken form of this food.

    Each form is judged **on its own** — the catalog name, each alias, and ``spoken``
    (the handle the brief commits the utterance to). Gluing them into one query would
    fake uniqueness: `Milk, whole` plus its alias "full fat milk" returns a single
    row, while the query a speaker would actually say ("whole milk") returns
    seventeen.

    Returns the **weakest** verdict across the forms: a food is only as findable as
    its worst spoken form.
    """
    forms: list[tuple[str, ...]] = []
    name_terms = tuple(search_words(name))
    if name_terms:
        forms.append(name_terms)
    for text in (*aliases, spoken):
        if text is None:
            continue
        terms = tuple(search_words(text))
        if terms and terms not in forms:
            forms.append(terms)
    if not forms:
        return LocatabilityVerdict(status="no_terms")
    verdicts = [_probe(list(terms), food_id, catalog) for terms in forms]
    return max(verdicts, key=lambda verdict: verdict.ordinal)


def judge_food(food_id: str, *, catalog) -> LocatabilityVerdict:
    """Judge one catalog food across every form the pipeline can put in an utterance.

    Ids alone are not enough to call this: the handle is derived from the entry the
    same way ``speech.pin_speech_portion`` derives it, so a caller holding only a food
    id still gets the verdict the author would have spoken. An id the catalog does not
    hold is `unmatched` rather than `no_terms`: nothing about a food was judged there,
    the id simply retrieves nothing.
    """
    entry = catalog.get(food_id)
    if not entry:
        return LocatabilityVerdict(status="unmatched")
    aliases = tuple(entry.get("aliases") or ())
    return search_locatability(
        food_id,
        entry.get("name"),
        catalog=catalog,
        aliases=aliases,
        spoken=spoken_identity(entry.get("name"), aliases=aliases),
    )
