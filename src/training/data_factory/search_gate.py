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

from src.training.data_factory.pool_filter import (
    ADDITION_MAX_WORDS,
    speakable_additions,
    spoken_identity,
)

__all__ = [
    "Locatability",
    "SpokenFix",
    "identifying_words",
    "judge_food",
    "qualifier_complement",
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


@dataclass(frozen=True)
class SpokenFix:
    """A form that reaches the pin uniquely, and what it took to get there."""

    phrase: str
    source: Literal["handle", "alias", "addition"]
    verdict: LocatabilityVerdict
    added: str = ""

    @property
    def changed(self) -> bool:
        """True when the brief's phrase is not the handle the record derives alone."""
        return self.source != "handle"


def qualifier_complement(
    food_id: str,
    *,
    catalog,
    extra_form: str | None = None,
    max_words: int = ADDITION_MAX_WORDS,
) -> SpokenFix | None:
    """How to speak this pin so the environment's search reaches it uniquely.

    ``None`` means no natural form does: the record distinguishes this food only by
    words nobody says ("NS as to fat", "skin / coating not eaten"), so the pin is not
    a task. That is the verdict the selection rule needs — not `ambiguous`.

    Three answers, cheapest first:

    - ``handle``: the phrase the brief already asks for, unchanged.
    - ``alias``: another surface form the binder accepts, unchanged.
    - ``addition``: the handle plus a phrase from the record's own words. Only the
      record's words can be added, because an AND query containing a word the catalog
      does not carry matches nothing at all.

    ``extra_form`` is one more form to try as it stands, for a caller holding a phrase
    the record does not derive (the lab's own tracer phrase, when a name has no usable
    segments). The search decides, never the shape of the string: a candidate counts
    only when the phrase actually returns this food and nothing else.
    """
    entry = catalog.get(food_id)
    if not entry:
        return None
    name = entry.get("name")
    aliases = tuple(
        str(alias) for alias in (entry.get("aliases") or ()) if str(alias).strip()
    )
    if extra_form:
        aliases = (*aliases, str(extra_form))
    handle = spoken_identity(name, aliases=tuple(entry.get("aliases") or ()))
    handle_terms = search_words(handle)
    handle_set = set(handle_terms)
    if handle_terms:
        verdict = _probe(list(handle_terms), food_id, catalog)
        if verdict.status == "unique":
            return SpokenFix(phrase=handle, source="handle", verdict=verdict)
    for alias in aliases:
        terms = search_words(alias)
        if not terms or set(terms) <= handle_set:
            continue
        verdict = _probe(list(terms), food_id, catalog)
        if verdict.status == "unique":
            return SpokenFix(phrase=alias, source="alias", verdict=verdict)
    for addition in speakable_additions(name):
        terms = search_words(addition)
        if not terms or len(terms) > max_words or set(terms) <= handle_set:
            continue
        verdict = _probe([*terms, *handle_terms], food_id, catalog)
        if verdict.status == "unique":
            return SpokenFix(
                phrase=" ".join(part for part in (addition, handle) if part),
                source="addition",
                verdict=verdict,
                added=addition,
            )
    return None


def identifying_words(food_id: str, *, catalog) -> tuple[str, ...]:
    """The words an utterance must carry for the agent's own search to reach this food.

    This is the form `qualifier_complement` measured, expressed as words rather than as
    a phrase, which is what lets the brief ask for content instead of for someone
    else's word order: `icing yeast-type doughnut` is the requirement, "an iced yeast
    doughnut" is the utterance. Empty when no natural form locates the food at all —
    such a pin is not a task, and the authoring gate refuses it (see speech.py).
    """
    fix = qualifier_complement(food_id, catalog=catalog)
    return tuple(search_words(fix.phrase)) if fix is not None else ()
