"""author — one intent → one authored nutrienv ``Task`` (spec §4.3, §6 step 5a).

An intent names a family and its authoring knobs; the matching **strategy**
produces either an accepted ``Task`` (via ``generate_one`` + the *injected*
expander — never constructed here) or an author-stage reject record routed to
``rejects/author.jsonl`` (spec §8). Strategies register per config family;
families without a strategy yet reject cleanly with ``author.unsupported_
family`` so the run continues (failure isolation, spec §4.1) — ticket 012
widens update / recommend / evaluate, ticket 013 adds the 3-leg composite.

This is a stage module: it imports nutrienv at module level (allowed by spec
§18); the package ``__init__`` stays nutrienv-free.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from nutrienv.bench.pipeline.generate_one import generate_one
from nutrienv.bench.pipeline.types import Rejected

from src.training.data_factory.roster_train import TRAIN_ROSTER

__all__ = ["AUTHOR_STRATEGIES", "author_task", "person_for_intent"]

REJECT_SCHEMA_VERSION = "nutrimind-v2-reject/1"

# user_id → person (TRAIN_ROSTER is the single source of people, spec US-16)
_ROSTER_BY_ID = {person.user_id: person for person in TRAIN_ROSTER}


def person_for_intent(intent: Mapping):
    """Resolve an intent's ``user_id`` to its ``TRAIN_ROSTER`` person."""
    person = _ROSTER_BY_ID.get(intent["user_id"])
    if person is None:
        raise KeyError(f"intent user_id {intent['user_id']!r} is not in TRAIN_ROSTER")
    return person


def _reject(intent: Mapping, failure_code: str, detail: str) -> dict:
    """The ``rejects/author.jsonl`` line shape (mirrors gates.rejects_record)."""
    return {
        "schema_version": REJECT_SCHEMA_VERSION,
        "task_id": intent["task_id"],
        "stage": "author",
        "status": "dropped",
        "failure_codes": [failure_code],
        "reason_detail": detail,
        "query": None,
        "intent": dict(intent),
    }


def _author_log(intent: Mapping, *, catalog, expander):
    """log: pool → injected expander → speech bind (generate_one)."""
    result = generate_one(
        catalog=catalog,
        family="log",
        person=person_for_intent(intent),
        seed=intent["seed"],
        occasion=intent["occasion"],
        scene=intent["scene"],
        amount_path=intent["amount_path"],
        expander=expander,
    )
    if result.accepted is not None:
        return result.accepted, None
    rejected: Rejected = result.rejected
    return None, _reject(
        intent,
        f"author.{rejected.reason}",
        f"generate_one rejected: {rejected.reason} ({rejected.query!r})",
    )


AUTHOR_STRATEGIES: dict[str, Callable[..., tuple]] = {
    "log": _author_log,
}


def author_task(intent: Mapping, *, catalog, expander) -> tuple:
    """Author one intent. Returns ``(task, None)`` or ``(None, reject_record)``.

    A single un-authorable intent never fails the run (spec §4.1) — it becomes
    an ``rejects/author.jsonl`` line and build continues.
    """
    strategy = AUTHOR_STRATEGIES.get(intent["family"])
    if strategy is None:
        return None, _reject(
            intent,
            "author.unsupported_family",
            f"no authoring strategy for family {intent['family']!r} yet "
            "(tickets 012/013)",
        )
    return strategy(intent, catalog=catalog, expander=expander)
