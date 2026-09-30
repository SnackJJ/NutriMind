"""author — one intent → one authored nutrienv ``Task`` (spec §4.3, §6 step 5a).

An intent names a family and its authoring knobs; the matching **strategy**
produces either an accepted ``Task`` (via ``generate_one`` + the *injected*
expander — never constructed here) or an author-stage reject record routed to
``rejects/author.jsonl`` (spec §8). Strategies register per config family;
unsupported families reject with ``author.unsupported_family`` so the run
continues (failure isolation, spec §4.1).

This is a stage module: it imports nutrienv at module level (allowed by spec
§18); the package ``__init__`` stays nutrienv-free.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
from collections.abc import Callable, Mapping, Sequence

from nutrienv.bench.pipeline.generate_one import generate_one
from nutrienv.bench.pipeline.templates import RECOMMEND_SHELLS, UPDATE_SHELLS, recommend_query
from nutrienv.bench.pipeline.types import Rejected
from nutrienv.bench.quality_gates import EVALUATE_TIERS
from nutrienv.bench.realize import Oracle, Task, compose_oracles
from nutrienv.world.daily_windows import plan_windows_for_meal
from nutrienv.world.types import ledger_totals

from src.training.data_factory.consistency import (
    foods_from_task,
    query_entity_consistency,
)
from src.training.data_factory.roster_train import TRAIN_ROSTER
from src.training.data_factory.search_gate import identifying_words
from src.training.data_factory.speech import bind_speech_context, revision_hint

__all__ = [
    "AUTHOR_STRATEGIES",
    "EVALUATE_TIERS",
    "TWO_LEG_COMPOSITE_STEPS",
    "author_task",
    "person_for_intent",
    "portion_table_gram_anchor",
]

# Batch-1 2-leg pairs (`generate_one` `_LEGAL_COMPOSITE_PAIRS`). Spec §2.1
# counts them as one `composite` family of 200; enumerate splits 1:1.
TWO_LEG_COMPOSITE_STEPS: tuple[tuple[str, ...], ...] = (
    ("log", "recommend"),
    ("update", "recommend"),
)

REJECT_SCHEMA_VERSION = "nutrimind-v2-reject/1"

_ROSTER_BY_ID = {person.user_id: person for person in TRAIN_ROSTER}

_SAFE_ALLERGENS = ("fish", "soy", "wheat")
_UPDATE_SHELL_CYCLE = (
    "upd-add-allergy-short",
    "upd-weight",
    "upd-phase-cut",
    "upd-phase-muscle",
    "upd-phase-maintain",
)
_REC_SHELL_BY_OCCASION = {
    "breakfast": "rec-breakfast",
    "lunch": "rec-lunch",
    "dinner": "rec-dinner",
    "snack": "rec-snack",
}

# Reject reasons a second attempt could fix. Structural faults (an unknown shell,
# an illegal step pair) are excluded: retrying them only burns calls.
_REWRITABLE_REASONS = frozenset(
    {
        "amount_path",
        "unresolvable",
        "steps",
        "query_foods_mismatch",
        "intent_conflict",
    }
)


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


def portion_table_gram_anchor(catalog) -> Callable[[str, str, str], float | None]:
    """Offline GramAnchor: first whitelist portion for the food, or None."""

    def anchor(food_id: str, expression: str, query: str) -> float | None:
        food = catalog.get(food_id) or {}
        portions = food.get("portions") or {}
        for grams in portions.values():
            try:
                value = float(grams)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    return anchor


def _result_or_reject(intent: Mapping, result) -> tuple:
    if result.accepted is not None:
        return result.accepted, None
    rejected: Rejected = result.rejected
    return None, _reject(
        intent,
        f"author.{rejected.reason}",
        f"generate_one rejected: {rejected.reason} ({rejected.query!r})",
    )


def _update_slots(person, shell: str) -> dict:
    if shell == "upd-add-allergy-short":
        have = set(person.allergies)
        allergen = next((tag for tag in _SAFE_ALLERGENS if tag not in have), "fish")
        return {"allergen": allergen}
    if shell == "upd-weight":
        return {"n": str(int(person.weight_kg) + 1)}
    return {}


def _author_log(intent: Mapping, *, catalog, expander, gram_anchor=None):
    result = generate_one(
        catalog=catalog,
        family="log",
        person=person_for_intent(intent),
        seed=intent["seed"],
        occasion=intent["occasion"],
        scene=intent.get("scene") or "empty",
        amount_path=intent["amount_path"],
        expander=expander,
        gram_anchor=gram_anchor,
        enable_semantic_vote=False,
    )
    task, reject = _result_or_reject(intent, result)
    if task is None:
        return None, reject
    return task, None


def _author_update(intent: Mapping, *, catalog, expander, gram_anchor=None):
    person = person_for_intent(intent)
    shell = intent.get("shell") or _UPDATE_SHELL_CYCLE[intent["seed"] % len(_UPDATE_SHELL_CYCLE)]
    if shell not in UPDATE_SHELLS:
        return None, _reject(intent, "author.unknown_shell", f"unknown update shell {shell!r}")
    slots = intent.get("slots") or _update_slots(person, shell)
    result = generate_one(
        catalog=catalog,
        family="update",
        person=person,
        seed=intent["seed"],
        shell=shell,
        slots=slots,
        enable_semantic_vote=False,
    )
    return _result_or_reject(intent, result)


def _author_recommend(intent: Mapping, *, catalog, expander, gram_anchor=None):
    person = person_for_intent(intent)
    occasion = intent["occasion"]
    if intent.get("shell"):
        shell = intent["shell"]
    elif person.persona == "gym":
        shell = "rec-post-gym"
    elif person.allergies:
        shell = "rec-named-dish"
    else:
        shell = _REC_SHELL_BY_OCCASION.get(occasion, "rec-dinner")
    if shell not in RECOMMEND_SHELLS:
        return None, _reject(intent, "author.unknown_shell", f"unknown recommend shell {shell!r}")
    slots = dict(intent.get("slots") or {})
    rec_occasion = "dinner" if shell == "rec-named-dish" else occasion
    if rec_occasion == "snack" and shell != "rec-snack":
        rec_occasion = "dinner"
    result = generate_one(
        catalog=catalog,
        family="recommend",
        person=person,
        seed=intent["seed"],
        occasion=rec_occasion,
        shell=shell,
        slots=slots,
        enable_semantic_vote=False,
    )
    return _result_or_reject(intent, result)


def _format_evaluate_food_phrase(food_id: str, grams: float, amount_path: str, catalog: Mapping) -> str:
    food = catalog.get(food_id) or {}
    name = food.get("name") or food_id
    portions = food.get("portions") or {}

    if amount_path == "explicit_grams":
        return f"{grams:g} g of {name}"

    unit_priority = ["cup", "slice", "piece", "can", "tbsp", "tsp", "fl_oz", "qns"]
    for u in unit_priority:
        if u in portions:
            try:
                base = float(portions[u])
            except (TypeError, ValueError):
                continue
            if base > 0:
                mult = grams / base
                if abs(mult - round(mult * 4) / 4) < 1e-2 and 0.2 < mult <= 10:
                    clean_mult = round(mult * 4) / 4
                    u_display = "serving" if u == "qns" else u
                    if clean_mult != 1 and not u_display.endswith("s"):
                        u_display += "s"
                    return f"{clean_mult:g} {u_display} of {name}"

    oz = grams / 28.35
    if abs(oz - round(oz * 2) / 2) < 0.05 and 0.5 <= oz <= 30:
        clean_oz = round(oz * 2) / 2
        return f"{clean_oz:g} oz of {name}"

    return f"{grams:g} g of {name}"


def _evaluate_rewriter(catalog):
    """Deterministic rewriter: speak a code-chosen evaluate plate with quantities."""

    def rewriter(items, *, intent, occasion, amount_path=None):
        bits = []
        foods = []
        path = amount_path or (intent.get("amount_path") if isinstance(intent, Mapping) else None) or "explicit_grams"
        for item in items:
            food_id = str(item.get("food_id") or "")
            grams = float(item.get("grams") or 0)
            phrase = _format_evaluate_food_phrase(food_id, grams, path, catalog)
            bits.append(phrase)
            foods.append(food_id)
        query = f"Is this {occasion} okay? I had " + " and ".join(bits) + "."
        return {"query": query, "foods": foods}

    return rewriter


def _author_evaluate(intent: Mapping, *, catalog, expander, gram_anchor=None):
    from nutrienv.bench.pipeline.roster import profile_for
    from nutrienv.bench.pipeline.sampler import sample_pools
    from nutrienv.bench.validator import fitting_plan
    from nutrienv.world.daily_windows import plan_windows_for_meal

    tier = intent.get("tier") or ""
    if tier not in EVALUATE_TIERS:
        return None, _reject(
            intent, "author.bad_tier", f"evaluate tier must be one of {EVALUATE_TIERS}, got {tier!r}"
        )
    person = person_for_intent(intent)
    occasion = intent["occasion"] if intent["occasion"] != "snack" else "lunch"
    amount_path = intent["amount_path"] or "explicit_grams"
    rewriter = _evaluate_rewriter(catalog)
    result = generate_one(
        catalog=catalog,
        family="evaluate",
        person=person,
        seed=intent["seed"],
        occasion=occasion,
        amount_path=amount_path,
        expander=expander,
        gram_anchor=gram_anchor,
        rewriter=rewriter,
        tier=tier,
        enable_semantic_vote=False,
    )
    if result.accepted is not None:
        return result.accepted, None
    profile = profile_for(person)
    windows = plan_windows_for_meal(profile.windows, {}, occasion)
    pools = sample_pools(
        catalog, seed=intent["seed"], family="evaluate", n_pools=1, pool_size=40
    )
    pool_ids = frozenset(food.food_id for food in pools[0].foods) if pools else None
    plate = (
        fitting_plan(
            catalog, windows, profile.allergies, allowed_food_ids=pool_ids
        )
        if windows is not None
        else None
    )
    if not plate:
        return _result_or_reject(intent, result)

    from nutrienv.bench.pipeline.review_harness import _match_portion

    multiples = (0.5, 1.0, 1.5, 2.0)
    item_candidates = []
    for item in plate:
        food_id = str(item["food_id"])
        grams = float(item["grams"])
        candidates = set()
        portions = (catalog.get(food_id) or {}).get("portions") or {}
        for one in portions.values():
            try:
                base = float(one)
            except (TypeError, ValueError):
                continue
            if base > 0:
                for qty in multiples:
                    cand = round(qty * base, 2)
                    if cand > 0 and _match_portion(portions, cand) != (None, None):
                        candidates.add(cand)
        sorted_cand = sorted(candidates, key=lambda value: abs(value - grams))
        item_candidates.append((food_id, grams, sorted_cand))

    accepted_plate = None
    if len(item_candidates) == 1:
        food_id, orig_grams, cands = item_candidates[0]
        for cand in cands:
            trial_plate = [{"food_id": food_id, "grams": cand}]
            trial_res = generate_one(
                catalog=catalog,
                family="evaluate",
                person=person,
                seed=intent["seed"],
                occasion=occasion,
                amount_path=amount_path,
                items=trial_plate,
                rewriter=rewriter,
                tier=tier,
                pool_size=40,
                enable_semantic_vote=False,
            )
            if trial_res.accepted is not None:
                accepted_plate = trial_plate
                result = trial_res
                break
    elif item_candidates:
        closest_plate = [{"food_id": f_id, "grams": cands[0]} for f_id, orig_g, cands in item_candidates if cands]
        if len(closest_plate) == len(item_candidates):
            trial_res = generate_one(
                catalog=catalog,
                family="evaluate",
                person=person,
                seed=intent["seed"],
                occasion=occasion,
                amount_path=amount_path,
                items=closest_plate,
                rewriter=rewriter,
                tier=tier,
                pool_size=40,
                enable_semantic_vote=False,
            )
            if trial_res.accepted is not None:
                accepted_plate = closest_plate
                result = trial_res

    if accepted_plate is None:
        result = generate_one(
            catalog=catalog,
            family="evaluate",
            person=person,
            seed=intent["seed"],
            occasion=occasion,
            amount_path=amount_path,
            items=plate,
            rewriter=rewriter,
            tier=tier,
            pool_size=40,
            enable_semantic_vote=False,
        )
    return _result_or_reject(intent, result)


def _apply_recovery_trap(task, trap: str):
    """Rewrite speech so the natural first action is illegal but recoverable.

    Oracle / s0 stay put so ``check_achievable`` still Passes (ADR-013).
    """
    if trap == "unknown_food":
        query = "Please log a bowl of leftover casserole for lunch."
    elif trap == "implausible_quantity":
        query = f"{task.query} Actually make that nine kilograms."
    else:
        return task
    return dataclasses.replace(task, query=query) if dataclasses.is_dataclass(task) else task


def _author_three_leg(intent: Mapping, *, catalog, expander, gram_anchor=None, fallback_expander=None):
    """update+log→recommend from public symbols only (spec §22.9)."""
    person = person_for_intent(intent)
    seed = intent["seed"]
    amount_path = intent.get("amount_path") or "named_measure"
    allergen = (intent.get("slots") or {}).get("allergen") or "fish"
    if allergen in person.allergies:
        allergen = next((tag for tag in _SAFE_ALLERGENS if tag not in person.allergies), "fish")
    upd = generate_one(
        catalog=catalog,
        family="update",
        person=person,
        seed=seed,
        shell="upd-add-allergy-short",
        slots={"allergen": allergen},
        enable_semantic_vote=False,
    )
    if upd.accepted is None:
        return _result_or_reject(intent, upd)

    log = generate_one(
        catalog=catalog,
        family="log",
        person=person,
        seed=seed,
        occasion="lunch",
        amount_path=amount_path,
        expander=expander,
        gram_anchor=gram_anchor,
        enable_semantic_vote=False,
    )
    if log.accepted is None and fallback_expander is not None:
        log = generate_one(
            catalog=catalog,
            family="log",
            person=person,
            seed=seed,
            occasion="lunch",
            amount_path=amount_path,
            expander=fallback_expander,
            gram_anchor=gram_anchor,
            enable_semantic_vote=False,
        )
    if log.accepted is None:
        return _result_or_reject(intent, log)

    update_task, log_task = upd.accepted, log.accepted
    expected = update_task.oracle.profile
    s0 = update_task.s0
    tail = list(log_task.oracle.ledger_tail)
    final_ledger = (*s0.ledger, *tail)
    plan_windows = plan_windows_for_meal(
        expected.windows, ledger_totals(list(final_ledger), catalog), "dinner"
    )
    if plan_windows is None:
        return None, _reject(intent, "author.recwin.empty_windows", "empty dinner windows")
    partial = copy.deepcopy(expected)
    update_sub = dataclasses.replace(update_task.oracle, ledger=None, ledger_tail=None)
    log_sub = Oracle(
        ledger_tail=list(tail), ledger=final_ledger, profile=copy.deepcopy(partial)
    )
    rec_sub = Oracle(
        profile=copy.deepcopy(partial),
        last_plan=[],
        ledger=final_ledger,
        plan_must_be_safe=True,
        plan_must_fit_windows=True,
        plan_windows=plan_windows,
    )
    rec_q = recommend_query("rec-dinner", {"occasion": "dinner"})
    query = f"{update_task.query} {log_task.query} {rec_q}"
    task = Task(
        intent["task_id"],
        "composite",
        query,
        s0,
        compose_oracles(update_sub, log_sub, rec_sub),
        ("multi_item_log",),
        person.persona,
        tier="",
    )
    return task, None


def _relabel_composite(task, intent: Mapping):
    """generate_one labels 2-leg Tasks as the first-leg family; §10 uses composite."""
    return dataclasses.replace(task, id=intent["task_id"], family="composite")


def _strip_stale_update_ledger(task):
    """Standalone update oracles carry ledger=(); composite update legs must not."""
    subs = task.oracle.sub_oracles
    if not subs:
        return task
    update_sub, *rest = subs
    if update_sub.ledger is None and update_sub.ledger_tail is None:
        return task
    update_sub = dataclasses.replace(update_sub, ledger=None, ledger_tail=None)
    return dataclasses.replace(task, oracle=compose_oracles(update_sub, *rest))


def _author_two_leg(intent: Mapping, *, catalog, expander, gram_anchor=None, fallback_expander=None):
    """2-leg log→recommend / update→recommend via public generate_one."""
    steps = tuple(intent.get("steps") or TWO_LEG_COMPOSITE_STEPS[0])
    if steps not in TWO_LEG_COMPOSITE_STEPS:
        return None, _reject(
            intent,
            "author.illegal_pair",
            f"2-leg composite steps must be one of {list(TWO_LEG_COMPOSITE_STEPS)}, got {steps!r}",
        )
    person = person_for_intent(intent)
    kwargs = dict(
        catalog=catalog,
        family="composite",
        person=person,
        seed=intent["seed"],
        steps=steps,
        occasion=intent["occasion"],
        enable_semantic_vote=False,
    )
    if steps == ("log", "recommend"):
        kwargs.update(
            amount_path=intent.get("amount_path") or "named_measure",
            expander=expander,
            gram_anchor=gram_anchor,
        )
    else:
        shell = intent.get("shell") or _UPDATE_SHELL_CYCLE[
            intent["seed"] % len(_UPDATE_SHELL_CYCLE)
        ]
        if shell not in UPDATE_SHELLS:
            return None, _reject(
                intent, "author.unknown_shell", f"unknown update shell {shell!r}"
            )
        kwargs["shell"] = shell
        kwargs["slots"] = intent.get("slots") or _update_slots(person, shell)

    result = generate_one(**kwargs)
    if (
        result.accepted is None
        and steps == ("log", "recommend")
        and fallback_expander is not None
    ):
        kwargs["expander"] = fallback_expander
        result = generate_one(**kwargs)
    task, reject = _result_or_reject(intent, result)
    if task is None:
        return None, reject
    if steps == ("update", "recommend"):
        task = _strip_stale_update_ledger(task)
    return _relabel_composite(task, intent), None


AUTHOR_STRATEGIES: dict[str, Callable[..., tuple]] = {
    "log": _author_log,
    "update": _author_update,
    "recommend": _author_recommend,
    "evaluate": _author_evaluate,
    "composite": _author_two_leg,
    "composite_update_log_recommend": _author_three_leg,
}


def author_task(
    intent: Mapping,
    *,
    catalog,
    expander,
    gram_anchor=None,
    fallback_expander=None,
    parse_retries: int = 1,
) -> tuple:
    """Author one intent. Returns ``(task, None)`` or ``(None, reject_record)``.

    A rejected attempt is retried when the reject is one a rewrite can fix, with
    the reason bound onto the expander so the next brief carries it. A single
    un-authorable intent never fails the run (spec §4.1) — it becomes an
    ``rejects/author.jsonl`` line and build continues.
    """
    strategy = AUTHOR_STRATEGIES.get(intent["family"])
    if strategy is None:
        return None, _reject(
            intent,
            "author.unsupported_family",
            f"no authoring strategy for family {intent['family']!r}",
        )
    expander = bind_speech_context(expander, intent)
    if fallback_expander is not None:
        fallback_expander = bind_speech_context(fallback_expander, intent)
    kwargs = dict(catalog=catalog, expander=expander, gram_anchor=gram_anchor)
    if intent["family"] in ("composite", "composite_update_log_recommend"):
        kwargs["fallback_expander"] = fallback_expander
    attempts = 1 + max(0, int(parse_retries))
    last_reject = None
    for _ in range(attempts):
        if last_reject is not None:
            _bind_revision(expander, last_reject, intent)
        task, reject = strategy(intent, **kwargs)
        if task is None:
            if _retryable_author_reject(reject) and _ < attempts - 1:
                last_reject = reject
                continue
            return None, reject
        foods = foods_from_task(task)
        pool_ids = getattr(expander, "last_pool_ids", None)
        allowed = set(pool_ids) if pool_ids else (set(foods) or None)
        code = query_entity_consistency(
            task.query,
            foods,
            intent=intent,
            catalog=catalog,
            allowed_ids=allowed,
            required_words=_required_words(expander, foods, catalog),
        )
        if code is None:
            trap = intent.get("recovery_trap")
            if trap:
                task = _apply_recovery_trap(task, trap)
            return task, None
        last_reject = _reject(
            intent,
            code,
            f"query/entity consistency: {code}",
        )
    return None, last_reject


def _required_words(expander, foods: Sequence[str], catalog) -> tuple[str, ...]:
    """The words the utterance has to be findable by, derived from the pinned food.

    Derived here rather than taken from the writer's own report: a model that dropped
    a word cannot also drop the requirement that it be there. Only an expander that
    speaks from a brief is held to this — an offline writer (`synth_expander`) is
    given no word list, so imposing one on it would measure a contract it never had.
    """
    if not getattr(expander, "speaks_from_brief", False):
        return ()
    words: list[str] = []
    for food_id in foods:
        for word in identifying_words(food_id, catalog=catalog):
            if word not in words:
                words.append(word)
    return tuple(words)


def _bind_revision(expander, reject: Mapping, intent: Mapping) -> None:
    """Tell the expander why the previous attempt was rejected, before retrying."""
    bind = getattr(expander, "bind_feedback", None)
    if bind is None:
        return
    codes = list(reject.get("failure_codes") or ())
    reason = codes[0].removeprefix("author.") if codes else ""
    bind(
        revision_hint(
            reason,
            portion=str(getattr(expander, "last_portion", "") or ""),
            query=_rejected_query(str(reject.get("reason_detail") or "")),
        )
    )


def _rejected_query(detail: str) -> str:
    """The utterance out of ``_result_or_reject``'s ``(repr(query))`` tail."""
    if not detail.endswith(")"):
        return ""
    tail = detail[detail.rfind("(") + 1 : -1]
    try:
        return str(ast.literal_eval(tail))
    except (ValueError, SyntaxError):
        return tail


def _retryable_author_reject(reject: Mapping | None) -> bool:
    """True when a second attempt could plausibly fix ``reject``.

    Only the reasons a rewrite addresses: the amount word, an unparseable amount,
    and the log-clause verb. Anything structural (a bad shell, an illegal pair)
    would burn calls for nothing.
    """
    if not reject:
        return False
    codes = list(reject.get("failure_codes") or ())
    return any(
        code.removeprefix("author.") in _REWRITABLE_REASONS for code in codes
    )
