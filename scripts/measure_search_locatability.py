#!/usr/bin/env python3
"""Measure how far the agent's own search gets from the pins a task commits.

`metrics.search_locatability` is reported by every build, and whether an ambiguous
or unreachable pin should be rejected is a data decision made on the number. A
number without its script is a claim, so this reruns it on the pinned catalog.

Three populations, all offline:

``catalog``   every food in the pinned catalog.
``pools``     foods the lab's own pool sampler draws (``--family``, ``--seeds``).
``planner``   the pins the authoring planner actually commits, i.e. the first pool
              food that survives ``pin_speech_portion``, together with the handle
              the brief asks the speaker to use.

Each is judged twice. The first column is the catalog record — name plus aliases,
which is what an earlier version of the judge measured. The second is the spoken
forms the utterance actually carries, which include the brief's handle. The gap
between the columns is the finding: a record name is full of FNDDS furniture no
speaker says, so judging it flatters every pin.

``--report complement`` answers the next question on the same population: for a pin
the utterance cannot reach, is there anything a speaker could say that would? The
answer is either the handle as it stands, another alias, the handle plus one phrase
taken from the record's own words, or nothing at all — and "nothing at all" is what
makes a pin unusable rather than merely ambiguous.

``--report selection`` covers the rule built on that answer. Pin selection now requires
a locatable food, so the report computes both picks on the same pool — the lab-only one
the planner made before the gate, and the live one — and prints what the gate cost
(intents that lost their pin, pins that moved) and whether the live picks are all
located. A live pin that is not uniquely located means the gate is not working.

Usage:
    python scripts/measure_search_locatability.py --source catalog
    python scripts/measure_search_locatability.py --source pools --family log --seeds 40
    python scripts/measure_search_locatability.py --source planner --family log
    python scripts/measure_search_locatability.py --source planner --report complement
    python scripts/measure_search_locatability.py --report selection --limit 400
    python scripts/measure_search_locatability.py --source planner --json /tmp/loc.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nutrienv.world.catalog_store import load_catalog  # noqa: E402

from src.training.data_factory.pool_filter import (  # noqa: E402
    filter_pool,
    is_suitable_meal_food,
    spoken_identity,
)
from src.training.data_factory.search_gate import (  # noqa: E402
    judge_food,
    qualifier_complement,
    search_locatability,
)

DEFAULT_FAMILY = "log"


def _entry_forms(catalog, food_id: str) -> tuple:
    entry = catalog.get(food_id) or {}
    aliases = tuple(entry.get("aliases") or ())
    return entry.get("name"), aliases


def catalog_population(catalog) -> list[tuple[str, str | None]]:
    """Every food in the catalog; its handle is derived, so pass ``None``."""
    return [(str(food_id), None) for food_id in catalog.keys()]


def pool_population(catalog, *, family: str, seeds: int, pool_size: int):
    """What the lab's sampler draws for ``family`` over ``seeds`` draws."""
    from nutrienv.bench.pipeline.sampler import sample_pools

    out: list[tuple[str, str | None]] = []
    for seed in range(seeds):
        for pool in sample_pools(
            catalog, seed=seed, family=family, n_pools=1, pool_size=pool_size
        ):
            out.extend((str(food.food_id), None) for food in pool.foods)
    return out


def planner_population(catalog, *, config_path, families: tuple[str, ...], limit: int):
    """The pins the authoring planner commits, with the handle it commits to.

    The real authoring path is run with a recording expander: ``generate_one`` draws
    the pool and calls the expander exactly as it does live, and the expander pins a
    food the same way ``speech.pin_speech_portion`` does — but returns an empty
    payload, so the task is rejected after the pin is recorded. Nothing here needs a
    model. An intent that never reaches the expander (it was rejected upstream) pins
    nothing and is reported as unattributed rather than silently dropped.
    """
    from src.training.data_factory import author as author_mod
    from src.training.data_factory.build import enumerate_intents
    from src.training.data_factory.config import load_config
    from src.training.data_factory.speech import pin_speech_portion

    config = load_config(config_path)
    intents = [
        intent
        for intent in enumerate_intents(config)
        if intent["family"] in families
    ]
    if limit and limit < len(intents):
        # stride, not head: the enumerator is family-major, so a head sample would
        # measure one family and call it the pipeline.
        stride = len(intents) // limit
        intents = intents[:: max(1, stride)][:limit]

    pins: list[tuple[str, str | None]] = []
    reached = 0

    def expander(pool, *, persona, family, amount_path=None):
        nonlocal reached
        reached += 1
        pool = filter_pool(pool, catalog=catalog)
        food, handle, _pin = pin_speech_portion(
            pool, amount_path=amount_path or "named_measure", catalog=catalog
        )
        if food is not None:
            pins.append((str(food.food_id), handle))
        return {"query": "", "foods": []}

    for intent in intents:
        try:
            author_mod.author_task(intent, catalog=catalog, expander=expander)
        except Exception as exc:  # noqa: BLE001 - a bad intent must not lose the sweep
            print(f"  ! {intent['task_id']}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return pins, {"intents": len(intents), "expander_calls": reached}


def judge(catalog, food_id: str, handle: str | None):
    """``(record-name verdict, spoken verdict)`` for one pin."""
    name, aliases = _entry_forms(catalog, food_id)
    record = search_locatability(food_id, name, catalog=catalog, aliases=aliases)
    if handle:
        spoken = search_locatability(
            food_id, name, catalog=catalog, aliases=aliases, spoken=handle
        )
    else:
        spoken = judge_food(food_id, catalog=catalog)
    return record, spoken


def phrase_locates(catalog, food_id: str, phrase: str):
    """The verdict for one spoken phrase alone, with no other form in the judgement.

    `judge_food` takes the weakest of the record name, the aliases and the handle —
    right for a metric that must price every form an utterance could carry, wrong for
    asking whether one phrase locates its food, which is the question pin selection
    answers.
    """
    return search_locatability(food_id, None, catalog=catalog, spoken=phrase)


def spoke_locates(catalog, food_id: str, phrase: str):
    """The verdict for one spoken form of one food, alongside the record's own forms.

    Not what pin selection asks (see `phrase_locates`), but what `metrics.search_locatability`
    reports on a real run, where every form the utterance could carry is priced in.
    """
    return search_locatability(
        food_id,
        (catalog.get(food_id) or {}).get("name"),
        catalog=catalog,
        spoken=phrase,
    )


def _row(catalog, food_id: str, handle: str | None, verdict) -> dict:
    name, _aliases = _entry_forms(catalog, food_id)
    return {
        "food_id": food_id,
        "name": name,
        "handle": handle if handle is not None else spoken_identity(name),
        "status": verdict.status,
        "terms": list(verdict.terms),
        "n_hits": len(verdict.hit_ids),
    }


def _distribution(counts: Counter) -> dict:
    total = sum(counts.values())
    return {
        "n": total,
        "counts": dict(sorted(counts.items())),
        "share": {
            status: round(100 * count / total, 1) if total else 0.0
            for status, count in sorted(counts.items())
        },
    }


def _print(title: str, dist: dict) -> None:
    shares = dist["share"]
    body = "  ".join(f"{status} {shares[status]}%" for status in shares)
    print(f"  {title:<20} n={dist['n']:<6} {body}")


# --------------------------------------------------------------------------- #
# what a speaker could say instead
# --------------------------------------------------------------------------- #

_FIX_BUCKETS = ("as spoken", "by addition", "by another alias", "not fixable")


def complement_report(catalog, population, *, show: int) -> dict:
    """For each pin: does the form the planner commits to locate it, and if not, what does?

    The form is the one `pin_speech_portion` returned, so with the gate live this
    counts the gate's own result. "not fixable" is the selection signal: the record
    distinguishes this food only by words no speaker says, so no utterance this
    pipeline can ask for will locate it. A pin the full record locates is counted
    separately — that is a record, not a phrase, and saying it is not speech.

    `judge_food` is reported alongside, because it is what `metrics.search_locatability`
    reads on a real run: it takes the weakest of the record name, every alias and the
    derived handle, so a pin whose committed phrase locates can still read `ambiguous`
    there. That residue is about forms the brief does not ask for, and it is printed so
    nobody reads the metric as the gate's own score.
    """
    counts: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    examples: dict[str, list[dict]] = {bucket: [] for bucket in _FIX_BUCKETS}
    record_only = 0
    metric_residue: Counter[str] = Counter()
    for food_id, handle in population:
        metric = judge_food(food_id, catalog=catalog)
        phrase = handle or spoken_identity(_entry_forms(catalog, food_id)[0])
        spoken = phrase_locates(catalog, food_id, phrase) if phrase else metric
        fix = None
        if spoken.status != "unique":
            fix = qualifier_complement(food_id, catalog=catalog)
        if fix is None and spoken.status != "unique":
            bucket = "not fixable"
            if search_locatability(
                food_id, _entry_forms(catalog, food_id)[0], catalog=catalog
            ).status == "unique":
                record_only += 1
        elif fix is not None and fix.source == "alias":
            bucket = "by another alias"
        elif fix is not None:
            bucket = "by addition"
        else:
            bucket = "as spoken"
        counts[bucket] += 1
        if bucket != "as spoken":
            statuses[metric.status] += 1
        elif metric.status != "unique":
            metric_residue[metric.status] += 1
        if bucket in ("by addition", "not fixable") and len(examples[bucket]) < show:
            name, _aliases = _entry_forms(catalog, food_id)
            examples[bucket].append(
                {
                    "food_id": food_id,
                    "name": name,
                    "verdict": spoken.status,
                    "speaks": fix.phrase if fix else None,
                    "added": fix.added if fix else None,
                }
            )
    total = sum(counts.values())
    return {
        "n": total,
        "counts": dict(sorted(counts.items())),
        "share": {
            bucket: round(100 * counts[bucket] / total, 1) if total else 0.0
            for bucket in _FIX_BUCKETS
        },
        "pin_status_when_not_spoken": dict(sorted(statuses.items())),
        "metric_residue": dict(sorted(metric_residue.items())),
        "record_name_only": record_only,
        "examples": examples,
    }


def _lab_accepts(pool, food, amount_path: str, catalog) -> bool:
    """The lab's own half of pin selection, re-derived so the gate stays measurable.

    `speech.pin_speech_portion` now gates on locatability, so the pin it returns is no
    longer evidence of what it would have returned without the gate. These are the
    lab-side checks it made before — suitable for a roster adult, a portion for this
    amount path, a tracer phrase that classifies as that path — spelled out here.
    """
    from nutrienv.bench.pipeline.sampler import speakable_tracer_food

    from src.training.data_factory.speech import _pin_for, _speech_amount_path

    if not is_suitable_meal_food((catalog.get(food.food_id) or {}).get("name")):
        return False
    if _pin_for(food, amount_path) is None:
        return False
    picked = speakable_tracer_food(
        type(pool)(pool_id=pool.pool_id, family=pool.family, foods=(food,)),
        catalog,
        amount_path=amount_path,
    )
    return picked is not None and _speech_amount_path(picked[1]) == amount_path


def selection_report(
    catalog, *, config_path, families: tuple[str, ...], limit: int, show: int
) -> dict:
    """What gating pin selection on locatability costs, and whether it holds.

    The pool is captured from the real authoring path. On it, two picks are computed:
    the lab-only one (what the planner pinned before the gate) and the live one (what
    `pin_speech_portion` pins now). The difference is the gate's cost — intents that
    lose their pin, and pins that move — and the live pick's own verdict is the
    verification: a pin that is not uniquely located means the gate is not working.
    """
    from src.training.data_factory import author as author_mod
    from src.training.data_factory.build import enumerate_intents
    from src.training.data_factory.config import load_config
    from src.training.data_factory.speech import pin_speech_portion

    config = load_config(config_path)
    intents = [
        intent
        for intent in enumerate_intents(config)
        if intent["family"] in families
    ]
    if limit and limit < len(intents):
        stride = len(intents) // limit
        intents = intents[:: max(1, stride)][:limit]

    rows: list[dict] = []

    def expander(pool, *, persona, family, amount_path=None):
        filtered = filter_pool(pool, catalog=catalog)
        path = amount_path or "named_measure"
        legacy = next(
            (
                food
                for food in filtered.foods
                if _lab_accepts(filtered, food, path, catalog)
            ),
            None,
        )
        live, handle, _pin = pin_speech_portion(
            filtered, amount_path=path, catalog=catalog
        )
        verdict = (
            phrase_locates(catalog, str(live.food_id), handle) if live else None
        )
        rows.append(
            {
                "pool": len(filtered.foods),
                "legacy": str(legacy.food_id) if legacy is not None else None,
                "live": str(live.food_id) if live is not None else None,
                "live_handle": handle,
                "live_status": verdict.status if verdict else None,
                "same": bool(
                    legacy is not None
                    and live is not None
                    and str(legacy.food_id) == str(live.food_id)
                ),
            }
        )
        return {"query": "", "foods": []}

    for intent in intents:
        try:
            author_mod.author_task(intent, catalog=catalog, expander=expander)
        except Exception as exc:  # noqa: BLE001 - a bad intent must not lose the sweep
            print(f"  ! {intent['task_id']}: {type(exc).__name__}: {exc}", file=sys.stderr)

    counts: Counter[str] = Counter()
    examples: list[dict] = []
    for row in rows:
        if row["legacy"] is None:
            counts["the lab picks nothing either way"] += 1
            continue
        counts["live pin found"] += 1 if row["live"] else 0
        counts["live pin lost to the gate"] += 0 if row["live"] else 1
        counts["live pin located uniquely"] += 1 if row["live_status"] == "unique" else 0
        if row["live"]:
            counts["same food as before"] += 1 if row["same"] else 0
            counts["a different food"] += 0 if row["same"] else 1
            if not row["same"] and len(examples) < show:
                examples.append(
                    {
                        "was": row["legacy"],
                        "becomes": row["live"],
                        "speaks": row["live_handle"],
                    }
                )
    total = len(rows)
    return {
        "n": total,
        "enumerated": len(intents),
        "counts": dict(sorted(counts.items())),
        "share": {
            key: round(100 * value / total, 1) if total else 0.0
            for key, value in sorted(counts.items())
        },
        "examples": examples,
    }


def _print_complement(report: dict, extra: dict) -> None:
    print(f"  pins n={report['n']}")
    for bucket in _FIX_BUCKETS:
        print(
            f"    {bucket:<18} {report['counts'].get(bucket, 0):<5} "
            f"{report['share'].get(bucket, 0.0)}%"
        )
    if report["pin_status_when_not_spoken"]:
        moved = "  ".join(
            f"{status} {count}"
            for status, count in report["pin_status_when_not_spoken"].items()
        )
        print(f"  verdicts of the pins that need something else: {moved}")
    if report["record_name_only"]:
        print(
            f"  of the unfixable, {report['record_name_only']} are located by the full "
            "record name — a record, not a phrase"
        )
    if report.get("metric_residue"):
        residue = "  ".join(
            f"{status} {count}" for status, count in report["metric_residue"].items()
        )
        print(
            "  of the pins that ARE spoken, this many still read weaker under "
            f"judge_food (record/alias forms the brief does not ask for): {residue}"
        )
    for bucket in ("by addition", "not fixable"):
        rows = report["examples"].get(bucket) or []
        if not rows:
            continue
        print(f"  {bucket}:")
        for row in rows:
            spoken = f" -> {row['speaks']!r}" if row["speaks"] else ""
            print(
                f"    [{row['verdict']:<11}] {row['food_id']}  "
                f"{str(row['name'])[:58]}{spoken}"
            )
    if extra:
        print(
            f"  intents={extra['intents']} reached the expander={extra['expander_calls']}"
        )


def _print_selection(report: dict, extra: dict) -> None:
    print(
        f"  intents enumerated: {report['enumerated']}, "
        f"reached the expander: {report['n']}"
    )
    for key, value in report["counts"].items():
        print(f"    {key:<34} {value:<5} {report['share'][key]}%")
    if report["examples"]:
        print("  pins that moved:")
        for row in report["examples"]:
            print(f"    {row['was']} -> {row['becomes']}  speaks {row['speaks']!r}")
    if extra:
        print(f"  intents enumerated={extra['intents']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=("catalog", "pools", "planner"), default="catalog"
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "data_factory.yaml"))
    parser.add_argument("--family", default=DEFAULT_FAMILY)
    parser.add_argument("--seeds", type=int, default=40, help="pools: draws to sample")
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="planner: cap intents")
    parser.add_argument("--show-worst", type=int, default=10)
    parser.add_argument(
        "--report",
        choices=("distribution", "complement", "selection"),
        default="distribution",
        help=(
            "complement: what a speaker could say instead of rejecting the pin; "
            "selection: what the planner would pin under that gate"
        ),
    )
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)

    catalog = load_catalog()
    extra: dict = {}
    if args.report == "selection":
        families = tuple(args.family.split(","))
        report = selection_report(
            catalog,
            config_path=args.config,
            families=families,
            limit=args.limit,
            show=args.show_worst,
        )
        print(f"source=planner report=selection catalog={len(catalog.keys())}")
        _print_selection(report, {})
        if args.json_path:
            pathlib.Path(args.json_path).write_text(
                json.dumps(
                    {"source": "planner", "family": args.family, **report},
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"wrote {args.json_path}")
        return 0

    if args.source == "catalog":
        population = catalog_population(catalog)
    elif args.source == "pools":
        population = pool_population(
            catalog,
            family=args.family,
            seeds=args.seeds,
            pool_size=args.pool_size,
        )
    else:
        families = tuple(args.family.split(","))
        population, extra = planner_population(
            catalog,
            config_path=args.config,
            families=families,
            limit=args.limit,
        )

    record_counts: Counter[str] = Counter()
    spoken_counts: Counter[str] = Counter()
    worst: list[dict] = []
    handles_seen = 0
    handle_mismatch = 0
    flips: Counter[str] = Counter()

    print(f"source={args.source} foods={len(population)} catalog={len(catalog.keys())}")
    if args.report == "complement":
        report = complement_report(catalog, population, show=args.show_worst)
        _print_complement(report, extra)
        if args.json_path:
            pathlib.Path(args.json_path).write_text(
                json.dumps(
                    {"source": args.source, "family": args.family, **report, **extra},
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"wrote {args.json_path}")
        return 0

    for food_id, handle in population:
        record, spoken = judge(catalog, food_id, handle)
        record_counts[record.status] += 1
        spoken_counts[spoken.status] += 1
        if handle:
            name, aliases = _entry_forms(catalog, food_id)
            derived = spoken_identity(name, aliases=aliases)
            handles_seen += 1
            if derived and derived != handle:
                handle_mismatch += 1
        if spoken.ordinal > record.ordinal:
            flips[f"{record.status}->{spoken.status}"] += 1
        if spoken.status in ("unreachable", "unmatched", "no_terms"):
            worst.append(_row(catalog, food_id, handle, spoken))

    record_dist = _distribution(record_counts)
    spoken_dist = _distribution(spoken_counts)

    if extra:
        print(
            f"  intents={extra['intents']} reached the expander={extra['expander_calls']} "
            f"pinned={len(population)}"
        )
    _print("record name+alias", record_dist)
    _print("spoken forms", spoken_dist)
    if flips:
        moved = "  ".join(f"{k} {v}" for k, v in sorted(flips.items()))
        print(f"  weaker when spoken: {moved}")
    if handles_seen:
        print(f"  planner handles checked: {handles_seen} (derived mismatch {handle_mismatch})")
    if worst and args.show_worst:
        worst.sort(key=lambda row: (row["status"] != "unreachable", row["n_hits"]))
        print(f"  worst {min(args.show_worst, len(worst))} of {len(worst)}:")
        for row in worst[: args.show_worst]:
            print(
                f"    {row['status']:<11} {row['food_id']}  "
                f"{str(row['name'])[:52]:<52} handle={row['handle']!r} "
                f"hits={row['n_hits']}"
            )

    if args.json_path:
        payload = {
            "source": args.source,
            "family": args.family,
            "population": len(population),
            "record_name": record_dist,
            "spoken_forms": spoken_dist,
            "flips": dict(sorted(flips.items())),
            "worst": worst,
            **extra,
        }
        pathlib.Path(args.json_path).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
