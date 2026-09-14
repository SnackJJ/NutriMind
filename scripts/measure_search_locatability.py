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

Usage:
    python scripts/measure_search_locatability.py --source catalog
    python scripts/measure_search_locatability.py --source pools --family log --seeds 40
    python scripts/measure_search_locatability.py --source planner --family log
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
    spoken_identity,
)
from src.training.data_factory.search_gate import (  # noqa: E402
    judge_food,
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
    if limit:
        intents = intents[:limit]

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
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)

    catalog = load_catalog()
    extra: dict = {}
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

    print(f"source={args.source} foods={len(population)} catalog={len(catalog.keys())}")
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
