"""TaskPackage materializer — Seam 3 (spec §9.1, §10, §19.1, §22.2).

Turns an authored, gate-passing ``Task`` into the canonical ``TaskPackage``
(the single artifact SFT / RLVR / eval materializers derive from), and writes
it to ``task_packages/<task_id>.json``.

The ``environment`` block comes from the verified public round-trip (ticket
002 Part A / spec OQ-2): ``task_to_item`` → ``freeze_tasks`` (which also runs
the oracle-gram gate) → ``load_split`` through a **transient scratch file** —
there is no public in-memory item→Task entry at the pinned rev. The scratch
file is deleted after materialization; no external dataset dependency.

Identifiers (spec §10, three distinct levels — never conflate):

- ``task_key``  = ``f"{family}--{'+'.join(steps)}--{user_id}"`` — logical task,
  no seed. Simple families use ``steps == (family,)``.
- ``task_id``   = ``f"{task_key}--{seed:06d}"`` — one concrete instance; the
  resume key, filename stem, sort/dedup key.
- ``attempt_id``= ``f"{task_id}--attempt-{n:02d}"`` — one teacher rollout.

A ``task_id`` seen twice within one run raises (``run_ctx.seen_task_ids`` is
the run-scoped registry); a re-run against an existing
``task_packages/<task_id>.json`` skips idempotently.

This is a stage module: it imports nutrienv at module level (allowed by spec
§18); the package ``__init__`` stays nutrienv-free.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
import tempfile
from datetime import datetime, timezone

from nutrienv.bench.pipeline.freezer import freeze_tasks, task_to_item
from nutrienv.bench.pipeline.types import catalog_digest
from nutrienv.bench.split import load_split
from nutrienv.harness.runner import DEFAULT_MAX_STEPS, FAMILY_MAX_STEPS, FINISH_OPS

from src.training.data_factory.concepts import (
    Provenance,
    RewardSemantics,
    TaskCatalogRef,
    TaskEnvironment,
    TaskOracle,
    TaskPackage,
    TaskVerifierRef,
    Termination,
)

__all__ = [
    "DuplicateTaskIdError",
    "RunContext",
    "attempt_id",
    "materialize",
    "write_package",
]

SCHEMA_VERSION = "nutrimind-v2-taskpackage/1"
VERIFIER_KIND = "nutrienv.bench.scorer.Scorer"
VERIFIER_CALL = "Scorer().score(end_state, oracle)"
RECONSTRUCT_WITH = (
    "nutrienv.bench.pipeline.freezer.task_to_item -> freeze_tasks([task], "
    "output_path=<transient scratch file>) -> nutrienv.bench.split.load_split"
)
ENVIRONMENT_NOTE = (
    "self-contained via the verified public round-trip (ticket 002 Part A, "
    "spec OQ-2 verdict A); the transient scratch file is the only bridge — "
    "no public in-memory item->Task entry exists at the pinned rev"
)


class DuplicateTaskIdError(ValueError):
    """The same ``task_id`` was materialized twice within one run."""


@dataclasses.dataclass
class RunContext:
    """Everything a materialization needs beyond the ``Task`` itself.

    Build constructs one per task (usually via ``dataclasses.replace`` on a
    run-wide template): run-wide fields come from the config + repo state,
    per-task fields (``steps`` / ``seed`` / ``intent_ref``) from the intent.
    ``seen_task_ids`` is the run-scoped duplicate registry — it is mutated by
    ``materialize`` and deliberately lives here so the spec §10 rule ("a
    task_id seen twice within one run raises") has exactly one home.
    """

    # run-wide
    catalog: object  # the loaded gold catalog (world facts)
    catalog_sha: str  # asserted == catalog_digest(catalog) on every call
    nutrienv_rev: str
    nutrimind_rev: str  # git sha of this repo
    config_sha: str
    rubric_version: str = "v2-r1"
    reward_version: str = "v2-r1"
    # per-task
    steps: tuple[str, ...] = ()  # composite legs; () -> (task.family,)
    seed: int = 0
    intent_ref: str = ""  # e.g. "intents/composite.jsonl#191"
    built_at: str | None = None  # default: now, ISO-8601 UTC
    # run-scoped registry (spec §10: duplicate task_id within a run raises)
    seen_task_ids: set[str] = dataclasses.field(default_factory=set)


def attempt_id(task_id: str, n: int) -> str:
    """One teacher rollout of a ``task_id`` (``n`` in ``1..k``)."""
    if n < 1:
        raise ValueError(f"attempt number must be >= 1, got {n}")
    return f"{task_id}--attempt-{n:02d}"


def _task_key(task, steps: tuple[str, ...]) -> str:
    return f"{task.family}--{'+'.join(steps)}--{task.s0.profile.user_id}"


def _termination(family: str) -> Termination:
    return Termination(
        finish_ops=sorted(FINISH_OPS),
        max_steps=FAMILY_MAX_STEPS.get(family, DEFAULT_MAX_STEPS),
    )


@contextlib.contextmanager
def _transient_round_trip(task, *, catalog: object, catalog_sha: str):
    """Write the 1-item frozen split to a transient scratch file, load it
    back, yield ``(item, rebuilt_task)``; the scratch file never survives."""
    with tempfile.TemporaryDirectory(prefix="nutrimind-taskpackage-") as scratch_dir:
        scratch = pathlib.Path(scratch_dir) / "one.json"
        payload, _ = freeze_tasks(
            [task], catalog=catalog, catalog_sha=catalog_sha,
            output_path=scratch, overwrite=True,
        )
        (rebuilt,) = load_split(scratch, catalog=catalog)
        yield payload["items"][0], rebuilt


def materialize(task, run_ctx: RunContext) -> TaskPackage:
    """Build the canonical ``TaskPackage`` for one gate-passing ``Task``.

    Raises ``DuplicateTaskIdError`` if this ``task_id`` was already
    materialized within the run, and asserts the pinned ``catalog_sha``
    matches the loaded catalog (spec §10's catalog_sha assertion).
    """
    digest = catalog_digest(run_ctx.catalog)
    if digest != run_ctx.catalog_sha:
        raise ValueError(
            f"catalog_sha mismatch: run context pinned {run_ctx.catalog_sha!r} "
            f"but the loaded catalog hashes to {digest!r}"
        )

    steps = tuple(run_ctx.steps) or (task.family,)
    key = _task_key(task, steps)
    task_id = f"{key}--{run_ctx.seed:06d}"
    if task_id in run_ctx.seen_task_ids:
        raise DuplicateTaskIdError(
            f"task_id {task_id!r} materialized twice within one run"
        )
    run_ctx.seen_task_ids.add(task_id)

    with _transient_round_trip(
        task, catalog=run_ctx.catalog, catalog_sha=run_ctx.catalog_sha
    ) as (item, rebuilt):
        # fail fast if the public round-trip ever stops being lossless
        if rebuilt.id != task.id or rebuilt.query != task.query:
            raise ValueError(
                "public round-trip is not lossless for "
                f"{task.id!r}: rebuilt id/query differ"
            )
        environment = TaskEnvironment(
            s0=item["s0"],
            reconstruct_with=RECONSTRUCT_WITH,
            note=ENVIRONMENT_NOTE,
        )
        oracle_payload = item["oracle"]

    return TaskPackage(
        schema_version=SCHEMA_VERSION,
        task_key=key,
        task_id=task_id,
        query=task.query,
        family=task.family,
        steps=list(steps),
        tier=task.tier,
        environment=environment,
        catalog=TaskCatalogRef(
            catalog_sha=run_ctx.catalog_sha,
            nutrienv_rev=run_ctx.nutrienv_rev,
        ),
        oracle=TaskOracle(
            payload=oracle_payload,
            oracle_version=f"nutrienv-{run_ctx.nutrienv_rev[:7]}",
            note="payload of the nutrienv Oracle via freezer (sub_oracles for composite)",
        ),
        verifier=TaskVerifierRef(kind=VERIFIER_KIND, call=VERIFIER_CALL),
        reward_semantics=RewardSemantics(
            reward_version=run_ctx.reward_version,
            kind="binary",
            map={"pass": 1.0, "fail": 0.0, "indeterminate": None},
        ),
        rubric_version=run_ctx.rubric_version,
        termination=_termination(task.family),
        seed=run_ctx.seed,
        provenance=Provenance(
            nutrimind_rev=run_ctx.nutrimind_rev,
            nutrienv_rev=run_ctx.nutrienv_rev,
            catalog_sha=run_ctx.catalog_sha,
            config_sha=run_ctx.config_sha,
            intent_ref=run_ctx.intent_ref,
            built_at=run_ctx.built_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )


def write_package(package: TaskPackage, output_dir) -> pathlib.Path | None:
    """Write ``task_packages/<task_id>.json`` deterministically.

    Idempotent across runs: an existing file is skipped (returns ``None``)
    rather than rewritten or compared.
    """
    target = pathlib.Path(output_dir) / f"{package.task_id}.json"
    if target.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(
        package.to_dict(), indent=2, ensure_ascii=False, sort_keys=True
    ) + "\n"
    target.write_text(blob, encoding="utf-8")
    return target
