"""build — the one-command orchestrator (spec §4.1, §6, §8; Seam 1).

``build(config, *, expander, teacher_complete=None, ...)`` runs the per-task
pipeline **through materialize** (the teacher path lands in ticket 011):

1. preflight — load the catalog, assert ``catalog_digest(catalog)`` equals the
   exam split's ``catalog_sha256`` and the installed nutrienv rev equals
   ``config.nutrienv_rev``; either mismatch **aborts the run** (partial
   ``run_manifest.json``, non-zero exit).
2. enumerate **intents** per family (deterministic ``task_id``s, sorted),
   written to ``intents/<family>.jsonl`` — byte-identical across runs.
3. per intent (skipping task_ids already terminal in this output dir unless
   ``force``): **author** → ``gates.run`` → ``materialize`` →
   ``task_packages/<task_id>.json``.

``--stop-after author`` writes ``intents/`` + ``tasks/`` and stops;
``--stop-after gate`` additionally writes ``task_packages/`` (debug / staged
artifacts, not a second pipeline — spec §4.1). A single task failure records a
``rejects/{author,gate,indeterminate}.jsonl`` line and the run continues; a
config / schema / dependency error fails the whole run immediately.

The ``expander`` (and from ticket 011 the ``teacher_complete``) are **injected**
— nothing in this module constructs them (spec §7, US-19). The CLI is the
composition root: it currently offers only the offline synthetic expander; the
production ark expander wrapper arrives with ticket 012.

This is a stage module: it imports nutrienv at module level (allowed by spec
§18); the package ``__init__`` stays nutrienv-free.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import subprocess
import sys
from collections.abc import Callable

from nutrienv.bench import EXAM_SPLIT_PATH, load_exam
from nutrienv.bench.pipeline.freezer import task_to_item
from nutrienv.bench.pipeline.types import catalog_digest
from nutrienv.world.catalog_store import load_catalog

from src.training.data_factory import author as author_mod
from src.training.data_factory import gates as gates_mod
from src.training.data_factory import materialize as mz
from src.training.data_factory import serialize as serialize_mod
from src.training.data_factory import verify as verify_mod
from src.training.data_factory.config import ConfigError, DataFactoryConfig, load_config
from src.training.data_factory.concepts import AttemptRecord, RolloutCache
from src.training.data_factory.gates import GateContext
from src.training.data_factory.roster_train import TRAIN_ROSTER
from src.training.data_factory.rollout_fc import rollout_tool_call
from src.training.data_factory.serialize import SerializeError

__all__ = ["BuildError", "build", "enumerate_intents", "main"]

MANIFEST_SCHEMA_VERSION = "nutrimind-v2-runmanifest/1"
INTENT_SCHEMA_VERSION = "nutrimind-v2-intent/1"
AUTHORED_TASK_SCHEMA_VERSION = "nutrimind-v2-authoredtask/1"

# config family → (canonical task family, steps) for the §10 task_key. The
# 2-leg composites land with ticket 013.
FAMILY_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "log": ("log", ("log",)),
    "update": ("update", ("update",)),
    "recommend": ("recommend", ("recommend",)),
    "evaluate": ("evaluate", ("evaluate",)),
    "composite_update_log_recommend": ("composite", ("update", "log", "recommend")),
}

_OCCASIONS = ("breakfast", "lunch", "dinner", "snack")
_AMOUNT_PATHS = ("named_measure", "explicit_grams")

REJECT_STAGE_FILES = {
    "author": "author.jsonl",
    "gate": "gate.jsonl",
    "indeterminate": "indeterminate.jsonl",
}


class BuildError(Exception):
    """A config / schema / dependency error — the whole run fails immediately
    (spec §4.1). A partial ``run_manifest.json`` is written before re-raising."""


class BuildError(Exception):
    """A config / schema / dependency error — the whole run fails immediately
    (spec §4.1). A partial ``run_manifest.json`` is written before re-raising."""


# --------------------------------------------------------------------------- #
# intent enumeration
# --------------------------------------------------------------------------- #


def enumerate_intents(config: DataFactoryConfig) -> list[dict]:
    """Deterministic per-family intents (spec §6 step 4), sorted by task_id.

    Pure function of the config — no timestamps, no environment — so two runs
    with the same config produce byte-identical ``intents/*.jsonl``.
    """
    intents: list[dict] = []
    for family in sorted(config.families):
        if family not in FAMILY_SPECS:
            raise BuildError(
                f"config families: no task-key spec for family {family!r}"
            )
        task_family, steps = FAMILY_SPECS[family]
        family_cfg = config.families[family]
        wanted = math.ceil(family_cfg.target_n * family_cfg.over_generate_x)
        for index in range(wanted):
            person = TRAIN_ROSTER[index % len(TRAIN_ROSTER)]
            task_key = f"{task_family}--{'+'.join(steps)}--{person.user_id}"
            intents.append(
                {
                    "schema_version": INTENT_SCHEMA_VERSION,
                    "task_id": f"{task_key}--{index:06d}",
                    "task_key": task_key,
                    "family": family,
                    "task_family": task_family,
                    "steps": list(steps),
                    "user_id": person.user_id,
                    "seed": index,
                    "occasion": _OCCASIONS[index % len(_OCCASIONS)],
                    "scene": "empty",
                    "shell": None,
                    "slots": None,
                    "amount_path": _AMOUNT_PATHS[index % len(_AMOUNT_PATHS)],
                    "knife": None,
                    "tier": "",
                }
            )
    if len(intents) > config.max_intents:
        raise BuildError(
            f"intent count {len(intents)} exceeds max_intents "
            f"{config.max_intents} (runaway guard)"
        )
    intents.sort(key=lambda intent: intent["task_id"])
    seen: set[str] = set()
    for intent in intents:
        if intent["task_id"] in seen:
            raise BuildError(
                f"enumerator produced task_id {intent['task_id']!r} twice (spec §10)"
            )
        seen.add(intent["task_id"])
    return intents


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    """Deterministic JSONL: sort_keys, no trailing whitespace, newline-ended."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(blob, encoding="utf-8")


def _append_jsonl(path: pathlib.Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #


def _installed_nutrienv_rev() -> str:
    """The installed nutrienv source tree's git HEAD (the ticket-001 pin
    check; mirrors the smoke test)."""
    import nutrienv

    src_root = pathlib.Path(nutrienv.__file__).resolve().parents[2]
    try:
        return subprocess.run(
            ["git", "-C", str(src_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BuildError(f"cannot resolve the installed nutrienv rev: {exc}") from exc


def _nutrimind_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BuildError(f"cannot resolve this repo's rev: {exc}") from exc


def _preflight(config: DataFactoryConfig) -> tuple:
    """(catalog, catalog_sha, gate_ctx). Aborts the run on any mismatch."""
    catalog = load_catalog(config.catalog_path)
    digest = catalog_digest(catalog)
    exam_path = pathlib.Path(config.exam_split_path or EXAM_SPLIT_PATH)
    exam_raw = json.loads(exam_path.read_text(encoding="utf-8"))
    if digest != exam_raw.get("catalog_sha256"):
        raise BuildError(
            "catalog SHA mismatch: the loaded catalog hashes to "
            f"{digest!r} but the exam split pins {exam_raw.get('catalog_sha256')!r} "
            "(train and exam must never diverge on world facts — US-2)"
        )
    installed_rev = _installed_nutrienv_rev()
    if installed_rev != config.nutrienv_rev:
        raise BuildError(
            f"nutrienv rev mismatch: installed {installed_rev!r} != pinned "
            f"{config.nutrienv_rev!r} (configs/data_factory.yaml nutrienv.rev)"
        )
    gate_ctx = GateContext.from_exam(load_exam(exam_path))
    return catalog, digest, gate_ctx


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #


def _terminal_task_ids(output_dir: pathlib.Path, *, include_packages: bool) -> set[str]:
    """task_ids already terminal here (spec §8 resume): a reject line, an
    accepted sft record, or — only when this run stops at or before
    materialize — a materialized package (a bare package is NOT terminal for
    a teacher-stage run: the task may still need its episode)."""
    terminal: set[str] = set()
    rejects = output_dir / "rejects"
    if rejects.is_dir():
        for path in rejects.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    terminal.add(json.loads(line)["task_id"])
    sft = output_dir / "sft" / "train.jsonl"
    if sft.is_file():
        for line in sft.read_text(encoding="utf-8").splitlines():
            if line.strip():
                terminal.add(json.loads(line)["task_id"])
    if include_packages:
        packages = output_dir / "task_packages"
        if packages.is_dir():
            terminal |= {path.stem for path in packages.glob("*.json")}
    return terminal


def _write_manifest(output_dir: pathlib.Path, manifest: dict) -> None:
    blob = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    target = output_dir / "run_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(blob, encoding="utf-8")
    tmp.replace(target)


# --------------------------------------------------------------------------- #
# the teacher stage (target=sft, spec §6 step 5d / §16)
# --------------------------------------------------------------------------- #


def _teacher_attempts_summary(cache: RolloutCache) -> list[dict]:
    """The compact per-attempt view for reject lines — full episodes live in
    ``rollouts/cache/<task_id>.json``."""
    return [
        {
            "attempt_id": attempt.attempt_id,
            "status": attempt.verification.status,
            "reward": attempt.verification.reward,
            "failure_codes": list(attempt.verification.failure_codes),
        }
        for attempt in cache.attempts
    ]


def _load_cache(path: pathlib.Path) -> RolloutCache:
    return RolloutCache.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _jsonable(value):
    """Recursively make an asdict tree JSON-writable. Only non-serializable
    leaves get dropped (the live ``FoodCatalog`` inside ``WorldState`` —
    huge and re-derivable from the pinned ``catalog_sha``); everything else
    (profiles, ledger rows, plans) is plain data and survives verbatim."""
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return {
            "__omitted__": type(value).__name__,
            "note": "not JSON-serializable; the catalog is pinned by catalog_sha",
        }
    return value


def _write_cache(path: pathlib.Path, cache: RolloutCache) -> None:
    """Atomic multi-attempt cache write (spec §9: temp + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(_jsonable(cache.to_dict()), ensure_ascii=False, sort_keys=True) + "\n"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(blob, encoding="utf-8")
    tmp.replace(path)


def _teacher_stage(
    package, task, *, config, family_cfg, teacher_complete, catalog, out: pathlib.Path
) -> RolloutCache:
    """Attempts 1..k (k = family ``teacher_k``, spec §16): attempt 1 at
    ``temperature_first``, 2..k at ``temperature_retry``, stopping at the
    first Pass. Every attempt that ran is recorded; ``selected_attempt`` is
    the 0-based index of the first Pass or None."""
    cache = RolloutCache(task_id=package.task_id)
    for n in range(1, family_cfg.teacher_k + 1):
        episode = rollout_tool_call(
            task,
            teacher_complete=teacher_complete,
            catalog=catalog,
            model=config.teacher.model_id,
        )
        verification = verify_mod.verify(package, episode)
        cache.attempts.append(
            AttemptRecord(
                attempt_id=mz.attempt_id(package.task_id, n),
                episode=episode,
                verification=verification,
            )
        )
        if verification.status == "pass":
            cache.selected_attempt = n - 1
            break
    _write_cache(out / "rollouts" / "cache" / f"{package.task_id}.json", cache)
    # serialize from the on-disk cache (cache-authoritative: a re-run loading
    # the same cache produces byte-identical records)
    return _load_cache(out / "rollouts" / "cache" / f"{package.task_id}.json")


# --------------------------------------------------------------------------- #
# the orchestrator
# --------------------------------------------------------------------------- #


def build(
    config: DataFactoryConfig,
    *,
    expander: Callable,
    teacher_complete: Callable | None = None,
    stop_after: str | None = None,
    force: bool = False,
    output_dir: str | pathlib.Path | None = None,
    config_path: str | pathlib.Path | None = None,
) -> dict:
    """Run the pipeline through materialize (teacher path: ticket 011).

    Returns the run manifest. Raises :class:`BuildError` (after writing a
    partial manifest) on any whole-run failure; single-task failures are
    recorded as reject lines and skipped.
    """
    from datetime import datetime, timezone

    if stop_after not in (None, "author", "gate"):
        raise BuildError(f"unknown --stop-after {stop_after!r}")
    out = pathlib.Path(output_dir if output_dir is not None else config.output_dir)
    if config_path is not None:
        config_sha = hashlib.sha256(
            pathlib.Path(config_path).read_bytes()
        ).hexdigest()
    else:
        config_sha = hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    manifest: dict = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "target": config.target,
        "stop_after": stop_after,
        "output_dir": str(out),
        "config_sha": config_sha,
        "counts": {
            "intents": 0, "skipped_terminal": 0, "authored": 0,
            "gate_kept": 0, "materialized": 0,
            "rejected": {"author": 0, "gate": 0, "indeterminate": 0},
        },
    }

    try:
        catalog, catalog_sha, gate_ctx = _preflight(config)
        manifest["catalog_sha"] = catalog_sha
        manifest["nutrienv_rev"] = config.nutrienv_rev
        manifest["nutrimind_rev"] = _nutrimind_rev()
        manifest["built_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )

        intents = enumerate_intents(config)
        manifest["counts"]["intents"] = len(intents)
        by_family: dict[str, list[dict]] = {}
        for intent in intents:
            by_family.setdefault(intent["family"], []).append(intent)
        for family, rows in sorted(by_family.items()):
            _write_jsonl(out / "intents" / f"{family}.jsonl", rows)

        run_ctx_base = mz.RunContext(
            catalog=catalog,
            catalog_sha=catalog_sha,
            nutrienv_rev=config.nutrienv_rev,
            nutrimind_rev=manifest["nutrimind_rev"],
            config_sha=config_sha,
        )
        gated = stop_after != "author"
        run_teacher = (
            config.target in ("sft", "all")
            and stop_after is None
            and teacher_complete is not None
        )
        if config.target in ("sft", "all") and stop_after is None and not run_teacher:
            raise BuildError(
                "target sft needs the teacher path: inject a teacher_complete "
                "(or pass --stop-after gate to stop at materialize)"
            )
        terminal = (
            set()
            if force
            else _terminal_task_ids(out, include_packages=not run_teacher)
        )
        accepted_records: list[dict] = []
        manifest["counts"].update(
            {"accepted": 0, "teacher_rejected": 0, "teacher_indeterminate": 0,
             "serialize_rejected": 0, "cache_reused": 0}
        )

        for intent in intents:
            task_id = intent["task_id"]
            if task_id in terminal:
                manifest["counts"]["skipped_terminal"] += 1
                continue

            task, reject = author_mod.author_task(
                intent, catalog=catalog, expander=expander
            )
            if task is None:
                _append_jsonl(out / "rejects" / "author.jsonl", reject)
                manifest["counts"]["rejected"]["author"] += 1
                continue
            manifest["counts"]["authored"] += 1
            _append_jsonl(
                out / "tasks" / f"{intent['family']}.jsonl",
                {
                    "schema_version": AUTHORED_TASK_SCHEMA_VERSION,
                    "task_id": task_id,
                    "query": task.query,
                    "gated": gated,
                    "item": task_to_item(task),
                },
            )
            if stop_after == "author":
                continue

            gate_result = gates_mod.run(task, gate_ctx)
            if not gate_result.keep:
                record = gates_mod.rejects_record(
                    gate_result, task, intent=intent
                )
                route = "indeterminate" if record["status"] == "indeterminate" else "gate"
                _append_jsonl(out / "rejects" / REJECT_STAGE_FILES[route], record)
                manifest["counts"]["rejected"][route] += 1
                continue
            manifest["counts"]["gate_kept"] += 1

            package = mz.materialize(
                task,
                dataclasses.replace(
                    run_ctx_base,
                    steps=tuple(intent["steps"]),
                    seed=intent["seed"],
                    intent_ref=f"intents/{intent['family']}.jsonl#{intent['seed']:06d}",
                ),
            )
            if package.task_id != task_id:
                raise BuildError(
                    f"author/materialize task_id mismatch for intent "
                    f"{task_id!r}: package says {package.task_id!r} (bad enumerator)"
                )
            mz.write_package(package, out / "task_packages")
            manifest["counts"]["materialized"] += 1

            if not run_teacher:
                continue

            # ---- teacher stage (spec §6 step 5d) ----
            cache_path = out / "rollouts" / "cache" / f"{task_id}.json"
            if cache_path.is_file():
                cache = _load_cache(cache_path)  # resume: never re-pay the teacher
                manifest["counts"]["cache_reused"] += 1
            else:
                cache = _teacher_stage(
                    package, task, config=config,
                    family_cfg=config.families[intent["family"]],
                    teacher_complete=teacher_complete, catalog=catalog, out=out,
                )

            common = {
                "task_id": task_id,
                "task_package_ref": f"task_packages/{task_id}.json",
                "rollouts_ref": f"rollouts/cache/{task_id}.json",
                "attempts": _teacher_attempts_summary(cache),
                "intent": dict(intent),
            }
            if cache.selected_attempt is not None:
                attempt = cache.attempts[cache.selected_attempt]
                try:
                    record = serialize_mod.serialize(
                        package, attempt.episode, attempt.verification,
                        config=config,
                        accepted_from_attempt=cache.selected_attempt + 1,
                    )
                except SerializeError as exc:
                    manifest["counts"]["serialize_rejected"] += 1
                    _append_jsonl(
                        out / "rejects" / "serialize.jsonl",
                        {
                            **common, "stage": "serialize",
                            "status": "indeterminate",
                            "failure_codes": [exc.code],
                            "reason_detail": exc.detail,
                        },
                    )
                    continue
                accepted_records.append(record)
                manifest["counts"]["accepted"] += 1
            elif any(
                a.verification.status == "fail" for a in cache.attempts
            ):
                # a completed legal episode that missed the hard contract:
                # an SFT-reject / analysis candidate, NEVER an RLVR negative (§4.2)
                first_fail = next(
                    a for a in cache.attempts if a.verification.status == "fail"
                )
                manifest["counts"]["teacher_rejected"] += 1
                _append_jsonl(
                    out / "rejects" / "teacher.jsonl",
                    {
                        **common, "stage": "teacher", "status": "fail",
                        "failure_codes": list(
                            first_fail.verification.failure_codes
                        ),
                    },
                )
            else:
                # no completed legal attempt at all: teacher error / no-finish /
                # invalid-op — nothing usable for SFT
                manifest["counts"]["teacher_indeterminate"] += 1
                _append_jsonl(
                    out / "rejects" / "indeterminate.jsonl",
                    {
                        **common, "stage": "teacher", "status": "indeterminate",
                        "failure_codes": list(
                            cache.attempts[-1].verification.failure_codes
                        ),
                    },
                )

        if run_teacher:
            # sorted by task_id, temp + atomic rename (spec §6 step 6)
            accepted_records.sort(key=lambda record: record["task_id"])
            train_path = out / "sft" / "train.jsonl"
            train_path.parent.mkdir(parents=True, exist_ok=True)
            blob = "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in accepted_records
            )
            tmp = train_path.with_suffix(".jsonl.tmp")
            tmp.write_text(blob, encoding="utf-8")
            tmp.replace(train_path)

        manifest["status"] = "complete"
        _write_manifest(out, manifest)
        return manifest
    except BuildError as exc:
        manifest["status"] = "aborted"
        manifest["abort_reason"] = str(exc)
        _write_manifest(out, manifest)
        raise


# --------------------------------------------------------------------------- #
# CLI (composition root — the only place adapters may be constructed)
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.training.data_factory.build",
        description="NutriMind v2 data factory build (spec §4/§6)",
    )
    parser.add_argument("--config", required=True, help="configs/data_factory.yaml")
    parser.add_argument(
        "--stop-after", choices=("author", "gate"), default=None,
        help="write the staged artifact and stop (debug, spec §4.1)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-run task_ids already terminal in this output dir",
    )
    parser.add_argument(
        "--expander", choices=("synthetic",), default=None,
        help="expander adapter (the production ark wrapper lands with ticket 012)",
    )
    parser.add_argument(
        "--teacher", choices=("ark",), default=None,
        help="teacher adapter for target=sft (network-guarded: "
        "NUTRIMIND_ALLOW_NETWORK=1 + ARK_API_KEY required at call time)",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    if args.expander is None:
        print(
            "refusing to run without an explicit --expander: the default would "
            "silently author with synthetic speech. Use --expander synthetic "
            "for offline runs; the production ark adapter lands with ticket 012.",
            file=sys.stderr,
        )
        return 1

    from src.training.data_factory.synthetic import synth_expander

    teacher_complete = None
    if args.teacher == "ark":
        from src.training.data_factory.rollout import make_ark_teacher_client

        teacher_complete = make_ark_teacher_client(config.teacher)

    try:
        manifest = build(
            config,
            expander=synth_expander(load_catalog(config.catalog_path)),
            teacher_complete=teacher_complete,
            stop_after=args.stop_after,
            force=args.force,
            config_path=args.config,
        )
    except BuildError as exc:
        print(f"build aborted: {exc}", file=sys.stderr)
        return 1
    counts = manifest["counts"]
    print(
        f"build complete: {counts['materialized']} materialized, "
        f"{counts.get('accepted', 0)} accepted, "
        f"{counts['rejected']['author'] + counts['rejected']['gate'] + counts['rejected']['indeterminate'] + counts.get('teacher_rejected', 0) + counts.get('teacher_indeterminate', 0) + counts.get('serialize_rejected', 0)} rejected, "
        f"{counts['skipped_terminal']} skipped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
