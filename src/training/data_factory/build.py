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
composition root: ``--expander synthetic`` or ``--expander commandcode``
(Command Code brief expander, ADR-011).

This is a stage module: it imports nutrienv at module level (allowed by spec
§18); the package ``__init__`` stays nutrienv-free.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import pathlib
import random
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable

from nutrienv.bench import EXAM_SPLIT_PATH, check_achievable, load_exam
from nutrienv.bench.pipeline.freezer import freeze_tasks, task_to_item
from nutrienv.bench.pipeline.types import catalog_digest
from nutrienv.world.catalog_store import load_catalog

from src.training.data_factory import author as author_mod
from src.training.data_factory import export_rlvr as rlvr_mod
from src.training.data_factory import gates as gates_mod
from src.training.data_factory import materialize as mz
from src.training.data_factory import serialize as serialize_mod
from src.training.data_factory import verify as verify_mod
from src.training.data_factory.config import ConfigError, DataFactoryConfig, load_config
from src.training.data_factory.concepts import AttemptRecord, RolloutCache, TaskPackage
from src.training.data_factory.gates import GateContext
from src.training.data_factory.query_identity import UniqueQueryIndex, unique_caps
from src.training.data_factory.roster_train import TRAIN_ROSTER
from src.training.data_factory.rollout_fc import rollout_tool_call
from src.training.data_factory.serialize import SerializeError
from src.training.data_factory.synthetic import qwen_max_fallback_expander

__all__ = [
    "BuildError",
    "build",
    "candidate_count",
    "enumerate_intents",
    "split_by_task_id",
    "main",
    "mini_exam_intents",
]

log = logging.getLogger(__name__)

_FROM_STAGES = (None, "author", "gate", "materialize", "rollout", "serialize")
_USD_PER_MTOK = 0.3
_RECOVERY_BAND = (0.15, 0.25)

MANIFEST_SCHEMA_VERSION = "nutrimind-v2-runmanifest/1"
INTENT_SCHEMA_VERSION = "nutrimind-v2-intent/1"
AUTHORED_TASK_SCHEMA_VERSION = "nutrimind-v2-authoredtask/1"

# config family → (canonical Task.family, default steps) for the §10 task_key.
# The 2-leg `composite` family cycles TWO_LEG_COMPOSITE_STEPS per intent
# (even index log→recommend, odd index update→recommend — 1:1). Spec §2.1
# pins the 200-count bucket, not a subtype mix.
FAMILY_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "log": ("log", ("log",)),
    "update": ("update", ("update",)),
    "recommend": ("recommend", ("recommend",)),
    "evaluate": ("evaluate", ("evaluate",)),
    "composite": ("composite", ("log", "recommend")),
    "composite_update_log_recommend": ("composite", ("update", "log", "recommend")),
}

_OCCASIONS = ("breakfast", "lunch", "dinner", "snack")
_LOG_OCCASIONS = ("breakfast", "lunch", "dinner")
_AMOUNT_PATHS = ("named_measure", "explicit_grams", "unspecified")
_PERSONA_AMOUNT_WEIGHTS: dict[str, tuple[tuple[str, float], ...]] = {
    "gym": (("explicit_grams", 0.60), ("named_measure", 0.30), ("unspecified", 0.10)),
    "everyday": (("explicit_grams", 0.15), ("named_measure", 0.70), ("unspecified", 0.15)),
    "cut": (("explicit_grams", 0.15), ("named_measure", 0.70), ("unspecified", 0.15)),
}
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

REJECT_STAGE_FILES = {
    "author": "author.jsonl",
    "gate": "gate.jsonl",
    "indeterminate": "indeterminate.jsonl",
}


class BuildError(Exception):
    """A config / schema / dependency error — the whole run fails immediately
    (spec §4.1). A partial ``run_manifest.json`` is written before re-raising."""


# Mini-exam val (ticket 017): reserved seeds, disjoint from Batch-1's 0..max_intents.
MINI_EXAM_N = 30
MINI_EXAM_SEED_BASE = 900_000
MINI_EXAM_POOL = 80


# --------------------------------------------------------------------------- #
# intent enumeration
# --------------------------------------------------------------------------- #


def split_by_task_id(task_id: str) -> str:
    """7:2:1 train/holdout/loss_val from a stable task_id hash (spec §22.16)."""
    digest = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest(), 16)
    bucket = digest % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "holdout"
    return "loss_val"


def candidate_count(
    estimated_unique_accepted_rate: float,
    *,
    target_n: int = 40,
    max_candidate_limit: int = 2000,
) -> int:
    """§6 ladder sizing for the 3-leg family (ticket 013)."""
    p = estimated_unique_accepted_rate
    if p <= 0:
        return max_candidate_limit
    return min(max_candidate_limit, max(120, math.ceil(target_n / p * 1.5)))


def _weighted_pick(rng: random.Random, pairs: tuple[tuple[str, float], ...]) -> str:
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return pairs[-1][0]
    cursor = rng.random() * total
    acc = 0.0
    for name, weight in pairs:
        acc += weight
        if cursor <= acc:
            return name
    return pairs[-1][0]


def _amount_path_for(person, index: int, family_cfg) -> tuple[str, bool]:
    rng = random.Random(f"{index}:{person.user_id}")
    if family_cfg.amount_path_weights:
        pairs = tuple(family_cfg.amount_path_weights.items())
    else:
        pairs = _PERSONA_AMOUNT_WEIGHTS.get(
            person.persona, _PERSONA_AMOUNT_WEIGHTS["everyday"]
        )
    path = _weighted_pick(rng, pairs)
    ounce = path == "named_measure" and rng.random() < 0.15
    return path, ounce


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
        if family == "composite_update_log_recommend":
            rate = 1.0 / family_cfg.over_generate_x
            wanted = candidate_count(
                rate,
                target_n=family_cfg.target_n,
                max_candidate_limit=config.max_intents,
            )
        if wanted > config.max_intents:
            raise BuildError(
                f"family {family!r} intent count {wanted} exceeds max_intents "
                f"{config.max_intents} (runaway guard)"
            )
        for index in range(wanted):
            person = TRAIN_ROSTER[index % len(TRAIN_ROSTER)]
            if family == "composite":
                steps = author_mod.TWO_LEG_COMPOSITE_STEPS[
                    index % len(author_mod.TWO_LEG_COMPOSITE_STEPS)
                ]
            task_key = f"{task_family}--{'+'.join(steps)}--{person.user_id}"
            if family in ("recommend",):
                occasion = _OCCASIONS[index % len(_OCCASIONS)]
            else:
                occasion = _LOG_OCCASIONS[index % len(_LOG_OCCASIONS)]
            amount_path, ounce = _amount_path_for(person, index, family_cfg)
            tier = ""
            shell = None
            slots = None
            if family == "evaluate":
                tier = author_mod.EVALUATE_TIERS[index % len(author_mod.EVALUATE_TIERS)]
            elif family == "update" or (
                family == "composite" and steps == ("update", "recommend")
            ):
                shell = _UPDATE_SHELL_CYCLE[index % len(_UPDATE_SHELL_CYCLE)]
            elif family == "recommend":
                if person.persona == "gym":
                    shell = "rec-post-gym"
                    occasion = "dinner"
                elif person.allergies:
                    shell = "rec-named-dish"
                    occasion = "dinner"
                else:
                    shell = _REC_SHELL_BY_OCCASION[occasion]
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
                    "occasion": occasion,
                    "scene": "empty",
                    "shell": shell,
                    "slots": slots,
                    "amount_path": amount_path,
                    "ounce_phrasing": ounce,
                    "knife": None,
                    "tier": tier,
                    "recovery_trap": (
                        "unknown_food" if family == "log" and index % 5 == 4 else None
                    ),
                    "gram_anchor": family_cfg.gram_anchor,
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


def _atomic_write(
    path: pathlib.Path, blob: str, *, before_replace=None
) -> None:
    """Temp + ``os.replace`` (spec §22.15). ``before_replace`` is a test hook."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(blob, encoding="utf-8")
    if before_replace is not None:
        before_replace(path, tmp)
    os.replace(tmp, path)


def _write_jsonl(path: pathlib.Path, rows: list[dict], *, before_replace=None) -> None:
    """Deterministic JSONL: sort_keys, no trailing whitespace, newline-ended."""
    blob = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_write(path, blob, before_replace=before_replace)


def _append_jsonl(path: pathlib.Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _finalize_jsonl(path: pathlib.Path) -> None:
    """Drop a trailing partial line from an append-only JSONL file."""
    if not path.is_file():
        return
    kept: list[bytes] = []
    for line in path.read_bytes().split(b"\n"):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue
        kept.append(line)
    path.write_bytes((b"\n".join(kept) + (b"\n" if kept else b"")))


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
    sft_dir = output_dir / "sft"
    if sft_dir.is_dir():
        for sft in sft_dir.glob("*.jsonl"):
            for line in sft.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    terminal.add(json.loads(line)["task_id"])
    if include_packages:
        packages = output_dir / "task_packages"
        if packages.is_dir():
            terminal |= {path.stem for path in packages.glob("*.json")}
    return terminal


def _write_manifest(output_dir: pathlib.Path, manifest: dict, *, before_replace=None) -> None:
    blob = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    _atomic_write(output_dir / "run_manifest.json", blob, before_replace=before_replace)


def _bump_codes(histogram: Counter[str], codes: list[str] | None) -> None:
    for code in codes or []:
        histogram[code] += 1


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def mini_exam_intents() -> list[dict]:
    """Reserved-seed log intents for ``--freeze-mini`` (ticket 017)."""
    intents: list[dict] = []
    task_family, steps = FAMILY_SPECS["log"]
    for index in range(MINI_EXAM_POOL):
        seed = MINI_EXAM_SEED_BASE + index
        person = TRAIN_ROSTER[index % len(TRAIN_ROSTER)]
        task_key = f"{task_family}--{'+'.join(steps)}--{person.user_id}"
        intents.append(
            {
                "schema_version": INTENT_SCHEMA_VERSION,
                "task_id": f"{task_key}--{seed:06d}",
                "task_key": task_key,
                "family": "log",
                "task_family": task_family,
                "steps": list(steps),
                "user_id": person.user_id,
                "seed": seed,
                "occasion": ("breakfast", "lunch", "dinner")[index % 3],
                "scene": "empty",
                "shell": None,
                "slots": None,
                "amount_path": _AMOUNT_PATHS[index % len(_AMOUNT_PATHS)],
                "knife": None,
                "tier": "",
            }
        )
    intents.sort(key=lambda intent: intent["task_id"])
    return intents


def _run_freeze_mini(
    *,
    expander: Callable,
    catalog,
    catalog_sha: str,
    gate_ctx: GateContext,
    out: pathlib.Path,
    manifest: dict,
) -> dict:
    """Author 30 TRAIN_ROSTER tasks, gate them, freeze to ``sft/val_mini.json``."""
    kept = []
    for intent in mini_exam_intents():
        if len(kept) >= MINI_EXAM_N:
            break
        task, reject = author_mod.author_task(
            intent, catalog=catalog, expander=expander
        )
        if task is None:
            _append_jsonl(out / "rejects" / "author.jsonl", reject)
            manifest["counts"]["rejected"]["author"] += 1
            continue
        manifest["counts"]["authored"] += 1
        task = dataclasses.replace(task, id=intent["task_id"])
        gate_result = gates_mod.run(task, gate_ctx)
        if not gate_result.keep:
            record = gates_mod.rejects_record(gate_result, task, intent=intent)
            route = "indeterminate" if record["status"] == "indeterminate" else "gate"
            _append_jsonl(out / "rejects" / REJECT_STAGE_FILES[route], record)
            manifest["counts"]["rejected"][route] += 1
            continue
        manifest["counts"]["gate_kept"] += 1
        kept.append(task)
    if len(kept) < MINI_EXAM_N:
        raise BuildError(
            f"--freeze-mini kept {len(kept)} tasks, need {MINI_EXAM_N}"
        )
    kept.sort(key=lambda task: task.id)
    report = check_achievable(kept)
    unreachable = [task.id for task in kept if task.id in report.unreachable]
    if unreachable:
        raise BuildError(f"--freeze-mini unreachable: {unreachable[:5]}")
    target = out / "sft" / "val_mini.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    freeze_tasks(
        kept,
        catalog=catalog,
        catalog_sha=catalog_sha,
        output_path=target,
        extra={"kind": "mini-exam-val", "n": MINI_EXAM_N},
        overwrite=True,
    )
    manifest["counts"]["eval_frozen"] = MINI_EXAM_N
    manifest["mini_exam"] = {
        "n": MINI_EXAM_N,
        "seed_base": MINI_EXAM_SEED_BASE,
        "path": "sft/val_mini.json",
    }
    return manifest


def _finalize_observability(
    manifest: dict,
    *,
    config: DataFactoryConfig,
    catalog_sha: str,
    reject_histogram: Counter[str],
    accepted_by_family: Counter[str],
    teacher_completed: int,
    teacher_error: int,
    teacher_no_finish: int,
    pass_count: int,
    serialized: int,
    attempted_task_ids: int,
    indeterminate_task_ids: int,
    accepted_records: list[dict],
    tokens: int,
    recovery_positive: int = 0,
    recovery_by_code: dict | None = None,
    budget_warned: bool = False,
    budget_stopped: bool = False,
    unique_queries: UniqueQueryIndex | None = None,
    unique_query_budget: int | None = None,
) -> None:
    """Fill §9.5 / §17 / §20 blocks on the run manifest (ticket 015)."""
    fail_n = manifest["counts"].get("teacher_rejected", 0)
    indeterminate_n = (
        manifest["counts"]["rejected"]["indeterminate"]
        + manifest["counts"].get("teacher_indeterminate", 0)
        + manifest["counts"].get("serialize_rejected", 0)
    )
    accepted_n = manifest["counts"].get("accepted", 0)
    manifest["counts"]["by_status"] = {
        "accepted": accepted_n,
        "fail": fail_n,
        "indeterminate": indeterminate_n,
    }
    manifest["counts"]["by_failure_code"] = dict(sorted(reject_histogram.items()))
    family_mix = {}
    for name, family in config.families.items():
        family_mix[name] = {
            "target": family.target_n,
            "actual": accepted_by_family.get(name, 0),
        }
    manifest["family_mix"] = family_mix
    usd_per_mtok = _USD_PER_MTOK
    manifest["cost"] = {
        "est_usd": round((tokens / 1_000_000) * usd_per_mtok, 6),
        "budget_usd": config.usd_budget,
        "teacher_tokens": tokens,
        "on_budget": config.on_budget,
        "budget_warned": budget_warned,
        "budget_stopped": budget_stopped,
    }
    accepted_n_rec = manifest["counts"].get("accepted", 0)
    rec_pos = recovery_positive
    rec_frac = rec_pos / accepted_n_rec if accepted_n_rec else None
    manifest.setdefault("metrics", {})
    unique_n = len(unique_queries) if unique_queries is not None else 0
    manifest["counts"]["unique_query_identities"] = unique_n
    manifest["counts"]["accepted_traces"] = accepted_n
    manifest["metrics"]["unique_query_count"] = unique_n
    manifest["metrics"]["unique_query_by_family"] = dict(
        unique_queries.by_family if unique_queries is not None else {}
    )
    manifest["metrics"]["unique_query_budget"] = unique_query_budget
    manifest["metrics"]["recovery_positive"] = rec_pos
    manifest["metrics"]["recovery_fraction"] = rec_frac
    manifest["metrics"]["recovery_by_code"] = recovery_by_code or {
        "semantic": {},
        "syntax": {},
    }
    versions = {
        "oracle_version": f"nutrienv-{config.nutrienv_rev[:7]}",
        "rubric_version": config.rubric_version,
        "reward_version": config.reward_version,
        "environment_version": f"nutrienv-{config.nutrienv_rev[:7]}",
        "task_schema_version": mz.SCHEMA_VERSION,
    }
    if accepted_records:
        meta = accepted_records[0]["meta"]
        for key in versions:
            if meta.get(key) != versions[key]:
                raise BuildError(
                    f"versions.{key} {versions[key]!r} != record meta {meta.get(key)!r}"
                )
    manifest["versions"] = versions

    flagged = reject_histogram.get("gate.draft_invalid", 0) + reject_histogram.get(
        "gate.unachievable", 0
    )
    reject_n = sum(reject_histogram.values())
    flagged_share = flagged / reject_n if reject_n else 0.0
    health: dict = {
        "catalog_sha_match": manifest.get("catalog_sha") == catalog_sha,
        "serialization_success_rate": _rate(serialized, pass_count),
        "teacher_completion_rate": _rate(
            teacher_completed,
            teacher_completed + teacher_error + teacher_no_finish,
        ),
        "teacher_pass_rate": _rate(pass_count, teacher_completed),
        "reject_histogram_ok": flagged_share <= 0.25,
        "reject_histogram_flagged_share": flagged_share,
    }
    if attempted_task_ids >= 40:
        health["indeterminate_rate"] = _rate(indeterminate_task_ids, attempted_task_ids)
        health["indeterminate_rate_note"] = None
    else:
        health["indeterminate_rate"] = None
        health["indeterminate_rate_note"] = (
            f"raw counts only (attempted_task_ids={attempted_task_ids} < 40): "
            f"indeterminate_task_ids={indeterminate_task_ids}"
        )
        health["indeterminate_task_ids"] = indeterminate_task_ids
        health["attempted_task_ids"] = attempted_task_ids
    lo, hi = _RECOVERY_BAND
    if rec_frac is None:
        health["recovery_fraction_in_band"] = True
    elif lo <= rec_frac <= hi:
        health["recovery_fraction_in_band"] = True
    else:
        health["recovery_fraction_in_band"] = False
        log.warning(
            "recovery_fraction %s outside [%s, %s] (health warning, run continues)",
            rec_frac, lo, hi,
        )
    manifest["health"] = health


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


def _task_from_package(package: TaskPackage, catalog):
    """Rebuild a runnable Task from a TaskPackage (public file round-trip)."""
    from nutrienv.bench import load_split

    item = {
        "id": package.task_id,
        "family": package.family,
        "persona": "everyday",
        "situations": [],
        "query": package.query,
        "s0": package.environment.s0,
        "oracle": package.oracle.payload,
    }
    if package.tier:
        item["tier"] = package.tier
    with tempfile.TemporaryDirectory(prefix="nutrimind-from-stage-") as scratch_dir:
        scratch = pathlib.Path(scratch_dir) / "item.json"
        scratch.write_text(
            json.dumps({"items": [item]}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (task,) = load_split(scratch, catalog=catalog)
    return task


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


def _write_cache(path: pathlib.Path, cache: RolloutCache, *, before_replace=None) -> None:
    """Atomic multi-attempt cache write (spec §9: temp + rename)."""
    blob = json.dumps(_jsonable(cache.to_dict()), ensure_ascii=False, sort_keys=True) + "\n"
    _atomic_write(path, blob, before_replace=before_replace)


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


def _pick_pass_attempt(cache: RolloutCache):
    """First Pass attempt, re-picking if ``selected_attempt`` is stale."""
    if cache.selected_attempt is not None:
        if 0 <= cache.selected_attempt < len(cache.attempts):
            attempt = cache.attempts[cache.selected_attempt]
            if attempt.verification.status == "pass":
                return attempt
    for index, attempt in enumerate(cache.attempts):
        if attempt.verification.status == "pass":
            cache.selected_attempt = index
            return attempt
    cache.selected_attempt = None
    return None


def _est_usd(tokens: int) -> float:
    return (tokens / 1_000_000) * _USD_PER_MTOK


def build(
    config: DataFactoryConfig,
    *,
    expander: Callable,
    teacher_complete: Callable | None = None,
    stop_after: str | None = None,
    from_stage: str | None = None,
    force: bool = False,
    output_dir: str | pathlib.Path | None = None,
    config_path: str | pathlib.Path | None = None,
    dry_run: bool = False,
    freeze_mini: bool = False,
    before_replace=None,
    unique_query_budget: int | None = None,
) -> dict:
    """Run the pipeline through materialize / teacher / serialize / rlvr.

    Returns the run manifest. Raises :class:`BuildError` (after writing a
    partial manifest) on any whole-run failure; single-task failures are
    recorded as reject lines and skipped.
    """
    from datetime import datetime, timezone

    if dry_run and freeze_mini:
        raise BuildError("cannot combine --dry-run and --freeze-mini")
    if stop_after not in (None, "author", "gate"):
        raise BuildError(f"unknown --stop-after {stop_after!r}")
    if from_stage not in _FROM_STAGES:
        raise BuildError(f"unknown --from-stage {from_stage!r}")
    if stop_after is not None and from_stage is not None:
        raise BuildError("cannot combine --stop-after and --from-stage")
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
        "dry_run": dry_run,
        "freeze_mini": freeze_mini,
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

        if freeze_mini:
            _run_freeze_mini(
                expander=expander,
                catalog=catalog,
                catalog_sha=catalog_sha,
                gate_ctx=gate_ctx,
                out=out,
                manifest=manifest,
            )
            manifest["status"] = "complete"
            _finalize_observability(
                manifest,
                config=config,
                catalog_sha=catalog_sha,
                reject_histogram=Counter(),
                accepted_by_family=Counter(),
                teacher_completed=0,
                teacher_error=0,
                teacher_no_finish=0,
                pass_count=0,
                serialized=0,
                attempted_task_ids=0,
                indeterminate_task_ids=0,
                accepted_records=[],
                tokens=0,
                unique_queries=UniqueQueryIndex(),
                unique_query_budget=unique_query_budget,
            )
            _write_manifest(out, manifest)
            return manifest

        intents = enumerate_intents(config)
        manifest["counts"]["intents"] = len(intents)
        if "composite_update_log_recommend" in config.families:
            three = config.families["composite_update_log_recommend"]
            rate = 1.0 / three.over_generate_x
            manifest["three_leg"] = {
                "candidate_count": candidate_count(
                    rate, target_n=three.target_n, max_candidate_limit=config.max_intents
                ),
                "estimated_unique_accepted_rate": rate,
                "seed_start": 0,
            }
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
        qwen_fallback = qwen_max_fallback_expander(catalog)

        gated = stop_after != "author" and not dry_run
        serialize_only = from_stage == "serialize"
        rerun_teacher = from_stage == "rollout"
        run_rlvr = (
            config.target in ("rlvr", "all")
            and stop_after is None
            and not dry_run
            and from_stage not in ("author", "gate")
        )
        run_teacher = (
            config.target in ("sft", "all")
            and stop_after is None
            and not dry_run
            and not serialize_only
        )
        if run_teacher and teacher_complete is None:
            raise BuildError(
                "target sft needs the teacher path: inject a teacher_complete "
                "(or pass --stop-after gate / --dry-run / --from-stage serialize)"
            )
        terminal = set()
        if not force and from_stage is None:
            terminal = _terminal_task_ids(out, include_packages=not run_teacher)
        accepted_records: list[dict] = []
        resume_sft = out / "sft" / "accepted.jsonl"
        if not resume_sft.is_file():
            resume_sft = out / "sft" / "train.jsonl"
        if resume_sft.is_file() and not force and from_stage is None:
            accepted_records = [
                json.loads(line)
                for line in resume_sft.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        manifest["counts"].update(
            {"accepted": 0, "teacher_rejected": 0, "teacher_indeterminate": 0,
             "serialize_rejected": 0, "cache_reused": 0, "rlvr_exported": 0}
        )
        reject_histogram: Counter[str] = Counter()
        accepted_by_family: Counter[str] = Counter()
        teacher_completed = teacher_error = teacher_no_finish = 0
        pass_count = serialized = 0
        attempted_task_ids = indeterminate_task_ids = 0
        tokens = 0
        recovery_positive = 0
        recovery_by_code: dict[str, dict[str, int]] = {"semantic": {}, "syntax": {}}
        budget_warned = budget_stopped = False
        seen_this_run: set[str] = set()
        unique_queries = UniqueQueryIndex()
        family_targets = {name: fam.target_n for name, fam in config.families.items()}
        overlay_caps = (
            unique_caps(family_targets, unique_query_budget)
            if unique_query_budget
            else None
        )

        def _note_recovery(episode, verification) -> None:
            nonlocal recovery_positive
            if verify_mod.is_recovery_positive(episode, verification):
                recovery_positive += 1
            for code in verify_mod.recovery_codes(episode):
                klass = verify_mod.ACTION_ERROR_CLASS.get(code)
                if klass is None:
                    continue
                bucket = recovery_by_code[klass]
                bucket[code] = bucket.get(code, 0) + 1

        def _serialize_cache(package, cache, intent) -> None:
            nonlocal pass_count, serialized, indeterminate_task_ids
            nonlocal teacher_completed, teacher_error, teacher_no_finish, tokens
            task_id = intent["task_id"]
            common = {
                "task_id": task_id,
                "task_package_ref": f"task_packages/{task_id}.json",
                "rollouts_ref": f"rollouts/cache/{task_id}.json",
                "attempts": _teacher_attempts_summary(cache),
                "intent": dict(intent),
            }
            for attempt in cache.attempts:
                status = attempt.verification.execution
                if status == "ok":
                    teacher_completed += 1
                elif status == "error":
                    teacher_error += 1
                elif status == "no_finish":
                    teacher_no_finish += 1
                for turn in attempt.episode.turns:
                    usage = turn.usage or {}
                    tokens += int(usage.get("prompt_tokens") or 0) + int(
                        usage.get("completion_tokens") or 0
                    )
            attempt = _pick_pass_attempt(cache)
            if attempt is not None:
                pass_count += 1
                try:
                    record = serialize_mod.serialize(
                        package, attempt.episode, attempt.verification,
                        config=config,
                        accepted_from_attempt=cache.selected_attempt + 1,
                    )
                except SerializeError as exc:
                    manifest["counts"]["serialize_rejected"] += 1
                    indeterminate_task_ids += 1
                    _append_jsonl(
                        out / "rejects" / "serialize.jsonl",
                        {
                            **common, "stage": "serialize",
                            "status": "indeterminate",
                            "failure_codes": [exc.code],
                            "reason_detail": exc.detail,
                        },
                    )
                    _bump_codes(reject_histogram, [exc.code])
                    return
                accepted_records.append(record)
                manifest["counts"]["accepted"] += 1
                serialized += 1
                accepted_by_family[intent["family"]] += 1
                unique_queries.add(
                    package.query, family=intent["family"], task_id=task_id
                )
                _note_recovery(attempt.episode, attempt.verification)
            elif any(a.verification.status == "fail" for a in cache.attempts):
                first_fail = next(
                    a for a in cache.attempts if a.verification.status == "fail"
                )
                manifest["counts"]["teacher_rejected"] += 1
                codes = list(first_fail.verification.failure_codes)
                _append_jsonl(
                    out / "rejects" / "teacher.jsonl",
                    {
                        **common, "stage": "teacher", "status": "fail",
                        "failure_codes": codes,
                    },
                )
                _bump_codes(reject_histogram, codes)
            else:
                manifest["counts"]["teacher_indeterminate"] += 1
                indeterminate_task_ids += 1
                codes = list(cache.attempts[-1].verification.failure_codes) if cache.attempts else []
                _append_jsonl(
                    out / "rejects" / "indeterminate.jsonl",
                    {
                        **common, "stage": "teacher", "status": "indeterminate",
                        "failure_codes": codes,
                    },
                )
                _bump_codes(reject_histogram, codes)

        for intent in intents:
            task_id = intent["task_id"]
            if task_id in seen_this_run:
                raise BuildError(
                    f"task_id {task_id!r} seen twice within one run (bad enumerator)"
                )
            seen_this_run.add(task_id)
            if task_id in terminal:
                manifest["counts"]["skipped_terminal"] += 1
                continue
            if overlay_caps is not None:
                family_cap = overlay_caps.get(intent["family"], 0)
                if unique_queries.by_family[intent["family"]] >= family_cap:
                    continue

            if serialize_only:
                cache_path = out / "rollouts" / "cache" / f"{task_id}.json"
                pkg_path = out / "task_packages" / f"{task_id}.json"
                if not cache_path.is_file() or not pkg_path.is_file():
                    continue
                package = TaskPackage.from_dict(
                    json.loads(pkg_path.read_text(encoding="utf-8"))
                )
                cache = _load_cache(cache_path)
                _serialize_cache(package, cache, intent)
                continue

            family_cfg = config.families[intent["family"]]
            gram_anchor = (
                author_mod.portion_table_gram_anchor(catalog)
                if family_cfg.gram_anchor
                else None
            )
            skip_upstream = (
                from_stage in ("rollout", "materialize", "gate")
                and (out / "task_packages" / f"{task_id}.json").is_file()
                and from_stage == "rollout"
            )
            if skip_upstream:
                package = TaskPackage.from_dict(
                    json.loads(
                        (out / "task_packages" / f"{task_id}.json").read_text(
                            encoding="utf-8"
                        )
                    )
                )
                task = None
            else:
                task, reject = author_mod.author_task(
                    intent,
                    catalog=catalog,
                    expander=expander,
                    gram_anchor=gram_anchor,
                    fallback_expander=(
                        qwen_fallback
                        if intent["family"]
                        in ("composite", "composite_update_log_recommend")
                        else None
                    ),
                    parse_retries=config.expander.parse_retries,
                )
                if task is None:
                    _append_jsonl(out / "rejects" / "author.jsonl", reject)
                    manifest["counts"]["rejected"]["author"] += 1
                    _bump_codes(reject_histogram, reject.get("failure_codes"))
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
                    unique_queries.add(
                        task.query, family=intent["family"], task_id=task_id
                    )
                    if unique_query_budget and len(unique_queries) >= unique_query_budget:
                        break
                    continue

                gate_result = gates_mod.run(task, gate_ctx)
                if not gate_result.keep:
                    record = gates_mod.rejects_record(
                        gate_result, task, intent=intent
                    )
                    route = "indeterminate" if record["status"] == "indeterminate" else "gate"
                    _append_jsonl(out / "rejects" / REJECT_STAGE_FILES[route], record)
                    manifest["counts"]["rejected"][route] += 1
                    _bump_codes(reject_histogram, record.get("failure_codes"))
                    if route == "indeterminate":
                        indeterminate_task_ids += 1
                    continue
                manifest["counts"]["gate_kept"] += 1
                if not run_teacher:
                    unique_queries.add(
                        task.query, family=intent["family"], task_id=task_id
                    )
                    if unique_query_budget and len(unique_queries) >= unique_query_budget:
                        break
                if dry_run:
                    continue

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
                if stop_after == "gate":
                    continue

            if run_rlvr:
                export = rlvr_mod.export_rlvr(package)
                rlvr_mod.write_rlvr(export, out / "rlvr")
                manifest["counts"]["rlvr_exported"] += 1

            if not run_teacher:
                continue
            if _est_usd(tokens) >= config.usd_budget and config.on_budget == "stop" and tokens > 0:
                budget_stopped = True
                break
            attempted_task_ids += 1

            cache_path = out / "rollouts" / "cache" / f"{task_id}.json"
            if cache_path.is_file() and not rerun_teacher:
                cache = _load_cache(cache_path)
                manifest["counts"]["cache_reused"] += 1
            else:
                if task is None:
                    task = _task_from_package(package, catalog)
                cache = _teacher_stage(
                    package, task, config=config,
                    family_cfg=family_cfg,
                    teacher_complete=teacher_complete, catalog=catalog, out=out,
                )
            _serialize_cache(package, cache, intent)
            if unique_query_budget and len(unique_queries) >= unique_query_budget:
                break

            est = _est_usd(tokens)
            if (
                config.usd_budget > 0
                and est >= 0.8 * config.usd_budget
                and not budget_warned
            ):
                log.warning(
                    "cost reached 80%% of usd_budget (est_usd=%s budget=%s)",
                    est, config.usd_budget,
                )
                budget_warned = True
            if config.usd_budget > 0 and est >= config.usd_budget:
                if config.on_budget == "stop":
                    budget_stopped = True
                    break

        if run_teacher or serialize_only:
            by_id = {record["task_id"]: record for record in accepted_records}
            merged = [by_id[key] for key in sorted(by_id)]
            blob = "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in merged
            )
            buckets = {"train": [], "holdout": [], "loss_val": []}
            for record in merged:
                buckets[split_by_task_id(record["task_id"])].append(record)
            for name, rows in buckets.items():
                blob = "".join(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    for record in rows
                )
                _atomic_write(
                    out / "sft" / f"{name}.jsonl", blob, before_replace=before_replace
                )
            # train.jsonl is the 70% split; keep a full accepted dump for resume
            full_blob = "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in merged
            )
            _atomic_write(
                out / "sft" / "accepted.jsonl", full_blob, before_replace=before_replace
            )

        rejects_dir = out / "rejects"
        if rejects_dir.is_dir():
            for path in rejects_dir.glob("*.jsonl"):
                _finalize_jsonl(path)

        if dry_run:
            report = {
                "schema_version": "nutrimind-v2-dryrun/1",
                "projected_accepts": {
                    "total": manifest["counts"]["gate_kept"],
                },
                "reject_histogram": dict(sorted(reject_histogram.items())),
                "intents": manifest["counts"]["intents"],
                "authored": manifest["counts"]["authored"],
                "gate_kept": manifest["counts"]["gate_kept"],
                "unique_query_count": len(unique_queries),
            }
            _atomic_write(
                out / "dry_run_report.json",
                json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                before_replace=before_replace,
            )

        manifest["status"] = "complete"
        if budget_stopped:
            manifest["status"] = "stopped_budget"
        _finalize_observability(
            manifest,
            config=config,
            catalog_sha=catalog_sha,
            reject_histogram=reject_histogram,
            accepted_by_family=accepted_by_family,
            teacher_completed=teacher_completed,
            teacher_error=teacher_error,
            teacher_no_finish=teacher_no_finish,
            pass_count=pass_count,
            serialized=serialized,
            attempted_task_ids=attempted_task_ids,
            indeterminate_task_ids=indeterminate_task_ids,
            accepted_records=accepted_records,
            tokens=tokens,
            recovery_positive=recovery_positive,
            recovery_by_code=recovery_by_code,
            budget_warned=budget_warned,
            budget_stopped=budget_stopped,
            unique_queries=unique_queries,
            unique_query_budget=unique_query_budget,
        )
        _write_manifest(out, manifest, before_replace=before_replace)
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
        "--from-stage",
        choices=("author", "gate", "materialize", "rollout", "serialize"),
        default=None,
        help="re-run terminal tasks from this stage (ticket 014)",
    )
    parser.add_argument(
        "--expander", choices=("synthetic", "commandcode", "ark"), default=None,
        help="speech expander: synthetic (offline) or commandcode "
        "(NUTRIMIND_ALLOW_NETWORK=1 + COMMANDCODE_API_KEY)",
    )
    parser.add_argument(
        "--teacher", choices=("ark", "commandcode"), default=None,
        help="teacher adapter for target=sft (network-guarded: "
        "NUTRIMIND_ALLOW_NETWORK=1 + COMMANDCODE_API_KEY required at call time)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="author + gate only; write dry_run_report.json (no teacher)",
    )
    parser.add_argument(
        "--freeze-mini", action="store_true",
        help="freeze 30 TRAIN_ROSTER tasks to sft/val_mini.json (no teacher)",
    )
    parser.add_argument(
        "--unique-query-budget",
        type=int,
        default=None,
        help="pilot overlay: stop after this many unique query identities "
        "(does not change production family target_n)",
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
            "for offline runs or --expander commandcode for live speech.",
            file=sys.stderr,
        )
        return 1

    catalog = load_catalog(config.catalog_path)
    if args.expander == "synthetic":
        from src.training.data_factory.synthetic import synth_expander

        expander = synth_expander(catalog)
    else:
        from src.training.data_factory.rollout import make_ark_expander_client
        from src.training.data_factory.speech import (
            complete_from_chat_client,
            make_brief_expander,
        )

        expander = make_brief_expander(
            complete=complete_from_chat_client(
                make_ark_expander_client(config.expander)
            ),
            catalog=catalog,
            parse_retries=config.expander.parse_retries,
        )

    teacher_complete = None
    if args.teacher in ("ark", "commandcode"):
        from src.training.data_factory.rollout import make_ark_teacher_client

        teacher_complete = make_ark_teacher_client(config.teacher)

    try:
        manifest = build(
            config,
            expander=expander,
            teacher_complete=teacher_complete,
            stop_after=args.stop_after,
            from_stage=args.from_stage,
            force=args.force,
            config_path=args.config,
            dry_run=args.dry_run,
            freeze_mini=args.freeze_mini,
            unique_query_budget=args.unique_query_budget,
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
