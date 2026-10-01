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
composition root: ``--expander synthetic`` or ``--expander deepseek`` (live
brief on the ``expander:`` channel). ``commandcode`` overlays the optional
channel. Cost (spec §17) is priced per
role and token type from the ``pricing:`` block: teacher usage comes from the
episode turns, expander usage from a :class:`TokenMeter` the CLI wraps around
the expander's chat client.

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
import time
from collections import Counter
from collections.abc import Callable, Mapping

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
from src.training.data_factory.config import (
    ConfigError,
    DataFactoryConfig,
    Pricing,
    TokenRates,
    load_config,
)
from src.training.data_factory.concepts import AttemptRecord, RolloutCache, TaskPackage
from src.training.data_factory.gates import GateContext
from src.training.data_factory.query_identity import UniqueQueryIndex, unique_caps
from src.training.data_factory.roster_train import TRAIN_ROSTER
from src.training.data_factory.rollout_fc import rollout_tool_call
from src.training.data_factory.search_gate import judge_food
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
    # NutriEnv ADR 0029 archetypes (archetypes.py / evaluate_hypo). The leading
    # step keeps their task_key apart from the plain family's.
    "evaluate_hypo": ("evaluate", ("hypo", "evaluate")),
    "recommend_inventory": ("recommend", ("inventory", "recommend")),
    "recommend_menu": ("recommend", ("menu", "recommend")),
    "composite_amend_recommend": ("composite", ("amend", "recommend")),
    "composite_refuse_recommend": ("composite", ("refuse", "recommend")),
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


class TokenMeter:
    """Running token totals for one priced role (teacher or expander).

    ``prompt_tokens`` includes ``cached_tokens`` (OpenAI usage semantics);
    ``completion_tokens`` includes reasoning tokens."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0

    def add(self, usage: Mapping | None) -> None:
        usage = usage or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.cached_tokens += int(usage.get("cached_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)

    def wrap(self, client: Callable[[dict], Mapping]) -> Callable[[dict], Mapping]:
        """A chat client that records each completion's ``usage`` here."""

        def metered(request: dict) -> Mapping:
            completion = client(request)
            self.add(completion.get("usage"))
            return completion

        return metered

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def est_usd(self, rates: TokenRates, usd_per_unit: float) -> float:
        cached = min(self.cached_tokens, self.prompt_tokens)
        cost = (
            (self.prompt_tokens - cached) * rates.input_per_mtok
            + cached * rates.cached_input_per_mtok
            + self.completion_tokens * rates.output_per_mtok
        ) / 1_000_000
        return cost * usd_per_unit

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
        }


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


# Families that share a lab family and an intent index would otherwise author
# the same pool (evaluate_hypo vs evaluate); each gets its own seed range.
_FAMILY_SEED_BASE = {
    "evaluate_hypo": 500_000,
    "recommend_inventory": 510_000,
    "recommend_menu": 520_000,
    "composite_amend_recommend": 530_000,
    "composite_refuse_recommend": 540_000,
}

# NutriEnv ADR 0017 Evaluate-unfit knives. ``swap`` stays out, as in the lab's
# own batch mill (legacy_run_batch._BATCH_KNIVES).
_REJECT_KNIVES = ("over_slot", "under_slot")


def _evaluate_knife(person, index: int) -> str | None:
    """About half the evaluate intents get a knife (a reject gold); an allergic
    person's reject is an allergy trap half the time (ADR 0024/0029: allergy
    rejects are a required share, not a rare draw)."""
    rng = random.Random(f"knife:{index}:{person.user_id}")
    if rng.random() < 0.5:
        return None
    if person.allergies and rng.random() < 0.5:
        return "allergy"
    return rng.choice(_REJECT_KNIVES)


def family_wave_size(config: DataFactoryConfig, family: str) -> int:
    """How many intents the first wave draws for ``family``.

    The 3-leg family uses :func:`candidate_count` (floor 120). Every other
    family uses ``ceil(target_n * over_generate_x)``.
    """
    family_cfg = config.families[family]
    if family == "composite_update_log_recommend":
        return candidate_count(
            1.0 / family_cfg.over_generate_x,
            target_n=family_cfg.target_n,
            max_candidate_limit=config.max_intents,
        )
    return math.ceil(family_cfg.target_n * family_cfg.over_generate_x)


def family_attempt_cap(config: DataFactoryConfig, family: str) -> int:
    """Per-family draw cap: two waves, and never above ``max_intents``.

    One wave is the old behavior (no top-up). The second wave is the room
    the quota refill is allowed to spend before it stops.
    """
    return min(config.max_intents, family_wave_size(config, family) * 2)


def should_top_up(
    *,
    accepted: int,
    target_n: int,
    draws: int,
    cap: int,
    total_intents: int,
    max_intents: int,
    budget_stopped: bool,
) -> bool:
    """Whether this family should draw one more intent."""
    if budget_stopped or accepted >= target_n:
        return False
    if draws >= cap or total_intents >= max_intents:
        return False
    return True


def intent_for(config: DataFactoryConfig, family: str, index: int) -> dict:
    """The intent ``enumerate_intents`` would emit at ``index`` for ``family``."""
    if family not in FAMILY_SPECS:
        raise BuildError(f"config families: no task-key spec for family {family!r}")
    if family not in config.families:
        raise BuildError(f"config families: {family!r} is not configured")
    task_family, steps = FAMILY_SPECS[family]
    family_cfg = config.families[family]
    person = TRAIN_ROSTER[index % len(TRAIN_ROSTER)]
    seed = config.seed_offset + _FAMILY_SEED_BASE.get(family, 0) + index
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
    knife = None
    if family in ("evaluate", "evaluate_hypo"):
        tier = author_mod.EVALUATE_TIERS[index % len(author_mod.EVALUATE_TIERS)]
        knife = _evaluate_knife(person, seed)
        if family == "evaluate_hypo":
            shell = "eval-hypo"
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
    return {
        "schema_version": INTENT_SCHEMA_VERSION,
        "task_id": f"{task_key}--{seed:06d}",
        "task_key": task_key,
        "family": family,
        "task_family": task_family,
        "steps": list(steps),
        "user_id": person.user_id,
        "seed": seed,
        "occasion": occasion,
        "scene": "empty",
        "shell": shell,
        "slots": slots,
        "amount_path": amount_path,
        "ounce_phrasing": ounce,
        "knife": knife,
        "tier": tier,
        "recovery_trap": (
            "unknown_food" if family == "log" and index % 5 == 4 else None
        ),
        "gram_anchor": family_cfg.gram_anchor,
    }


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
        wanted = family_wave_size(config, family)
        if wanted > config.max_intents:
            raise BuildError(
                f"family {family!r} intent count {wanted} exceeds max_intents "
                f"{config.max_intents} (runaway guard)"
            )
        for index in range(wanted):
            intents.append(intent_for(config, family, index))
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


# Locatability statuses that judge the pipeline rather than a food: `unavailable`
# (the package would not rebuild) and `no_foods` (its oracle exposes no ids). They are
# reported in the block, never averaged into the rate.
NOT_A_FOOD_VERDICT = frozenset({"unavailable", "no_foods"})


def _locatability_status(catalog, package: TaskPackage, task=None) -> str:
    """How the agent's own search would reach the foods this task pins.

    Reported per accepted task as `metrics.search_locatability`. This is an
    observable, not a gate: whether an ambiguous pin should be rejected needs the
    number first.

    The verdict is the **weakest** across every pinned food — a task is only as
    findable as its least findable food — and per food it is the weakest across the
    forms the utterance can carry, including the brief's handle (see `search_gate`).

    Foods come from the ``Task``: the live one when the caller has it, otherwise one
    rebuilt out of the package with the same public loader a resume uses (the
    package's oracle payload stores freezer *references*, ``ledger="s0_plus_tail"``,
    not rows, so the ids are only readable through a loaded task).
    """
    if task is None:
        try:
            task = _task_from_package(package, catalog)
        except (KeyError, ValueError, OSError, TypeError) as exc:
            # A package that will not rebuild is a real defect on resume paths; the
            # metric must not turn it into a silent data point.
            log.warning("locatability: cannot rebuild %s: %s", package.task_id, exc)
            return "unavailable"
    verdicts = [
        judge_food(food_id, catalog=catalog)
        for food_id in _oracle_food_ids(getattr(task, "oracle", None))
    ]
    if not verdicts:
        # An accepted task with no pinned food is a defect, not a findability verdict.
        log.warning("locatability: %s pins no food to judge", package.task_id)
        return "no_foods"
    return max(verdicts, key=lambda verdict: verdict.ordinal).status


def _oracle_food_ids(oracle) -> list[str]:
    """The foods an oracle expects, in source order, deduplicated."""
    if oracle is None:
        return []
    sources = list(getattr(oracle, "sub_oracles", None) or ()) or [oracle]
    ids: list[str] = []
    for entry in sources:
        rows = getattr(entry, "ledger_tail", None) or getattr(entry, "ledger", None) or ()
        if isinstance(rows, str):
            continue
        for row in rows:
            food_id = row.get("food_id") if isinstance(row, Mapping) else getattr(row, "food_id", None)
            if food_id and str(food_id) not in ids:
                ids.append(str(food_id))
    return ids


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
    teacher_usage: TokenMeter | None = None,
    expander_usage: TokenMeter | None = None,
    recovery_positive: int = 0,
    recovery_by_code: dict | None = None,
    budget_warned: bool = False,
    budget_stopped: bool = False,
    unique_queries: UniqueQueryIndex | None = None,
    unique_query_budget: int | None = None,
    search_locatability_counts: Counter[str] | None = None,
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
    teacher_usage = teacher_usage or TokenMeter()
    expander_usage = expander_usage or TokenMeter()
    pricing = config.pricing
    from datetime import datetime, timezone

    priced_at = datetime.now(timezone.utc)
    teacher_usd = teacher_usage.est_usd(
        pricing.rates("teacher", priced_at), pricing.usd_per_unit
    )
    expander_usd = expander_usage.est_usd(
        pricing.rates("expander", priced_at), pricing.usd_per_unit
    )
    pricing_block = {
        "currency": pricing.currency,
        "cny_per_usd": pricing.cny_per_usd,
        "fx_as_of": pricing.fx_as_of,
    }
    if pricing.peak is not None:
        pricing_block["tier"] = "peak" if pricing.is_peak(priced_at) else "off_peak"
        pricing_block["tier_as_of"] = priced_at.isoformat(timespec="seconds")
    manifest["cost"] = {
        "est_usd": round(teacher_usd + expander_usd, 6),
        "budget_usd": config.usd_budget,
        "teacher_tokens": teacher_usage.total,
        "expander_tokens": expander_usage.total,
        "on_budget": config.on_budget,
        "budget_warned": budget_warned,
        "budget_stopped": budget_stopped,
        "by_role": {
            "teacher": {**teacher_usage.to_dict(), "est_usd": round(teacher_usd, 6)},
            "expander": {**expander_usage.to_dict(), "est_usd": round(expander_usd, 6)},
        },
        "pricing": pricing_block,
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
    locatability = dict(sorted((search_locatability_counts or Counter()).items()))
    manifest["metrics"]["search_locatability"] = locatability
    # `unavailable` (the package would not rebuild) and `no_foods` (its oracle exposes
    # no ids) are defects in the pipeline, not verdicts about a food, so neither may
    # dilute the rate. Both stay in the block above, where they are visible.
    judged = sum(
        count
        for status, count in locatability.items()
        if status not in NOT_A_FOOD_VERDICT
    )
    manifest["metrics"]["search_locatability_usable_rate"] = _rate(
        int(locatability.get("unique", 0)), judged
    )
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


def _is_infra_error(error: str | None) -> bool:
    if not error:
        return False
    lower = error.lower()
    return any(
        term in lower
        for term in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "timeout",
            "timed out",
            "connection",
            "network",
            "toolcallinfraerror",
            "provider request failed",
        )
    )


def _extract_submission(att: AttemptRecord):
    ep = att.episode
    turns = getattr(ep, "turns", None) or []
    for turn in reversed(turns):
        op = getattr(turn, "executed_op", None) if hasattr(turn, "executed_op") else (turn.get("executed_op") if isinstance(turn, dict) else None)
        if op and isinstance(op, dict):
            op_name = op.get("op")
            if op_name in ("submit_plan", "log_meal", "update_profile", "finish"):
                return op
    for turn in reversed(turns):
        op = getattr(turn, "executed_op", None) if hasattr(turn, "executed_op") else (turn.get("executed_op") if isinstance(turn, dict) else None)
        if op:
            return op
    return None


def _canonical_submission(op: dict | None):
    if not op or not isinstance(op, dict):
        return None
    name = op.get("op")
    if name == "submit_plan":
        items = tuple(sorted((str(i.get("food_id")), round(float(i.get("grams", 0)), 2)) for i in (op.get("items") or [])))
        verdict = op.get("verdict")
        reasons = tuple(sorted(op.get("reasons") or []))
        return ("submit_plan", verdict, items, reasons)
    if name == "log_meal":
        return ("log_meal", str(op.get("food_id")), round(float(op.get("grams", 0)), 2), op.get("eaten_at"))
    if name == "update_profile":
        patch = op.get("patch") or {}
        return ("update_profile", tuple(sorted((str(k), str(v)) for k, v in patch.items())))
    return (name, tuple(sorted((str(k), str(v)) for k, v in op.items() if k != "op")))


def _is_identical_failure(prev: AttemptRecord, curr: AttemptRecord) -> bool:
    if prev.verification.status != "fail" or curr.verification.status != "fail":
        return False
    prev_codes = tuple(sorted(prev.verification.failure_codes or []))
    curr_codes = tuple(sorted(curr.verification.failure_codes or []))
    if not prev_codes or prev_codes != curr_codes:
        return False
    prev_sub = _canonical_submission(_extract_submission(prev))
    curr_sub = _canonical_submission(_extract_submission(curr))
    return prev_sub == curr_sub


def _teacher_stage(
    package, task, *, config, family_cfg, teacher_complete, catalog, out: pathlib.Path,
    teacher_usage: TokenMeter | None = None, expander_usage: TokenMeter | None = None,
) -> RolloutCache:
    """Attempts 1..k (k = family ``teacher_k``, spec §16): attempt 1 at
    ``temperature_first``, 2..k at ``temperature_retry``, stopping at the
    first Pass. Every attempt that ran is recorded; ``selected_attempt`` is
    the 0-based index of the first Pass or None."""
    cache = RolloutCache(task_id=package.task_id)
    attempt_num = 1
    infra_retry_count = 0
    max_infra_retries = 5

    while attempt_num <= family_cfg.teacher_k:
        if (
            config.usd_budget > 0
            and config.on_budget == "stop"
            and teacher_usage is not None
            and expander_usage is not None
        ):
            spent = teacher_usage.total + expander_usage.total
            est = _est_usd(config.pricing, teacher_usage, expander_usage)
            if est >= config.usd_budget and spent > 0:
                break

        episode = rollout_tool_call(
            task,
            teacher_complete=teacher_complete,
            catalog=catalog,
            model=config.teacher.model_id,
            evaluate_hint=True,
        )

        # Infra errors: exponential backoff retry without consuming teacher_k
        if episode.error and _is_infra_error(episode.error):
            infra_retry_count += 1
            if infra_retry_count <= max_infra_retries:
                time.sleep(1.0 * (2 ** (infra_retry_count - 1)))
                continue
            else:
                verification = verify_mod.verify(package, episode)
                cache.attempts.append(
                    AttemptRecord(
                        attempt_id=mz.attempt_id(package.task_id, attempt_num),
                        episode=episode,
                        verification=verification,
                    )
                )
                break

        verification = verify_mod.verify(package, episode)
        cache.attempts.append(
            AttemptRecord(
                attempt_id=mz.attempt_id(package.task_id, attempt_num),
                episode=episode,
                verification=verification,
            )
        )
        if verification.status == "pass":
            cache.selected_attempt = attempt_num - 1
            break

        # Deterministic early stopping after 2 consecutive identical failures
        if len(cache.attempts) >= 2:
            if _is_identical_failure(cache.attempts[-2], cache.attempts[-1]):
                break

        attempt_num += 1

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


def _est_usd(pricing: Pricing, teacher: TokenMeter, expander: TokenMeter, when=None) -> float:
    """Price the accumulated meters at the tier in force at ``when``.

    A boundary crossing reprices the whole total; calls are not stamped.
    """
    from datetime import datetime, timezone

    when = when or datetime.now(timezone.utc)
    return teacher.est_usd(
        pricing.rates("teacher", when), pricing.usd_per_unit
    ) + expander.est_usd(pricing.rates("expander", when), pricing.usd_per_unit)


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
    expander_meter: TokenMeter | None = None,
) -> dict:
    """Run the pipeline through materialize / teacher / serialize / rlvr.

    ``expander_meter`` is the meter wrapped around a live expander's chat
    client (None → the expander spends nothing). Returns the run manifest. Raises :class:`BuildError` (after writing a
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
    expander_usage = expander_meter if expander_meter is not None else TokenMeter()
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
            "intents": 0, "skipped_terminal": 0, "skipped_quota_met": 0, "authored": 0,
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
                expander_usage=expander_usage,
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
                "seed_start": config.seed_offset,
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
        # Search locatability of every pinned food (see search_gate): reported, not
        # enforced. Whether to reject an ambiguous pin is a data decision that needs
        # the number first.
        search_locatability_counts: Counter[str] = Counter()
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
        for record in accepted_records:
            family_name = (record.get("meta") or {}).get("family")
            if isinstance(family_name, str):
                accepted_by_family[family_name] += 1
        teacher_completed = teacher_error = teacher_no_finish = 0
        pass_count = serialized = 0
        attempted_task_ids = indeterminate_task_ids = 0
        teacher_usage = TokenMeter()
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

        def _serialize_cache(package, cache, intent, task=None) -> None:
            nonlocal pass_count, serialized, indeterminate_task_ids
            nonlocal teacher_completed, teacher_error, teacher_no_finish
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
                    teacher_usage.add(turn.usage)
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
                _bump_codes(
                    search_locatability_counts,
                    [_locatability_status(catalog, package, task)],
                )
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

        queue = list(intents)
        next_index = {
            family: sum(1 for row in intents if row["family"] == family)
            for family in config.families
        }
        draws: Counter[str] = Counter()
        cursor = 0

        def _note_attempt(intent) -> None:
            if not run_teacher or budget_stopped or serialize_only:
                return
            if (
                config.usd_budget > 0
                and _est_usd(config.pricing, teacher_usage, expander_usage)
                >= config.usd_budget
            ):
                return
            family = intent["family"]
            draws[family] += 1
            cap = family_attempt_cap(config, family)
            # next_index is how many intents are already minted. draws counts
            # finished ones, so a draws<cap check would still enqueue past cap.
            if next_index[family] >= cap:
                return
            if not should_top_up(
                accepted=accepted_by_family[family],
                target_n=config.families[family].target_n,
                draws=draws[family],
                cap=cap,
                total_intents=manifest["counts"]["intents"],
                max_intents=config.max_intents,
                budget_stopped=budget_stopped,
            ):
                return
            extra = intent_for(config, family, next_index[family])
            next_index[family] += 1
            queue.append(extra)
            manifest["counts"]["intents"] += 1
            _append_jsonl(out / "intents" / f"{family}.jsonl", extra)

        while cursor < len(queue):
            intent = queue[cursor]
            cursor += 1
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
            if (
                config.usd_budget > 0
                and config.on_budget == "stop"
                and (teacher_usage.total + expander_usage.total) > 0
            ):
                est = _est_usd(config.pricing, teacher_usage, expander_usage)
                if est >= config.usd_budget:
                    budget_stopped = True
                    break

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
                    _note_attempt(intent)
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
                    _note_attempt(intent)
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
            family = intent["family"]
            family_cfg = config.families[family]
            if accepted_by_family[family] >= family_cfg.target_n:
                manifest["counts"]["skipped_quota_met"] += 1
                continue
            spent = teacher_usage.total + expander_usage.total
            est = _est_usd(config.pricing, teacher_usage, expander_usage)
            if est >= config.usd_budget and config.on_budget == "stop" and spent > 0:
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
                    teacher_usage=teacher_usage, expander_usage=expander_usage,
                )
            _serialize_cache(package, cache, intent, task)
            if unique_query_budget and len(unique_queries) >= unique_query_budget:
                break

            est = _est_usd(config.pricing, teacher_usage, expander_usage)
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
            _note_attempt(intent)

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
            teacher_usage=teacher_usage,
            expander_usage=expander_usage,
            recovery_positive=recovery_positive,
            recovery_by_code=recovery_by_code,
            budget_warned=budget_warned,
            budget_stopped=budget_stopped,
            unique_queries=unique_queries,
            unique_query_budget=unique_query_budget,
            search_locatability_counts=search_locatability_counts,
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
        "--expander", choices=("synthetic", "deepseek", "commandcode"), default=None,
        help="speech expander: synthetic (offline), deepseek (yaml expander: "
        "channel), or commandcode (optional overlay). Live calls need "
        "NUTRIMIND_ALLOW_NETWORK=1 and the channel's credential_env",
    )
    parser.add_argument(
        "--teacher", choices=("deepseek", "commandcode"), default=None,
        help="teacher adapter for target=sft: deepseek uses teacher:, "
        "commandcode overlays the optional channel "
        "(NUTRIMIND_ALLOW_NETWORK=1 + the channel credential)",
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
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="override config output_dir (e.g. data/student/v2-batch1-200)",
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
            "for offline runs or --expander deepseek for live speech.",
            file=sys.stderr,
        )
        return 1

    catalog = load_catalog(config.catalog_path)
    expander_meter = TokenMeter()

    def _overlay(role_config):
        if config.commandcode is None:
            print(
                "config error: --commandcode needs a commandcode: block",
                file=sys.stderr,
            )
            return None
        return dataclasses.replace(
            role_config,
            model_id=config.commandcode.model_id,
            endpoint=config.commandcode.endpoint,
            credential_env=config.commandcode.credential_env,
        )

    if args.expander == "synthetic":
        from src.training.data_factory.synthetic import synth_expander

        expander = synth_expander(catalog)
    else:
        from src.training.data_factory.rollout import make_expander_client
        from src.training.data_factory.speech import (
            complete_from_chat_client,
            make_brief_expander,
        )

        expander_cfg = config.expander
        if args.expander == "commandcode":
            expander_cfg = _overlay(config.expander)
            if expander_cfg is None:
                return 1
        expander = make_brief_expander(
            complete=complete_from_chat_client(
                expander_meter.wrap(make_expander_client(expander_cfg))
            ),
            catalog=catalog,
            parse_retries=config.expander.parse_retries,
        )

    teacher_complete = None
    if args.teacher in ("deepseek", "commandcode"):
        from src.training.data_factory.rollout import make_teacher_client

        teacher_cfg = config.teacher
        if args.teacher == "commandcode":
            teacher_cfg = _overlay(config.teacher)
            if teacher_cfg is None:
                return 1
        teacher_complete = make_teacher_client(teacher_cfg)

    try:
        manifest = build(
            config,
            output_dir=args.output,
            expander=expander,
            teacher_complete=teacher_complete,
            stop_after=args.stop_after,
            from_stage=args.from_stage,
            force=args.force,
            config_path=args.config,
            dry_run=args.dry_run,
            freeze_mini=args.freeze_mini,
            unique_query_budget=args.unique_query_budget,
            expander_meter=expander_meter,
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
