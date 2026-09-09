"""Shared seam concept types for the v2 data factory (ticket 003).

These are the plain data records the pure seams exchange (spec §19.7). The module
is deliberately **logic-free**: field definitions, docstrings, and trivial
``to_dict`` / ``from_dict`` only. An AST test in
``tests/training/data_factory/test_concepts.py`` enforces that constraint.

Ownership (who produces / consumes each type):

- ``GateResult``            — Seam 2 output   (ticket 005 ``gates.run``).
- ``TurnMeta``              — produced by the instrumented ReActHarness (ticket 009),
                              consumed by the verifier (007) and serializer (008).
                              Lives here so neither ticket depends on the other.
- ``EpisodeResult``         — one teacher rollout (ticket 009); the single input the
                              verifier (Seam 3, ticket 007) and serializer (Seam 4,
                              ticket 008) take.
- ``VerificationResult``    — Seam 3 output, spec §12 shape (three axes + status).
- ``RolloutCache``          — the on-disk ``rollouts/cache/<task_id>.json`` container
                              (written by ticket 011, re-read by ticket 014's
                              ``--from-stage serialize``).
- ``TaskPackage`` (+ its nested blocks) — the canonical artifact, spec §9.1 shape
                              (materialized by ticket 006).

``Any``-typed fields (``EpisodeResult.end_state`` / ``.task``, ``TurnMeta.executed_op``,
``TaskEnvironment.s0``) are opaque passthrough: live objects in-process, plain
JSON-safe dicts after a ``to_dict`` round-trip. This module never imports nutrienv
(ADR-012 / spec §18 keep nutrienv imports inside the stage modules that need them).
"""

from __future__ import annotations

import dataclasses
from typing import Any

__all__ = [
    "AttemptRecord",
    "EpisodeResult",
    "GateResult",
    "Provenance",
    "RewardSemantics",
    "RolloutCache",
    "TaskCatalogRef",
    "TaskEnvironment",
    "TaskOracle",
    "TaskPackage",
    "TaskVerifierRef",
    "Termination",
    "TurnMeta",
    "VerificationResult",
]


# --------------------------------------------------------------------------- #
# Seam 2 — gates (ticket 005)
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class GateResult:
    """Outcome of ``gates.run(task, ctx)`` (spec §11 gate.*, ordered, first wins).

    ``keep`` is True iff every gate passed. On a reject, ``failure_code`` is the
    stable slug (``gate.verbatim_query_collision`` / ``gate.semantic_key_collision``
    / ``gate.slot_value_overlaps_exam`` / ``gate.stage_a`` / ``gate.draft_invalid``
    / ``gate.unachievable``) and ``reason_detail`` carries the human text.
    """

    keep: bool
    failure_code: str | None = None
    reason_detail: str | None = None
    stage: str = "gate"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GateResult":
        return cls(
            keep=data["keep"],
            failure_code=data.get("failure_code"),
            reason_detail=data.get("reason_detail"),
            stage=data.get("stage", "gate"),
        )


# --------------------------------------------------------------------------- #
# Teacher rollout metadata (ticket 009 producer; 007 / 008 consumers)
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class TurnMeta:
    """One ReAct turn, as recorded by v2's instrumented harness (spec §12).

    ``raw_action_text`` is the assistant text whose action was executed;
    ``executed_op`` the action ``NutriEnv.step`` actually received. ``parse_status``
    / ``fallback_used`` / ``fallback_reason`` come from v2's **own** re-parse of
    ``raw_action_text`` — never from ``nutrienv.harness.react._parse_action``
    internals. ``content`` / ``reasoning_content`` / ``finish_reason`` / ``usage``
    are the teacher client's completion payload for the turn (spec §7);
    ``observation`` is the env observation this turn produced (the next ``user``
    message in the serialized trajectory, spec §9.2).
    """

    raw_action_text: str | None = None
    executed_op: dict | None = None
    parse_status: str | None = None  # v2 re-parse verdict, e.g. "ok" | "invalid"
    fallback_used: bool = False
    fallback_reason: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None  # {prompt_tokens, completion_tokens, reasoning_tokens}
    observation: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TurnMeta":
        return cls(
            raw_action_text=data.get("raw_action_text"),
            executed_op=data.get("executed_op"),
            parse_status=data.get("parse_status"),
            fallback_used=data.get("fallback_used", False),
            fallback_reason=data.get("fallback_reason"),
            content=data.get("content"),
            reasoning_content=data.get("reasoning_content"),
            finish_reason=data.get("finish_reason"),
            usage=data.get("usage"),
            observation=data.get("observation"),
        )


@dataclasses.dataclass
class EpisodeResult:
    """One teacher rollout — the single input the verifier and serializer take.

    ``end_state`` is the world state the rollout produced (a live nutrienv
    ``WorldState`` in-process; a dict after a JSON round-trip). ``task`` is the
    resolved ``Task`` the episode ran against (spec §17). ``error`` is set when the
    attempt died abnormally (API error, timeout); ``reached_finish`` is True iff a
    FINISH op terminated the episode inside the step budget.
    ``reset_observation`` is the env's ``reset`` observation — the FIRST user
    message of the serialized trajectory (each ``TurnMeta.observation`` is the
    one that turn PRODUCED, i.e. the next user message; spec §9.2).
    """

    end_state: Any = None
    turns: list[TurnMeta] = dataclasses.field(default_factory=list)
    reached_finish: bool = False
    error: str | None = None
    task: Any = None
    latency_s: float | None = None
    reset_observation: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeResult":
        return cls(
            end_state=data.get("end_state"),
            turns=[TurnMeta.from_dict(t) for t in data.get("turns", [])],
            reached_finish=data.get("reached_finish", False),
            error=data.get("error"),
            task=data.get("task"),
            latency_s=data.get("latency_s"),
            reset_observation=data.get("reset_observation"),
        )


# --------------------------------------------------------------------------- #
# Seam 3 — tri-state verification (ticket 007), spec §12
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class VerificationResult:
    """Tri-state verification outcome — three independent axes + derived status.

    ``status`` is derived by the verifier from the axes (pass iff
    ``execution == "ok" and oracle_exec == "ok" and scorer == "pass"``; fail iff
    the first two are ok and ``scorer == "fail"``; indeterminate otherwise), never
    from ``Scorer`` alone. An exception is never turned into ``fail``.
    ``reward`` is 1.0 / 0.0 / None per the binary v2-r1 map. ``diagnostic_scores``
    (soft rubric) NEVER affects status or reward in v2.0 (spec §13).
    """

    status: str  # "pass" | "fail" | "indeterminate"
    execution: str  # "ok" | "no_finish" | "invalid_op" | "error"
    oracle_exec: str  # "ok" | "error" | "env_mismatch"
    scorer: str | None  # "pass" | "fail" | None
    reward: float | None  # 1.0 | 0.0 | None
    oracle_version: str  # "nutrienv-<rev>" for v2.0
    rubric_version: str  # "v2-r1"
    reward_version: str  # "v2-r1"
    failure_codes: list[str] = dataclasses.field(default_factory=list)
    evidence: list[Any] = dataclasses.field(default_factory=list)
    diagnostic_scores: dict | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VerificationResult":
        return cls(
            status=data["status"],
            execution=data["execution"],
            oracle_exec=data["oracle_exec"],
            scorer=data.get("scorer"),
            reward=data.get("reward"),
            oracle_version=data["oracle_version"],
            rubric_version=data.get("rubric_version", "v2-r1"),
            reward_version=data.get("reward_version", "v2-r1"),
            failure_codes=list(data.get("failure_codes", [])),
            evidence=list(data.get("evidence", [])),
            diagnostic_scores=data.get("diagnostic_scores"),
        )


# --------------------------------------------------------------------------- #
# Rollout cache (ticket 011 writes; ticket 014 re-reads)
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class AttemptRecord:
    """One teacher attempt (1..k) of a ``task_id`` inside a ``RolloutCache``."""

    attempt_id: str  # f"{task_id}--attempt-{n:02d}"
    episode: EpisodeResult
    verification: VerificationResult

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AttemptRecord":
        return cls(
            attempt_id=data["attempt_id"],
            episode=EpisodeResult.from_dict(data["episode"]),
            verification=VerificationResult.from_dict(data["verification"]),
        )


@dataclasses.dataclass
class RolloutCache:
    """The on-disk ``rollouts/cache/<task_id>.json`` container (spec §17).

    Multi-attempt by construction: one ``AttemptRecord`` per teacher attempt 1..k
    that ran (attempt 1 at ``temperature_first``, 2..k at ``temperature_retry``;
    the loop stops at the first Pass). ``selected_attempt`` is the **0-based index**
    into ``attempts`` of the first Pass, or ``None`` if no attempt passed.
    """

    task_id: str
    attempts: list[AttemptRecord] = dataclasses.field(default_factory=list)
    selected_attempt: int | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RolloutCache":
        return cls(
            task_id=data["task_id"],
            attempts=[AttemptRecord.from_dict(a) for a in data.get("attempts", [])],
            selected_attempt=data.get("selected_attempt"),
        )


# --------------------------------------------------------------------------- #
# TaskPackage (canonical artifact, spec §9.1) — ticket 006 materializes
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class TaskEnvironment:
    """``environment`` block — the serialized ``s0`` plus the public reconstruction
    path (ticket 002 Part A: ``task_to_item`` → ``freeze_tasks`` → ``load_split``
    through a transient scratch file; no public in-memory entry exists)."""

    s0: dict
    reconstruct_with: str
    note: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskEnvironment":
        return cls(
            s0=data["s0"],
            reconstruct_with=data["reconstruct_with"],
            note=data.get("note"),
        )


@dataclasses.dataclass
class TaskCatalogRef:
    """``catalog`` block — the world facts the task was authored against."""

    catalog_sha: str
    nutrienv_rev: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskCatalogRef":
        return cls(catalog_sha=data["catalog_sha"], nutrienv_rev=data["nutrienv_rev"])


@dataclasses.dataclass
class TaskOracle:
    """``oracle`` block — the freezer payload of ``Task.oracle`` (``sub_oracles``
    for composite) plus the oracle version."""

    payload: dict
    oracle_version: str
    note: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskOracle":
        return cls(
            payload=data["payload"],
            oracle_version=data["oracle_version"],
            note=data.get("note"),
        )


@dataclasses.dataclass
class TaskVerifierRef:
    """``verifier`` block — which scorer judges this task and how it is called."""

    kind: str  # "nutrienv.bench.scorer.Scorer"
    call: str  # "Scorer().score(end_state, oracle)"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskVerifierRef":
        return cls(kind=data["kind"], call=data["call"])


@dataclasses.dataclass
class RewardSemantics:
    """``reward_semantics`` block — binary v2-r1: pass→1.0, fail→0.0,
    indeterminate→null."""

    reward_version: str
    kind: str  # "binary"
    map: dict  # {"pass": 1.0, "fail": 0.0, "indeterminate": None}

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RewardSemantics":
        return cls(
            reward_version=data["reward_version"],
            kind=data["kind"],
            map=dict(data["map"]),
        )


@dataclasses.dataclass
class Termination:
    """``termination`` block — FINISH ops and the family step budget
    (``nutrienv.harness.runner.FINISH_OPS`` + ``FAMILY_MAX_STEPS``)."""

    finish_ops: list[str]
    max_steps: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Termination":
        return cls(
            finish_ops=list(data["finish_ops"]),
            max_steps=data["max_steps"],
        )


@dataclasses.dataclass
class Provenance:
    """``provenance`` block — full version trail on every record (spec §10)."""

    nutrimind_rev: str
    nutrienv_rev: str
    catalog_sha: str
    config_sha: str
    intent_ref: str  # e.g. "intents/composite.jsonl#191"
    built_at: str  # ISO-8601 timestamp

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Provenance":
        return cls(
            nutrimind_rev=data["nutrimind_rev"],
            nutrienv_rev=data["nutrienv_rev"],
            catalog_sha=data["catalog_sha"],
            config_sha=data["config_sha"],
            intent_ref=data["intent_ref"],
            built_at=data["built_at"],
        )


@dataclasses.dataclass
class TaskPackage:
    """The canonical task artifact (spec §9.1) — the single source SFT, RLVR, and
    evaluation materializers derive from. Authored once per intent by ticket 006;
    artifacts are never reverse-inferred from each other."""

    schema_version: str  # "nutrimind-v2-taskpackage/1"
    task_key: str  # f"{family}--{'+'.join(steps)}--{person.user_id}" (no seed)
    task_id: str  # f"{task_key}--{seed:06d}" — resume key / sort+dedup key
    query: str
    family: str
    steps: list[str]
    tier: str  # "" except evaluate (EVALUATE_TIERS); never the batch number
    environment: TaskEnvironment
    catalog: TaskCatalogRef
    oracle: TaskOracle
    verifier: TaskVerifierRef
    reward_semantics: RewardSemantics
    rubric_version: str
    termination: Termination
    seed: int
    provenance: Provenance

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskPackage":
        return cls(
            schema_version=data["schema_version"],
            task_key=data["task_key"],
            task_id=data["task_id"],
            query=data["query"],
            family=data["family"],
            steps=list(data["steps"]),
            tier=data["tier"],
            environment=TaskEnvironment.from_dict(data["environment"]),
            catalog=TaskCatalogRef.from_dict(data["catalog"]),
            oracle=TaskOracle.from_dict(data["oracle"]),
            verifier=TaskVerifierRef.from_dict(data["verifier"]),
            reward_semantics=RewardSemantics.from_dict(data["reward_semantics"]),
            rubric_version=data["rubric_version"],
            termination=Termination.from_dict(data["termination"]),
            seed=data["seed"],
            provenance=Provenance.from_dict(data["provenance"]),
        )
