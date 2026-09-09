"""Tri-state verifier — Seam 3 (spec §11, §12, §19.2, §19.3).

``verify(task_package, episode) -> VerificationResult`` wraps nutri-env's
binary ``Scorer`` with the ``indeterminate`` state a data factory needs. Three
axes are recorded **separately** so a ``Scorer`` miss and an execution problem
are never conflated:

- ``execution``   — ok / no_finish / invalid_op / error  (v2's own re-parse of
  ``raw_action_text`` vs ``executed_op``; never nutri-env's private
  ``_parse_action`` path).
- ``oracle_exec`` — ok / error / env_mismatch  (reconstruct the environment
  from ``task_package.environment`` via the public file round-trip, compare
  lineage to the episode's end state; a raise → error).
- ``scorer``      — pass / fail / None (None while an axis above is not ok).

``status`` is derived: pass iff all three clean and scorer pass; fail iff both
axes ok and scorer fail; indeterminate otherwise. An exception is **never**
turned into fail, and a completed legal episode with ``Scorer.passed is
False`` is **never** promoted to indeterminate.

Pure given the ``EpisodeResult`` (the only I/O is the sanctioned transient
scratch file of the public round-trip — no network, no global state). This is
a stage module: it imports nutrienv at module level (allowed by spec §18).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import tempfile
import traceback

from nutrienv.bench import Scorer, load_split
from nutrienv.bench.realize import scored_oracles
from nutrienv.bench.pipeline.types import catalog_digest
from nutrienv.harness.runner import FINISH_OPS

from src.training.data_factory.concepts import (
    EpisodeResult,
    TaskPackage,
    VerificationResult,
)

__all__ = [
    "V2_ACTION_OPS",
    "derive_execution",
    "parse_action_text",
    "verify",
]

# v2's own legal-op vocabulary — a mirror of NutriEnv.step's dispatch table at
# the pinned rev. nutri-env's ``OPS`` is NOT in its ``__all__`` (ADR-012:
# private), so v2 cannot import it; the benchmark is frozen, and the guard
# tests would catch a drift on a rev bump.
V2_ACTION_OPS = frozenset(
    {
        "search_foods",
        "amend_meal",
        "get_profile",
        "get_ledger",
        "update_profile",
        "log_meal",
        "get_food",
        "submit_plan",
        "get_dri",
        "update_plan",
    }
) | frozenset(FINISH_OPS)

_FENCED = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)


def _normalize_action(action: dict) -> dict:
    """Mirror the normalization the ReAct harness applies before env.step:
    an accepted ``submit_plan`` drops its free-form ``reasons`` field. v2's
    re-parse must normalize identically or a genuine plan would falsely
    mismatch the action the env received."""
    if (
        action.get("op") == "submit_plan"
        and (action.get("verdict") == "accept" or "verdict" not in action)
    ):
        action = {k: v for k, v in action.items() if k != "reasons"}
    return action


def parse_action_text(text: str | None) -> tuple[dict | None, str]:
    """v2's own re-parse of one assistant turn (spec §12 "Action legality").

    Returns ``(action, status)``: ``("ok", action)`` when the text contains a
    well-formed action JSON with a legal op, else ``(None, reason)`` with
    reason one of ``empty`` / ``no_json`` / ``not_object`` / ``no_op`` /
    ``illegal_op``. Never consults nutri-env's private parser.
    """
    if not text or not text.strip():
        return None, "empty"
    stripped = text.strip()
    candidates: list[str] = []
    fenced = _FENCED.search(stripped)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(stripped)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                data, _ = decoder.raw_decode(candidate, match.start())
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            op = data.get("op")
            if not isinstance(op, str) or not op:
                return None, "no_op"
            if op not in V2_ACTION_OPS:
                return None, "illegal_op"
            return _normalize_action(data), "ok"
    return None, "no_json"


def derive_execution(episode: EpisodeResult) -> str:
    """The execution axis: ok / no_finish / invalid_op / error (spec §12).

    Priority: a recorded error outranks a missing finish, which outranks a
    per-turn legality problem — the episode is judged by its most fundamental
    defect first.
    """
    if episode.error is not None:
        return "error"
    if not episode.reached_finish:
        return "no_finish"
    for turn in episode.turns:
        action, status = parse_action_text(turn.raw_action_text)
        if action is None:
            return "invalid_op"
        if turn.executed_op is None or action != turn.executed_op:
            return "invalid_op"
    return "ok"


# --------------------------------------------------------------------------- #
# environment reconstruction (oracle_exec axis)
# --------------------------------------------------------------------------- #


def _reconstruct_task(task_package: TaskPackage, episode: EpisodeResult):
    """Rebuild the runnable ``Task`` from the package blocks via the public
    file round-trip (ticket 002 Part A). persona / situations / id come from
    the episode's task — the package's reconstruction contract is the
    environment + oracle payload."""
    task = episode.task
    item = {
        "id": task.id,
        "family": task_package.family,
        "persona": task.persona,
        "situations": list(task.situations),
        "query": task_package.query,
        "s0": task_package.environment.s0,
        "oracle": task_package.oracle.payload,
    }
    if task_package.tier:
        item["tier"] = task_package.tier
    with tempfile.TemporaryDirectory(prefix="nutrimind-verify-") as scratch_dir:
        scratch = pathlib.Path(scratch_dir) / "item.json"
        payload = {"items": [item]}
        scratch.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (rebuilt,) = load_split(scratch)
    return rebuilt


def _lineage(task_package: TaskPackage, episode: EpisodeResult, rebuilt) -> list:
    """env_mismatch evidence: everything that must hold for the episode to
    belong to this package (same world, same person, s0 ledger is a prefix of
    the end ledger, same query)."""
    problems: list = []
    end = episode.end_state
    if task_package.query != episode.task.query:
        problems.append(
            {"check": "query", "package": task_package.query, "episode": episode.task.query}
        )
    if rebuilt.s0.profile.user_id != end.profile.user_id:
        problems.append(
            {
                "check": "user_id",
                "package": rebuilt.s0.profile.user_id,
                "episode": end.profile.user_id,
            }
        )
    expected_prefix = tuple(rebuilt.s0.ledger)
    actual_prefix = tuple(end.ledger)[: len(expected_prefix)]
    if expected_prefix != actual_prefix:
        problems.append(
            {
                "check": "s0_ledger_prefix",
                "package": len(expected_prefix),
                "episode": len(end.ledger),
            }
        )
    return problems


# --------------------------------------------------------------------------- #
# fail evidence enrichment (spec §12: the concrete number that missed)
# --------------------------------------------------------------------------- #


def _plan_totals(end_state, items) -> dict[str, float]:
    totals: dict[str, float] = {}
    for entry in items or []:
        food = end_state.catalog.get(entry.get("food_id")) or {}
        for key, amount in (food.get("nutrients") or {}).items():
            totals[key] = totals.get(key, 0.0) + float(amount) * float(
                entry.get("grams", 0.0)
            ) / 100.0
    return totals


def _fail_detail(end_state, oracle, tag: str, submitted_items=None):
    """A short human-readable record of WHAT missed (spec §12 evidence).

    ``submitted_items`` is the episode's last ``submit_plan`` payload — the
    env rejects malformed plans (e.g. unknown food_id) without storing them,
    so the last_plan alone cannot always name the offender."""
    if tag == "window":
        windows = oracle.plan_windows if oracle.plan_windows is not None else (
            oracle.profile.windows if oracle.profile is not None else {}
        )
        totals = _plan_totals(end_state, end_state.last_plan)
        misses = []
        for nutrient, bounds in windows.items():
            lo, hi = bounds
            amount = totals.get(nutrient, 0.0)
            if amount < lo or amount > hi:
                misses.append(f"{nutrient} {amount:.1f} outside [{lo}, {hi}]")
        return {"window_misses": misses} if misses else {"window_misses": "none computed"}
    if tag == "allergy":
        profile = oracle.profile if oracle.profile is not None else end_state.profile
        prohibited = set(profile.allergies)
        offenders = []
        for entry in end_state.last_plan or []:
            food = end_state.catalog.get(entry.get("food_id")) or {}
            hit = prohibited & set(food.get("allergen_tags") or [])
            if hit:
                offenders.append({"food_id": entry.get("food_id"), "allergens": sorted(hit)})
        return {"allergy_offenders": offenders}
    if tag == "inventory_miss":
        allowed = oracle.allowed_food_ids
        if allowed is None:
            allowed = end_state.allowed_food_ids
        allowed_set = set(allowed or ())
        off = [
            entry.get("food_id")
            for entry in end_state.last_plan or []
            if entry.get("food_id") not in allowed_set
        ]
        return {"off_allowed_food_ids": off}
    if tag == "wrong_goal":
        # the env drops rejected plans: fall back to what was actually submitted
        items = end_state.last_plan or submitted_items or []
        bad = [
            entry.get("food_id")
            for entry in items
            if entry.get("food_id") not in end_state.catalog
        ]
        if bad:
            return {"nonexistent_food_ids": bad, "rejected_by_env": not end_state.last_plan}
        if oracle.last_plan:
            return {
                "plan_items": len(end_state.last_plan or []),
                "oracle_items": len(oracle.last_plan),
            }
        return {"plan": "malformed or missing"}
    if tag == "log_miss":
        tail = oracle.ledger_tail or []
        got = [
            (row.food_id, row.grams)
            for row in end_state.ledger[-len(tail):]
        ] if tail else []
        return {
            "expected_tail": [(r.food_id, r.grams) for r in tail],
            "ledger_tail": got,
        }
    if tag == "update_miss":
        if oracle.profile is not None:
            names = {f.name for f in dataclasses.fields(type(end_state.profile))}
            differing = sorted(
                name
                for name in names
                if getattr(end_state.profile, name, None)
                != getattr(oracle.profile, name, None)
            )
            return {"profile_diff": differing}
        return {"profile": "oracle profile missing"}
    return {"tag": tag}


# --------------------------------------------------------------------------- #
# the verifier
# --------------------------------------------------------------------------- #


def verify(task_package: TaskPackage, episode: EpisodeResult) -> VerificationResult:
    """Judge one episode against one TaskPackage (three axes, derived status)."""
    execution = derive_execution(episode)
    evidence: list = []

    oracle_exec = "ok"
    rebuilt = None
    try:
        sha = catalog_digest(episode.end_state.catalog)
        if sha != task_package.catalog.catalog_sha:
            oracle_exec = "env_mismatch"
            evidence.append(
                {
                    "check": "catalog_sha",
                    "episode": sha,
                    "package": task_package.catalog.catalog_sha,
                }
            )
        else:
            rebuilt = _reconstruct_task(task_package, episode)
            problems = _lineage(task_package, episode, rebuilt)
            if problems:
                oracle_exec = "env_mismatch"
                evidence.extend(problems)
    except Exception:
        oracle_exec = "error"
        evidence.append({"traceback": traceback.format_exc()})

    scorer_result = None
    submitted_items = None
    for turn in episode.turns:
        if (
            isinstance(turn.executed_op, dict)
            and turn.executed_op.get("op") == "submit_plan"
        ):
            submitted_items = turn.executed_op.get("items")
    if execution == "ok" and oracle_exec == "ok":
        try:
            scorer_result = Scorer().score(episode.end_state, rebuilt.oracle)
        except Exception:
            oracle_exec = "error"
            evidence.append({"traceback": traceback.format_exc()})

    reward_map = task_package.reward_semantics.map
    if execution == "ok" and oracle_exec == "ok" and scorer_result is not None:
        status = "pass" if scorer_result["passed"] is True else "fail"
    else:
        status = "indeterminate"

    failure_codes: list[str] = []
    if status == "fail":
        failure_codes.append("task_fail")
        tag = scorer_result["tag"]
        failure_codes.append(tag)
        evidence.append({"scorer_tag": tag})
        if "sub_tags" in scorer_result:
            evidence.append({"sub_tags": list(scorer_result["sub_tags"])})
        # composite: detail for the first failing sub-oracle, with its index
        if getattr(rebuilt.oracle, "sub_oracles", None):
            for index, sub in enumerate(scored_oracles(rebuilt.oracle)):
                sub_tag = Scorer().score(episode.end_state, sub)["tag"]
                if sub_tag != "pass":
                    evidence.append(
                        {
                            "failing_sub_oracle": index,
                            **_fail_detail(
                                episode.end_state, sub, sub_tag,
                                submitted_items=submitted_items,
                            ),
                        }
                    )
                    break
        else:
            evidence.append(
                _fail_detail(
                    episode.end_state, rebuilt.oracle, tag,
                    submitted_items=submitted_items,
                )
            )
    elif status == "pass":
        evidence.append({"scorer_tag": "pass"})
        if "sub_tags" in scorer_result:
            evidence.append({"sub_tags": list(scorer_result["sub_tags"])})
    else:
        if execution == "error":
            failure_codes.append("teacher_error")
        elif execution == "no_finish":
            failure_codes.append("teacher_no_finish")
        elif execution == "invalid_op":
            failure_codes.append("teacher_invalid_op")
            bad = next(
                (
                    i
                    for i, turn in enumerate(episode.turns)
                    if parse_action_text(turn.raw_action_text)[0] != turn.executed_op
                ),
                None,
            )
            evidence.append({"failing_turn": bad})
        if oracle_exec == "error":
            failure_codes.append("oracle_error")
        elif oracle_exec == "env_mismatch":
            failure_codes.append("env_reconstruction_mismatch")

    reward = (
        reward_map["pass"]
        if status == "pass"
        else reward_map["fail"] if status == "fail" else reward_map["indeterminate"]
    )

    if scorer_result is None:
        scorer_axis = None
    else:
        scorer_axis = "pass" if scorer_result["passed"] is True else "fail"

    return VerificationResult(
        status=status,
        execution=execution,
        oracle_exec=oracle_exec,
        scorer=scorer_axis,
        reward=reward,
        oracle_version=task_package.oracle.oracle_version,
        rubric_version=task_package.rubric_version,
        reward_version=task_package.reward_semantics.reward_version,
        failure_codes=failure_codes,
        evidence=evidence,
        diagnostic_scores=None,  # soft rubric: never computed in v2.0 (spec §13)
    )
