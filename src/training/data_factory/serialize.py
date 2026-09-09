"""serialize — Seam 4: the pure v2 SFT record writer (spec §9.2 / §11 / §19.4).

``serialize(task_package, episode, verification, *, config, ...)`` builds the
OpenAI-shaped record: ``system`` (the frozen ``react_manual("v2")``) once, the
``Task:`` user turn, then strictly alternating observation / assistant turns,
ending in an assistant FINISH turn. Parallel ``segments`` / ``train_on``
arrays; ``train_on`` is true exactly on ``step`` / ``final`` messages for
Batch 1. ``assistant.content = f"{plan}\\n{op_json}"`` where ``plan`` is the
teacher ``reasoning_content`` truncated to ``plan_max_tokens`` (token-exact
with an injected tokenizer, else a ~4 chars/token heuristic — the mode is
recorded in ``meta.plan_truncation``) and ``op_json`` the action actually
executed against ``NutriEnv``.

No token-level ``loss_mask`` is stored — the v2 loader (ticket 019) derives it
from ``train_on``. ``validate_record`` (also exported) is the loader-mirror
structural check serialize runs on its own output; ticket 019 can reuse it.

Failures raise :class:`SerializeError` with a spec §11 slug
(``serialize.empty_episode`` / ``no_system_turn`` / ``consecutive_assistant`` /
``missing_observation`` / ``turn_count_mismatch`` / ``last_turn_not_finish`` /
``too_long`` / ``no_plan_any_turn``); build routes them to
``rejects/serialize.jsonl`` as indeterminate. One turn without a plan is
tolerated (``plan=""`` + ``meta.n_turns_without_plan``); every turn without a
plan is ``no_plan_any_turn``.

Stage module: imports nutrienv at module level (allowed by spec §18).
"""

from __future__ import annotations

import json

from nutrienv.env import NutriEnv
from nutrienv.harness.react import react_manual
from nutrienv.harness.runner import DEFAULT_MAX_STEPS, FAMILY_MAX_STEPS, FINISH_OPS

from src.training.data_factory.concepts import (
    EpisodeResult,
    TaskPackage,
    VerificationResult,
)
from src.training.data_factory.verify import parse_action_text

__all__ = ["SerializeError", "serialize", "validate_record"]

SCHEMA_VERSION = "nutrimind-v2-sft/1"

# v1 markers must never appear in a v2 assistant turn (spec §9.2) — stripped
# from the teacher plan text before composing the content
_V1_MARKERS = ("<think>", "</think>", "<tool_call>", "</tool_call>",
               "<|im_start|>", "<|im_end|>")


class SerializeError(Exception):
    """A serialize-stage failure (spec §11); ``code`` is the stable slug."""

    def __init__(self, code: str, detail: str | None = None):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}" + (f": {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _token_count(text: str, tokenizer) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return (len(text) + 3) // 4  # ~4 chars/token heuristic (spec §9.2)


def _truncate_plan(plan: str, *, plan_max_tokens: int, tokenizer) -> str:
    if tokenizer is not None:
        ids = tokenizer.encode(plan)
        if len(ids) <= plan_max_tokens:
            return plan
        return tokenizer.decode(ids[:plan_max_tokens])
    return plan[: plan_max_tokens * 4]


def _sanitize_plan(plan: str) -> str:
    for marker in _V1_MARKERS:
        plan = plan.replace(marker, "")
    return plan


def _step_budget(max_steps: int, turn_index: int) -> str:
    remaining = max(0, max_steps - turn_index)
    return f"Step budget: {remaining} action(s) remaining, including this turn."


def _reset_observation(episode: EpisodeResult) -> str:
    """The first user message's observation. Prefer the recorded one; fall
    back to re-deriving from the episode's task s0 (deterministic)."""
    if episode.reset_observation:
        return episode.reset_observation
    if episode.task is None:
        raise SerializeError("serialize.missing_observation", "reset observation")
    observation = NutriEnv().reset(episode.task.s0)
    return json.dumps(observation, default=str)[:6000]


# --------------------------------------------------------------------------- #
# the loader-mirror structural check
# --------------------------------------------------------------------------- #


def validate_record(record: dict) -> None:
    """Structural invariants (spec §9.2 rules + §11 serialize codes). Raises
    ``SerializeError`` with the first violated code; reused by the v2 loader."""
    messages = record.get("messages") or []
    segments = record.get("segments") or []
    train_on = record.get("train_on") or []

    if not messages:
        raise SerializeError("serialize.empty_episode")
    if messages[0].get("role") != "system" or sum(
        1 for m in messages if m.get("role") == "system"
    ) != 1:
        raise SerializeError("serialize.no_system_turn")
    if len(messages) < 2 or messages[1].get("role") != "user":
        raise SerializeError("serialize.no_system_turn", "missing Task turn")
    for previous, current in zip(messages, messages[1:]):
        if previous["role"] == "assistant" and current["role"] == "assistant":
            raise SerializeError("serialize.consecutive_assistant")
    if not (len(messages) == len(segments) == len(train_on)):
        raise SerializeError("serialize.turn_count_mismatch")
    # after system (+task), user/assistant alternate strictly
    if len(messages) > 2:
        for index in range(2, len(messages)):
            expected = "user" if index % 2 == 0 else "assistant"
            if messages[index]["role"] != expected:
                raise SerializeError("serialize.turn_count_mismatch")
    if messages[-1]["role"] != "assistant" or segments[-1] != "final":
        raise SerializeError("serialize.last_turn_not_finish")
    for index, (segment, flag) in enumerate(zip(segments, train_on)):
        if flag != (segment in ("step", "final")):
            raise SerializeError("serialize.turn_count_mismatch")
    for message in messages:
        if message["role"] == "assistant":
            if any(marker in message["content"] for marker in _V1_MARKERS):
                # cannot happen from serialize() (plans are sanitized); this is
                # the loader-mirror invariant (spec §9.2 v1-record rejection)
                raise SerializeError("serialize.v1_marker")


# --------------------------------------------------------------------------- #
# the record writer
# --------------------------------------------------------------------------- #


def serialize(
    task_package: TaskPackage,
    episode: EpisodeResult,
    verification: VerificationResult,
    *,
    config,
    accepted_from_attempt: int = 1,
    batch: int = 1,
    tokenizer=None,
) -> dict:
    """Build one v2 SFT record (spec §9.2) from a verified-PASS episode.

    Raises ``SerializeError`` (spec §11 slugs) on any serialize-edge failure;
    build routes those to ``rejects/serialize.jsonl`` as indeterminate.
    """
    if not episode.turns:
        raise SerializeError("serialize.empty_episode")
    if not episode.reached_finish or (
        episode.turns[-1].executed_op or {}
    ).get("op") not in FINISH_OPS:
        raise SerializeError("serialize.last_turn_not_finish")
    for index, turn in enumerate(episode.turns):
        parsed, _status = parse_action_text(turn.raw_action_text)
        if parsed != turn.executed_op:
            # defensive: verify() already routes these to teacher_invalid_op,
            # so build never sends a non-genuine episode here (spec §12)
            raise SerializeError(
                "serialize.invalid_op_turn", f"turn {index} not a genuine parse"
            )

    max_steps = FAMILY_MAX_STEPS.get(task_package.family, DEFAULT_MAX_STEPS)
    reset_observation = _reset_observation(episode)

    messages: list[dict] = [
        {"role": "system", "content": react_manual("v2")},
        {"role": "user", "content": f"Task:\n{task_package.query}"},
    ]
    segments = ["system", "task"]
    train_on = [False, False]

    turns_without_plan = 0
    task = episode.task
    persona = (
        task.get("persona")
        if isinstance(task, dict)
        else getattr(task, "persona", None)
    )  # dict-backed episode: a cache round-trip (spec §17) has no live Task
    for index, turn in enumerate(episode.turns):
        if index == 0:
            observation = reset_observation
        else:
            observation = episode.turns[index - 1].observation
        if not observation:
            raise SerializeError(
                "serialize.missing_observation", f"turn {index}"
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{_step_budget(max_steps, index)}\nObservation:\n{observation}"
                ),
            }
        )
        segments.append("observation")
        train_on.append(False)

        plan = _sanitize_plan(turn.reasoning_content or "")
        if not plan:
            turns_without_plan += 1
        plan = _truncate_plan(
            plan, plan_max_tokens=config.plan_max_tokens, tokenizer=tokenizer
        )
        op_json = json.dumps(turn.executed_op)
        content = f"{plan}\n{op_json}" if plan else op_json
        messages.append({"role": "assistant", "content": content})
        segments.append("final" if index == len(episode.turns) - 1 else "step")
        train_on.append(True)

    if turns_without_plan == len(episode.turns):
        raise SerializeError("serialize.no_plan_any_turn")

    record = {
        "schema_version": SCHEMA_VERSION,
        "task_key": task_package.task_key,
        "task_id": task_package.task_id,
        "accepted_from_attempt": accepted_from_attempt,
        "task_package_ref": f"task_packages/{task_package.task_id}.json",
        "messages": messages,
        "segments": segments,
        "train_on": train_on,
        "meta": {
            "family": task_package.family,
            "steps": list(task_package.steps),
            "tier": task_package.tier,
            "persona": persona,
            "batch": batch,
            "seed": task_package.seed,
            "teacher": config.teacher.model_id,
            "teacher_params": {
                "thinking": dict(config.teacher.thinking),
                "temperature_first": config.teacher.temperature_first,
                "temperature_retry": config.teacher.temperature_retry,
            },
            "expander": config.expander.model_id,
            "verification": {
                "status": verification.status,
                "reward": verification.reward,
                "failure_codes": list(verification.failure_codes),
                "evidence": list(verification.evidence),
            },
            "oracle_version": task_package.oracle.oracle_version,
            "rubric_version": task_package.rubric_version,
            "reward_version": task_package.reward_semantics.reward_version,
            "environment_version": f"nutrienv-{config.nutrienv_rev[:7]}",
            "task_schema_version": task_package.schema_version,
            "catalog_sha": task_package.catalog.catalog_sha,
            "nutrienv_rev": config.nutrienv_rev,
            "nutrimind_rev": task_package.provenance.nutrimind_rev,
            "n_steps": len(episode.turns),
            "n_turns_without_plan": turns_without_plan,
            "plan_truncation": "token" if tokenizer is not None else "chars4",
        },
    }

    record_tokens = sum(_token_count(m["content"], tokenizer) for m in messages)
    if record_tokens > config.max_seq_tokens:
        raise SerializeError(
            "serialize.too_long", f"{record_tokens} > {config.max_seq_tokens}"
        )

    validate_record(record)
    return record
