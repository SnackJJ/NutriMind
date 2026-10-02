"""serialize — Seam 4: the pure v2 SFT record writer (spec §9.2 / ticket 026).

``serialize`` writes ADR-014 native tool calling: ``system`` is
``TOOL_SYSTEM_PROMPT``, assistant turns carry ``tool_calls`` plus truncated
``reasoning_content``, observations are ``role=tool`` keyed by
``tool_call_id``. Ticket 008's text-op serializer stays CLOSED; this is the
production path.

No token-level ``loss_mask`` is stored — the v2 loader (ticket 028) derives it
from ``train_on``. ``validate_record`` is the loader-mirror structural check.

Stage module: imports nutrienv at module level (allowed by spec §18).
"""

from __future__ import annotations

import json

from nutrienv.harness.runner import FINISH_OPS
from nutrienv.harness.tools_schema import TOOL_SYSTEM_PROMPT

from src.training.data_factory.concepts import (
    EpisodeResult,
    TaskPackage,
    TurnMeta,
    VerificationResult,
)

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


def _tool_name(turn: TurnMeta) -> str | None:
    calls = turn.tool_calls or []
    if not calls:
        return None
    func = calls[0].get("function") or {}
    name = func.get("name")
    return name if isinstance(name, str) else None


def _tool_call_id(turn: TurnMeta) -> str | None:
    if turn.tool_call_id:
        return turn.tool_call_id
    calls = turn.tool_calls or []
    if calls and isinstance(calls[0].get("id"), str):
        return calls[0]["id"]
    return None


def _is_text_op_blob(content: str | None) -> bool:
    if not content:
        return False
    stripped = content.strip()
    if '{"op"' in stripped or "\n{\"op\"" in stripped:
        return True
    return stripped.startswith("{") and '"op"' in stripped


def _message_text(message: dict) -> str:
    parts: list[str] = []
    if message.get("content"):
        parts.append(str(message["content"]))
    if message.get("reasoning_content"):
        parts.append(str(message["reasoning_content"]))
    if message.get("tool_calls"):
        parts.append(json.dumps(message["tool_calls"], ensure_ascii=False))
    return "\n".join(parts)


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
    if messages[-1]["role"] != "assistant" or segments[-1] != "final":
        raise SerializeError("serialize.last_turn_not_finish")
    last_calls = messages[-1].get("tool_calls") or []
    if not last_calls:
        raise SerializeError("serialize.last_turn_not_finish")
    last_name = (last_calls[0].get("function") or {}).get("name")
    if last_name not in FINISH_OPS and last_name != "submit_plan":
        raise SerializeError("serialize.last_turn_not_finish")
    for index, (segment, flag) in enumerate(zip(segments, train_on)):
        if flag != (segment in ("step", "final")):
            raise SerializeError("serialize.turn_count_mismatch")
        if segment == "tool" and messages[index].get("role") != "tool":
            raise SerializeError("serialize.turn_count_mismatch")
        if segment in ("step", "final") and messages[index].get("role") != "assistant":
            raise SerializeError("serialize.turn_count_mismatch")
    for message in messages:
        if message["role"] != "assistant":
            continue
        if not message.get("tool_calls"):
            raise SerializeError("serialize.no_tool_calls")
        if _is_text_op_blob(message.get("content")):
            raise SerializeError("serialize.no_tool_calls", "text-op blob")
        blob = _message_text(message)
        if any(marker in blob for marker in _V1_MARKERS):
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
    last_name = _tool_name(episode.turns[-1])
    if not episode.reached_finish or (
        last_name not in FINISH_OPS and last_name != "submit_plan"
    ):
        raise SerializeError("serialize.last_turn_not_finish")
    for index, turn in enumerate(episode.turns):
        if not turn.tool_calls:
            raise SerializeError("serialize.no_tool_calls", f"turn {index}")
        if _is_text_op_blob(turn.content) and not turn.tool_calls:
            raise SerializeError("serialize.no_tool_calls", f"turn {index} text-op")

    messages: list[dict] = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
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
    last_index = len(episode.turns) - 1
    for index, turn in enumerate(episode.turns):
        plan = _sanitize_plan(turn.reasoning_content or "")
        if not plan:
            turns_without_plan += 1
        is_final = index == last_index
        budget = config.plan_max_tokens
        if is_final and getattr(config, "final_plan_max_tokens", None):
            budget = config.final_plan_max_tokens  # the hand-in: carries the verdict
        plan = _truncate_plan(plan, plan_max_tokens=budget, tokenizer=tokenizer)
        messages.append({
            "role": "assistant",
            "content": None,
            "reasoning_content": plan or None,
            "tool_calls": list(turn.tool_calls),
        })
        segments.append("final" if is_final else "step")
        train_on.append(True)
        if is_final:
            continue
        observation = turn.observation
        if not observation:
            raise SerializeError(
                "serialize.missing_observation", f"turn {index}"
            )
        call_id = _tool_call_id(turn)
        if not call_id:
            raise SerializeError("serialize.no_tool_calls", f"turn {index} missing id")
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": observation,
        })
        segments.append("tool")
        train_on.append(False)

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
            "plan_max_tokens": config.plan_max_tokens,
            "final_plan_max_tokens": (getattr(config, "final_plan_max_tokens", None)
                                      or config.plan_max_tokens),
        },
    }

    record_tokens = sum(_token_count(_message_text(m), tokenizer) for m in messages)
    if record_tokens > config.max_seq_tokens:
        raise SerializeError(
            "serialize.too_long", f"{record_tokens} > {config.max_seq_tokens}"
        )

    validate_record(record)
    return record
