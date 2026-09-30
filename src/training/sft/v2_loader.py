"""v2 SFT loader for native-FC records (spec §9.2 / ticket 028).

Applies the student chat template with the lab ``NUTRIENV_TOOLS`` schema.
``train_on`` drives the token mask. Retired text-op and v1 XML records are
hard-rejected. The v1 loader in ``train.py`` is not imported here.
"""

from __future__ import annotations

import json

from nutrienv.harness.tools_schema import NUTRIENV_TOOLS

from src.training.data_factory.serialize import SerializeError, validate_record

__all__ = ["LoadError", "tokenize_v2_record"]

_V1_MARKERS = ("<think>", "</think>", "<tool_call>", "</tool_call>",
               "<|im_start|>", "<|im_end|>")


class LoadError(ValueError):
    """A v2 SFT record is structurally invalid or a retired protocol."""


def _is_text_op_blob(content: str | None) -> bool:
    if not content:
        return False
    stripped = content.strip()
    if '{"op"' in stripped or '\n{"op"' in stripped:
        return True
    return stripped.startswith("{") and '"op"' in stripped


def _reject_retired(record: dict) -> None:
    for message in record.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        has_calls = bool(message.get("tool_calls"))
        if _is_text_op_blob(content) and not has_calls:
            raise LoadError(
                "retired text-op record (plan\\n{\"op\":…} without tool_calls)"
            )
        blob = str(content or "")
        if any(marker in blob for marker in _V1_MARKERS):
            raise LoadError(
                "v1 XML record (<tool_call>/<think>/<|im_start|> as action channel)"
            )


def _ids(rendered) -> list:
    if not isinstance(rendered, list):
        raise LoadError("tokenizer.apply_chat_template(..., tokenize=True) must return ids")
    return rendered


def tokenize_v2_record(record: dict, tokenizer, *, tools=None) -> dict:
    """Tokenize one §9.2 FC record. Labels follow ``train_on``.

    ``tools`` defaults to the lab schema (same object eval declares).
    """
    if not isinstance(record, dict):
        raise LoadError(f"expected a mapping, got {type(record).__name__}")
    _reject_retired(record)
    try:
        validate_record(record)
    except SerializeError as exc:
        raise LoadError(str(exc)) from exc

    schema = NUTRIENV_TOOLS if tools is None else tools
    messages = record["messages"]
    train_on = record["train_on"]
    full = tokenizer.apply_chat_template(
        messages, tools=schema, tokenize=True, add_generation_prompt=False
    )
    if not isinstance(full, list):
        raise LoadError("tokenizer.apply_chat_template(..., tokenize=True) must return ids")
    labels = [-100] * len(full)
    # Only trained turns are rendered as prefixes. Each prefix keeps the system
    # + Task turns (validate_record: train_on[0:2] is False), which templates
    # such as Qwen3.5's require ("No messages" / "No user query" otherwise).
    for index, flag in enumerate(train_on):
        if not flag:
            continue
        start = len(_ids(tokenizer.apply_chat_template(
            messages[:index], tools=schema, tokenize=True, add_generation_prompt=False
        )))
        end = len(_ids(tokenizer.apply_chat_template(
            messages[: index + 1], tools=schema, tokenize=True, add_generation_prompt=False
        )))
        if start < end:
            labels[start:end] = full[start:end]
    if len(full) != len(labels):
        raise LoadError("input_ids / labels length mismatch")
    return {"input_ids": full, "labels": labels, "tools": schema}
