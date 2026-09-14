"""Shared train/eval prompt construction (RL ticket 002 / spec D2).

System prompt and ``tools`` are the lab objects. ``parallel_tool_calls`` is
false. Train (student_rollout) and eval import this function; they do not
keep a second copy.
"""

from __future__ import annotations

from nutrienv.harness.tools_schema import NUTRIENV_TOOLS, TOOL_SYSTEM_PROMPT

from src.training.data_factory.concepts import TaskPackage

__all__ = ["prompt_for_package", "tokenize_prompt"]


def prompt_for_package(task_package: TaskPackage) -> dict:
    """Byte-stable prompt payload for one TaskPackage."""
    return {
        "system": TOOL_SYSTEM_PROMPT,
        "tools": NUTRIENV_TOOLS,
        "task": f"Task:\n{task_package.query}",
        "parallel_tool_calls": False,
    }


def tokenize_prompt(payload: dict, tokenizer) -> list:
    """Tokenize the shared payload with the student chat template + tools."""
    messages = [
        {"role": "system", "content": payload["system"]},
        {"role": "user", "content": payload["task"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tools=payload["tools"],
        tokenize=True,
        add_generation_prompt=True,
    )
