"""One-task mock of the v2 exam entry. Does not load a model and does not call vLLM.

The GPU command is ``scripts/run_exam_baseline.sh`` (zero-shot Qwen3.5-2B,
native FC, ``RUNS=3``). This process only checks that ``episode_row`` can
score one frozen exam task when the completion is injected.

Requires ``PYTHONPATH`` to point at a nutri-env checkout whose git HEAD is
the ADR-012 pin (``configs/data_factory.yaml`` ``nutrienv.rev``). The pin
commit's ``harness/__init__.py`` imports ``buddy``, which that commit does
not contain; the pilot worktree carries a two-name stub for that import.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "student" / "pilot-20260929" / "exam_mock.json"


def main() -> int:
    from nutrienv.bench import EXAM_SPLIT_PATH, load_split
    from nutrienv.world.catalog_store import load_catalog

    from src.training.rl.eval_exam import episode_row
    from src.training.rl.exam_gate import assert_exam_byte_identical, assert_lab_at_rev

    exam_path = Path(EXAM_SPLIT_PATH)
    head = assert_lab_at_rev()
    assert_exam_byte_identical(exam_path)
    tasks = load_split(exam_path)
    if len(tasks) != 63:
        raise SystemExit(f"expected 63 exam tasks, got {len(tasks)} at {exam_path}")
    task = tasks[0]

    def complete(_request: dict) -> dict:
        return {
            "content": None,
            "reasoning_content": "mock finish; no model weights loaded",
            "tool_calls": [
                {
                    "id": "call_mock_finish",
                    "type": "function",
                    "function": {"name": "finish", "arguments": "{}"},
                }
            ],
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0},
        }

    row = episode_row(task, complete, catalog=load_catalog(), model="mock-no-weights", run=0)
    payload = {
        "exam_path": str(exam_path),
        "n_exam_tasks": len(tasks),
        "lab_head": head,
        "mocked_task_id": row["task_id"],
        "family": row["family"],
        "run": row["run"],
        "status": row["status"],
        "execution": row["execution"],
        "finished": row["finished"],
        "error": row["error"],
        "n_steps": row["n_steps"],
        "model_loaded": False,
        "production_command": (
            "NUTRIENV_SRC=/tmp/nutri-env-0ee68ea/src "
            "bash scripts/run_exam_baseline.sh"
        ),
        "production_note": (
            "scripts/run_exam_baseline.sh serves Qwen/Qwen3.5-2B with vLLM and "
            "calls eval_exam --runs 3. It requires a GPU and does load the model. "
            "This mock does not invoke that script."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if row["error"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    raise SystemExit(main())
