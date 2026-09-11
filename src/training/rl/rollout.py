"""student_rollout seam (RL ticket 001 / spec D3).

Reuses the factory's lab FC loop (``rollout_tool_call``) and returns factory
``EpisodeResult``. Policy is injected; tests pass a scripted complete.
``parallel_tool_calls`` is false. No network without
``NUTRIMIND_ALLOW_NETWORK=1``.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

from nutrienv.bench import load_split
from nutrienv.world.catalog_store import load_catalog

from src.training.data_factory.concepts import EpisodeResult, TaskPackage
from src.training.data_factory.rollout_fc import rollout_tool_call

__all__ = ["student_rollout"]


def _task_from_package(task_package: TaskPackage, catalog):
    item = {
        "id": task_package.task_id,
        "family": task_package.family,
        "persona": "everyday",
        "situations": [],
        "query": task_package.query,
        "s0": task_package.environment.s0,
        "oracle": task_package.oracle.payload,
    }
    if task_package.tier:
        item["tier"] = task_package.tier
    with tempfile.TemporaryDirectory(prefix="nutrimind-rl-task-") as scratch_dir:
        scratch = pathlib.Path(scratch_dir) / "item.json"
        scratch.write_text(
            json.dumps({"items": [item]}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (task,) = load_split(scratch, catalog=catalog)
    return task


def student_rollout(
    policy_spec: dict,
    task_package: TaskPackage,
    *,
    k: int,
    seed: int,
) -> list[EpisodeResult]:
    """k independent FC episodes on one TaskPackage. ``seed`` is recorded for
    callers; the injected policy is the source of determinism in tests."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if policy_spec.get("parallel_tool_calls", False):
        raise ValueError("parallel_tool_calls must be false (ADR-014)")
    complete = policy_spec.get("complete")
    if complete is None:
        raise ValueError("policy_spec.complete is required (injected policy)")
    catalog = policy_spec.get("catalog") or load_catalog()
    task = _task_from_package(task_package, catalog)
    _ = seed
    episodes: list[EpisodeResult] = []
    for _ in range(k):
        episodes.append(
            rollout_tool_call(
                task,
                teacher_complete=complete,
                catalog=catalog,
                model=policy_spec.get("model", "scripted-student"),
            )
        )
    return episodes
