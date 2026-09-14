"""RLVR export — TaskPackage → ``rlvr/<task_id>.json`` (spec §9.3 / ticket 027).

Prompt is native FC: ``TOOL_SYSTEM_PROMPT`` + ``NUTRIENV_TOOLS``. No teacher,
no messages. Input is a TaskPackage, never an SFT record.
"""

from __future__ import annotations

import json
import os
import pathlib

from nutrienv.harness.tools_schema import NUTRIENV_TOOLS, TOOL_SYSTEM_PROMPT

from src.training.data_factory.concepts import TaskPackage

__all__ = ["RLVR_SCHEMA_VERSION", "export_rlvr", "write_rlvr"]

RLVR_SCHEMA_VERSION = "nutrimind-v2-rlvr/1"


def export_rlvr(package: TaskPackage) -> dict:
    """Pure projection of a TaskPackage to the §9.3 RLVR export object."""
    if not isinstance(package, TaskPackage):
        raise TypeError(
            "export_rlvr takes a TaskPackage, never an SFT record "
            f"(got {type(package).__name__})"
        )
    return {
        "schema_version": RLVR_SCHEMA_VERSION,
        "task_id": package.task_id,
        "task_package_ref": f"task_packages/{package.task_id}.json",
        "prompt": {
            "system": TOOL_SYSTEM_PROMPT,
            "tools": NUTRIENV_TOOLS,
            "task": f"Task:\n{package.query}",
        },
        "environment": package.environment.to_dict(),
        "verifier": {
            "kind": package.verifier.kind,
            "oracle": package.oracle.payload,
            "oracle_version": package.oracle.oracle_version,
        },
        "reward": {
            "adapter": package.reward_semantics.kind,
            "reward_version": package.reward_semantics.reward_version,
            "map": dict(package.reward_semantics.map),
        },
        "termination": package.termination.to_dict(),
        "seed": package.seed,
        "meta": {
            "family": package.family,
            "steps": list(package.steps),
            "tier": package.tier,
            "catalog_sha": package.catalog.catalog_sha,
            "nutrienv_rev": package.catalog.nutrienv_rev,
            "nutrimind_rev": package.provenance.nutrimind_rev,
            "oracle_version": package.oracle.oracle_version,
            "rubric_version": package.rubric_version,
            "reward_version": package.reward_semantics.reward_version,
            "task_schema_version": package.schema_version,
        },
    }


def write_rlvr(export: dict, output_dir) -> pathlib.Path:
    """Atomic write of ``rlvr/<task_id>.json``."""
    target = pathlib.Path(output_dir) / f"{export['task_id']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(export, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(blob, encoding="utf-8")
    os.replace(tmp, target)
    return target
