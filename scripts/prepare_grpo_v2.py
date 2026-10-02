"""Export checkpoint-probed TaskPackages and disjoint SFT holdouts for veRL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.data_factory.concepts import TaskPackage
from src.training.data_factory.export_rlvr import export_rlvr


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(sources):
    packages = {}
    for source in sources:
        source = Path(source)
        for line in source.read_text().splitlines():
            row = json.loads(line)
            package = TaskPackage.from_dict(json.loads(
                (source.parent.parent / row["task_package_ref"]).read_text()))
            if package.task_id in packages:
                raise ValueError(f"duplicate task: {package.task_id}")
            packages[package.task_id] = package
    return packages


def task_identity(package):
    """Task IDs/provenance are not evidence of different learning problems."""
    payload = {"query": package.query, "s0": package.environment.s0,
               "oracle": package.oracle.payload}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def clean_holdouts(train, holdouts):
    trained = {task_identity(p) for p in train}
    seen = set()
    clean, overlap, duplicates = [], [], []
    for package in holdouts:
        identity = task_identity(package)
        if identity in trained:
            overlap.append(package.task_id)
        elif identity in seen:
            duplicates.append(package.task_id)
        else:
            seen.add(identity)
            clean.append(package)
    return clean, {"sft_seen_task_ids": overlap, "duplicate_holdout_task_ids": duplicates}


def training_row(package):
    exported = export_rlvr(package)
    return {
        "data_source": "nutrimind_v2",
        "agent_name": "nutrimind_v2",
        "prompt": [{"role": "system", "content": exported["prompt"]["system"]},
                   {"role": "user", "content": exported["prompt"]["task"]}],
        "ability": package.family,
        "reward_model": {"style": "rule", "ground_truth": package.task_id},
        "extra_info": {"task_id": package.task_id, "family": package.family,
                       "task_package_json": json.dumps(package.to_dict(), ensure_ascii=False)},
    }


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from nutrienv.bench.pipeline.freezer import catalog_digest
    from nutrienv.world.catalog_store import load_catalog

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--lab", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    checkpoint, probe, out = map(Path, (args.checkpoint, args.probe, args.out))
    if out.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {out}")
    manifest = json.loads((checkpoint / "run_manifest.json").read_text())
    selection = json.loads((probe / "grpo_candidates.json").read_text())
    probe_manifest = json.loads((probe / "manifest.json").read_text())
    adapter_sha = sha(checkpoint / "adapter_model.safetensors")
    if adapter_sha != selection["adapter_sha256"]:
        raise ValueError("candidate checkpoint checksum mismatch")
    rev = subprocess.check_output(["git", "-C", args.lab, "rev-parse", "HEAD"], text=True).strip()
    if rev != manifest["nutrienv_rev_installed"]:
        raise ValueError("environment does not match SFT")
    sources = manifest["data"]["train"]["path"]
    for source, expected in zip(sources, manifest["data"]["train"]["sha256"], strict=True):
        if sha(source) != expected:
            raise ValueError(f"SFT source changed: {source}")
    train_pool = collect(sources)
    val_sources = [str(Path(source).with_name("holdout.jsonl")) for source in sources]
    val_pool = collect(val_sources)
    if set(train_pool) & set(val_pool):
        raise ValueError("train/holdout task overlap")
    excluded = set(manifest["data"]["train"].get("refused_hand_in_task_ids", []))
    excluded.update(manifest["data"]["train"].get("over_max_length_task_ids", []))
    train = [train_pool[tid] for tid in selection["mixed_task_ids"]
             if train_pool[tid].family in {"composite", "evaluate", "recommend"}]
    if any(p.task_id in excluded for p in train):
        raise ValueError("selected an excluded SFT task")
    catalog_sha = catalog_digest(load_catalog())
    for package in train + list(val_pool.values()):
        if package.catalog.nutrienv_rev != rev or package.catalog.catalog_sha != catalog_sha:
            raise ValueError(f"environment/catalog mismatch: {package.task_id}")
    for package in train:
        actual = hashlib.sha256(json.dumps(package.to_dict(), sort_keys=True).encode()).hexdigest()
        if actual != probe_manifest["package_sha256"][package.task_id]:
            raise ValueError(f"probed task changed: {package.task_id}")
    # Prefer short, genuinely mixed tasks for hardware smoke; full train remains unchanged.
    report = json.loads((probe / "report.json").read_text())
    rates = {t["task_id"]: t["pass_rate_valid"] for t in report["tasks"]}
    smoke = sorted(train, key=lambda p: (p.family != "evaluate", abs(rates[p.task_id] - .5), p.task_id))[:4]
    val, holdout_audit = clean_holdouts(
        [p for tid, p in train_pool.items() if tid not in excluded], val_pool.values())
    val = sorted(val, key=lambda p: (p.family, p.task_id))
    out.mkdir(parents=True)
    artifacts = {}
    for name, packages in (("train", train), ("val", val), ("smoke_train", smoke), ("smoke_val", val[:2])):
        target = out / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist([training_row(p) for p in packages]), target)
        artifacts[name] = {"tasks": len(packages), "sha256": sha(target), "ids": [p.task_id for p in packages]}
    metadata = {"schema": "nutrimind-verl-v2/1", "adapter_sha256": adapter_sha,
                "nutrienv_rev": rev, "catalog_sha256": catalog_sha,
                "selection_sha256": sha(probe / "grpo_candidates.json"),
                "holdout_audit": holdout_audit,
                "sft_train_problem_sha256": sorted({task_identity(p) for tid, p in train_pool.items() if tid not in excluded}),
                "validation_problem_sha256": [task_identity(p) for p in val],
                "holdout_sources": [{"path": p, "sha256": sha(p)} for p in val_sources],
                "reward": {"pass": 1, "fail": 0, "infrastructure_invalid": -999},
                "artifacts": artifacts}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v["tasks"] for k, v in artifacts.items()}))


if __name__ == "__main__":
    main()
