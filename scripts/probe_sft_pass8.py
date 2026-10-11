"""Probe the checkpoint's actual SFT train tasks with independent FC episodes."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.data_factory.concepts import TaskPackage
from src.training.rl.eval_exam import VLLMChatClient, check_server, episode_row, sampling_preset
from src.training.rl.rollout import _task_from_package


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows, task_ids, runs):
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(row)
    tasks = []
    for tid in task_ids:
        group = groups[tid]
        passed = sum(r["status"] == "pass" for r in group)
        valid = sum(r["status"] != "indeterminate" for r in group)
        p = passed / valid if valid else None
        tasks.append({"task_id": tid, "family": group[0]["family"] if group else None,
                      "completed": len(group), "valid": valid, "passed": passed,
                      "pass_rate_valid": p, "pass_at_k_observed": passed > 0,
                      "reward_variance": p * (1-p) if p is not None else None,
                      "mixed_group": 0 < passed < valid,
                      "complete": len(group) == runs})
    complete = [t for t in tasks if t["complete"]]
    families = {}
    for family in sorted({t["family"] for t in complete}):
        subset = [t for t in complete if t["family"] == family]
        families[family] = {"tasks": len(subset),
                            "pass_at_k": sum(t["passed"] > 0 for t in subset)/len(subset),
                            "mixed_groups": sum(t["mixed_group"] for t in subset),
                            "all_fail": sum(t["valid"] == runs and t["passed"] == 0 for t in subset),
                            "all_pass": sum(t["passed"] == runs for t in subset)}
    return {"k": runs, "expected_episodes": len(task_ids)*runs,
            "completed_episodes": len(rows), "completed_groups": len(complete),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "pass_at_1_mean": sum(r["status"] == "pass" for r in rows)/len(rows) if rows else None,
            "pass_at_k": sum(t["passed"] > 0 for t in complete)/len(complete) if complete else None,
            "mixed_groups": sum(t["mixed_group"] for t in complete),
            "by_family": families, "tasks": tasks}


def finalize(out):
    manifest = json.loads((out/"manifest.json").read_text())
    latest = {}
    for line in (out/"episodes.jsonl").read_text().splitlines():
        row = json.loads(line)
        latest[(row["task_id"], row["run"])] = row
    report = summarize(list(latest.values()), manifest["task_ids"], manifest["runs"])
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    k = manifest["runs"]
    selection = {"checkpoint": manifest["checkpoint"],
                 "adapter_sha256": manifest["adapter_sha256"],
                 "k": k, "criterion": "k valid episodes, with both Pass and Fail",
                 "mixed_task_ids": [], "all_pass_task_ids": [],
                 "all_fail_task_ids": [], "incomplete_or_indeterminate_task_ids": []}
    for task in report["tasks"]:
        if not task["complete"] or task["valid"] != k:
            key = "incomplete_or_indeterminate_task_ids"
        elif task["mixed_group"]:
            key = "mixed_task_ids"
        elif task["passed"] == k:
            key = "all_pass_task_ids"
        else:
            key = "all_fail_task_ids"
        selection[key].append(task["task_id"])
    (out/"grpo_candidates.json").write_text(json.dumps(selection, indent=2)+"\n")
    lines = [f"# SFT train Pass@{k} probe", "",
             f"Checkpoint: `{manifest['checkpoint']}`", "",
             f"Completed episodes: {report['completed_episodes']}/{report['expected_episodes']}", "",
             f"Single-rollout Pass: {report['pass_at_1_mean']:.2%}", "",
             f"Observed Pass@{k}: {report['pass_at_k']:.2%}", "",
             "| Family | Tasks | Single-rollout Pass | Pass@k | Mixed | All-pass | All-fail |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for family, stats in report["by_family"].items():
        family_rows = [r for r in latest.values() if r["family"] == family]
        p = sum(r["status"] == "pass" for r in family_rows)/len(family_rows)
        lines.append(f"| {family} | {stats['tasks']} | {p:.2%} | {stats['pass_at_k']:.2%} | {stats['mixed_groups']} | {stats['all_pass']} | {stats['all_fail']} |")
    lines.extend(["", "These are in-training-task diagnostics, not held-out benchmark results.",
                  "Eight failures do not prove zero underlying success probability.", ""])
    (out/"report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--lab", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3.5-2B")
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        finalize(Path(args.out_dir))
        return
    if args.runs < 1 or args.workers < 1:
        parser.error("runs and workers must be positive")
    os.environ["NUTRIMIND_ALLOW_NETWORK"] = "1"
    check_server(args.base_url, args.model)
    checkpoint = Path(args.checkpoint).resolve()
    training = json.loads((checkpoint/"run_manifest.json").read_text())
    lab_rev = subprocess.check_output(["git", "-C", args.lab, "rev-parse", "HEAD"], text=True).strip()
    if lab_rev != training["nutrienv_rev_installed"]:
        raise RuntimeError("lab revision differs from the SFT training environment")
    excluded = set(training["data"]["train"].get("refused_hand_in_task_ids", []))
    excluded.update(training["data"]["train"].get("over_max_length_task_ids", []))
    packages = {}
    sources = []
    for source, expected in zip(training["data"]["train"]["path"], training["data"]["train"]["sha256"], strict=True):
        path = Path(source)
        if digest(path) != expected:
            raise RuntimeError(f"training source checksum mismatch: {path}")
        sources.append({"path": str(path), "sha256": expected})
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["task_id"] in excluded:
                continue
            package_path = path.parent.parent/row["task_package_ref"]
            package = TaskPackage.from_dict(json.loads(package_path.read_text()))
            if package.catalog.nutrienv_rev != lab_rev:
                raise RuntimeError(f"task revision mismatch: {package.task_id}")
            packages[package.task_id] = package
    if args.limit:
        # A small smoke covers several task families.
        ordered = sorted(packages.values(), key=lambda p: (p.family, p.task_id))
        first = {}
        for package in ordered:
            first.setdefault(package.family, package)
        chosen = list(first.values())[:args.limit]
        packages = {p.task_id: p for p in chosen}
    from nutrienv.world.catalog_store import load_catalog
    from nutrienv.bench.pipeline.freezer import catalog_digest
    catalog = load_catalog()
    if any(p.catalog.catalog_sha != catalog_digest(catalog) for p in packages.values()):
        raise RuntimeError("catalog digest mismatch")
    tasks = {tid: _task_from_package(p, catalog) for tid, p in packages.items()}
    sampling = sampling_preset(True)
    manifest = {"kind": "sft-train-pass-k-probe", "checkpoint": str(checkpoint),
                "adapter_sha256": digest(checkpoint/"adapter_model.safetensors"),
                "lab_rev": lab_rev, "sources": sources, "excluded": sorted(excluded),
                "task_ids": list(tasks), "package_sha256": {
                    tid: hashlib.sha256(json.dumps(p.to_dict(), sort_keys=True).encode()).hexdigest()
                    for tid, p in packages.items()},
                "runs": args.runs, "sampling": sampling, "model": args.model,
                "base_url": args.base_url, "workers": args.workers}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("existing probe manifest differs")
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    episode_path = out/"episodes.jsonl"
    latest = {}
    if episode_path.exists():
        for line in episode_path.read_text().splitlines():
            row = json.loads(line)
            latest[(row["task_id"], row["run"])] = row
    done = {key for key, row in latest.items() if row.get("error_kind") != "transport"}
    jobs = [(tid, run) for tid in tasks for run in range(args.runs) if (tid, run) not in done]
    os.environ["NUTRIMIND_ALLOW_NETWORK"] = "1"

    def execute(tid, run):
        seed = int.from_bytes(hashlib.sha256(f"{tid}:{run}".encode()).digest()[:4], "big") & 0x7fffffff
        client = VLLMChatClient(base_url=args.base_url, model=args.model, sampling=sampling, seed=seed)
        row = episode_row(tasks[tid], client, catalog=catalog, model=args.model, run=run)
        row["seed"] = seed
        row["reward"] = 1 if row["status"] == "pass" else (0 if row["status"] == "fail" else None)
        return row

    print(f"START tasks={len(tasks)} runs={args.runs} remaining={len(jobs)}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, tid, run): (tid, run) for tid, run in jobs}
        with episode_path.open("a") as handle:
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                handle.write(json.dumps(row)+"\n")
                handle.flush()
                os.fsync(handle.fileno())
                latest[(row["task_id"], row["run"])] = row
                report = summarize(list(latest.values()), list(tasks), args.runs)
                temp = out/"report.json.tmp"
                temp.write_text(json.dumps(report, indent=2)+"\n")
                temp.replace(out/"report.json")
                print(f"PROGRESS {len(latest)}/{len(tasks)*args.runs} groups={report['completed_groups']} mixed={report['mixed_groups']} statuses={report['status_counts']}", flush=True)
    finalize(out)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
