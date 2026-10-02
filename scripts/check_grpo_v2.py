"""Fail-closed preflight for the pinned, portable v2 GRPO stack."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess

MIMO_COMMIT = "a2ad9f6160b03ff2d47e59832bfb6b289f37c917"
ENV_COMMIT = "47367d9c569d0a46cbd1c97d5f08afb3a7d573ac"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(root, lab, mimo, dataset):
    from nutrienv.bench.pipeline.freezer import catalog_digest
    from nutrienv.world.catalog_store import load_catalog
    import torch
    from transformers import AutoConfig

    for repo, expected in ((lab, ENV_COMMIT), (mimo, MIMO_COMMIT)):
        actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise RuntimeError(f"revision mismatch: {repo}: {actual}")
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest["nutrienv_rev"] != ENV_COMMIT:
        raise RuntimeError("dataset environment revision mismatch")
    if manifest["catalog_sha256"] != catalog_digest(load_catalog()):
        raise RuntimeError("catalog changed")
    train_ids = set(manifest["artifacts"]["train"]["ids"])
    val_ids = set(manifest["artifacts"]["val"]["ids"])
    if train_ids & val_ids:
        raise RuntimeError("train/validation overlap")
    validation_identity = manifest.get("validation_problem_sha256")
    if validation_identity is None:
        raise RuntimeError("dataset needs query/state/oracle holdout audit; re-export it")
    if set(validation_identity) & set(manifest["sft_train_problem_sha256"]):
        raise RuntimeError("validation contains an SFT-seen problem")
    if len(set(validation_identity)) != len(validation_identity):
        raise RuntimeError("validation contains duplicate problems")
    for name, artifact in manifest["artifacts"].items():
        if digest(dataset / f"{name}.parquet") != artifact["sha256"]:
            raise RuntimeError(f"dataset checksum mismatch: {name}")
    checkpoint = root / "data/student/models/sft_v2_lora_b12"
    if digest(checkpoint / "adapter_model.safetensors") != manifest["adapter_sha256"]:
        raise RuntimeError("SFT checkpoint changed")
    merged = checkpoint / "merged"
    config = AutoConfig.from_pretrained(merged)
    if config.model_type not in {"qwen3_5", "qwen3_5_text"}:
        raise RuntimeError("expected Qwen3.5 SFT model")
    versions = {p: metadata.version(p) for p in
        ("torch", "vllm", "transformers", "verl", "ray", "datasets", "peft", "tensordict", "transferqueue",
         "nutrimind-vllm-compat", "flash-linear-attention")}
    for package, expected in {"torch": "2.11.0", "vllm": "0.24.0", "transformers": "5.9.0",
            "verl": "0.9.0.dev0", "datasets": "4.0.0", "transferqueue": "0.1.8",
            "flash-linear-attention": "0.5.2"}.items():
        if versions[package] != expected:
            raise RuntimeError(f"dependency drift: {package}={versions[package]}, expected {expected}")
    from transformers.models.qwen3_5 import modeling_qwen3_5
    if modeling_qwen3_5.chunk_gated_delta_rule is None:
        raise RuntimeError("fla kernels unavailable; Qwen3.5 training would use the torch GDN fallback")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("this baseline requires exactly one visible CUDA GPU")
    props = torch.cuda.get_device_properties(0)
    return {"mimo_commit": MIMO_COMMIT, "nutrienv_commit": ENV_COMMIT,
        "dataset_manifest_sha256": digest(dataset / "manifest.json"),
        "versions": versions,
        "source_sha256": {p: digest(root / p) for p in
            ("src/training/rl/verl_agent_loop.py", "src/training/rl/verl_trainer.py",
             "src/training/rl/train_verl.py", "scripts/run_grpo_v2.sh", "scripts/prepare_grpo_v2.py",
             "configs/grpo_v2_4090.yaml", "configs/grpo_v2_a800.yaml", "configs/grpo_v2_a800_fast.yaml",
             "configs/grpo_v2_agent.yaml",
             "infra/grpo/vllm_plugin/nutrimind_vllm_compat.py",
             "infra/grpo/vllm_plugin/nutrimind_vllm_compat_model.py")},
        "gpu": props.name, "gpu_memory_gib": round(props.total_memory / 2**30, 2),
        "cuda_runtime": torch.version.cuda, "train_tasks": len(train_ids), "val_tasks": len(val_ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", default="data/rl/grpo_v2")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    report = check(root, Path(os.environ.get("NUTRIENV_ROOT", "/root/nutri-env-pin")),
        Path(os.environ.get("MIMO_VERL_ROOT", "/root/autodl-tmp/mimo-verl")),
        root / args.dataset)
    print(json.dumps(report, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
