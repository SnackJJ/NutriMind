"""Merge an RL LoRA from a veRL FSDP checkpoint into the SFT merged model.

``python scripts/export_grpo_lora.py --actor <global_step_N/actor> --base <sft merged dir> --out <dir>``

The checkpoint's frozen base weights must equal the SFT model's, which also
checks the key mapping before any merged weight is written.
"""

import argparse
import json
from pathlib import Path
import shutil

from safetensors.torch import load_file, save_file
import torch

PREFIX = "base_model.model."


def local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    meta = json.loads((args.actor / "lora_train_meta.json").read_text())
    scale = meta["lora_alpha"] / meta["r"]
    state = torch.load(args.actor / "model_world_size_1_rank_0.pt", map_location="cpu", weights_only=False)
    weights = load_file(args.base / "model.safetensors")

    modules = sorted({k[len(PREFIX):-len(".lora_A.default.weight")] for k in state if k.endswith(".lora_A.default.weight")})
    max_delta = 0.0
    for module in modules:
        # The SFT safetensors keep Qwen3.5's multimodal naming (model.language_model.*).
        key = f"{module}.weight".replace("model.", "model.language_model.", 1)
        frozen = local(state[f"{PREFIX}{module}.base_layer.weight"])
        if key not in weights or not torch.equal(weights[key], frozen):
            raise RuntimeError(f"checkpoint base weight does not match SFT model: {key}")
        a = local(state[f"{PREFIX}{module}.lora_A.default.weight"]).float()
        b = local(state[f"{PREFIX}{module}.lora_B.default.weight"]).float()
        delta = scale * (b @ a)
        max_delta = max(max_delta, delta.abs().max().item())
        weights[key] = (weights[key].float() + delta).to(weights[key].dtype)
    if max_delta == 0:
        raise RuntimeError("LoRA delta is zero; the checkpoint has no RL update")

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(weights, args.out / "model.safetensors", metadata={"format": "pt"})
    for path in args.base.iterdir():
        if path.name != "model.safetensors":
            shutil.copy2(path, args.out / path.name)
    print(json.dumps({"modules": len(modules), "scale": scale, "max_abs_delta": max_delta, "out": str(args.out)}))


if __name__ == "__main__":
    main()
