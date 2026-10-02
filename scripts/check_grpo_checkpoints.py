"""Verify finite RL LoRA weights and updates across two trusted local checkpoints."""

import argparse
import json
from pathlib import Path

import torch


def adapters(path):
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    result = {key: tensor for key, tensor in state.items() if ".lora_" in key}
    for key, tensor in result.items():
        if hasattr(tensor, "to_local"):
            if tensor.device_mesh.size() != 1:
                raise RuntimeError("this verifier only supports single-rank checkpoints")
            result[key] = tensor.to_local()
    if not result or not any(".lora_B" in key for key in result):
        raise RuntimeError("checkpoint contains no RL LoRA B matrices")
    if not all(torch.isfinite(tensor).all().item() for tensor in result.values()):
        raise RuntimeError("non-finite RL adapter weights")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    first, second = adapters(args.first), adapters(args.second)
    if first.keys() != second.keys():
        raise RuntimeError("adapter tensor keys changed")
    b_nonzero = sum(torch.count_nonzero(t).item() for k, t in first.items() if ".lora_B" in k)
    changed = [k for k in first if not torch.equal(first[k], second[k])]
    if not b_nonzero or not changed:
        raise RuntimeError("RL adapter did not update")
    report = {"first": args.first, "second": args.second, "finite": True,
              "first_lora_b_nonzero_elements": b_nonzero,
              "adapter_tensors": len(first), "changed_tensors": len(changed),
              "max_absolute_update": max((first[k].float() - second[k].float()).abs().max().item()
                                         for k in changed)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
