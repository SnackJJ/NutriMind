"""v2 SFT entry point: Qwen3.5-2B + LoRA on TRL ``SFTTrainer`` (ADR-015).

    python -m src.training.sft.train_v2 --config configs/sft_v2_lora.yaml [--dry-run]

Records are v2 native-FC SFT records (ADR-014). Each is rendered with the
student chat template + lab ``NUTRIENV_TOOLS`` through ``tokenize_v2_record``
(labels from ``train_on``: assistant turns only). Every record must pass the
train/eval identity check against ``tokenize_prompt`` before anything trains.
Records longer than ``data.max_length`` are dropped and counted (ADR-011:
full-log, never slid).

``--dry-run`` is CPU-only and loads no weights: data + tokenizer, render,
token-length distribution, over-length count, masked-token fraction, identity.
The Unsloth v1 path (``train.py``) is not used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import statistics
import subprocess
import types

import yaml

from src.training.rl.prompt import as_ids, prompt_for_package, tokenize_prompt
from src.training.sft.v2_loader import tokenize_v2_record

__all__ = ["encode_records", "for_template", "prompt_identity", "main"]

_TASK_PREFIX = "Task:\n"


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def for_template(record: dict) -> dict:
    """Copy with tool-call ``arguments`` as mappings.

    Records keep the OpenAI JSON string; vLLM parses it before rendering, and
    the Qwen3.5 template iterates ``arguments|items``. Training renders the same.
    """
    record = copy.deepcopy(record)
    for message in record["messages"]:
        for call in message.get("tool_calls") or []:
            func = call.get("function") or {}
            if isinstance(func.get("arguments"), str):
                func["arguments"] = json.loads(func["arguments"])
    return record


def prompt_identity(record: dict, tokenizer, train_ids: list) -> dict:
    """Train/eval prompt identity for one record.

    ``context``: the system + tools + Task render of ``tokenize_prompt`` (for
    this record's query) is an id-identical prefix of the training ids.
    ``generation_prompt``: all of ``tokenize_prompt`` (with its assistant
    header) is a prefix too. ``generation_prompt_thinking``: same with
    ``enable_thinking=True`` (the header a with-reasoning eval must send).
    """
    task = record["messages"][1].get("content") or ""
    if not task.startswith(_TASK_PREFIX):
        return {"context": False, "generation_prompt": False,
                "generation_prompt_thinking": False}
    payload = prompt_for_package(types.SimpleNamespace(query=task[len(_TASK_PREFIX):]))
    messages = [
        {"role": "system", "content": payload["system"]},
        {"role": "user", "content": payload["task"]},
    ]
    context = as_ids(tokenizer.apply_chat_template(
        messages, tools=payload["tools"], tokenize=True, add_generation_prompt=False
    ))
    eval_ids = tokenize_prompt(payload, tokenizer)
    thinking = as_ids(tokenizer.apply_chat_template(
        messages, tools=payload["tools"], tokenize=True,
        add_generation_prompt=True, enable_thinking=True,
    ))

    def is_prefix(ids):
        return train_ids[: len(ids)] == ids

    return {
        "context": is_prefix(context) and eval_ids[: len(context)] == context,
        "generation_prompt": is_prefix(eval_ids),
        "generation_prompt_thinking": is_prefix(thinking),
    }


def _distribution(values: list) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(q):
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {"min": ordered[0], "p50": pct(0.5), "p90": pct(0.9), "p95": pct(0.95),
            "max": ordered[-1], "mean": round(statistics.fmean(ordered), 1),
            "sum": sum(ordered)}


def encode_records(records: list[dict], tokenizer, *, max_length: int) -> tuple[list, dict]:
    """(examples, stats). Examples carry ``input_ids`` + ``labels`` only."""
    examples, lengths, over = [], [], []
    trained = total = 0
    identity = {"context": 0, "generation_prompt": 0, "generation_prompt_thinking": 0}
    for record in records:
        out = tokenize_v2_record(for_template(record), tokenizer)
        ids, labels = out["input_ids"], out["labels"]
        for key, ok in prompt_identity(record, tokenizer, ids).items():
            identity[key] += int(ok)
        lengths.append(len(ids))
        if len(ids) > max_length:
            over.append(record.get("task_id"))
            continue
        trained += sum(1 for label in labels if label != -100)
        total += len(ids)
        examples.append({"input_ids": ids, "labels": labels})
    stats = {
        "n_records": len(records),
        "n_kept": len(examples),
        "n_over_max_length": len(over),
        "over_max_length_task_ids": over,
        "max_length": max_length,
        "tokens": _distribution(lengths),
        "trained_token_fraction": round(trained / total, 4) if total else 0.0,
        "masked_token_fraction": round(1 - trained / total, 4) if total else 0.0,
        "identity": identity,
    }
    return examples, stats


def _sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _git(*args, cwd=None) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _installed_nutrienv_rev() -> str | None:
    import nutrienv

    return _git("rev-parse", "HEAD", cwd=pathlib.Path(nutrienv.__file__).resolve().parents[2])


def run_manifest(config_path, config: dict, splits: dict, records: dict) -> dict:
    record_revs = sorted({
        (record.get("meta") or {}).get("nutrienv_rev")
        for rows in records.values() for record in rows
    } - {None})
    return {
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "config": config,
        "data": {
            name: {"path": config["data"][name], "sha256": _sha256(config["data"][name]),
                   **splits[name]}
            for name in splits
        },
        "nutrimind_rev": _git("rev-parse", "HEAD"),
        "nutrimind_dirty": bool(_git("status", "--porcelain")),
        "nutrienv_rev_records": record_revs,
        "nutrienv_rev_installed": _installed_nutrienv_rev(),
    }


def _train(config: dict, tokenizer, train: list, loss_val: list, output_dir, *, merge: bool) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["id"],
        dtype=getattr(torch, config["model"]["dtype"]),
        attn_implementation=config["model"]["attn_implementation"],
    )
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(output_dir),
            max_length=config["data"]["max_length"],
            **config["sft"],
        ),
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(loss_val),
        processing_class=tokenizer,
        peft_config=LoraConfig(task_type="CAUSAL_LM", **config["lora"]),
    )
    metrics = {"train": trainer.train().metrics, "loss_val": trainer.evaluate()}
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    if merge:
        merged = pathlib.Path(output_dir) / "merged"
        trainer.model.merge_and_unload().save_pretrained(str(merged))
        tokenizer.save_pretrained(str(merged))
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/sft_v2_lora.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU only, no weights: render + stats + identity")
    parser.add_argument("--tokenizer", help="tokenizer path/id (default: model.id)")
    parser.add_argument("--merge", action="store_true",
                        help="also save a merged bf16 model under <output_dir>/merged")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    config = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or config["model"]["id"])
    max_length = config["data"]["max_length"]

    records, examples, splits = {}, {}, {}
    for name in ("train", "loss_val"):
        records[name] = read_jsonl(config["data"][name])
        examples[name], splits[name] = encode_records(
            records[name], tokenizer, max_length=max_length
        )
    report = run_manifest(args.config, config, splits, records)
    print(json.dumps({k: report[k] for k in ("data", "nutrienv_rev_records",
                                              "nutrienv_rev_installed")}, indent=2))

    broken = {name: s["n_records"] - s["identity"]["context"] for name, s in splits.items()}
    if any(broken.values()):
        raise SystemExit(f"train/eval prompt identity broken (records per split): {broken}")
    if not examples["train"]:
        raise SystemExit("no train records within max_length")
    if args.dry_run:
        return 0

    output_dir = pathlib.Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report["metrics"] = _train(
        config, tokenizer, examples["train"], examples["loss_val"], output_dir,
        merge=args.merge,
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
