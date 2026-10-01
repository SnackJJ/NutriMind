"""v2 SFT entry point: Qwen3.5-2B + LoRA on TRL ``SFTTrainer`` (ADR-015).

    python -m src.training.sft.train_v2 --config configs/sft_v2_lora.yaml [--dry-run]

Records are v2 native-FC SFT records (ADR-014). Each is first replayed through
the lab FC loop (``eval_context``) so its context is what the official eval
sends, then rendered with the student chat template + lab ``NUTRIENV_TOOLS``
through ``tokenize_v2_record`` (labels from ``train_on``: assistant turns
only). Every record must pass the train/eval identity check (every turn's
request is a token prefix) before anything trains.
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

from src.training.data_factory.concepts import TaskPackage
from src.training.data_factory.rollout_fc import rollout_tool_call
from src.training.rl.prompt import as_ids, prompt_for_package, tokenize_prompt
from src.training.rl.rollout import _task_from_package
from src.training.sft.v2_loader import tokenize_v2_record

__all__ = ["ReplayError", "encode_records", "eval_context", "for_template", "load_tasks",
           "prompt_identity", "main"]

_TASK_PREFIX = "Task:\n"
_OBSERVATION_SEP = "\nObservation:\n"   # nutrienv.harness.tool_call._observation_turn


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


class ReplayError(ValueError):
    """The record's last turn does not end the episode in the lab loop."""


def eval_context(record: dict, task, reset_observation: str, catalog) -> dict:
    """``record`` as the lab FC loop (the official eval) presents it.

    The data factory stores every tool call the teacher emitted and bare
    observations; the loop executes only the first call per turn, keeps only
    that call in the history, opens with a ``Step budget`` + reset-observation
    user turn and wraps each tool reply the same way. Replaying the recorded
    turns through the loop (``rollout_tool_call``) and keeping the messages it
    builds makes training see exactly those requests.

    Observation text stays the live episode's (``reset_observation`` from the
    rollout cache, tool replies from the record): the task package was written
    with ``sort_keys``, so the replay's dict order differs from the official
    split's. Each replayed observation must equal the live one as JSON.
    """
    turns = [m for m in record["messages"] if m["role"] == "assistant"]
    live = [reset_observation] + [m["content"] for m in record["messages"]
                                  if m["role"] == "tool"]
    requests = []

    def replay(request):
        requests.append(copy.deepcopy(request["messages"]))
        turn = turns[len(requests) - 1]
        return {"content": turn.get("content"),
                "reasoning_content": turn.get("reasoning_content"),
                "tool_calls": turn["tool_calls"][:1]}

    episode = rollout_tool_call(task, teacher_complete=replay, catalog=catalog)
    task_id = record.get("task_id")
    if len(requests) < len(turns) or (episode.error and len(requests) == len(turns)):
        raise SystemExit(f"{task_id}: replay ran {len(requests)}/{len(turns)} turns "
                         f"({episode.error})")
    final = turns[-1]
    messages = requests[len(turns) - 1] + [{"role": "assistant", "content": final.get("content"),
                                "reasoning_content": final.get("reasoning_content"),
                                "tool_calls": final["tool_calls"][:1]}]
    observed = [m for m in messages[2:] if m["role"] in ("user", "tool")]
    if len(observed) != len(live):
        raise SystemExit(f"{task_id}: replay has {len(observed)} observations, "
                         f"live episode {len(live)}")
    for message, text in zip(observed, live):
        head, sep, body = message["content"].partition(_OBSERVATION_SEP)
        if not sep or json.loads(body) != json.loads(text):
            raise SystemExit(f"{task_id}: replayed observation differs from the live one")
        message["content"] = head + sep + text
    if len(requests) > len(turns):
        # rollout_fc._build_turns stops at the first submit_plan even when Env
        # refuses it; the loop went on, so the record ends on a refused hand-in.
        raise ReplayError(f"{task_id}: final hand-in refused by Env")

    segments = []
    for index, message in enumerate(messages):
        role = message["role"]
        if index < 2:
            segments.append(("system", "task")[index])
        elif role == "assistant":
            segments.append("final" if index == len(messages) - 1 else "step")
        else:
            segments.append("tool" if role == "tool" else "observation")
    return {**record, "messages": messages, "segments": segments,
            "train_on": [s in ("step", "final") for s in segments]}


def load_tasks(records: list[dict], batch_dir, catalog) -> list[tuple]:
    """(Task, live reset observation) per record. ``task_package_ref`` is
    batch-relative; the reset observation is the accepted attempt's in
    ``rollouts/cache/<task_id>.json`` (``accepted_from_attempt`` is 1-based)."""
    batch_dir = pathlib.Path(batch_dir)
    loaded = []
    for record in records:
        package = TaskPackage.from_dict(json.loads(
            (batch_dir / record["task_package_ref"]).read_text(encoding="utf-8")))
        cache = json.loads((batch_dir / "rollouts" / "cache" / f"{record['task_id']}.json")
                           .read_text(encoding="utf-8"))
        episode = cache["attempts"][record["accepted_from_attempt"] - 1]["episode"]
        loaded.append((_task_from_package(package, catalog), episode["reset_observation"]))
    return loaded


def prompt_identity(record: dict, tokenizer, train_ids: list) -> dict:
    """Train/eval prompt identity for one record.

    ``context``: the system + tools + Task render of ``tokenize_prompt`` (for
    this record's query) is an id-identical prefix of the training ids.
    ``turns``: for every assistant turn, the messages before it rendered with
    the generation header are a prefix too (after ``eval_context`` those are
    the loop's requests). The header is the thinking one (``enable_thinking``,
    what a with-reasoning eval sends) for a turn with a plan, and the closed
    empty-think one for a turn without: there the record's empty think
    tokenizes its two newlines as one token, which the open thinking header
    (ending in one newline) splits.
    """
    task = record["messages"][1].get("content") or ""
    if not task.startswith(_TASK_PREFIX):
        return {"context": False, "turns": False}
    payload = prompt_for_package(types.SimpleNamespace(query=task[len(_TASK_PREFIX):]))
    messages = [
        {"role": "system", "content": payload["system"]},
        {"role": "user", "content": payload["task"]},
    ]
    context = as_ids(tokenizer.apply_chat_template(
        messages, tools=payload["tools"], tokenize=True, add_generation_prompt=False
    ))
    eval_ids = tokenize_prompt(payload, tokenizer)

    def is_prefix(ids):
        return train_ids[: len(ids)] == ids

    history = for_template(record)["messages"]
    turns = all(
        is_prefix(as_ids(tokenizer.apply_chat_template(
            history[:index], tools=payload["tools"], tokenize=True,
            add_generation_prompt=True, enable_thinking=bool(message.get("reasoning_content")),
        )))
        for index, message in enumerate(history) if message["role"] == "assistant"
    )
    return {"context": is_prefix(context) and eval_ids[: len(context)] == context,
            "turns": turns}


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
    identity = {"context": 0, "turns": 0}
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

    from nutrienv.world.catalog_store import load_catalog

    catalog = load_catalog()
    records, examples, splits = {}, {}, {}
    for name in ("train", "loss_val"):
        records[name] = read_jsonl(config["data"][name])
        # <batch>/sft/<split>.jsonl; task_package_ref is relative to <batch>
        batch_dir = pathlib.Path(config["data"][name]).parents[1]
        replayed, refused = [], []
        for record, (task, reset) in zip(records[name],
                                         load_tasks(records[name], batch_dir, catalog)):
            try:
                replayed.append(eval_context(record, task, reset, catalog))
            except ReplayError:
                refused.append(record.get("task_id"))
        examples[name], splits[name] = encode_records(replayed, tokenizer,
                                                      max_length=max_length)
        splits[name].update(n_records=len(records[name]),
                            n_refused_hand_in=len(refused),
                            refused_hand_in_task_ids=refused)
    report = run_manifest(args.config, config, splits, records)
    print(json.dumps({k: report[k] for k in ("data", "nutrienv_rev_records",
                                              "nutrienv_rev_installed")}, indent=2))

    broken = {name: s["n_records"] - s["n_refused_hand_in"]
                    - min(s["identity"]["context"], s["identity"]["turns"])
              for name, s in splits.items()}
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
