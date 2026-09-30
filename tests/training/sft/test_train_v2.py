"""v2 SFT entry point (B5): render, assistant-only labels, over-length
accounting, train/eval prompt identity. Offline: hand-built §9.2 records and a
Qwen3.5-shaped toy tokenizer; no weights, no downloads."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.harness.tools_schema import TOOL_SYSTEM_PROMPT  # noqa: E402

from src.training.sft import train_v2  # noqa: E402
from src.training.sft.train_v2 import encode_records, for_template, prompt_identity  # noqa: E402


class QwenLikeTokenizer:
    """Byte-level stand-in for the Qwen3.5 template's load-bearing behaviour:
    raises on no messages / no user turn, requires mapping ``arguments``,
    renders ``<think>`` on assistant turns, and the generation prompt closes
    an empty think unless ``enable_thinking=True``."""

    def apply_chat_template(self, messages, tools=None, tokenize=False,
                            add_generation_prompt=False, enable_thinking=None):
        if not messages:
            raise ValueError("No messages provided.")
        if not any(m.get("role") == "user" for m in messages):
            raise ValueError("No user query found in messages.")
        text = "<s>system\n" + "TOOLS" + json.dumps(tools, sort_keys=True)
        text += "\n" + messages[0]["content"] + "</s>\n"
        for message in messages[1:]:
            role = message["role"]
            if role == "user":
                text += "<s>user\n" + message["content"] + "</s>\n"
            elif role == "tool":
                text += "<s>user\n<tool_response>" + message["content"] + "</tool_response></s>\n"
            else:
                text += "<s>assistant\n<think>\n" + (message.get("reasoning_content") or "")
                text += "\n</think>\n\n"
                for call in message.get("tool_calls") or []:
                    args = call["function"]["arguments"]
                    if not isinstance(args, dict):
                        raise TypeError("Can only get item pairs from a mapping.")
                    text += "<tool_call>" + call["function"]["name"]
                    text += "".join(f"<{k}>{v}" for k, v in args.items()) + "</tool_call>"
                text += "</s>\n"
        if add_generation_prompt:
            text += "<s>assistant\n<think>\n" + ("" if enable_thinking else "\n</think>\n\n")
        return list(text.encode("utf-8")) if tokenize else text


def _call(name, args, call_id):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def make_record(task_id="t1", *, n_logs=1, observation="OBS" * 5, system=TOOL_SYSTEM_PROMPT):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Task:\nI had 150 g of oatmeal for lunch."}]
    segments, train_on = ["system", "task"], [False, False]
    for index in range(n_logs):
        call_id = f"call_{index}"
        messages += [
            {"role": "assistant", "content": None, "reasoning_content": "PLAN-log",
             "tool_calls": [_call("log_meal", {"food_id": "F1", "grams": 150}, call_id)]},
            {"role": "tool", "tool_call_id": call_id, "content": observation},
        ]
        segments += ["step", "tool"]
        train_on += [True, False]
    messages.append({"role": "assistant", "content": None, "reasoning_content": "PLAN-done",
                     "tool_calls": [_call("done", {}, "call_done")]})
    segments.append("final")
    train_on.append(True)
    return {"task_id": task_id, "messages": messages, "segments": segments,
            "train_on": train_on, "meta": {"nutrienv_rev": "0" * 40}}


def _labelled_text(example) -> str:
    return bytes(label for label in example["labels"] if label != -100).decode("utf-8")


def test_labels_only_on_assistant_turns():
    tok = QwenLikeTokenizer()
    (example,), stats = encode_records([make_record(n_logs=2)], tok, max_length=10**6)
    labelled = _labelled_text(example)
    assert labelled.count("<s>assistant\n<think>\nPLAN-log") == 2
    assert "PLAN-done" in labelled and "<tool_call>done</tool_call></s>" in labelled
    assert "<tool_call>log_meal<food_id>F1<grams>150</tool_call>" in labelled
    for masked in ("TOOLS", "Task:", "OBS", "tool_response", TOOL_SYSTEM_PROMPT[:40]):
        assert masked not in labelled
    assert 0 < stats["trained_token_fraction"] < 1
    assert stats["trained_token_fraction"] + stats["masked_token_fraction"] == pytest.approx(1)


def test_over_max_length_is_dropped_and_counted_not_truncated():
    tok = QwenLikeTokenizer()
    short, long = make_record("short"), make_record("long", observation="X" * 5000)
    _, probe = encode_records([short, long], tok, max_length=10**6)
    cap = probe["tokens"]["min"]
    examples, stats = encode_records([short, long], tok, max_length=cap)
    assert stats["n_records"] == 2 and stats["n_kept"] == 1
    assert stats["n_over_max_length"] == 1 and stats["over_max_length_task_ids"] == ["long"]
    assert len(examples[0]["input_ids"]) == cap == len(examples[0]["labels"])


def test_arguments_rendered_as_mapping_without_mutating_the_record():
    record = make_record()
    tok = QwenLikeTokenizer()
    with pytest.raises(TypeError):
        tok.apply_chat_template(record["messages"], tools=[], tokenize=True)
    encode_records([record], tok, max_length=10**6)
    assert isinstance(record["messages"][2]["tool_calls"][0]["function"]["arguments"], str)
    assert for_template(record)["messages"][2]["tool_calls"][0]["function"]["arguments"] == {
        "food_id": "F1", "grams": 150}


def test_prompt_identity_against_tokenize_prompt():
    tok = QwenLikeTokenizer()
    record = make_record()
    ids = tok.apply_chat_template(for_template(record)["messages"],
                                  tools=_tools(), tokenize=True)
    identity = prompt_identity(record, tok, ids)
    assert identity["context"] is True
    # default (non-thinking) header closes an empty think; the record has a plan
    assert identity["generation_prompt"] is False
    assert identity["generation_prompt_thinking"] is True

    drifted = make_record(system=TOOL_SYSTEM_PROMPT + " drift")
    drifted_ids = tok.apply_chat_template(for_template(drifted)["messages"],
                                          tools=_tools(), tokenize=True)
    assert prompt_identity(drifted, tok, drifted_ids)["context"] is False


def _tools():
    from nutrienv.harness.tools_schema import NUTRIENV_TOOLS

    return NUTRIENV_TOOLS


def _write_run(tmp_path, train_records, val_records):
    paths = {}
    for name, rows in (("train", train_records), ("loss_val", val_records)):
        paths[name] = tmp_path / f"{name}.jsonl"
        paths[name].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    config = {"model": {"id": "toy"}, "data": {**{k: str(v) for k, v in paths.items()},
                                                "max_length": 10**6},
              "output_dir": str(tmp_path / "out")}
    config_path = tmp_path / "sft.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


@pytest.fixture
def toy_auto_tokenizer(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *_a, **_k: QwenLikeTokenizer()))


def test_dry_run_passes_and_writes_nothing(tmp_path, toy_auto_tokenizer, capsys):
    config_path = _write_run(tmp_path, [make_record("a"), make_record("b", n_logs=3)],
                             [make_record("c")])
    assert train_v2.main(["--config", str(config_path), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["data"]["train"]["n_kept"] == 2
    assert report["data"]["train"]["identity"]["context"] == 2
    assert report["nutrienv_rev_records"] == ["0" * 40]
    assert not (tmp_path / "out").exists()


def test_dry_run_refuses_a_broken_identity(tmp_path, toy_auto_tokenizer):
    config_path = _write_run(tmp_path, [make_record(system="some other prompt")],
                             [make_record("c")])
    with pytest.raises(SystemExit, match="identity"):
        train_v2.main(["--config", str(config_path), "--dry-run"])
