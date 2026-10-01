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
    assert prompt_identity(record, tok, ids) == {"context": True, "turns": True}

    drifted = make_record(system=TOOL_SYSTEM_PROMPT + " drift")
    drifted_ids = tok.apply_chat_template(for_template(drifted)["messages"],
                                          tools=_tools(), tokenize=True)
    assert prompt_identity(drifted, tok, drifted_ids)["context"] is False


def test_turn_identity_uses_the_empty_think_header_for_a_planless_turn():
    tok = QwenLikeTokenizer()
    record = make_record(n_logs=2)
    record["messages"][4]["reasoning_content"] = None
    ids = tok.apply_chat_template(for_template(record)["messages"],
                                  tools=_tools(), tokenize=True)
    assert prompt_identity(record, tok, ids)["turns"] is True
    assert prompt_identity(record, tok, ids[:-1] + [0])["turns"] is True
    # a history that is not the training prefix of a later turn
    assert prompt_identity(record, tok, ids[:40] + [0] + ids[41:])["turns"] is False


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
    # main() replays records through the lab loop; the toy records have no task package
    monkeypatch.setattr(train_v2, "load_tasks", lambda records, *_a: [(None, None)] * len(records))
    monkeypatch.setattr(train_v2, "eval_context", lambda record, *_a: record)


def test_dry_run_passes_and_writes_nothing(tmp_path, toy_auto_tokenizer, capsys):
    config_path = _write_run(tmp_path, [make_record("a"), make_record("b", n_logs=3)],
                             [make_record("c")])
    assert train_v2.main(["--config", str(config_path), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["data"]["train"]["n_kept"] == 2
    assert report["data"]["train"]["identity"] == {"context": 2, "turns": 2}
    assert report["nutrienv_rev_records"] == ["0" * 40]
    assert not (tmp_path / "out").exists()


def test_dry_run_refuses_a_broken_identity(tmp_path, toy_auto_tokenizer):
    config_path = _write_run(tmp_path, [make_record(system="some other prompt")],
                             [make_record("c")])
    with pytest.raises(SystemExit, match="identity"):
        train_v2.main(["--config", str(config_path), "--dry-run"])


# --------------------------------------------------------------------------- #
# eval_context: replay through the real lab FC loop (pinned NutriEnv catalog)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def exam():
    from nutrienv.bench import EXAM_SPLIT_PATH, load_split
    from nutrienv.world.catalog_store import load_catalog

    catalog = load_catalog()
    return {t.id: t for t in load_split(EXAM_SPLIT_PATH, catalog=catalog)}, catalog


def _live_record(task, catalog, actions):
    """A factory-shaped record: every turn keeps an extra (never executed)
    parallel call, tool replies are bare observations, no opening observation."""
    from nutrienv.env import NutriEnv

    env = NutriEnv()
    reset = json.dumps(env.reset(task.s0), default=str)
    messages = [{"role": "system", "content": TOOL_SYSTEM_PROMPT},
                {"role": "user", "content": f"Task:\n{task.query}"}]
    for index, (name, args) in enumerate(actions):
        calls = [_call(name, args, f"call_{index}"), _call("get_ledger", {}, f"extra_{index}")]
        messages.append({"role": "assistant", "content": None,
                         "reasoning_content": f"PLAN-{index}", "tool_calls": calls})
        if index < len(actions) - 1:
            result = env.step({"op": name, **args})
            obs = result["observation"] if result.get("ok") else {"error": result.get("error")}
            messages.append({"role": "tool", "tool_call_id": f"call_{index}",
                             "content": json.dumps(obs, default=str)[:6000]})
    segments = ["system", "task"] + ["step" if m["role"] == "assistant" else "tool"
                                     for m in messages[2:]]
    segments[-1] = "final"
    record = {"task_id": task.id, "messages": messages, "segments": segments,
              "train_on": [s in ("step", "final") for s in segments]}
    return record, reset


def test_eval_context_matches_the_lab_loop(exam):
    from nutrienv.harness.runner import FAMILY_MAX_STEPS

    tasks, catalog = exam
    task = tasks["adr20-log-5001"]
    record, reset = _live_record(task, catalog, [("get_profile", {}), ("get_ledger", {}),
                                                 ("done", {})])
    out = train_v2.eval_context(record, task, reset, catalog)
    budget = FAMILY_MAX_STEPS[task.family]
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["system", "user", "user", "assistant", "tool", "assistant", "tool",
                     "assistant"]
    assert out["messages"][2]["content"] == (
        f"Step budget: {budget} action(s) remaining.\nObservation:\n{reset}")
    assert out["messages"][4]["content"] == (
        f"Step budget: {budget - 1} action(s) remaining.\nObservation:\n"
        + record["messages"][3]["content"])
    assert out["messages"][6]["content"].startswith(f"Step budget: {budget - 2} action(s)")
    assert all(len(m["tool_calls"]) == 1 for m in out["messages"] if m["role"] == "assistant")
    assert out["segments"] == ["system", "task", "observation", "step", "tool", "step", "tool",
                               "final"]
    assert out["train_on"] == [s in ("step", "final") for s in out["segments"]]
    assert len(record["messages"][2]["tool_calls"]) == 2  # input untouched
    _, stats = encode_records([out], QwenLikeTokenizer(), max_length=10**6)
    assert stats["identity"] == {"context": 1, "turns": 1}
    assert "extra_" not in json.dumps(out)


def test_eval_context_keeps_the_live_dict_order(exam):
    tasks, catalog = exam
    task = tasks["adr20-log-5001"]
    record, reset = _live_record(task, catalog, [("get_profile", {}), ("done", {})])
    shuffled = json.dumps(json.loads(reset), sort_keys=True)
    out = train_v2.eval_context(record, task, shuffled, catalog)
    assert out["messages"][2]["content"].endswith(shuffled)


def test_eval_context_refuses_a_drifted_observation(exam):
    tasks, catalog = exam
    task = tasks["adr20-log-5001"]
    record, reset = _live_record(task, catalog, [("get_profile", {}), ("done", {})])
    record["messages"][3]["content"] = json.dumps({"op": "get_profile", "profile": {}})
    with pytest.raises(SystemExit, match="differs"):
        train_v2.eval_context(record, task, reset, catalog)


def test_eval_context_flags_a_refused_final_hand_in(exam):
    tasks, catalog = exam
    task = tasks["adr20-eval-5010"]
    record, reset = _live_record(task, catalog, [
        ("get_profile", {}),
        ("submit_plan", {"items": [], "verdict": "accept", "reasons": ["x"]}),
    ])
    with pytest.raises(train_v2.ReplayError, match="refused"):
        train_v2.eval_context(record, task, reset, catalog)


def test_dry_run_concatenates_several_train_batches(tmp_path, toy_auto_tokenizer, capsys):
    config_path = _write_run(tmp_path, [make_record("a")], [make_record("c")])
    second = tmp_path / "train2.jsonl"
    second.write_text(json.dumps(make_record("b", n_logs=2)) + "\n", encoding="utf-8")
    config = json.loads(config_path.read_text())
    config["data"]["train"] = [config["data"]["train"], str(second)]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert train_v2.main(["--config", str(config_path), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["data"]["train"]["n_kept"] == 2
    assert len(report["data"]["train"]["sha256"]) == 2
