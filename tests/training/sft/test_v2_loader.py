"""Ticket 028 — v2 SFT loader for FC records; v1 loader untouched."""

from __future__ import annotations

import json
import pathlib

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.harness.tools_schema import NUTRIENV_TOOLS  # noqa: E402
from nutrienv.world.catalog_store import load_catalog  # noqa: E402

from src.training.data_factory.config import load_config  # noqa: E402
from src.training.data_factory.materialize import RunContext, catalog_digest, materialize  # noqa: E402
from src.training.data_factory.rollout_fc import ScriptedFCTeacher, rollout_tool_call  # noqa: E402
from src.training.data_factory.serialize import serialize  # noqa: E402
from src.training.sft.v2_loader import LoadError, tokenize_v2_record  # noqa: E402

from tests.training.data_factory import _fixtures as fx  # noqa: E402
from tests.training.data_factory.test_serialize import _pass_verification, fc_log_script  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]


class ToyTokenizer:
    """Deterministic chat-template stand-in: shared prefix across prefixes."""

    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        parts = []
        if tools is not None:
            parts.append("TOOLS:" + json.dumps(tools, sort_keys=True))
        for message in messages:
            parts.append(
                message.get("role", "")
                + ":"
                + str(message.get("content") or "")
                + str(message.get("reasoning_content") or "")
                + json.dumps(message.get("tool_calls") or [], sort_keys=True)
            )
        if add_generation_prompt:
            parts.append("ASSISTANT:")
        text = "\n".join(parts)
        if tokenize:
            return list(text.encode("utf-8"))
        return text


@pytest.fixture(scope="module")
def fc_record():
    catalog = load_catalog()
    task = fx.make_log_task(catalog, fx.first_person(), seed=30)
    episode = rollout_tool_call(
        task, teacher_complete=ScriptedFCTeacher(fc_log_script(task)), catalog=catalog
    )
    package = materialize(
        task,
        RunContext(
            catalog=catalog,
            catalog_sha=catalog_digest(catalog),
            nutrienv_rev="0ee68eaa6c246e8079915761c95fc986c53d4979",
            nutrimind_rev="d" * 40,
            config_sha="0" * 64,
            seed=30,
            built_at="2026-09-11T12:00:00+00:00",
        ),
    )
    config = load_config("configs/data_factory.yaml")
    return serialize(
        package, episode, _pass_verification(package), config=config, accepted_from_attempt=1
    )


def test_tokenize_sets_labels_from_train_on(fc_record):
    tok = ToyTokenizer()
    out = tokenize_v2_record(fc_record, tok)
    assert out["tools"] is NUTRIENV_TOOLS
    assert len(out["input_ids"]) == len(out["labels"])
    assert any(label != -100 for label in out["labels"])
    assert any(label == -100 for label in out["labels"])
    # tool / system / task turns are train_on false → those spans are -100
    messages = fc_record["messages"]
    train_on = fc_record["train_on"]
    prev = tok.apply_chat_template([], tools=NUTRIENV_TOOLS, tokenize=True)
    for index, (message, flag) in enumerate(zip(messages, train_on)):
        current = tok.apply_chat_template(
            messages[: index + 1], tools=NUTRIENV_TOOLS, tokenize=True
        )
        start, end = len(prev), len(current)
        chunk = out["labels"][start:end]
        if not flag:
            assert chunk == [-100] * (end - start)
            assert message["role"] in ("system", "user", "tool")
        else:
            assert chunk == out["input_ids"][start:end]
            assert message["role"] == "assistant"
            assert message.get("tool_calls")
        prev = current


def test_reject_text_op_and_v1_xml(fc_record):
    tok = ToyTokenizer()
    text_op = json.loads(json.dumps(fc_record))
    for message in text_op["messages"]:
        if message["role"] == "assistant":
            message["content"] = 'plan\n{"op": "log_meal"}'
            message.pop("tool_calls", None)
    with pytest.raises(LoadError, match="text-op"):
        tokenize_v2_record(text_op, tok)

    v1 = json.loads(json.dumps(fc_record))
    for message in v1["messages"]:
        if message["role"] == "assistant":
            message["content"] = "<think>x</think>\n<tool_call>{}</tool_call>"
    with pytest.raises(LoadError, match="v1 XML"):
        tokenize_v2_record(v1, tok)


def test_v1_loader_untouched_and_not_pointed_at_v2():
    train_py = (REPO / "src" / "training" / "sft" / "train.py").read_text(encoding="utf-8")
    assert "data/student" not in train_py
    assert "tokenize_v2_record" not in train_py
    v2 = (REPO / "src" / "training" / "sft" / "v2_loader.py").read_text(encoding="utf-8")
    assert "from src.training.sft.train" not in v2
    assert "from src.training.sft.normalize" not in v2
