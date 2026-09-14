"""RL ticket 002 — train/eval prompt identity for one TaskPackage."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.harness.tools_schema import NUTRIENV_TOOLS, TOOL_SYSTEM_PROMPT  # noqa: E402
from nutrienv.world.catalog_store import load_catalog  # noqa: E402

from src.training.data_factory.materialize import RunContext, catalog_digest, materialize  # noqa: E402
from src.training.data_factory.rollout_fc import ScriptedFCTeacher  # noqa: E402
from src.training.rl.eval_report import prompt_for_package as eval_prompt  # noqa: E402
from src.training.rl.prompt import prompt_for_package, tokenize_prompt  # noqa: E402
from src.training.rl.rollout import prompt_for_package as train_prompt  # noqa: E402
from src.training.rl.rollout import student_rollout  # noqa: E402

from tests.training.data_factory import _fixtures as fx  # noqa: E402
from tests.training.rl.test_student_rollout import _call, _pass_script  # noqa: E402


class ToyTokenizer:
    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        text = json.dumps(
            {"messages": messages, "tools": tools, "gen": add_generation_prompt},
            sort_keys=True,
        )
        if tokenize:
            return list(text.encode("utf-8"))
        return text


@pytest.fixture(scope="module")
def package():
    catalog = load_catalog()
    task = fx.make_log_task(catalog, fx.first_person(), seed=30)
    return materialize(
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
    ), catalog, task


def test_train_and_eval_share_the_function(package):
    pkg, _catalog, _task = package
    assert train_prompt is prompt_for_package
    assert eval_prompt is prompt_for_package
    payload = prompt_for_package(pkg)
    assert payload["system"] is TOOL_SYSTEM_PROMPT
    assert payload["tools"] is NUTRIENV_TOOLS
    assert payload["parallel_tool_calls"] is False
    tok = ToyTokenizer()
    train_ids = tokenize_prompt(train_prompt(pkg), tok)
    eval_ids = tokenize_prompt(eval_prompt(pkg), tok)
    assert train_ids == eval_ids


def test_student_rollout_payload_matches_shared_prompt(package):
    pkg, catalog, task = package
    seen = []
    teacher = ScriptedFCTeacher(_pass_script(task))

    def complete(request):
        seen.append(request)
        return teacher(request)

    spec = {"complete": complete, "catalog": catalog, "parallel_tool_calls": False}
    student_rollout(spec, pkg, k=1, seed=0)
    assert seen
    payload = prompt_for_package(pkg)
    assert seen[0]["tools"] == payload["tools"]
    assert seen[0]["parallel_tool_calls"] is False
    assert seen[0]["messages"][0]["content"] == payload["system"]
    assert payload["task"] in seen[0]["messages"][1]["content"]
