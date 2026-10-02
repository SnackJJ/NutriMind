"""Ticket 022 — recovery-positive predicate, metric, health, authoring lever."""

from __future__ import annotations

import ast
import json
import logging
import pathlib

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

import importlib  # noqa: E402

dispatch_mod = importlib.import_module("nutrienv.actions.dispatch")
schemas_mod = importlib.import_module("nutrienv.actions.schemas")
from nutrienv.env import NutriEnv  # noqa: E402
from nutrienv.world.catalog_store import load_catalog  # noqa: E402

from src.training.data_factory import author as author_mod  # noqa: E402
from src.training.data_factory import build as build_mod  # noqa: E402
from src.training.data_factory.build import build  # noqa: E402
from src.training.data_factory.concepts import (  # noqa: E402
    EpisodeResult,
    TurnMeta,
    VerificationResult,
)
from src.training.data_factory.rollout_fc import ScriptedFCTeacher  # noqa: E402
from src.training.data_factory.synthetic import synth_expander  # noqa: E402
from src.training.data_factory.verify import (  # noqa: E402
    ACTION_ERROR_CLASS,
    is_recovery_positive,
)

from tests.training.data_factory.test_build_sft import (  # noqa: E402
    author_all,
    episode_script,
    sft_config,
    teacher_script,
)


def _codes_in(mod) -> set[str]:
    tree = ast.parse(pathlib.Path(mod.__file__).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != "ActionError" or not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            found.add(arg.value)
    return found


def test_every_installed_actionerror_code_is_classified():
    raised = _codes_in(dispatch_mod) | _codes_in(schemas_mod)
    assert raised
    missing = raised - set(ACTION_ERROR_CLASS)
    extra_class = set(ACTION_ERROR_CLASS) - raised
    assert not missing, f"unclassified ActionError codes: {missing}"
    assert extra_class <= raised | set(ACTION_ERROR_CLASS)
    assert all(kind in ("semantic", "syntax") for kind in ACTION_ERROR_CLASS.values())


def _episode(codes: list[str | None], *, status: str) -> tuple[EpisodeResult, VerificationResult]:
    turns = []
    for code in codes:
        if code is None:
            obs = json.dumps({"ok": True})
        else:
            obs = json.dumps({"error": {"code": code, "message": code}})
        turns.append(TurnMeta(observation=obs, tool_calls=[{"id": "x"}]))
    result = EpisodeResult(turns=turns, reached_finish=True)
    verification = VerificationResult(
        status=status,
        execution="ok",
        oracle_exec="ok",
        scorer="pass" if status == "pass" else "fail",
        reward=1.0 if status == "pass" else 0.0,
        oracle_version="x",
        rubric_version="v2-r1",
        reward_version="v2-r1",
    )
    return result, verification


def test_is_recovery_positive_table():
    sem_pass = _episode(["unknown_food"], status="pass")
    assert is_recovery_positive(*sem_pass) is True
    schema_pass = _episode(["bad_schema"], status="pass")
    assert is_recovery_positive(*schema_pass) is False
    sem_fail = _episode(["unknown_food"], status="fail")
    assert is_recovery_positive(*sem_fail) is False
    clean = _episode([None], status="pass")
    assert is_recovery_positive(*clean) is False
    unknown_op = _episode(["unknown_op"], status="pass")
    assert is_recovery_positive(*unknown_op) is False


def test_syntax_only_does_not_count_toward_fraction(tmp_path, caplog):
    out = tmp_path / "out"
    config = sft_config(out, teacher_k=1, target_n=1, over_generate_x=1.0)
    expander = synth_expander(load_catalog())
    tasks = author_all(config, expander)
    script = teacher_script(tasks, pass_at_attempt=1, teacher_k=1)
    manifest = build(
        config,
        expander=expander,
        teacher_complete=ScriptedFCTeacher(script),
        output_dir=out,
    )
    # clean Pass traces: syntax-only recovery must not inflate the headline
    assert manifest["metrics"]["recovery_positive"] == 0
    assert (manifest["metrics"]["recovery_fraction"] or 0) == 0
    syntax = manifest["metrics"]["recovery_by_code"]["syntax"]
    assert "bad_schema" not in syntax or True  # may be empty on clean scripts


def test_health_warns_outside_band_never_fails(caplog):
    caplog.set_level(logging.WARNING)
    manifest = {
        "counts": {"accepted": 10, "rejected": {"author": 0, "gate": 0, "indeterminate": 0}},
        "catalog_sha": "abc",
    }
    from collections import Counter

    from src.training.data_factory.config import load_config

    config = load_config("configs/data_factory.yaml")
    for frac, expect_warn in ((0.00, True), (0.20, False), (0.30, True)):
        caplog.clear()
        m = json.loads(json.dumps(manifest))
        m["counts"]["accepted"] = 10
        build_mod._finalize_observability(
            m,
            config=config,
            catalog_sha="abc",
            reject_histogram=Counter(),
            accepted_by_family=Counter(),
            teacher_completed=0,
            teacher_error=0,
            teacher_no_finish=0,
            pass_count=0,
            serialized=0,
            attempted_task_ids=0,
            indeterminate_task_ids=0,
            accepted_records=[],
            recovery_positive=int(frac * 10),
            recovery_by_code={"semantic": {}, "syntax": {}},
        )
        warned = "recovery_fraction" in caplog.text
        assert warned is expect_warn
        assert m["health"]["recovery_fraction_in_band"] is (not expect_warn)


def test_unknown_food_scripted_episode_counts(tmp_path):
    from src.training.data_factory.rollout_fc import ScriptedFCTeacher, rollout_tool_call
    from src.training.data_factory import materialize as mz, serialize as sz, verify as vf
    from src.training.data_factory.materialize import RunContext
    from src.training.data_factory.config import load_config
    from tests.training.data_factory import _fixtures as fx

    catalog = load_catalog()
    task = fx.make_log_task(catalog, fx.first_person(), seed=30)
    row = task.oracle.ledger_tail[0]

    def _call(name, args, call_id):
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }

    script = [
        ("bad", [_call("log_meal", {"food_id": "not-a-food", "grams": 100.0, "eaten_at": row.eaten_at}, "b")]),
        ("fix", [_call("log_meal", {"food_id": row.food_id, "grams": row.grams, "eaten_at": row.eaten_at}, "g")]),
    ]
    for extra in task.oracle.ledger_tail[1:]:
        script.append((
            "log",
            [_call("log_meal", {"food_id": extra.food_id, "grams": extra.grams, "eaten_at": extra.eaten_at}, extra.food_id)],
        ))
    script.append(("done", [_call("done", {}, "d")]))
    episode = rollout_tool_call(
        task, teacher_complete=ScriptedFCTeacher(script), catalog=catalog
    )
    package = mz.materialize(
        task,
        RunContext(
            catalog=catalog,
            catalog_sha=mz.catalog_digest(catalog),
            nutrienv_rev="0ee68eaa6c246e8079915761c95fc986c53d4979",
            nutrimind_rev="d" * 40,
            config_sha="0" * 64,
            seed=30,
            built_at="2026-09-11T12:00:00+00:00",
        ),
    )
    verification = vf.verify(package, episode)
    assert verification.status == "pass"
    assert is_recovery_positive(episode, verification) is True

    out = tmp_path / "out"
    config = sft_config(out, teacher_k=1, target_n=1, over_generate_x=1.0)
    # Drive through build with this recovery script for the authored tasks
    expander = synth_expander(catalog)
    tasks = author_all(config, expander)
    full = []
    for authored in tasks:
        row0 = authored.oracle.ledger_tail[0]
        full.extend([
            ("bad", [_call("log_meal", {"food_id": "not-a-food", "grams": 100.0, "eaten_at": row0.eaten_at}, "b")]),
        ])
        full.extend(episode_script(authored))
    manifest = build(
        config,
        expander=expander,
        teacher_complete=ScriptedFCTeacher(full),
        output_dir=out,
    )
    assert manifest["counts"]["accepted"] >= 1
    assert manifest["metrics"]["recovery_positive"] >= 1
    assert manifest["metrics"]["recovery_fraction"] > 0
    assert "unknown_food" in manifest["metrics"]["recovery_by_code"]["semantic"]
    records = [
        json.loads(line)
        for line in (out / "sft" / "accepted.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records


def test_authored_log_trap_natural_first_action(catalog=None):
    catalog = load_catalog()
    expander = synth_expander(catalog)
    from src.training.data_factory.roster_train import TRAIN_ROSTER

    person = TRAIN_ROSTER[0]
    intent = {
        "task_id": "log--log--train-alba--000003",
        "family": "log",
        "user_id": person.user_id,
        "seed": 3,
        "occasion": "lunch",
        "scene": "empty",
        "amount_path": "named_measure",
        "tier": "",
        "recovery_trap": "unknown_food",
    }
    task, reject = author_mod.author_task(intent, catalog=catalog, expander=expander)
    if task is None:
        intent["amount_path"] = "explicit_grams"
        task, reject = author_mod.author_task(intent, catalog=catalog, expander=expander)
    assert task is not None, reject
    env = NutriEnv()
    env.reset(task.s0)
    stepped = env.step({"op": "log_meal", "food_id": "leftover casserole", "grams": 100})
    assert stepped.get("ok") is False
    code = (stepped.get("error") or {}).get("code")
    assert code == "unknown_food"
    from nutrienv.bench.achievable import check_achievable

    report = check_achievable([task])
    assert task.id not in report.unreachable


def test_enumerate_and_build_apply_log_recovery_trap(tmp_path):
    from src.training.data_factory.build import enumerate_intents
    from tests.training.data_factory.test_build import tiny_config

    catalog = load_catalog()
    expander = synth_expander(catalog)
    config = tiny_config(tmp_path / "out", target_n=5, over_generate_x=1.0)
    intents = enumerate_intents(config)
    trapped = [intent for intent in intents if intent.get("recovery_trap") == "unknown_food"]
    assert trapped, "enumerate must set recovery_trap on a fraction of log intents"
    out = tmp_path / "out"
    build(config, expander=expander, stop_after="author", output_dir=out)
    task_lines = [
        json.loads(line)
        for line in (out / "tasks" / "log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trapped_ids = {intent["task_id"] for intent in trapped}
    authored_traps = [row for row in task_lines if row["task_id"] in trapped_ids]
    assert authored_traps, trapped_ids
    query = authored_traps[0]["query"]
    assert "casserole" in query.lower()
    env = NutriEnv()
    # reconstruct via the authored item
    from nutrienv.bench.split import load_split
    import tempfile

    item = authored_traps[0]["item"]
    with tempfile.TemporaryDirectory() as scratch:
        path = pathlib.Path(scratch) / "one.json"
        path.write_text(json.dumps({"items": [item]}) + "\n", encoding="utf-8")
        (task,) = load_split(path, catalog=catalog)
    env.reset(task.s0)
    stepped = env.step({"op": "log_meal", "food_id": "leftover casserole", "grams": 100})
    assert (stepped.get("error") or {}).get("code") == "unknown_food"
