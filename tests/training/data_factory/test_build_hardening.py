"""Ticket 014 — resume, --from-stage, atomic writes, cost budget."""

from __future__ import annotations

import json
import logging

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.world.catalog_store import load_catalog  # noqa: E402

from src.training.data_factory.build import BuildError, build, enumerate_intents  # noqa: E402
from src.training.data_factory.concepts import RolloutCache  # noqa: E402
from src.training.data_factory.rollout_fc import ScriptedFCTeacher  # noqa: E402
from src.training.data_factory.synthetic import synth_expander  # noqa: E402

from tests.training.data_factory.test_build_sft import (  # noqa: E402
    author_all,
    sft_config,
    teacher_script,
)


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.fixture(scope="module")
def expander(catalog):
    return synth_expander(catalog)


def test_crash_between_write_and_rename_leaves_train_intact(tmp_path, expander):
    out = tmp_path / "out"
    config = sft_config(out, teacher_k=1, target_n=1, over_generate_x=1.0)
    tasks = author_all(config, expander)
    script = teacher_script(tasks, pass_at_attempt=1, teacher_k=1)
    build(
        config,
        expander=expander,
        teacher_complete=ScriptedFCTeacher(script),
        output_dir=out,
    )
    original = (out / "sft" / "accepted.jsonl").read_bytes()

    def boom(path, tmp):
        if path.name == "accepted.jsonl":
            raise RuntimeError("injected crash before replace")

    with pytest.raises(RuntimeError, match="injected crash"):
        build(
            config,
            expander=expander,
            teacher_complete=ScriptedFCTeacher(script),
            output_dir=out,
            force=True,
            before_replace=boom,
        )
    assert (out / "sft" / "accepted.jsonl").read_bytes() == original
    # no half-written JSON line in the live file
    for line in original.splitlines():
        json.loads(line)


def test_second_run_reuses_cache_zero_teacher(tmp_path, expander):
    out = tmp_path / "out"
    config = sft_config(out, teacher_k=1, target_n=1, over_generate_x=1.0)
    tasks = author_all(config, expander)
    script = teacher_script(tasks, pass_at_attempt=1, teacher_k=1)
    first_calls = []

    def counting(complete):
        def wrapped(request):
            first_calls.append(1)
            return complete(request)
        return wrapped

    teacher = ScriptedFCTeacher(script)
    build(
        config, expander=expander, teacher_complete=counting(teacher), output_dir=out
    )
    n_first = len(first_calls)
    assert n_first > 0
    train = (out / "sft" / "accepted.jsonl").read_bytes()

    second_calls = []

    def must_not(request):
        second_calls.append(1)
        raise AssertionError("teacher must not run on cache reuse")

    second = build(config, expander=expander, teacher_complete=must_not, output_dir=out)
    assert second_calls == []
    assert second["counts"]["skipped_terminal"] >= 1
    assert (out / "sft" / "accepted.jsonl").read_bytes() == train


def test_from_stage_serialize_zero_teacher_and_null_selected(tmp_path, expander):
    out = tmp_path / "out"
    config = sft_config(out, teacher_k=1, target_n=1, over_generate_x=1.0)
    tasks = author_all(config, expander)
    script = teacher_script(tasks, pass_at_attempt=1, teacher_k=1)
    build(
        config,
        expander=expander,
        teacher_complete=ScriptedFCTeacher(script),
        output_dir=out,
    )
    train = (out / "sft" / "accepted.jsonl").read_bytes()

    calls = []

    def must_not(request):
        calls.append(1)
        raise AssertionError("serialize-only must not call the teacher")

    again = build(
        config,
        expander=expander,
        teacher_complete=must_not,
        output_dir=out,
        from_stage="serialize",
    )
    assert calls == []
    assert (out / "sft" / "accepted.jsonl").read_bytes() == train
    assert again["counts"]["accepted"] >= 1

    # selected_attempt = null stays a reject
    cache_path = next((out / "rollouts" / "cache").glob("*.json"))
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["selected_attempt"] = None
    for attempt in payload["attempts"]:
        attempt["verification"]["status"] = "fail"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    third = build(
        config,
        expander=expander,
        teacher_complete=must_not,
        output_dir=out,
        from_stage="serialize",
        force=True,
    )
    assert third["counts"]["accepted"] == 0
    assert third["counts"]["teacher_rejected"] >= 1


def test_from_stage_rollout_reruns_teacher(tmp_path, expander):
    out = tmp_path / "out"
    config = sft_config(out, teacher_k=1, target_n=1, over_generate_x=1.0)
    tasks = author_all(config, expander)
    script = teacher_script(tasks, pass_at_attempt=1, teacher_k=1)
    build(
        config,
        expander=expander,
        teacher_complete=ScriptedFCTeacher(script),
        output_dir=out,
    )
    packages_before = {
        p.name: p.read_bytes() for p in (out / "task_packages").glob("*.json")
    }
    calls = []
    teacher = ScriptedFCTeacher(script)

    def counting(request):
        calls.append(1)
        return teacher(request)

    build(
        config,
        expander=expander,
        teacher_complete=counting,
        output_dir=out,
        from_stage="rollout",
    )
    assert calls
    after = {p.name: p.read_bytes() for p in (out / "task_packages").glob("*.json")}
    assert after == packages_before


def test_on_budget_stop_writes_manifest(tmp_path, expander, caplog):
    out = tmp_path / "out"
    import dataclasses

    config = sft_config(out, teacher_k=1, target_n=2, over_generate_x=1.0)
    tasks = author_all(config, expander)
    script = teacher_script(tasks, pass_at_attempt=1, teacher_k=1)
    teacher = ScriptedFCTeacher(script)

    def with_tokens(request):
        body = teacher(request)
        body.setdefault("usage", {})
        body["usage"]["prompt_tokens"] = 1_000_000
        body["usage"]["completion_tokens"] = 0
        return body

    # budget 0.0001 so the first 1M-token attempt trips 100%
    import dataclasses

    config = dataclasses.replace(config, usd_budget=0.0001, on_budget="stop")
    manifest = build(
        config, expander=expander, teacher_complete=with_tokens, output_dir=out
    )
    assert manifest["status"] == "stopped_budget"
    written = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert written["status"] == "stopped_budget"
    assert written["cost"]["budget_usd"] == 0.0001
    assert written["cost"]["budget_stopped"] is True
    # two intents; halt after the first teacher spend
    assert written["counts"]["intents"] == 2
    attempted = written["health"].get("attempted_task_ids") or written["counts"].get(
        "accepted", 0
    ) + written["counts"].get("teacher_rejected", 0) + written["counts"].get(
        "teacher_indeterminate", 0
    )
    assert attempted < written["counts"]["intents"]


def test_on_budget_warn_completes(tmp_path, expander, caplog):
    import dataclasses

    out = tmp_path / "out"
    config = dataclasses.replace(
        sft_config(out, teacher_k=1, target_n=1, over_generate_x=1.0),
        usd_budget=0.0001,
        on_budget="warn",
    )
    tasks = author_all(config, expander)
    script = teacher_script(tasks, pass_at_attempt=1, teacher_k=1)
    teacher = ScriptedFCTeacher(script)

    def with_tokens(request):
        body = teacher(request)
        body.setdefault("usage", {})
        body["usage"]["prompt_tokens"] = 1_000_000
        body["usage"]["completion_tokens"] = 0
        return body

    caplog.set_level(logging.WARNING)
    manifest = build(
        config, expander=expander, teacher_complete=with_tokens, output_dir=out
    )
    assert manifest["status"] == "complete"
    assert manifest["cost"]["budget_warned"] is True
    assert "80%" in caplog.text or manifest["cost"]["budget_warned"]


def test_duplicate_task_id_raises(tmp_path):
    config = sft_config(tmp_path, teacher_k=1, target_n=1, over_generate_x=1.0)
    intents = enumerate_intents(config)
    # enumerator itself must not emit duplicates
    ids = [i["task_id"] for i in intents]
    assert len(ids) == len(set(ids))
    with pytest.raises(BuildError, match="twice"):
        # poison: two families that collapse to the same task_id is not possible
        # through enumerate; the run-loop raise is covered by injecting a dup list
        from src.training.data_factory import build as build_mod

        dup = [intents[0], intents[0]]

        def fake_enum(_config):
            return dup

        original = build_mod.enumerate_intents
        build_mod.enumerate_intents = fake_enum
        try:
            build(
                config,
                expander=synth_expander(load_catalog()),
                teacher_complete=lambda *_: (_ for _ in ()).throw(AssertionError()),
                stop_after="author",
                output_dir=tmp_path / "out",
            )
        finally:
            build_mod.enumerate_intents = original
