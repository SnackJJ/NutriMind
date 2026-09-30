"""Ticket 011 — build --target sft end-to-end for the log family (Seam 1).

Offline: the synthetic expander authors, a scripted ``teacher_complete`` queue
drives real episodes through the real env, and build wires author → gate →
materialize → rollout (attempts 1..k) → verify → serialize → sft/train.jsonl,
with rollouts/cache/<task_id>.json as the multi-attempt container.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import get_args

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.world.catalog_store import load_catalog  # noqa: E402

from src.training.data_factory import build as build_mod  # noqa: E402
from src.training.data_factory import author as author_mod  # noqa: E402
from src.training.data_factory.build import (  # noqa: E402
    NOT_A_FOOD_VERDICT,
    build,
    enumerate_intents,
    family_attempt_cap,
    intent_for,
)
from src.training.data_factory.search_gate import Locatability  # noqa: E402


def sft_lines(out: pathlib.Path) -> list[str]:
    accepted = out / "sft" / "accepted.jsonl"
    if accepted.is_file():
        return [line for line in accepted.read_text(encoding="utf-8").splitlines() if line.strip()]
    train = out / "sft" / "train.jsonl"
    if train.is_file():
        return [line for line in train.read_text(encoding="utf-8").splitlines() if line.strip()]
    return []
from src.training.data_factory.config import load_config  # noqa: E402
from src.training.data_factory.concepts import RolloutCache  # noqa: E402
from src.training.data_factory.rollout_fc import ScriptedFCTeacher  # noqa: E402
from src.training.data_factory.synthetic import synth_expander  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "data_factory.yaml"


def sft_config(output_dir, *, teacher_k=2, target_n=2, over_generate_x=1.0):
    """The real config, shrunk to a tiny log-only sft run."""
    base = load_config(CONFIG_PATH)
    log_cfg = dataclasses.replace(
        base.families["log"],
        target_n=target_n, over_generate_x=over_generate_x, teacher_k=teacher_k,
    )
    return dataclasses.replace(
        base, target="sft", families={"log": log_cfg}, output_dir=str(output_dir)
    )


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.fixture(scope="module")
def expander(catalog):
    return synth_expander(catalog)


def author_all(config, expander):
    """Pre-author the config's intents (for building the teacher script)."""
    tasks = []
    for intent in enumerate_intents(config):
        task, _ = author_mod.author_task(intent, catalog=load_catalog(), expander=expander)
        tasks.append(task)
    return tasks


def episode_script(task, *, grams_scale=1.0):
    """One scripted FC episode for a log task: the tail rows, then done."""
    turns = []
    for index, row in enumerate(task.oracle.ledger_tail, start=1):
        args = {
            "op": "log_meal",
            "food_id": row.food_id,
            "grams": round(row.grams * grams_scale, 2),
            "eaten_at": row.eaten_at,
        }
        # tool arguments omit the env "op" key
        tool_args = {k: v for k, v in args.items() if k != "op"}
        turns.append((
            "I should log my lunch.",
            [{
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": "log_meal",
                    "arguments": json.dumps(tool_args, ensure_ascii=False),
                },
            }],
        ))
    turns.append((
        "All logged, finishing.",
        [{
            "id": f"call_{len(turns) + 1}",
            "type": "function",
            "function": {"name": "done", "arguments": "{}"},
        }],
    ))
    return turns


def teacher_script(tasks, *, pass_at_attempt=1, teacher_k=2):
    """The full queue: per task, (pass_at_attempt - 1) failing episodes then a
    passing one (attempts beyond the Pass never run). ``pass_at_attempt``
    beyond k means every attempt fails."""
    script = []
    for task in tasks:
        attempts = min(pass_at_attempt, teacher_k)
        for attempt in range(1, attempts + 1):
            if attempt < pass_at_attempt:
                script.extend(episode_script(task, grams_scale=1.5))  # beyond ±15 %
            else:
                script.extend(episode_script(task))
    return script


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #


def test_scripted_pass_to_train_jsonl(tmp_path, catalog, expander):
    config = sft_config(tmp_path / "out", teacher_k=2)
    tasks = author_all(config, expander)
    script = teacher_script(tasks, pass_at_attempt=1, teacher_k=2)
    out = tmp_path / "out"
    manifest = build(
        config, expander=expander, teacher_complete=ScriptedFCTeacher(script),
        output_dir=out,
    )
    assert manifest["status"] == "complete"
    assert manifest["counts"]["accepted"] == 2

    # the search-locatability observable is written on the accept path, one
    # verdict per accepted task (an empty block here would mean the wiring is dead)
    locatability = manifest["metrics"]["search_locatability"]
    assert sum(locatability.values()) == manifest["counts"]["accepted"]
    # The allowed set is derived from the verdict Literal plus the build-level defect
    # statuses; a hardcoded list once excluded `no_terms`, which is a legal verdict.
    assert set(locatability) <= set(get_args(Locatability)) | NOT_A_FOOD_VERDICT
    judged = sum(
        count for status, count in locatability.items()
        if status not in NOT_A_FOOD_VERDICT
    )
    assert manifest["metrics"]["search_locatability_usable_rate"] == pytest.approx(
        locatability.get("unique", 0) / judged
    )

    lines = sft_lines(out)
    records = [json.loads(line) for line in lines]
    assert len(records) == 2  # one per task, exactly
    assert [r["task_id"] for r in records] == sorted(r["task_id"] for r in records)
    for record in records:
        assert record["schema_version"] == "nutrimind-v2-sft/1"
        assert record["meta"]["verification"]["status"] == "pass"
        assert record["meta"]["verification"]["reward"] == 1.0
        assert record["accepted_from_attempt"] == 1
    # pass on attempt 1 → one attempt per cache, selected at 0
    for cache_path in (out / "rollouts" / "cache").glob("*.json"):
        cache = RolloutCache.from_dict(
            json.loads(cache_path.read_text(encoding="utf-8"))
        )
        assert len(cache.attempts) == 1
        assert cache.selected_attempt == 0
        assert cache.attempts[0].verification.status == "pass"


def test_retry_until_pass_then_stop(tmp_path, catalog, expander):
    """Attempt 1 fails, attempt 2 passes: k attempts ≤ teacher_k, the loop
    stops at the Pass, selected_attempt points at it."""
    config = sft_config(tmp_path / "out", teacher_k=2)
    tasks = author_all(config, expander)
    script = teacher_script(tasks, pass_at_attempt=2, teacher_k=2)
    out = tmp_path / "out"
    manifest = build(
        config, expander=expander, teacher_complete=ScriptedFCTeacher(script),
        output_dir=out,
    )
    assert manifest["counts"]["accepted"] == 2
    records = [
        json.loads(line)
        for line in sft_lines(out)
    ]
    assert all(r["accepted_from_attempt"] == 2 for r in records)
    for cache_path in (out / "rollouts" / "cache").glob("*.json"):
        cache = RolloutCache.from_dict(
            json.loads(cache_path.read_text(encoding="utf-8"))
        )
        assert len(cache.attempts) == 2  # stopped at the Pass, not k=2 beyond
        assert cache.selected_attempt == 1
        assert cache.attempts[0].verification.status == "fail"
        assert cache.attempts[1].verification.status == "pass"


def test_byte_identical_train_jsonl_across_runs(tmp_path, catalog, expander):
    config_a = sft_config(tmp_path / "a")
    config_b = sft_config(tmp_path / "b")
    tasks = author_all(config_a, expander)
    script_a = teacher_script(tasks, pass_at_attempt=2, teacher_k=2)
    script_b = teacher_script(tasks, pass_at_attempt=2, teacher_k=2)
    build(config_a, expander=expander, teacher_complete=ScriptedFCTeacher(script_a),
          output_dir=tmp_path / "a")
    build(config_b, expander=expander, teacher_complete=ScriptedFCTeacher(script_b),
          output_dir=tmp_path / "b")
    assert (tmp_path / "a" / "sft" / "accepted.jsonl").read_bytes() == (
        tmp_path / "b" / "sft" / "accepted.jsonl"
    ).read_bytes()


# --------------------------------------------------------------------------- #
# reject routing after k attempts
# --------------------------------------------------------------------------- #


def test_all_attempts_fail_to_teacher_jsonl(tmp_path, catalog, expander):
    config = sft_config(tmp_path / "out", teacher_k=2)
    # A zero-accept family refills up to the attempt cap, so the script has to
    # cover that many tasks, not just the first wave.
    cap = family_attempt_cap(config, "log")
    tasks = []
    for index in range(cap):
        task, reject = author_mod.author_task(
            intent_for(config, "log", index),
            catalog=load_catalog(),
            expander=expander,
        )
        assert task is not None, reject
        tasks.append(task)
    script = teacher_script(tasks, pass_at_attempt=99, teacher_k=2)  # never passes
    out = tmp_path / "out"
    manifest = build(
        config, expander=expander, teacher_complete=ScriptedFCTeacher(script),
        output_dir=out,
    )
    assert manifest["counts"]["accepted"] == 0
    assert manifest["counts"]["intents"] == cap
    assert manifest["counts"]["teacher_rejected"] == cap
    assert not sft_lines(out)

    lines = [
        json.loads(line)
        for line in (out / "rejects" / "teacher.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(lines) == cap
    for reject in lines:
        assert reject["stage"] == "teacher"
        assert reject["status"] == "fail"
        assert reject["failure_codes"][0] == "task_fail"
        assert len(reject["failure_codes"]) == 2  # ["task_fail", <Scorer tag>]
        assert len(reject["attempts"]) == 2  # all k attempts stored
        assert {a["status"] for a in reject["attempts"]} == {"fail"}
        assert reject["task_package_ref"] == f"task_packages/{reject['task_id']}.json"
        assert reject["rollouts_ref"] == f"rollouts/cache/{reject['task_id']}.json"
        # an analysis candidate — nothing may mark it an RLVR negative (§4.2)
        assert "rlvr" not in json.dumps(reject).lower()


def test_no_finish_to_indeterminate_jsonl(tmp_path, catalog, expander):
    config = sft_config(tmp_path / "out", teacher_k=1)
    cap = family_attempt_cap(config, "log")
    # one episode per draw: 12 idle reads hit the log step budget. Zero accepts
    # refill until the cap (two waves), so the script covers cap episodes.
    script = [
        (
            "checking.",
            [{
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": "get_profile", "arguments": "{}"},
            }],
        )
        for i in range(12 * cap)
    ]
    out = tmp_path / "out"
    manifest = build(
        config, expander=expander, teacher_complete=ScriptedFCTeacher(script),
        output_dir=out,
    )
    assert manifest["counts"]["accepted"] == 0
    assert manifest["counts"]["intents"] == cap
    assert manifest["counts"]["teacher_indeterminate"] == cap
    lines = [
        json.loads(line)
        for line in (out / "rejects" / "indeterminate.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(lines) == cap
    for reject in lines:
        assert reject["status"] == "indeterminate"
        assert reject["failure_codes"] == ["teacher_no_finish"]
    assert not sft_lines(out)


def test_one_bad_intent_among_good_lands(tmp_path, catalog, expander):
    """An un-authorable intent rejects at author; the good ones still run the
    full teacher path and land in train.jsonl."""
    config = sft_config(tmp_path / "out", teacher_k=1)
    tasks = author_all(config, expander)
    script = []
    for task in tasks:
        script.extend(episode_script(task))
    # fail exactly one intent at author stage
    calls = {"n": 0}

    inner_bind = getattr(expander, "bind_intent", None)

    def bind_intent(intent):
        bound = inner_bind(intent) if inner_bind is not None else expander

        def wrapped(pool, *, persona, family, amount_path=None):
            calls["n"] += 1
            if calls["n"] == 2:
                return {"query": "", "foods": []}
            result = bound(
                pool, persona=persona, family=family, amount_path=amount_path
            )
            wrapped.last_pool_ids = getattr(bound, "last_pool_ids", None)
            return result

        return wrapped

    def flaky(pool, *, persona, family, amount_path=None):
        return bind_intent({})(pool, persona=persona, family=family, amount_path=amount_path)

    flaky.bind_intent = bind_intent

    out = tmp_path / "out"
    manifest = build(
        config, expander=flaky, teacher_complete=ScriptedFCTeacher(script),
        output_dir=out,
    )
    assert manifest["status"] == "complete"
    assert manifest["counts"]["rejected"]["author"] == 1
    assert manifest["counts"]["accepted"] == 1
    records = [
        json.loads(line)
        for line in sft_lines(out)
    ]
    assert len(records) == 1
    assert records[0]["meta"]["verification"]["status"] == "pass"


# --------------------------------------------------------------------------- #
# the cache contract
# --------------------------------------------------------------------------- #


def test_cache_round_trips_full_attempt_records(tmp_path, catalog, expander):
    config = sft_config(tmp_path / "out", teacher_k=2)
    tasks = author_all(config, expander)
    script = teacher_script(tasks, pass_at_attempt=2, teacher_k=2)
    out = tmp_path / "out"
    build(config, expander=expander, teacher_complete=ScriptedFCTeacher(script),
          output_dir=out)

    cache_paths = sorted((out / "rollouts" / "cache").glob("*.json"))
    assert len(cache_paths) == 2
    for path, task in zip(cache_paths, tasks):
        cache = RolloutCache.from_dict(json.loads(path.read_text(encoding="utf-8")))
        assert cache.task_id == path.stem
        assert cache.selected_attempt == 1
        assert len(cache.attempts) == 2
        for attempt in cache.attempts:
            assert attempt.attempt_id.startswith(cache.task_id + "--attempt-")
            episode = attempt.episode
            # EpisodeResult: per-step messages with assistant content +
            # reasoning_content + finish_reason + usage, and the end state
            assert episode.reached_finish is True
            assert episode.reset_observation
            for turn in episode.turns:
                assert turn.tool_calls
                assert turn.reasoning_content  # kept separate from content
                assert turn.usage is not None
                name = turn.tool_calls[0]["function"]["name"]
                if name not in ("done", "finish", "stop"):
                    assert turn.executed_op is not None
            assert episode.end_state is not None
            # VerificationResult
            assert attempt.verification.status in ("pass", "fail")
            assert attempt.verification.reward in (0.0, 1.0)
        # the first failing attempt's episode details survive the round-trip
        assert cache.attempts[0].verification.failure_codes[0] == "task_fail"


def test_rerun_reuses_cache_without_teacher(tmp_path, catalog, expander):
    """A re-run after a crash-between-cache-and-train (task not yet terminal)
    loads the cache and re-serializes — the teacher is never re-paid."""
    config = sft_config(tmp_path / "out", teacher_k=1)
    tasks = author_all(config, expander)
    script = teacher_script(tasks, pass_at_attempt=1, teacher_k=1)
    out = tmp_path / "out"
    build(config, expander=expander, teacher_complete=ScriptedFCTeacher(script),
          output_dir=out)
    first = (out / "sft" / "accepted.jsonl").read_bytes()

    # simulate the interrupted state: accepted record + reject lines gone,
    # cache + packages intact
    for name in ("train.jsonl", "accepted.jsonl", "holdout.jsonl", "loss_val.jsonl"):
        path = out / "sft" / name
        if path.exists():
            path.unlink()
    for reject in (out / "rejects").glob("*.jsonl"):
        reject.unlink()

    # NO teacher injected at all — the cache alone must carry the run
    manifest = build(config, expander=expander, teacher_complete=lambda req: (_ for _ in ()).throw(
        AssertionError("teacher must not be called")
    ), output_dir=out)
    assert manifest["counts"]["cache_reused"] == 2
    assert manifest["counts"]["accepted"] == 2
    assert (out / "sft" / "accepted.jsonl").read_bytes() == first  # identical records
