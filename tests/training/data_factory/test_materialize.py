"""Ticket 006 — TaskPackage materializer (Seam 3, spec §9.1 / §10 / §19.1).

Offline: tasks come from ``generate_one`` (synthetic expander where needed),
the environment round-trip runs through the public freezer/split API with a
transient scratch file, and nothing touches the network.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench import Task, check_achievable, load_split  # noqa: E402
from nutrienv.bench.pipeline import freezer  # noqa: E402
from nutrienv.bench.pipeline.types import catalog_digest  # noqa: E402
from nutrienv.env import NutriEnv  # noqa: E402
from nutrienv.harness.runner import DEFAULT_MAX_STEPS, FAMILY_MAX_STEPS  # noqa: E402

from src.training.data_factory import materialize as mz  # noqa: E402
from src.training.data_factory.concepts import TaskPackage  # noqa: E402
from src.training.data_factory.materialize import (  # noqa: E402
    DuplicateTaskIdError,
    RunContext,
    attempt_id,
)
from src.training.data_factory.roster_train import TRAIN_ROSTER  # noqa: E402

from tests.training.data_factory import _fixtures as fx  # noqa: E402

NUTRIENV_REV = "203d807b19953a86b5486303ba6f7dd3b9cf7bb6"
NUTRIMIND_REV = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
CONFIG_SHA = "0" * 64


def make_ctx(catalog, **overrides) -> RunContext:
    fields = dict(
        catalog=catalog,
        catalog_sha=catalog_digest(catalog),
        nutrienv_rev=NUTRIENV_REV,
        nutrimind_rev=NUTRIMIND_REV,
        config_sha=CONFIG_SHA,
        built_at="2026-09-09T12:00:00+00:00",
    )
    fields.update(overrides)
    return RunContext(**fields)


@pytest.fixture(scope="module")
def catalog():
    return fx.gold_catalog()


@pytest.fixture(scope="module")
def person():
    # a TRAIN person (ticket 004): train-* isolation carries into packages
    return next(p for p in TRAIN_ROSTER if not p.allergies and p.persona == "everyday")


@pytest.fixture(scope="module")
def log_task(catalog, person):
    return fx.make_log_task(catalog, person, seed=30)


@pytest.fixture(scope="module")
def three_leg(catalog):
    task, reason = fx.assemble_three_leg(catalog, seed=101, allergen="fish")
    assert task is not None, f"3-leg assembly failed: {reason}"
    return task


# --------------------------------------------------------------------------- #
# identifiers (spec §10)
# --------------------------------------------------------------------------- #


def test_identifier_helpers():
    key = "composite--update+log+recommend--train-alba"
    assert mz.attempt_id(f"{key}--000191", 1) == f"{key}--000191--attempt-01"
    assert mz.attempt_id(f"{key}--000191", 12) == f"{key}--000191--attempt-12"
    with pytest.raises(ValueError):
        mz.attempt_id(f"{key}--000191", 0)


def test_task_key_and_id_format(catalog, log_task):
    pkg = mz.materialize(log_task, make_ctx(catalog, seed=30, intent_ref="intents/log.jsonl#30"))
    # simple family: steps == (family,) — key has no seed, id appends it
    assert pkg.task_key == f"log--log--{log_task.s0.profile.user_id}"
    assert pkg.task_id == f"{pkg.task_key}--000030"
    assert pkg.seed == 30
    assert pkg.steps == ["log"]


def test_three_leg_key_carries_steps(catalog, three_leg):
    pkg = mz.materialize(
        three_leg,
        make_ctx(
            catalog,
            steps=("update", "log", "recommend"),
            seed=101,
            intent_ref="intents/composite.jsonl#101",
        ),
    )
    assert pkg.task_key == (
        f"composite--update+log+recommend--{three_leg.s0.profile.user_id}"
    )
    assert pkg.task_id == f"{pkg.task_key}--000101"
    assert pkg.steps == ["update", "log", "recommend"]


# --------------------------------------------------------------------------- #
# §9.1 schema completeness
# --------------------------------------------------------------------------- #


def test_package_has_all_required_fields(catalog, log_task):
    pkg = mz.materialize(log_task, make_ctx(catalog, seed=30))
    data = pkg.to_dict()
    required = [
        "schema_version", "task_key", "task_id", "query", "family", "steps",
        "tier", "environment", "catalog", "oracle", "verifier",
        "reward_semantics", "rubric_version", "termination", "seed",
        "provenance",
    ]
    for field in required:
        assert field in data, f"missing §9.1 field: {field}"
    assert data["schema_version"] == "nutrimind-v2-taskpackage/1"
    assert data["verifier"] == {
        "kind": "nutrienv.bench.scorer.Scorer",
        "call": "Scorer().score(end_state, oracle)",
    }
    assert data["reward_semantics"] == {
        "reward_version": "v2-r1",
        "kind": "binary",
        "map": {"pass": 1.0, "fail": 0.0, "indeterminate": None},
    }
    assert data["rubric_version"] == "v2-r1"
    assert data["tier"] == log_task.tier
    # and the package round-trips through the 003 concepts (to/from_dict)
    assert TaskPackage.from_dict(data) == pkg


def test_termination_per_family(catalog, log_task, three_leg):
    log_pkg = mz.materialize(log_task, make_ctx(catalog))
    three_pkg = mz.materialize(
        three_leg, make_ctx(catalog, steps=("update", "log", "recommend"))
    )
    assert log_pkg.termination.finish_ops == sorted({"done", "finish", "stop"})
    assert log_pkg.termination.max_steps == FAMILY_MAX_STEPS["log"] == 12
    assert three_pkg.termination.max_steps == FAMILY_MAX_STEPS["composite"] == 30
    # unknown family falls back to the default budget
    term = mz._termination("no-such-family")
    assert term.max_steps == DEFAULT_MAX_STEPS
    assert term.finish_ops == sorted({"done", "finish", "stop"})


def test_provenance_complete(catalog, log_task):
    pkg = mz.materialize(
        log_task, make_ctx(catalog, seed=30, intent_ref="intents/log.jsonl#30")
    )
    prov = pkg.provenance
    assert prov.nutrimind_rev == NUTRIMIND_REV
    assert prov.nutrienv_rev == NUTRIENV_REV
    assert prov.catalog_sha == catalog_digest(catalog)
    assert prov.config_sha == CONFIG_SHA
    assert prov.intent_ref == "intents/log.jsonl#30"
    assert prov.built_at  # ISO-8601 timestamp present


def test_built_at_defaults_to_now(catalog, log_task):
    pkg = mz.materialize(log_task, make_ctx(catalog, built_at=None))
    assert pkg.provenance.built_at.startswith("20")
    assert "T" in pkg.provenance.built_at


def test_oracle_version_is_rev_prefixed(catalog, log_task):
    pkg = mz.materialize(log_task, make_ctx(catalog))
    assert pkg.oracle.oracle_version == "nutrienv-203d807"


# --------------------------------------------------------------------------- #
# §19.1 environment round-trip
# --------------------------------------------------------------------------- #


def _reconstruct_from_package(pkg: TaskPackage, task: Task, catalog):
    """The documented consumer path: package blocks -> 1-item frozen file ->
    load_split -> runnable Task (persona/situations come from the source Task;
    the package's reconstruction contract is the environment)."""
    item = {
        "id": task.id,
        "family": pkg.family,
        "persona": task.persona,
        "situations": list(task.situations),
        "query": pkg.query,
        "s0": pkg.environment.s0,
        "oracle": pkg.oracle.payload,
    }
    with tempfile.TemporaryDirectory() as d:
        fp = pathlib.Path(d) / "from-package.json"
        payload = {
            "version": freezer.PIPELINE_VERSION,
            "catalog": freezer.CATALOG_V1_RELPATH,
            "catalog_sha256": pkg.catalog.catalog_sha,
            "items": [item],
        }
        fp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (rebuilt,) = load_split(fp, catalog=catalog)
    return rebuilt


def _s0_view(s0):
    p = s0.profile
    return (
        p.user_id,
        tuple(p.allergies),
        {k: tuple(v) for k, v in p.windows.items()},
        p.phase,
        p.activity,
        tuple((r.food_id, r.grams, r.eaten_at) for r in s0.ledger),
        tuple(s0.allowed_food_ids) if s0.allowed_food_ids is not None else None,
    )


@pytest.mark.parametrize("which", ["log", "three_leg"])
def test_environment_round_trips_runnable_and_reachable(
    catalog, which, log_task, three_leg
):
    task = log_task if which == "log" else three_leg
    steps = () if which == "log" else ("update", "log", "recommend")
    pkg = mz.materialize(task, make_ctx(catalog, steps=steps, seed=1))

    # package s0 == the item s0 (the serialization the freezer produces)
    assert pkg.environment.s0 == freezer.task_to_item(task)["s0"]

    rebuilt = _reconstruct_from_package(pkg, task, catalog)
    assert _s0_view(rebuilt.s0) == _s0_view(task.s0), "s0 not preserved"
    assert rebuilt.query == task.query
    # catalog_sha the package pinned is the one the world hashes to
    assert pkg.catalog.catalog_sha == catalog_digest(catalog)
    # runnable …
    obs = NutriEnv().reset(rebuilt.s0)
    assert isinstance(obs, dict)
    # … and Pass-reachable
    assert rebuilt.id not in check_achievable([rebuilt]).unreachable


def test_composite_payload_has_sub_oracles(catalog, three_leg):
    pkg = mz.materialize(
        three_leg, make_ctx(catalog, steps=("update", "log", "recommend"))
    )
    assert list(pkg.oracle.payload) == ["sub_oracles"]
    assert len(pkg.oracle.payload["sub_oracles"]) == 3


# --------------------------------------------------------------------------- #
# run semantics (spec §10)
# --------------------------------------------------------------------------- #


def test_duplicate_task_id_within_run_raises(catalog, log_task):
    ctx = make_ctx(catalog, seed=30)
    mz.materialize(log_task, ctx)
    with pytest.raises(DuplicateTaskIdError, match="twice within one run"):
        mz.materialize(log_task, ctx)
    # a different seed is a different task_id — allowed
    other = mz.materialize(log_task, make_ctx(catalog, seed=31))
    assert other.task_id.endswith("--000031")


def test_catalog_sha_mismatch_raises(catalog, log_task):
    ctx = make_ctx(catalog, catalog_sha="f" * 64)
    with pytest.raises(ValueError, match="catalog_sha mismatch"):
        mz.materialize(log_task, ctx)


def test_write_package_is_idempotent(catalog, log_task, tmp_path):
    pkg = mz.materialize(log_task, make_ctx(catalog, seed=30))
    written = mz.write_package(pkg, tmp_path)
    assert written == tmp_path / f"{pkg.task_id}.json"
    assert written.exists()
    first_blob = written.read_text(encoding="utf-8")

    # re-run: existing file is skipped, not rewritten
    assert mz.write_package(pkg, tmp_path) is None
    assert written.read_text(encoding="utf-8") == first_blob

    # the written file parses back into the same package
    reloaded = TaskPackage.from_dict(json.loads(first_blob))
    assert reloaded == pkg


def test_scratch_file_is_deleted(catalog, log_task, tmp_path, monkeypatch):
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    # route tempfile's default dir at our probe dir; restored on teardown
    monkeypatch.setattr(tempfile, "tempdir", str(scratch_root))
    mz.materialize(log_task, make_ctx(catalog, seed=30))
    leftovers = [p for p in scratch_root.rglob("*")]
    assert leftovers == [], f"orphan scratch files: {leftovers}"


def test_materialize_fails_fast_on_lossy_round_trip(
    catalog, log_task, monkeypatch
):
    """If the public round-trip ever stops being lossless, materialization
    raises instead of writing a package silently."""
    from nutrienv.bench.split import load_split as real_load_split

    def lossy(path, *, catalog=None):
        tasks = real_load_split(path, catalog=catalog)
        import dataclasses

        return [dataclasses.replace(t, query=t.query + " (mutated)") for t in tasks]

    monkeypatch.setattr(mz, "load_split", lossy)
    with pytest.raises(ValueError, match="not lossless"):
        mz.materialize(log_task, make_ctx(catalog, seed=30))
