"""Ticket 002 Part A / spec OQ-2 — single frozen item -> runnable NutriEnv.

Verifies the TaskPackage `environment` block is reconstructable through the **public**
nutrienv API only. There is no public *in-memory* `item -> Task` entry
(`nutrienv.bench.split.__all__` is all file-based; `_item` / `_s0` / `_oracle` are
private), so reconstruction round-trips through a transient 1-item split file:

    Task --task_to_item--> item --freeze_tasks--> <tmp>.json --load_split--> Task'

The TaskPackage carries the full `s0`; the temp file is scratch, not an external
dataset dependency. Verdict: self-contained (see ticket 002 Part A).
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench import Scorer, check_achievable, load_split  # noqa: E402
from nutrienv.bench.pipeline.freezer import freeze_tasks, task_to_item  # noqa: E402
from nutrienv.bench.pipeline.generate_one import generate_one  # noqa: E402
from nutrienv.bench.pipeline.types import catalog_digest  # noqa: E402
from nutrienv.bench.realize import scored_oracles  # noqa: E402
from nutrienv.env import NutriEnv  # noqa: E402
from nutrienv.world.catalog_store import GOLD_CATALOG_PATH, load_catalog  # noqa: E402


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(GOLD_CATALOG_PATH)


def _no_expander_task(catalog, kind):
    """Composite(update, recommend), update, recommend — none need an LLM expander."""
    if kind == "update":
        r = generate_one(
            catalog=catalog, family="update", seed=11,
            shell="upd-weight", slots={"n": "71"},
        )
    elif kind == "recommend":
        r = generate_one(
            catalog=catalog, family="recommend", seed=12,
            occasion="dinner", shell="rec-dinner",
        )
    elif kind == "composite":
        r = generate_one(
            catalog=catalog, family="composite", steps=("update", "recommend"),
            seed=13, occasion="dinner",
            shell="upd-add-allergy-short", slots={"allergen": "soy"},
        )
    else:  # pragma: no cover
        raise ValueError(kind)
    assert r.accepted is not None, f"{kind}: generate_one rejected: {r.rejected}"
    return r.accepted


def _s0_view(task):
    p = task.s0.profile
    return {
        "user_id": p.user_id,
        "allergies": tuple(p.allergies),
        "windows": {k: tuple(v) for k, v in p.windows.items()},
        "phase": p.phase,
        "activity": p.activity,
        "weight_kg": p.weight_kg,
        "ledger": tuple((r.food_id, r.grams, r.eaten_at) for r in task.s0.ledger),
        "allowed_food_ids": task.s0.allowed_food_ids,
    }


def _oracle_sig(oracle):
    out = []
    for so in scored_oracles(oracle):
        out.append({
            "profile_user": so.profile.user_id if so.profile else None,
            "windows": sorted(so.profile.windows.items()) if so.profile else None,
            "last_plan": so.last_plan,
            "plan_must_be_safe": so.plan_must_be_safe,
            "plan_must_fit_windows": so.plan_must_fit_windows,
            "plan_windows": sorted(so.plan_windows.items()) if so.plan_windows else None,
            "last_verdict": so.last_verdict,
            "update_band": so.update_band,
            "ledger_tail": (
                tuple((r.food_id, r.grams, r.eaten_at) for r in so.ledger_tail)
                if so.ledger_tail else None
            ),
            "allowed_food_ids": so.allowed_food_ids,
        })
    return out


def _round_trip(task, catalog):
    sha = catalog_digest(catalog)
    with tempfile.TemporaryDirectory() as d:
        fp = pathlib.Path(d) / "one.json"
        freeze_tasks([task], catalog=catalog, catalog_sha=sha,
                     output_path=fp, overwrite=True)
        (rebuilt,) = load_split(fp, catalog=catalog)
    return rebuilt


def test_no_public_in_memory_item_to_task():
    import nutrienv.bench.split as split_mod

    assert not any(
        "item" in name.lower() or "payload" in name.lower()
        for name in split_mod.__all__
    ), "a public in-memory reconstructor appeared — simplify this test / update spec §9.1"


@pytest.mark.parametrize("kind", ["update", "recommend", "composite"])
def test_frozen_item_round_trips_to_runnable_env(kind, catalog):
    task = _no_expander_task(catalog, kind)
    rebuilt = _round_trip(task, catalog)

    assert _s0_view(rebuilt) == _s0_view(task), "s0 not preserved"
    assert _oracle_sig(rebuilt.oracle) == _oracle_sig(task.oracle), "oracle not preserved"
    assert (rebuilt.family, rebuilt.persona, rebuilt.tier, rebuilt.query) == (
        task.family, task.persona, task.tier, task.query,
    ), "task meta not preserved"

    obs = NutriEnv().reset(rebuilt.s0)
    assert isinstance(obs, dict), "reconstructed s0 is not runnable"

    assert rebuilt.id not in check_achievable([rebuilt]).unreachable, (
        "reconstructed task is no longer Pass-reachable"
    )
