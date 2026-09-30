"""Quota refill: one wave, then top up until accepted == target_n, the cap, or the budget."""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

from src.training.data_factory.build import (
    BuildError,
    enumerate_intents,
    family_attempt_cap,
    family_wave_size,
    intent_for,
    should_top_up,
)
from src.training.data_factory.config import load_config

CONFIG = pathlib.Path(__file__).resolve().parents[3] / "configs" / "data_factory.yaml"


def _log_only():
    config = load_config(CONFIG)
    return dataclasses.replace(
        config,
        families={"log": dataclasses.replace(
            config.families["log"], target_n=2, over_generate_x=1.5
        )},
        max_intents=20,
    )


def test_one_wave_does_not_itself_refill():
    config = _log_only()
    intents = enumerate_intents(config)
    assert len(intents) == 3  # ceil(2 * 1.5); the loop, not the enumerator, tops up
    assert family_wave_size(config, "log") == 3
    assert family_attempt_cap(config, "log") == 6


def test_should_top_up_stops_at_target_cap_budget_and_max_intents():
    common = dict(target_n=2, cap=6, max_intents=10)
    assert should_top_up(
        accepted=0, draws=3, total_intents=3, budget_stopped=False, **common
    )
    assert not should_top_up(
        accepted=2, draws=3, total_intents=3, budget_stopped=False, **common
    )
    assert not should_top_up(
        accepted=0, draws=6, total_intents=6, budget_stopped=False, **common
    )
    assert not should_top_up(
        accepted=0, draws=3, total_intents=10, budget_stopped=False, **common
    )
    assert not should_top_up(
        accepted=0, draws=3, total_intents=3, budget_stopped=True, **common
    )


def test_top_up_index_continues_the_wave():
    config = _log_only()
    wave = enumerate_intents(config)
    extra = intent_for(config, "log", len(wave))
    assert extra["task_id"] not in {row["task_id"] for row in wave}
    assert extra["seed"] == len(wave)
    assert extra["family"] == "log"


def test_simulated_refill_reaches_target_within_the_cap():
    """No network. Accept every other draw until target_n, and stop at the cap."""
    config = _log_only()
    target = config.families["log"].target_n
    cap = family_attempt_cap(config, "log")
    produced = enumerate_intents(config)
    accepted = 0
    draws = 0
    cursor = 0
    while cursor < len(produced):
        cursor += 1
        draws += 1
        if draws % 2 == 0:
            accepted += 1
        if should_top_up(
            accepted=accepted,
            target_n=target,
            draws=draws,
            cap=cap,
            total_intents=len(produced),
            max_intents=config.max_intents,
            budget_stopped=False,
        ):
            produced.append(intent_for(config, "log", len(produced)))
    assert accepted >= target
    assert draws <= cap
    assert len(produced) > family_wave_size(config, "log")
    assert len({row["task_id"] for row in produced}) == len(produced)


def test_production_three_leg_joins_at_target_20_and_hits_the_floor():
    config = load_config(CONFIG)
    family = config.families["composite_update_log_recommend"]
    assert family.target_n == 20
    wave = family_wave_size(config, "composite_update_log_recommend")
    # p = 1/3 → ceil(20 / (1/3) * 1.5) = 90 < 120, so the 120 floor binds.
    assert wave == 120
    assert family_attempt_cap(config, "composite_update_log_recommend") == 240
    assert sum(family_wave_size(config, name) for name in config.families) <= config.max_intents


def test_small_batch_three_leg_trips_the_total_guard():
    """The 120 floor is capped by max_intents per family. The error is the sum."""
    config = load_config(CONFIG)
    three = config.families["composite_update_log_recommend"]
    log = dataclasses.replace(config.families["log"], target_n=2, over_generate_x=1.5)
    small = dataclasses.replace(
        config,
        families={"composite_update_log_recommend": three, "log": log},
        max_intents=100,
    )
    # 3-leg alone would be min(100, 180) = 100 and would not raise.
    alone = dataclasses.replace(small, families={"composite_update_log_recommend": three})
    assert len(enumerate_intents(alone)) == 100
    with pytest.raises(BuildError, match=r"intent count \d+ exceeds max_intents 100"):
        enumerate_intents(small)
