"""Pilot ticket 006 — RL unique-query pool, expansion, OPD budget 0."""

from __future__ import annotations

from src.training.rl.pilot_pool import (
    OPD_UNIQUE,
    RL_PILOT_UNIQUE,
    build_rl_unique_query_pool,
    pool_expansion_needed,
    select_opd_queries,
)
from src.training.rl.reward import REWARD_MAP


def test_reward_map_unchanged_binary():
    assert REWARD_MAP == {"pass": 1.0, "fail": 0.0, "indeterminate": None}
    pool = build_rl_unique_query_pool([], size=200)
    assert pool["reward_map"]["pass"] == 1.0
    assert pool["reward_map"]["fail"] == 0.0
    assert pool["reward_map"]["indeterminate"] is None


def test_rl_pool_counts_unique_queries_with_explicit_overlap_and_held_out():
    packages = [
        {"task_id": f"t{i}", "family": "log", "query": f"For lunch I had meal {i}."}
        for i in range(10)
    ]
    packages.append(
        {"task_id": "dup", "family": "log", "query": "For lunch I had meal 0."}
    )
    sft = ["For lunch I had meal 0.", "For lunch I had meal 1."]
    pool = build_rl_unique_query_pool(
        packages, sft_identities=sft, size=8, held_out_fraction=0.25
    )
    assert pool["unique_query_count"] == 8
    assert pool["target"] == 8
    assert pool["unique_query_count"] != 11
    assert {row["identity"] for row in pool["overlap_with_sft"]}
    assert pool["held_out"]
    overlap_ids = {row["task_id"] for row in pool["overlap_with_sft"]}
    held_ids = {row["task_id"] for row in pool["held_out"]}
    assert overlap_ids.isdisjoint(held_ids)
    assert RL_PILOT_UNIQUE == 200


def test_noisy_curve_with_healthy_mixed_groups_does_not_expand():
    assert not pool_expansion_needed(
        {"noisy_curve": True, "mixed_reward_groups": 12}
    )


def test_listed_shortages_do_expand():
    assert pool_expansion_needed({"too_few_mixed_groups": True})
    assert pool_expansion_needed({"empty_family_coverage": True})
    assert pool_expansion_needed({"empty_composite_coverage": True})
    assert pool_expansion_needed({"train_up_held_out_flat": True})
    assert pool_expansion_needed({"verifier_shortcut": True})


def test_opd_budget_zero_and_refuses_ambiguous_or_disagreement():
    assert OPD_UNIQUE == 0
    states = [
        {
            "query": "ok",
            "catalog_ambiguous": False,
            "teacher_status": "pass",
            "verifier_status": "pass",
        },
        {
            "query": "amb",
            "catalog_ambiguous": True,
            "teacher_status": "pass",
            "verifier_status": "pass",
        },
        {
            "query": "disagree",
            "catalog_ambiguous": False,
            "teacher_status": "pass",
            "verifier_status": "fail",
        },
    ]
    assert select_opd_queries(states) == []
    assert select_opd_queries(states, budget=0) == []
    picked = select_opd_queries(states, budget=5)
    assert [row["query"] for row in picked] == ["ok"]
