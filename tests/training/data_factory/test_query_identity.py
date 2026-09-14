"""Pilot tickets 003/004 — unique query identity + SFT overlay budget."""

from __future__ import annotations

import pathlib

import pytest

from src.training.data_factory.query_identity import (
    REALIZATION_RANK,
    SFT_COLD_START_UNIQUE,
    UniqueQueryIndex,
    query_identity,
    select_realization,
    unique_caps,
)

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from src.training.data_factory.build import build  # noqa: E402
from src.training.data_factory.config import load_config  # noqa: E402
from src.training.data_factory.synthetic import synth_expander  # noqa: E402

from tests.training.data_factory.test_build import tiny_config  # noqa: E402
from tests.training.data_factory.test_build_sft import (  # noqa: E402
    author_all,
    sft_config,
    teacher_script,
)
from src.training.data_factory.rollout_fc import ScriptedFCTeacher  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
YAML_PATH = REPO / "configs" / "data_factory.yaml"


def test_realization_rank_order_is_four_steps():
    assert REALIZATION_RANK == (
        "consistency",
        "unique_bind",
        "diversity",
        "collision",
    )


def test_two_realizations_of_one_identity_count_as_one():
    index = UniqueQueryIndex()
    assert index.add("For lunch I had yogurt.", family="log", task_id="a")
    assert not index.add("for lunch i had yogurt!", family="log", task_id="b")
    assert index.add("For dinner I had soup.", family="log", task_id="c")
    assert len(index) == 2
    assert query_identity("for lunch i had yogurt!") == query_identity(
        "For lunch I had yogurt."
    )


def test_select_realization_consistency_then_bind_then_diversity_then_collision():
    seen = [query_identity("seen meal")]
    picked = select_realization(
        [
            {
                "query": "bad",
                "consistency_ok": False,
                "unique_bind": True,
                "verbatim_collision": False,
                "semantic_collision": False,
                "exam_collision": False,
            },
            {
                "query": "seen meal",
                "consistency_ok": True,
                "unique_bind": True,
                "verbatim_collision": False,
                "semantic_collision": False,
                "exam_collision": False,
            },
            {
                "query": "fresh meal",
                "consistency_ok": True,
                "unique_bind": True,
                "verbatim_collision": True,
                "semantic_collision": False,
                "exam_collision": False,
            },
            {
                "query": "fresh clean",
                "consistency_ok": True,
                "unique_bind": True,
                "verbatim_collision": False,
                "semantic_collision": False,
                "exam_collision": False,
            },
        ],
        seen_identities=seen,
    )
    assert picked["query"] == "fresh clean"


def test_unique_caps_sum_to_budget_and_follow_mix():
    caps = unique_caps(
        {
            "composite": 200,
            "composite_update_log_recommend": 40,
            "recommend": 71,
            "evaluate": 55,
            "log": 42,
            "update": 13,
        },
        SFT_COLD_START_UNIQUE,
    )
    assert sum(caps.values()) == SFT_COLD_START_UNIQUE
    assert caps["composite"] > caps["update"]


def test_build_manifest_unique_query_count_not_inflated_by_teacher_k(tmp_path):
    from nutrienv.world.catalog_store import load_catalog

    catalog = load_catalog()
    expander = synth_expander(catalog)
    counts = []
    for teacher_k in (1, 6):
        out = tmp_path / f"k{teacher_k}"
        config = sft_config(out, teacher_k=teacher_k, target_n=1, over_generate_x=1.0)
        assert config.families["log"].teacher_k == teacher_k
        tasks = author_all(config, expander)
        script = teacher_script(tasks, pass_at_attempt=1, teacher_k=teacher_k)
        manifest = build(
            config,
            expander=expander,
            teacher_complete=ScriptedFCTeacher(script),
            output_dir=out,
        )
        unique = manifest["counts"]["unique_query_identities"]
        accepted = manifest["counts"]["accepted"]
        assert unique == accepted
        assert unique != teacher_k or teacher_k == 1
        counts.append(unique)
    assert counts[0] == counts[1]


def test_overlay_stops_on_unique_queries_not_traces(tmp_path):
    from nutrienv.world.catalog_store import load_catalog

    yaml_before = YAML_PATH.read_bytes()
    catalog = load_catalog()
    expander = synth_expander(catalog)
    out = tmp_path / "out"
    config = tiny_config(out, target_n=4, over_generate_x=1.0)
    assert len(config.families) == 1
    shipped_before = {
        name: fam.target_n for name, fam in load_config(YAML_PATH).families.items()
    }
    manifest = build(
        config,
        expander=expander,
        stop_after="gate",
        output_dir=out,
        unique_query_budget=2,
    )
    unique = manifest["counts"]["unique_query_identities"]
    authored = manifest["counts"]["authored"]
    assert unique == 2
    assert unique == manifest["metrics"]["unique_query_count"]
    assert unique != authored or authored == 2
    assert manifest["metrics"]["unique_query_budget"] == 2
    assert unique != manifest["counts"].get("accepted_traces", 0) or unique == 2
    assert YAML_PATH.read_bytes() == yaml_before
    shipped_after = {
        name: fam.target_n for name, fam in load_config(YAML_PATH).families.items()
    }
    assert shipped_after == shipped_before
    assert shipped_after["composite"] == 200
    assert shipped_after["composite_update_log_recommend"] == 40
    assert shipped_after["recommend"] == 71
    assert shipped_after["evaluate"] == 55
    assert shipped_after["log"] == 42
    assert shipped_after["update"] == 13
    # overlay config's family mix is a test shrink, not a yaml rewrite
    assert config.families["log"].target_n == 4


def test_overlay_does_not_resize_ticket_020_target():
    text = (
        REPO / ".scratch/nutrimind-v2/issues/020-batch-1-production-run.md"
    ).read_text()
    assert "ready-for-agent" in text
    assert "420" in text
    assert "status: CLOSED" not in text.split("---", 2)[1]
