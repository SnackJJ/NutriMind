"""Ticket 003 — shared seam concept types: field contracts + JSON round-trips.

``concepts.py`` must stay logic-free (field definitions and trivial
``to_dict`` / ``from_dict`` only); the AST scan below enforces that so later
tickets cannot quietly grow business logic into the shared vocabulary.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import pytest

from src.training.data_factory import (
    AttemptRecord,
    EpisodeResult,
    GateResult,
    Provenance,
    RewardSemantics,
    RolloutCache,
    TaskCatalogRef,
    TaskEnvironment,
    TaskOracle,
    TaskPackage,
    TaskVerifierRef,
    Termination,
    TurnMeta,
    VerificationResult,
)
from src.training.data_factory import concepts as concepts_mod

_CONCEPTS_PATH = pathlib.Path(concepts_mod.__file__)

_ALL_CONCEPTS = [
    AttemptRecord,
    EpisodeResult,
    GateResult,
    Provenance,
    RewardSemantics,
    RolloutCache,
    TaskCatalogRef,
    TaskEnvironment,
    TaskOracle,
    TaskPackage,
    TaskVerifierRef,
    Termination,
    TurnMeta,
    VerificationResult,
]

# Modules the concept types must never (transitively) pull in at import time.
_BANNED_IMPORT_ROOTS = {
    "nutrienv",
    "socket",
    "httpx",
    "http",
    "requests",
    "urllib",
    "openai",
    "dashscope",
    "subprocess",
    "os",
}

_ALLOWED_FUNCTIONS = {"to_dict", "from_dict", "__post_init__"}


def test_every_concept_type_is_a_dataclass():
    for cls in _ALL_CONCEPTS:
        assert dataclasses.is_dataclass(cls), cls


def test_concepts_module_is_logic_free():
    """Only field definitions + trivial (de)serialization; no imports beyond
    dataclasses/typing, no functions outside to_dict/from_dict."""
    tree = ast.parse(_CONCEPTS_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            roots = {n.split(".")[0] for n in names}
            assert not roots & _BANNED_IMPORT_ROOTS, (
                f"concepts.py imports {roots & _BANNED_IMPORT_ROOTS} — the shared "
                "vocabulary must stay side-effect-free"
            )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert node.name in _ALLOWED_FUNCTIONS, (
                f"concepts.py defines function {node.name!r} — concept types carry "
                "no business logic (ticket 003)"
            )


def test_gate_result_contract():
    keep = GateResult(keep=True)
    drop = GateResult(
        keep=False,
        failure_code="gate.semantic_key_collision",
        reason_detail="matches exam task adr20-log-1001",
    )
    assert keep.keep and keep.failure_code is None
    assert drop.stage == "gate"
    assert GateResult.from_dict(drop.to_dict()) == drop


def test_turn_meta_round_trip():
    turn = TurnMeta(
        raw_action_text='{"op": "search_foods", "q": "oatmeal"}',
        executed_op={"op": "search_foods", "q": "oatmeal"},
        parse_status="ok",
        fallback_used=False,
        content='{"op": "search_foods", "q": "oatmeal"}',
        reasoning_content="find oatmeal in the catalog",
        finish_reason="stop",
        usage={"prompt_tokens": 210, "completion_tokens": 18, "reasoning_tokens": 739},
        observation='{"hits": [...]}',
    )
    assert TurnMeta.from_dict(turn.to_dict()) == turn


def test_turn_meta_fc_round_trip():
    """Ticket 024: native tool-calling turn stores tool_calls + tool_call_id."""
    turn = TurnMeta(
        tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "search_foods", "arguments": '{"q": "oatmeal"}'},
        }],
        tool_call_id="call_1",
        executed_op={"op": "search_foods", "q": "oatmeal"},
        reasoning_content="find oatmeal in the catalog",
        content=None,
        raw_action_text=None,
        parse_status=None,
        fallback_used=False,
        fallback_reason=None,
        observation='{"hits": [...]}',
    )
    restored = TurnMeta.from_dict(turn.to_dict())
    assert restored == turn
    assert restored.tool_calls[0]["id"] == "call_1"
    assert restored.tool_call_id == "call_1"
    assert restored.executed_op == {"op": "search_foods", "q": "oatmeal"}
    assert restored.raw_action_text is None
    assert restored.parse_status is None
    assert restored.fallback_used is False
    assert restored.fallback_reason is None


def test_turn_meta_no_tool_calls_is_distinct():
    """A turn with no tool_calls is representable and is not coerced into an op."""
    empty = TurnMeta(
        tool_calls=[],
        tool_call_id=None,
        executed_op=None,
        reasoning_content="nothing to call",
        raw_action_text=None,
        parse_status=None,
        fallback_used=False,
        fallback_reason=None,
    )
    restored = TurnMeta.from_dict(empty.to_dict())
    assert restored == empty
    assert restored.tool_calls == []
    assert restored.executed_op is None
    fallback = TurnMeta(
        executed_op={"op": "get_profile"},
        fallback_used=True,
        fallback_reason="no_json",
        parse_status="no_json",
        raw_action_text="sure, let me look that up",
    )
    assert restored != fallback
    assert restored.executed_op != {"op": "get_profile"}


def test_episode_result_round_trip():
    episode = EpisodeResult(
        end_state={"profile": {"weight_kg": 70}, "ledger": []},  # dict form (opaque Any)
        turns=[TurnMeta(raw_action_text='{"op": "finish"}', executed_op={"op": "finish"})],
        reached_finish=True,
        error=None,
        task={"id": "log--log--train-ada--000042"},
        latency_s=12.5,
    )
    restored = EpisodeResult.from_dict(episode.to_dict())
    assert restored == episode
    assert restored.turns[0].executed_op == {"op": "finish"}


def test_verification_result_round_trip():
    result = VerificationResult(
        status="pass",
        execution="ok",
        oracle_exec="ok",
        scorer="pass",
        reward=1.0,
        oracle_version="nutrienv-203d807",
        rubric_version="v2-r1",
        reward_version="v2-r1",
        failure_codes=[],
        evidence=[{"sub_tags": ["pass", "pass", "pass"]}],
    )
    assert VerificationResult.from_dict(result.to_dict()) == result

    indeterminate = VerificationResult(
        status="indeterminate",
        execution="invalid_op",
        oracle_exec="ok",
        scorer=None,
        reward=None,
        oracle_version="nutrienv-203d807",
        rubric_version="v2-r1",
        reward_version="v2-r1",
        failure_codes=["teacher_invalid_op"],
    )
    restored = VerificationResult.from_dict(indeterminate.to_dict())
    assert restored == indeterminate
    assert restored.reward is None  # null, never 0.0 (spec §12)


def _sample_task_package() -> TaskPackage:
    return TaskPackage(
        schema_version="nutrimind-v2-taskpackage/1",
        task_key="composite--update+log+recommend--train-ada",
        task_id="composite--update+log+recommend--train-ada--000191",
        query="Please add milk to my allergies. For lunch I had two slices of white "
        "bread. What should I have for dinner?",
        family="composite",
        steps=["update", "log", "recommend"],
        tier="",
        environment=TaskEnvironment(
            s0={"profile": {"user_id": "train-ada"}, "ledger": [], "allowed_food_ids": None},
            reconstruct_with="nutrienv.bench.pipeline.freezer.task_to_item -> freeze_tasks "
            "-> load_split (transient scratch file; ticket 002 Part A)",
        ),
        catalog=TaskCatalogRef(
            catalog_sha="57184b2bbce4519076b4238a8d64861950db46fdc793d0e43055f07f43c28b5f",
            nutrienv_rev="203d807b19953a86b5486303ba6f7dd3b9cf7bb6",
        ),
        oracle=TaskOracle(
            payload={"sub_oracles": [{"family": "update"}, {"family": "log"}]},
            oracle_version="nutrienv-203d807",
        ),
        verifier=TaskVerifierRef(
            kind="nutrienv.bench.scorer.Scorer", call="Scorer().score(end_state, oracle)"
        ),
        reward_semantics=RewardSemantics(
            reward_version="v2-r1", kind="binary",
            map={"pass": 1.0, "fail": 0.0, "indeterminate": None},
        ),
        rubric_version="v2-r1",
        termination=Termination(finish_ops=["finish", "done", "stop"], max_steps=30),
        seed=191,
        provenance=Provenance(
            nutrimind_rev="0" * 40,
            nutrienv_rev="203d807b19953a86b5486303ba6f7dd3b9cf7bb6",
            catalog_sha="57184b2bbce4519076b4238a8d64861950db46fdc793d0e43055f07f43c28b5f",
            config_sha="9" * 12,
            intent_ref="intents/composite.jsonl#191",
            built_at="2026-09-09T12:00:00+00:00",
        ),
    )


def test_task_package_round_trip():
    package = _sample_task_package()
    assert TaskPackage.from_dict(package.to_dict()) == package
    # the three identifiers are distinct fields, never conflated (spec §10)
    assert package.task_key not in package.task_id or package.task_id.endswith("--000191")
    assert package.task_id == f"{package.task_key}--{package.seed:06d}"


def test_rollout_cache_round_trip_multi_attempt():
    """RolloutCache supports multiple attempts + selected_attempt (ticket 003 /
    ticket 011: one entry per attempt that ran; selected = first Pass or None)."""
    def _attempt(n: int, status: str) -> AttemptRecord:
        return AttemptRecord(
            attempt_id=f"log--log--train-ada--000042--attempt-{n:02d}",
            episode=EpisodeResult(
                end_state={"ledger": []},
                turns=[TurnMeta(raw_action_text='{"op": "finish"}', executed_op={"op": "finish"})],
                reached_finish=True,
            ),
            verification=VerificationResult(
                status=status,
                execution="ok",
                oracle_exec="ok",
                scorer=status,
                reward=1.0 if status == "pass" else 0.0,
                oracle_version="nutrienv-203d807",
                rubric_version="v2-r1",
                reward_version="v2-r1",
            ),
        )

    cache = RolloutCache(
        task_id="log--log--train-ada--000042",
        attempts=[_attempt(1, "fail"), _attempt(2, "pass")],
        selected_attempt=1,  # 0-based index of the first Pass
    )
    restored = RolloutCache.from_dict(cache.to_dict())
    assert restored == cache
    assert restored.selected_attempt == 1
    assert restored.attempts[restored.selected_attempt].verification.status == "pass"

    no_pass = RolloutCache(task_id="t", attempts=[_attempt(1, "fail")], selected_attempt=None)
    assert RolloutCache.from_dict(no_pass.to_dict()).selected_attempt is None


def test_attempt_record_requires_verification():
    """The cache shape pins {attempt_id, EpisodeResult, VerificationResult} — an
    attempt without a verification result cannot be constructed."""
    with pytest.raises(TypeError):
        AttemptRecord(attempt_id="x--attempt-01", episode=EpisodeResult())
