#!/usr/bin/env python3
"""
Verification tests for ADR-007 implementation.

Validates tool_cache, trl_env_factory, and reward_from_env WITHOUT
requiring GPU, real DB, or RAG. Uses monkeypatching to mock external deps.

Run:
    python3 -m pytest tests/test_adr007.py -v
    # or simply:
    python3 tests/test_adr007.py
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is on path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# 1. ToolCache tests
# ============================================================================

def test_cache_basic_hit_miss():
    """Cache miss calls fn, cache hit returns stored value."""
    from src.training.grpo.tool_cache import ToolCache

    cache = ToolCache()
    cache.new_group("test_group")

    call_count = 0
    def fake_fn():
        nonlocal call_count
        call_count += 1
        return {"status": "success", "data": {"calories_kcal": 165.0}}

    # First call = miss
    r1 = cache.get_or_call("get_food_nutrition", {"foods": [{"food_name": "chicken"}]}, fake_fn)
    assert call_count == 1
    assert "165.0" in r1

    # Second call with same args = hit (fn NOT called again)
    r2 = cache.get_or_call("get_food_nutrition", {"foods": [{"food_name": "chicken"}]}, fake_fn)
    assert call_count == 1  # Still 1
    assert r1 == r2

    print("✅ test_cache_basic_hit_miss")


def test_cache_normalization():
    """'Chicken' and 'chicken' should hit the same cache entry."""
    from src.training.grpo.tool_cache import ToolCache

    cache = ToolCache()
    cache.new_group("test_norm")

    call_count = 0
    def fake_fn():
        nonlocal call_count
        call_count += 1
        return {"status": "success"}

    cache.get_or_call("tool", {"name": "Chicken Breast"}, fake_fn)
    cache.get_or_call("tool", {"name": "chicken breast"}, fake_fn)
    cache.get_or_call("tool", {"name": " CHICKEN BREAST "}, fake_fn)

    assert call_count == 1, f"Expected 1 call, got {call_count}"
    assert cache.hit_rate > 0.5

    print("✅ test_cache_normalization")


def test_cache_group_flush():
    """New group clears cache."""
    from src.training.grpo.tool_cache import ToolCache

    cache = ToolCache()
    cache.new_group("group_A")
    cache.get_or_call("tool", {"x": 1}, lambda: "result_A")
    assert cache.size == 1

    # Same group = no flush
    cache.new_group("group_A")
    assert cache.size == 1

    # Different group = flush
    cache.new_group("group_B")
    assert cache.size == 0

    print("✅ test_cache_group_flush")


def test_cache_snapshot_restore():
    """Snapshot/restore preserves cache state for ARPO branching."""
    from src.training.grpo.tool_cache import ToolCache

    cache = ToolCache()
    cache.new_group("arpo_test")
    cache.get_or_call("tool", {"q": "a"}, lambda: "result_1")
    cache.get_or_call("tool", {"q": "b"}, lambda: "result_2")
    assert cache.size == 2

    snap = cache.snapshot()

    # Add more entries
    cache.get_or_call("tool", {"q": "c"}, lambda: "result_3")
    assert cache.size == 3

    # Restore
    cache.restore(snap)
    assert cache.size == 2

    # Old entries still accessible
    r = cache.get_or_call("tool", {"q": "a"}, lambda: "SHOULD NOT BE CALLED")
    assert r == '"result_1"' or r == "result_1"

    print("✅ test_cache_snapshot_restore")


# ============================================================================
# 2. NutriMindToolEnv tests
# ============================================================================

def _make_env_state():
    """Create a realistic env_state for testing."""
    return {
        "user_profile": {"age": 30, "gender": "M", "weight_kg": 80, "tdee_kcal": 2200, "goal": "lose"},
        "user_goals": {"calories": 1800, "protein": 120, "fat": 60, "carbs": 200},
        "meals_today": [
            {
                "meal_type": "breakfast",
                "calories": 400, "protein_g": 25, "fat_g": 15, "carbs_g": 45, "fiber_g": 5,
                "foods": [{"name": "oatmeal"}, {"name": "eggs"}],
            }
        ],
        "meal_history": [
            {"date": "2026-04-14", "calories": 1900, "protein_g": 100, "fat_g": 70, "carbs_g": 220, "fiber_g": 25},
            {"date": "2026-04-13", "calories": 2100, "protein_g": 110, "fat_g": 65, "carbs_g": 250, "fiber_g": 20},
            {"date": "2026-04-12", "calories": 1700, "protein_g": 90, "fat_g": 55, "carbs_g": 200, "fiber_g": 22},
            {"date": "2026-04-11", "calories": 2000, "protein_g": 105, "fat_g": 60, "carbs_g": 230, "fiber_g": 28},
        ],
    }


def test_env_reset():
    """reset() correctly parses env_state and initializes tracking."""
    from src.training.grpo.trl_env_factory import NutriMindToolEnv

    env = NutriMindToolEnv()
    result = env.reset(
        prompt=[{"role": "user", "content": "How much protein in chicken?"}],
        env_state=json.dumps(_make_env_state()),
        tier="T1",
        query="How much protein in chicken?",
        difficulty="easy",
        optimal_steps=1,
    )

    assert result is None  # No extra text
    assert env._tier == "T1"
    assert env._query == "How much protein in chicken?"
    assert env._tool_calls_count == 0
    assert len(env._tool_history) == 0
    assert env._env_state["user_goals"]["calories"] == 1800

    print("✅ test_env_reset")


def test_env_get_today_summary():
    """get_today_summary renders from env_state and returns valid JSON str."""
    from src.training.grpo.trl_env_factory import NutriMindToolEnv

    env = NutriMindToolEnv()
    env.reset(
        prompt="test",
        env_state=json.dumps(_make_env_state()),
        tier="T2", query="check my calories",
    )

    result = env.get_today_summary()

    # Must be a string
    assert isinstance(result, str), f"Expected str, got {type(result)}"

    parsed = json.loads(result)
    assert parsed["status"] == "success"
    assert parsed["data"]["total_calories"] == 400.0
    assert parsed["data"]["calorie_budget"] == 1800.0
    assert parsed["data"]["remaining_calories"] == 1400.0
    assert parsed["data"]["meal_count"] == 1
    assert len(parsed["data"]["meals_logged"]) == 1

    # Tool history recorded
    assert env._tool_calls_count == 1
    assert env._tool_history[0]["tool_name"] == "get_today_summary"
    assert env._tool_history[0]["success"] is True

    print("✅ test_env_get_today_summary")


def test_env_get_history():
    """get_history renders from env_state with trends and goal adherence."""
    from src.training.grpo.trl_env_factory import NutriMindToolEnv

    env = NutriMindToolEnv()
    env.reset(prompt="test", env_state=json.dumps(_make_env_state()), tier="T2", query="my week")

    result = env.get_history(days=4, metric="all", compare_to_goal=True)

    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["status"] == "success"
    assert parsed["data"]["days_with_data"] == 4
    assert "daily_averages" in parsed["data"]
    assert "goal_adherence" in parsed["data"]
    assert "calories" in parsed["data"]["goal_adherence"]

    print("✅ test_env_get_history")


def test_env_get_history_empty():
    """get_history with no meal_history returns error."""
    from src.training.grpo.trl_env_factory import NutriMindToolEnv

    env = NutriMindToolEnv()
    env.reset(prompt="test", env_state=json.dumps({"meal_history": []}), tier="T1", query="my week")

    result = env.get_history(days=7)
    parsed = json.loads(result)
    assert parsed["status"] == "error"
    assert parsed["error_type"] == "no_data_in_range"

    print("✅ test_env_get_history_empty")


def test_env_set_goal_mock():
    """set_goal returns mock success without touching DB."""
    from src.training.grpo.trl_env_factory import NutriMindToolEnv

    env = NutriMindToolEnv()
    env.reset(prompt="test", env_state=json.dumps(_make_env_state()), tier="T1", query="set protein")

    result = env.set_goal(metric="protein", target_value=150.0, goal_type="gain")

    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["status"] == "success"
    assert parsed["data"]["metric"] == "protein"
    assert parsed["data"]["previous_value"] == 120  # From env_state
    assert parsed["data"]["new_value"] == 150.0

    print("✅ test_env_set_goal_mock")


def test_env_set_goal_invalid():
    """set_goal with invalid metric returns error."""
    from src.training.grpo.trl_env_factory import NutriMindToolEnv

    env = NutriMindToolEnv()
    env.reset(prompt="test", env_state="{}", tier="T1", query="set goal")

    result = env.set_goal(metric="sugar", target_value=50.0)
    parsed = json.loads(result)
    assert parsed["status"] == "error"

    print("✅ test_env_set_goal_invalid")


def test_env_log_meal_mock():
    """log_meal returns mock success with realistic calories."""
    from src.training.grpo.trl_env_factory import NutriMindToolEnv

    mock_nutrition = {
        "status": "success",
        "data": {"total": {"calories_kcal": 250.0, "protein_g": 30.0, "fat_g": 8.0, "carbs_g": 0.0, "fiber_g": 0.0}},
    }

    env = NutriMindToolEnv()
    env.reset(prompt="test", env_state="{}", tier="T2", query="log lunch")

    # Mock at the point of use inside the _mock closure. The closure does
    # `from src.tools.get_food_nutrition import get_food_nutrition` which
    # requires the full tool chain (DB, RAG, loguru). We intercept it via
    # sys.modules so the import itself returns our mock module.
    fake_module = MagicMock()
    fake_module.get_food_nutrition = MagicMock(return_value=mock_nutrition)
    with patch.dict("sys.modules", {"src.tools.get_food_nutrition": fake_module}):
        result = env.log_meal(meal_type="lunch", foods=[{"food_name": "chicken", "amount_grams": 150}])

    parsed = json.loads(result)
    assert parsed["status"] == "success"
    assert parsed["total_calories"] == 250.0
    assert "train_" in parsed["meal_id"]
    assert env._tool_calls_count == 1

    print("✅ test_env_log_meal_mock")


def test_env_cache_determinism():
    """Same tool call in same group returns identical results across instances."""
    from src.training.grpo.trl_env_factory import NutriMindToolEnv

    state = json.dumps(_make_env_state())
    prompt = [{"role": "user", "content": "check today"}]

    env1 = NutriMindToolEnv()
    env1.reset(prompt=prompt, env_state=state, tier="T1", query="check today")
    r1 = env1.get_today_summary()

    env2 = NutriMindToolEnv()
    env2.reset(prompt=prompt, env_state=state, tier="T1", query="check today")
    r2 = env2.get_today_summary()

    assert r1 == r2, "Same prompt group should return identical results"

    print("✅ test_env_cache_determinism")


# ============================================================================
# 3. Reward tests
# ============================================================================

def test_reward_from_env_basic():
    """reward_from_env builds trajectory from env state and scores it."""
    from src.training.grpo.trl_env_factory import NutriMindToolEnv
    from src.training.grpo.reward import reward_from_env

    # Force v2 for this test (v3 needs group-level scoring)
    old_ver = os.environ.get("NUTRIMIND_REWARD_VERSION")
    os.environ["NUTRIMIND_REWARD_VERSION"] = "v2"

    env = NutriMindToolEnv()
    env.reset(prompt="test", env_state=json.dumps(_make_env_state()), tier="T1", query="protein in chicken")

    # Simulate a tool call recorded in env
    env._tool_history = [{
        "tool_name": "get_food_nutrition",
        "args": {"foods": [{"food_name": "chicken breast", "amount_grams": 100}]},
        "result": {
            "status": "success",
            "data": {
                "total": {"calories_kcal": 165.0, "protein_g": 31.0, "fat_g": 3.6, "carbs_g": 0.0, "fiber_g": 0.0},
            },
        },
        "success": True,
    }]
    env._tool_calls_count = 1

    # Fake completion with final answer referencing the tool result
    completion = "Chicken breast (100g) has 165 calories and 31g of protein."

    scores = reward_from_env(
        environments=[env],
        completions=[completion],
        tier=["T1"],
        query=["protein in chicken"],
        difficulty=["easy"],
        optimal_steps=[1],
        branch_condition=[""],
    )

    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0
    assert scores[0] > 0.3, f"Expected decent score for correct T1 tool use, got {scores[0]}"

    # Restore env var
    if old_ver is None:
        os.environ.pop("NUTRIMIND_REWARD_VERSION", None)
    else:
        os.environ["NUTRIMIND_REWARD_VERSION"] = old_ver

    print(f"✅ test_reward_from_env_basic (score={scores[0]:.3f})")


def test_reward_from_env_no_tools():
    """T1 query with no tool calls should get low reward (hard gate)."""
    from src.training.grpo.trl_env_factory import NutriMindToolEnv
    from src.training.grpo.reward import reward_from_env

    old_ver = os.environ.get("NUTRIMIND_REWARD_VERSION")
    os.environ["NUTRIMIND_REWARD_VERSION"] = "v2"

    env = NutriMindToolEnv()
    env.reset(prompt="test", env_state="{}", tier="T1", query="calories in rice")
    # No tool calls — empty history

    scores = reward_from_env(
        environments=[env],
        completions=["Rice has about 130 calories per 100g."],
        tier=["T1"],
        query=["calories in rice"],
        difficulty=["easy"],
        optimal_steps=[1],
        branch_condition=[""],
    )

    assert scores[0] == 0.0, f"T1 with no tool calls should get 0.0 (hard gate), got {scores[0]}"

    if old_ver is None:
        os.environ.pop("NUTRIMIND_REWARD_VERSION", None)
    else:
        os.environ["NUTRIMIND_REWARD_VERSION"] = old_ver

    print("✅ test_reward_from_env_no_tools")


def test_reward_from_env_t4_safety():
    """T4 safety query should reward safety declaration."""
    from src.training.grpo.trl_env_factory import NutriMindToolEnv
    from src.training.grpo.reward import reward_from_env

    old_ver = os.environ.get("NUTRIMIND_REWARD_VERSION")
    os.environ["NUTRIMIND_REWARD_VERSION"] = "v2"

    env = NutriMindToolEnv()
    env.reset(prompt="test", env_state="{}", tier="T4", query="I have kidney disease")

    scores = reward_from_env(
        environments=[env],
        completions=["Your situation involves complex medical nutrition. Please consult your physician or a registered dietitian."],
        tier=["T4"],
        query=["I have kidney disease"],
        difficulty=["hard"],
        optimal_steps=[0],
        branch_condition=[""],
    )

    assert scores[0] > 0.5, f"T4 with safety declaration should score well, got {scores[0]}"

    if old_ver is None:
        os.environ.pop("NUTRIMIND_REWARD_VERSION", None)
    else:
        os.environ["NUTRIMIND_REWARD_VERSION"] = old_ver

    print(f"✅ test_reward_from_env_t4_safety (score={scores[0]:.3f})")


def test_extract_final_answer_conversational():
    """Test final answer extraction from TRL conversational format."""
    from src.training.grpo.reward import _extract_final_answer_from_completion

    # Conversational format: list of message dicts
    conv = [
        {"role": "assistant", "content": '<tool_call>{"name": "get_food_nutrition", "arguments": {}}</tool_call>'},
        {"role": "tool", "content": '{"status": "success"}'},
        {"role": "assistant", "content": "Chicken breast has 165 calories per 100g."},
    ]
    answer = _extract_final_answer_from_completion(conv)
    assert answer == "Chicken breast has 165 calories per 100g."

    # String format
    text = '<tool_call>{"name":"foo"}</tool_call>\n<tool_response>{"ok":true}</tool_response>\nThe answer is 42.'
    answer = _extract_final_answer_from_completion(text)
    assert "42" in answer

    # No final answer (ends with tool call)
    conv_no_answer = [
        {"role": "assistant", "content": '<tool_call>{"name": "foo"}</tool_call>'},
    ]
    answer = _extract_final_answer_from_completion(conv_no_answer)
    assert answer is None

    print("✅ test_extract_final_answer_conversational")


# ============================================================================
# 4. Integration: type hints check for TRL schema generation
# ============================================================================

def test_tool_method_signatures():
    """All public tool methods have proper type hints and docstrings."""
    import inspect
    from src.training.grpo.trl_env_factory import NutriMindToolEnv

    env = NutriMindToolEnv()
    public_methods = [
        name for name, method in inspect.getmembers(env, predicate=inspect.ismethod)
        if not name.startswith("_") and name != "reset"
    ]

    expected_tools = {"get_food_nutrition", "log_meal", "get_today_summary",
                      "get_history", "retrieve_knowledge", "set_goal"}

    assert set(public_methods) == expected_tools, (
        f"Public methods mismatch.\nExpected: {expected_tools}\nGot: {set(public_methods)}"
    )

    for name in public_methods:
        method = getattr(env, name)
        sig = inspect.signature(method)
        hints = method.__func__.__annotations__

        # Must have return type annotation
        assert "return" in hints, f"{name} missing return type hint"
        assert hints["return"] == str, f"{name} return type should be str, got {hints['return']}"

        # Must have docstring
        assert method.__doc__, f"{name} missing docstring"
        assert "Args:" in method.__doc__ or "Returns:" in method.__doc__, (
            f"{name} docstring missing Args/Returns section"
        )

    print(f"✅ test_tool_method_signatures ({len(public_methods)} tools verified)")


# ============================================================================
# 5. Reward v3 (RULER-style LLM Judge) tests
# ============================================================================

def test_judge_parse_scores_array():
    """GroupJudge._parse_scores handles legacy array format."""
    from src.training.grpo.reward import GroupJudge

    judge = GroupJudge()
    # Legacy array format
    result = judge._parse_scores('{"scores": [7, 5, 8, 3], "reasoning": "test"}', 4)
    assert result is not None
    scores, dims = result
    assert len(scores) == 4
    assert abs(scores[0] - 6/9) < 0.01  # (7-1)/9
    assert abs(scores[2] - 7/9) < 0.01  # (8-1)/9
    # Legacy format has empty dimension details
    assert dims == [{}] * 4

    print("✅ test_judge_parse_scores_array")


def test_judge_parse_scores_dict():
    """GroupJudge._parse_scores handles legacy dict format."""
    from src.training.grpo.reward import GroupJudge

    judge = GroupJudge()
    result = judge._parse_scores(
        '{"scores": {"score_1": 8, "score_2": 4, "score_3": 6}, "reasoning": "ok"}', 3
    )
    assert result is not None
    scores, dims = result
    assert len(scores) == 3
    assert scores[0] > scores[1]  # score_1 (8) > score_2 (4)

    print("✅ test_judge_parse_scores_dict")


def test_judge_parse_multidimensional():
    """GroupJudge._parse_scores handles multi-dimensional candidate format."""
    from src.training.grpo.reward import GroupJudge, JUDGE_DIMENSION_WEIGHTS

    judge = GroupJudge()
    raw = json.dumps({"candidates": [
        {"id": 1, "accuracy": 9, "helpfulness": 8, "tool_use": 7, "communication": 6},
        {"id": 2, "accuracy": 4, "helpfulness": 5, "tool_use": 3, "communication": 4},
    ]})
    result = judge._parse_scores(raw, 2)
    assert result is not None
    scores, dims = result
    assert len(scores) == 2
    assert len(dims) == 2
    # First candidate has higher scores → higher composite
    assert scores[0] > scores[1]
    # Dimension details are present
    assert dims[0]["accuracy"] == 9.0
    assert dims[0]["helpfulness"] == 8.0
    assert dims[1]["tool_use"] == 3.0
    # Verify weighted calculation for candidate 1
    expected = (
        JUDGE_DIMENSION_WEIGHTS["accuracy"] * (9 - 1) / 9 +
        JUDGE_DIMENSION_WEIGHTS["helpfulness"] * (8 - 1) / 9 +
        JUDGE_DIMENSION_WEIGHTS["tool_use"] * (7 - 1) / 9 +
        JUDGE_DIMENSION_WEIGHTS["communication"] * (6 - 1) / 9
    )
    assert abs(scores[0] - expected) < 0.001, f"Expected {expected:.4f}, got {scores[0]:.4f}"

    print("✅ test_judge_parse_multidimensional")


def test_judge_parse_scores_markdown_wrapped():
    """GroupJudge._parse_scores handles markdown code block wrapping."""
    from src.training.grpo.reward import GroupJudge

    judge = GroupJudge()
    raw = '```json\n{"scores": [9, 3, 7, 5]}\n```'
    result = judge._parse_scores(raw, 4)
    assert result is not None
    scores, _ = result
    assert len(scores) == 4
    assert scores[0] > scores[1]  # 9 > 3

    print("✅ test_judge_parse_scores_markdown_wrapped")


def test_judge_parse_scores_invalid():
    """GroupJudge._parse_scores returns None on invalid input."""
    from src.training.grpo.reward import GroupJudge

    judge = GroupJudge()
    assert judge._parse_scores("not json", 4) is None
    assert judge._parse_scores('{"scores": [1, 2]}', 4) is None  # Wrong count
    assert judge._parse_scores('{"no_scores": true}', 4) is None
    assert judge._parse_scores('{"scores": "bad"}', 4) is None
    # Multi-dim missing a dimension
    bad_dim = json.dumps({"candidates": [
        {"id": 1, "accuracy": 8, "helpfulness": 7},  # missing tool_use, communication
    ]})
    assert judge._parse_scores(bad_dim, 1) is None

    print("✅ test_judge_parse_scores_invalid")


def test_judge_parse_scores_clamping():
    """Scores outside [1,10] are clamped."""
    from src.training.grpo.reward import GroupJudge

    judge = GroupJudge()
    result = judge._parse_scores('{"scores": [0, 15, 5, -3]}', 4)
    assert result is not None
    scores, _ = result
    # 0 → clamped to 1 → normalized to 0.0
    assert scores[0] == 0.0
    # 15 → clamped to 10 → normalized to 1.0
    assert scores[1] == 1.0

    print("✅ test_judge_parse_scores_clamping")


def test_judge_build_candidates_block():
    """Candidates block formats trajectories correctly."""
    from src.training.grpo.reward import GroupJudge
    from src.training.grpo.environment import RolloutTrajectory, RolloutStep, ToolExecutionResult

    judge = GroupJudge()

    traj1 = RolloutTrajectory(prompt="test")
    traj1.steps.append(RolloutStep(
        step_idx=0, model_output="", think_content=None, action_type="tool_call",
        tool_execution=ToolExecutionResult(
            tool_name="get_food_nutrition", tool_args={}, result={}, success=True,
        ),
    ))
    traj1.final_answer = "Chicken has 165 calories."
    traj1.terminated = True
    traj1.termination_reason = "final_answer"

    traj2 = RolloutTrajectory(prompt="test")
    traj2.terminated = True
    traj2.termination_reason = "max_tokens"

    block = judge._build_candidates_block([traj1, traj2])
    assert "Candidate 1" in block
    assert "Candidate 2" in block
    assert "get_food_nutrition" in block
    assert "TRUNCATED" in block

    print("✅ test_judge_build_candidates_block")


def test_reward_v3_group_without_judge():
    """reward_v3_group with mock judge that returns uniform scores falls back to v2."""
    from src.training.grpo.reward import reward_v3_group, GroupJudge
    from src.training.grpo.environment import (
        RolloutTrajectory, TaskMetadata, RolloutStep, ToolExecutionResult,
    )

    # Create a mock judge that returns uniform scores (simulating failure)
    class MockJudge(GroupJudge):
        def score_group(self, trajectories, task_metadata):
            n = len(trajectories)
            return [0.5] * n, [{}] * n

    # Build 4 trajectories with varying quality
    meta = TaskMetadata(query="protein in chicken", tier="T1", optimal_steps=1)
    trajectories = []
    for i in range(4):
        t = RolloutTrajectory(prompt="protein in chicken")
        if i < 3:  # First 3 have tool calls
            t.steps.append(RolloutStep(
                step_idx=0, model_output="", think_content=None, action_type="tool_call",
                tool_execution=ToolExecutionResult(
                    tool_name="get_food_nutrition",
                    tool_args={"foods": [{"food_name": "chicken"}]},
                    result={"status": "success", "data": {
                        "calories_kcal": 165.0, "protein_g": 31.0,
                    }},
                    success=True,
                ),
            ))
            t.total_tool_calls = 1
            t.final_answer = f"Chicken has 165 calories and 31g protein." if i < 2 else "Chicken is food."
            t.terminated = True
            t.termination_reason = "final_answer"
        else:  # Last one has no tool calls
            t.final_answer = "I think chicken has protein."
            t.terminated = True
            t.termination_reason = "final_answer"
        trajectories.append(t)

    breakdowns = reward_v3_group(trajectories, meta, judge=MockJudge())

    assert len(breakdowns) == 4
    # When judge returns uniform, should be pure v2 scores
    # The last trajectory (no tools, T1) should get 0.0 (hard gate)
    assert breakdowns[3].total == 0.0
    # First two should score decently
    assert breakdowns[0].total > 0.3
    # Judge was effectively not applied
    assert breakdowns[0].details["judge_applied"] is False

    print("✅ test_reward_v3_group_without_judge")


def test_reward_v3_group_with_variance():
    """reward_v3_group with judge that produces variance uses hybrid scoring."""
    from src.training.grpo.reward import reward_v3_group, GroupJudge
    from src.training.grpo.environment import (
        RolloutTrajectory, TaskMetadata, RolloutStep, ToolExecutionResult,
    )

    class MockVariantJudge(GroupJudge):
        def score_group(self, trajectories, task_metadata):
            scores = [0.9, 0.3, 0.7, 0.1]
            dims = [
                {"accuracy": 9, "helpfulness": 9, "tool_use": 8, "communication": 8},
                {"accuracy": 3, "helpfulness": 4, "tool_use": 3, "communication": 2},
                {"accuracy": 7, "helpfulness": 7, "tool_use": 6, "communication": 7},
                {"accuracy": 2, "helpfulness": 1, "tool_use": 1, "communication": 1},
            ]
            return scores, dims

    meta = TaskMetadata(query="protein in chicken", tier="T1", optimal_steps=1)
    trajectories = []
    for i in range(4):
        t = RolloutTrajectory(prompt="protein in chicken")
        t.steps.append(RolloutStep(
            step_idx=0, model_output="", think_content=None, action_type="tool_call",
            tool_execution=ToolExecutionResult(
                tool_name="get_food_nutrition",
                tool_args={"foods": [{"food_name": "chicken"}]},
                result={"status": "success", "data": {"calories_kcal": 165.0}},
                success=True,
            ),
        ))
        t.total_tool_calls = 1
        t.final_answer = f"Answer {i}: Chicken has 165 calories."
        t.terminated = True
        t.termination_reason = "final_answer"
        trajectories.append(t)

    breakdowns = reward_v3_group(trajectories, meta, judge=MockVariantJudge())

    assert len(breakdowns) == 4
    # Judge was applied
    assert breakdowns[0].details["judge_applied"] is True
    # Scores should vary (not all the same) due to judge variance
    scores = [b.total for b in breakdowns]
    assert len(set(round(s, 4) for s in scores)) > 1, f"Expected variance, got {scores}"
    # First should be highest (judge=0.9)
    assert scores[0] > scores[3]  # 0.9 judge > 0.1 judge
    # Dimension details should be in breakdown
    assert "judge_dimensions" in breakdowns[0].details
    assert breakdowns[0].details["judge_dimensions"]["accuracy"] == 9

    print(f"✅ test_reward_v3_group_with_variance (scores={[f'{s:.3f}' for s in scores]})")


def test_reward_from_env_v3_integration():
    """reward_from_env dispatches to v3 when NUTRIMIND_REWARD_VERSION=v3."""
    from src.training.grpo.trl_env_factory import NutriMindToolEnv
    from src.training.grpo.reward import reward_from_env, _default_judge, GroupJudge
    import src.training.grpo.reward as reward_module

    # Save and set env var
    old_version = os.environ.get("NUTRIMIND_REWARD_VERSION")
    os.environ["NUTRIMIND_REWARD_VERSION"] = "v3"

    # Mock the judge to avoid real API calls
    class MockJudge(GroupJudge):
        def score_group(self, trajectories, task_metadata):
            n = len(trajectories)
            # Return descending scores with dimension details
            scores = [1.0 - i * 0.2 for i in range(n)]
            dims = [
                {"accuracy": 10 - i*2, "helpfulness": 9 - i*2,
                 "tool_use": 8 - i*2, "communication": 7 - i*2}
                for i in range(n)
            ]
            return scores, dims

    old_judge = reward_module._default_judge
    reward_module._default_judge = MockJudge()

    try:
        state = json.dumps(_make_env_state())
        envs = []
        completions = []
        for i in range(4):
            env = NutriMindToolEnv()
            env.reset(
                prompt=[{"role": "user", "content": "check my calories"}],
                env_state=state,
                tier="T2",
                query="check my calories",
                optimal_steps=1,
            )
            env.get_today_summary()
            envs.append(env)
            completions.append([
                {"role": "assistant", "content": f"Answer {i}: You consumed 400 calories today."},
            ])

        scores = reward_from_env(
            environments=envs,
            completions=completions,
            tier=["T2"] * 4,
            query=["check my calories"] * 4,
            difficulty=["medium"] * 4,
            optimal_steps=[1] * 4,
            branch_condition=[""] * 4,
        )

        assert len(scores) == 4
        # Should have variance from judge
        assert len(set(round(s, 4) for s in scores)) > 1
        # First should score highest
        assert scores[0] >= scores[3]

        print(f"✅ test_reward_from_env_v3_integration (scores={[f'{s:.3f}' for s in scores]})")
    finally:
        # Restore
        reward_module._default_judge = old_judge
        if old_version is None:
            os.environ.pop("NUTRIMIND_REWARD_VERSION", None)
        else:
            os.environ["NUTRIMIND_REWARD_VERSION"] = old_version


# ============================================================================
# Runner
# ============================================================================

if __name__ == "__main__":
    tests = [
        # Cache
        test_cache_basic_hit_miss,
        test_cache_normalization,
        test_cache_group_flush,
        test_cache_snapshot_restore,
        # Env
        test_env_reset,
        test_env_get_today_summary,
        test_env_get_history,
        test_env_get_history_empty,
        test_env_set_goal_mock,
        test_env_set_goal_invalid,
        test_env_log_meal_mock,
        test_env_cache_determinism,
        # Reward v2
        test_reward_from_env_basic,
        test_reward_from_env_no_tools,
        test_reward_from_env_t4_safety,
        test_extract_final_answer_conversational,
        # Reward v3 (RULER-style judge)
        test_judge_parse_scores_array,
        test_judge_parse_scores_dict,
        test_judge_parse_multidimensional,
        test_judge_parse_scores_markdown_wrapped,
        test_judge_parse_scores_invalid,
        test_judge_parse_scores_clamping,
        test_judge_build_candidates_block,
        test_reward_v3_group_without_judge,
        test_reward_v3_group_with_variance,
        test_reward_from_env_v3_integration,
        # Signatures
        test_tool_method_signatures,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
    print("All tests passed! ✅")
