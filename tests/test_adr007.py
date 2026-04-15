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

    print(f"✅ test_reward_from_env_basic (score={scores[0]:.3f})")


def test_reward_from_env_no_tools():
    """T1 query with no tool calls should get low reward (hard gate)."""
    from src.training.grpo.trl_env_factory import NutriMindToolEnv
    from src.training.grpo.reward import reward_from_env

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

    print("✅ test_reward_from_env_no_tools")


def test_reward_from_env_t4_safety():
    """T4 safety query should reward safety declaration."""
    from src.training.grpo.trl_env_factory import NutriMindToolEnv
    from src.training.grpo.reward import reward_from_env

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
        # Reward
        test_reward_from_env_basic,
        test_reward_from_env_no_tools,
        test_reward_from_env_t4_safety,
        test_extract_final_answer_conversational,
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
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
    print("All tests passed! ✅")
