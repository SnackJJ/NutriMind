"""Tests for NutriMindEnv._execute_tool edge cases."""

import pytest
from src.training.grpo.environment import NutriMindEnv, ToolExecutionResult


def _make_env(snapshot=None, tool_registry=None):
    """Create a NutriMindEnv with mock tool registry."""
    if tool_registry is None:
        tool_registry = {
            "get_food_nutrition": lambda **kwargs: {
                "status": "success",
                "data": {"calories_kcal": 165},
            },
            "get_today_summary": lambda: {
                "status": "success",
                "data": {"total_calories": 0},
            },
            "retrieve_knowledge": lambda **kwargs: {
                "status": "success",
                "data": {"chunks": []},
            },
        }
    return NutriMindEnv(
        tool_registry=tool_registry,
        user_state_snapshot=snapshot,
    )


class TestExecuteTool:
    def test_stateless_tool_with_snapshot_present(self):
        """get_food_nutrition should work even when snapshot is set."""
        env = _make_env(snapshot={"meals_today": []})
        result = env._execute_tool(
            "get_food_nutrition",
            {"foods": [{"food_name": "chicken", "amount_grams": 100}]},
        )
        assert result.success is True
        assert result.result["status"] == "success"

    def test_stateless_tool_with_empty_snapshot(self):
        """get_food_nutrition should work when snapshot is empty dict."""
        env = _make_env(snapshot={})
        result = env._execute_tool(
            "get_food_nutrition",
            {"foods": [{"food_name": "rice", "amount_grams": 200}]},
        )
        assert result.success is True

    def test_get_today_summary_with_snapshot(self):
        """get_today_summary with snapshot should use mocked data."""
        env = _make_env(snapshot={
            "meals_today": [{"calories": 500, "protein_g": 30, "fat_g": 20, "carbs_g": 50, "fiber_g": 5, "foods": [{"name": "chicken"}]}],
            "user_goals": {"calories": 2000},
        })
        result = env._execute_tool("get_today_summary", {})
        assert result.success is True
        assert result.result["data"]["total_calories"] == 500.0

    def test_get_today_summary_without_snapshot(self):
        """get_today_summary without snapshot should call real function."""
        env = _make_env(snapshot=None)
        result = env._execute_tool("get_today_summary", {})
        assert result.success is True

    def test_unknown_tool(self):
        """Unknown tool should return error."""
        env = _make_env()
        result = env._execute_tool("nonexistent_tool", {})
        assert result.success is False
        assert "Unknown tool" in result.error_message
