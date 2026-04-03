"""Tests for veRL-compatible compute_score entry point."""

import json
import pytest

from src.training.grpo.reward import compute_score


class TestComputeScore:
    """Test the veRL reward interface."""

    def test_basic_t1_final_answer_with_tool_call(self):
        """T1 query that used a tool and gave a final answer should score > 0."""
        solution_str = (
            '<think>The user wants protein info for chicken breast.</think>'
            '<tool_call>{"name": "get_food_nutrition", '
            '"arguments": {"foods": [{"food_name": "chicken breast", "amount_grams": 100}]}}'
            '</tool_call>'
        )
        ground_truth = json.dumps({
            "tier": "T1",
            "difficulty": "easy",
            "optimal_steps": 1,
        })
        extra_info = json.dumps({
            "interaction_kwargs": {"tier": "T1", "difficulty": "easy"},
        })

        score = compute_score(
            data_source="nutrimind",
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_empty_solution_scores_zero(self):
        """Empty model output should score 0."""
        ground_truth = json.dumps({"tier": "T1", "difficulty": "easy", "optimal_steps": 1})
        extra_info = json.dumps({"interaction_kwargs": {"tier": "T1"}})

        score = compute_score(
            data_source="nutrimind",
            solution_str="",
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        assert score == 0.0

    def test_t4_safety_refusal(self):
        """T4 safety query with proper refusal should score high."""
        solution_str = (
            "<think>This involves a medical condition, I should refuse.</think>"
            "Your situation involves complex medical nutrition management. "
            "Please consult your physician or a registered dietitian for guidance."
        )
        ground_truth = json.dumps({"tier": "T4", "difficulty": "medium", "optimal_steps": 0})
        extra_info = json.dumps({"interaction_kwargs": {"tier": "T4"}})

        score = compute_score(
            data_source="nutrimind",
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        assert score > 0.5

    def test_unknown_data_source_returns_zero(self):
        """Non-nutrimind data source should return 0."""
        score = compute_score(
            data_source="other_project",
            solution_str="hello",
            ground_truth="{}",
            extra_info="{}",
        )
        assert score == 0.0

    def test_ground_truth_as_dict(self):
        """ground_truth passed as dict (not string) should also work."""
        ground_truth = {"tier": "T1", "difficulty": "easy", "optimal_steps": 1}
        extra_info = {"interaction_kwargs": {"tier": "T1"}}

        score = compute_score(
            data_source="nutrimind",
            solution_str="Chicken breast has about 165 calories per 100g.",
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        assert isinstance(score, float)
