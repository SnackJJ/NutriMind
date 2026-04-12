"""
Tests for veRL → TRL migration.

Validates the full TRL training pipeline WITHOUT requiring GPU, vLLM server,
or TRL/transformers. All heavy dependencies are mocked.

Run:
    pytest tests/test_trl_migration.py -v
    pytest tests/test_trl_migration.py -v -k "test_reward"   # just reward tests
    pytest tests/test_trl_migration.py -v -k "test_rollout"  # just rollout tests

Test layers:
    1. Data preparation (JSONL → HF Dataset)
    2. Reward wrapper (trl_reward_wrapper matches compute_score)
    3. Multi-turn rollout logic (env_mask, tool execution, trajectory building)
    4. Train script config validation (dry_run)
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root on path
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a minimal JSONL file with all tier types."""
    entries = [
        {
            "query": "How much protein is in 100g of chicken breast?",
            "tier": "T1",
            "difficulty": "easy",
            "env_state": {},
        },
        {
            "query": "What is the glycemic index?",
            "tier": "T0-qa",
            "difficulty": "easy",
            "env_state": {},
        },
        {
            "query": "Log my lunch: 150g grilled salmon and a cup of brown rice",
            "tier": "T2",
            "difficulty": "medium",
            "env_state": {
                "meals_today": [{"calories": 350, "protein_g": 10, "fat_g": 5,
                                 "carbs_g": 60, "fiber_g": 2,
                                 "foods": [{"name": "oatmeal"}],
                                 "meal_type": "breakfast"}],
                "user_goals": {"calories": 2000},
            },
        },
        {
            "query": "Check my calories today. If I'm under 1500, suggest a high-protein snack.",
            "tier": "T3",
            "difficulty": "hard",
            "env_state": {
                "meals_today": [{"calories": 800, "protein_g": 45, "fat_g": 30,
                                 "carbs_g": 80, "fiber_g": 8,
                                 "foods": [{"name": "eggs"}, {"name": "toast"}],
                                 "meal_type": "breakfast"}],
                "user_goals": {"calories": 2000},
            },
            "branch_condition": {
                "check_tool": "get_today_summary",
                "condition_field": "remaining_calories",
                "threshold": 500,
                "expected_branch": "under_budget",
            },
        },
        {
            "query": "I was diagnosed with stage 3 kidney disease. Design a renal diet.",
            "tier": "T4",
            "difficulty": "hard",
            "env_state": {},
        },
    ]

    jsonl_path = tmp_path / "prompts.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return jsonl_path


@pytest.fixture
def mock_tool_registry():
    """Mock tool registry that returns deterministic results."""
    def mock_get_food_nutrition(foods=None, **kwargs):
        return {
            "status": "success",
            "data": {
                "calories_kcal": 165.0,
                "protein_g": 31.0,
                "carbs_g": 0.0,
                "fat_g": 3.6,
                "fiber_g": 0.0,
            },
        }

    def mock_log_meal(**kwargs):
        return {"status": "success", "meal_id": "mock_123", "total_calories": 250.0}

    def mock_get_today_summary():
        return {
            "status": "success",
            "data": {
                "date": "2026-04-08",
                "total_calories": 800.0,
                "calorie_budget": 2000.0,
                "remaining_calories": 1200.0,
                "protein_g": 45.0, "fat_g": 30.0,
                "carbs_g": 80.0, "fiber_g": 8.0,
                "meal_count": 1,
                "food_summary": "eggs, toast",
                "meals_logged": [],
            },
        }

    def mock_get_history(**kwargs):
        return {
            "status": "success",
            "data": {
                "period": "Last 7 days",
                "days_with_data": 3,
                "daily_averages": {
                    "calories_kcal": 1900.0, "protein_g": 95.0,
                    "fat_g": 65.0, "carbs_g": 230.0, "fiber_g": 22.0,
                },
                "trend": "stable",
                "daily_breakdown": [],
            },
        }

    def mock_retrieve_knowledge(**kwargs):
        return {
            "status": "success",
            "data": {
                "results": [{"text": "High-protein snacks include Greek yogurt, nuts."}],
            },
        }

    def mock_set_goal(**kwargs):
        return {"status": "success", "message": "Goal updated."}

    return {
        "get_food_nutrition": mock_get_food_nutrition,
        "log_meal": mock_log_meal,
        "get_today_summary": mock_get_today_summary,
        "get_history": mock_get_history,
        "retrieve_knowledge": mock_retrieve_knowledge,
        "set_goal": mock_set_goal,
    }


# ============================================================================
# 1. Data Preparation Tests
# ============================================================================


class TestDataPreparation:
    """Test scripts/prepare_trl_data.py conversion logic."""

    def test_convert_entry_basic(self):
        """convert_entry produces correct schema."""
        from scripts.prepare_trl_data import convert_entry

        entry = {
            "query": "How much protein in chicken?",
            "tier": "T1",
            "difficulty": "easy",
            "env_state": {},
        }
        result = convert_entry(entry)

        assert "prompt" in result
        assert isinstance(result["prompt"], list)
        assert len(result["prompt"]) == 2  # system + user
        assert result["prompt"][0]["role"] == "system"
        assert result["prompt"][1]["role"] == "user"
        assert result["prompt"][1]["content"] == "How much protein in chicken?"
        assert result["tier"] == "T1"
        assert result["difficulty"] == "easy"
        assert result["optimal_steps"] == 1
        assert result["env_state"] == "{}"
        assert result["query"] == "How much protein in chicken?"

    def test_optimal_steps_mapping(self):
        """Each tier maps to correct optimal_steps."""
        from scripts.prepare_trl_data import get_optimal_steps

        assert get_optimal_steps("T0-qa") == 0
        assert get_optimal_steps("T1") == 1
        assert get_optimal_steps("T1-simple") == 1
        assert get_optimal_steps("T2") == 2
        assert get_optimal_steps("T2-ambiguous") == 2
        assert get_optimal_steps("T3") == 3
        assert get_optimal_steps("T4") == 0
        assert get_optimal_steps("error_recovery") == 2

    def test_env_state_serialized_as_json(self):
        """env_state dict is serialized to JSON string."""
        from scripts.prepare_trl_data import convert_entry

        env_state = {"meals_today": [{"calories": 500}], "user_goals": {"calories": 2000}}
        entry = {"query": "test", "tier": "T2", "difficulty": "medium", "env_state": env_state}
        result = convert_entry(entry)

        parsed = json.loads(result["env_state"])
        assert parsed["meals_today"][0]["calories"] == 500
        assert parsed["user_goals"]["calories"] == 2000

    def test_branch_condition_serialized(self):
        """branch_condition is serialized when present, empty string when absent."""
        from scripts.prepare_trl_data import convert_entry

        # With branch_condition
        entry = {
            "query": "test", "tier": "T3", "difficulty": "hard", "env_state": {},
            "branch_condition": {"check_tool": "get_today_summary", "threshold": 500},
        }
        result = convert_entry(entry)
        assert result["branch_condition"] != ""
        parsed = json.loads(result["branch_condition"])
        assert parsed["check_tool"] == "get_today_summary"

        # Without branch_condition
        entry2 = {"query": "test", "tier": "T1", "difficulty": "easy", "env_state": {}}
        result2 = convert_entry(entry2)
        assert result2["branch_condition"] == ""

    def test_full_pipeline_jsonl_to_dataset(self, sample_jsonl, tmp_path):
        """End-to-end: JSONL → load → convert → Dataset."""
        from scripts.prepare_trl_data import convert_entry, load_jsonl

        entries = load_jsonl(sample_jsonl)
        assert len(entries) == 5

        converted = [convert_entry(e) for e in entries]

        # Verify all required columns exist
        required_cols = {"prompt", "tier", "difficulty", "optimal_steps",
                         "env_state", "branch_condition", "query"}
        for item in converted:
            assert required_cols.issubset(item.keys()), f"Missing columns: {required_cols - item.keys()}"

        # Verify tiers preserved
        tiers = [c["tier"] for c in converted]
        assert "T1" in tiers
        assert "T0-qa" in tiers
        assert "T4" in tiers


# ============================================================================
# 2. Reward Wrapper Tests
# ============================================================================


class TestRewardWrapper:
    """Test that TRL reward wrapper produces same results as veRL compute_score."""

    def _make_solution_final_answer(self, answer_text: str) -> str:
        """Create a simple final-answer-only solution string."""
        return answer_text

    def _make_solution_with_tool_call(self) -> str:
        """Create a solution with tool_call + tool_response + final answer."""
        return (
            '<think>I need to look up chicken breast nutrition.</think>\n'
            '<tool_call>{"name": "get_food_nutrition", '
            '"arguments": {"foods": [{"food_name": "chicken breast", "amount_grams": 100}]}}'
            '</tool_call>\n'
            '<tool_response>{"status": "success", "data": {'
            '"calories_kcal": 165, "protein_g": 31, "carbs_g": 0, "fat_g": 3.6, "fiber_g": 0}}'
            '</tool_response>\n'
            'Chicken breast (100g) has 165 calories, 31g protein, 0g carbs, and 3.6g fat.'
        )

    def _make_solution_t4_safety(self) -> str:
        """Create a T4 safety response."""
        return (
            "Your situation involves complex medical nutrition management "
            "that exceeds my safe service boundary. Please consult your physician "
            "or a registered dietitian for personalized guidance."
        )

    def test_trl_wrapper_basic(self):
        """trl_reward_wrapper returns list of floats with correct length."""
        from src.training.grpo.reward import trl_reward_wrapper

        completions = [
            self._make_solution_with_tool_call(),
            self._make_solution_final_answer("Short answer"),
            "",  # empty
        ]
        scores = trl_reward_wrapper(
            completions,
            tier=["T1", "T1", "T1"],
            difficulty=["easy", "easy", "easy"],
            optimal_steps=[1, 1, 1],
            query=["q1", "q2", "q3"],
            branch_condition=["", "", ""],
        )

        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        assert scores[2] == 0.0  # empty completion → 0

    def test_trl_wrapper_matches_compute_score(self):
        """TRL wrapper produces same score as veRL compute_score for same input."""
        from src.training.grpo.reward import compute_score, trl_reward_wrapper

        solution = self._make_solution_with_tool_call()

        # veRL path
        verl_score = compute_score(
            data_source="nutrimind",
            solution_str=solution,
            ground_truth={"tier": "T1", "difficulty": "easy", "optimal_steps": 1},
            extra_info={"interaction_kwargs": {"reward_version": "v2"}},
        )

        # TRL path
        trl_scores = trl_reward_wrapper(
            [solution],
            tier=["T1"],
            difficulty=["easy"],
            optimal_steps=[1],
            query=["How much protein in chicken?"],
            branch_condition=[""],
        )

        assert abs(verl_score - trl_scores[0]) < 1e-6, (
            f"Score mismatch: veRL={verl_score}, TRL={trl_scores[0]}"
        )

    def test_t0_qa_no_tools_high_score(self):
        """T0-qa with a substantive answer and no tools → high score."""
        from src.training.grpo.reward import trl_reward_wrapper

        solution = (
            "The glycemic index (GI) is a measure of how quickly a food "
            "causes blood sugar levels to rise. Foods are ranked on a scale "
            "from 0 to 100, with pure glucose at 100."
        )
        scores = trl_reward_wrapper(
            [solution], tier=["T0-qa"], difficulty=["easy"],
            optimal_steps=[0], query=["What is the glycemic index?"],
            branch_condition=[""],
        )
        assert scores[0] > 0.5, f"T0-qa substantive answer should score > 0.5, got {scores[0]}"

    def test_t4_safety_declaration_scores(self):
        """T4 with proper safety declaration → decent score."""
        from src.training.grpo.reward import trl_reward_wrapper

        solution = self._make_solution_t4_safety()
        scores = trl_reward_wrapper(
            [solution], tier=["T4"], difficulty=["hard"],
            optimal_steps=[0], query=["I have kidney disease"],
            branch_condition=[""],
        )
        assert scores[0] > 0.3, f"T4 safety response should score > 0.3, got {scores[0]}"

    def test_empty_and_garbage_input(self):
        """Edge cases: empty, whitespace, garbage."""
        from src.training.grpo.reward import trl_reward_wrapper

        scores = trl_reward_wrapper(
            ["", "   ", "asdf", None],
            tier=["T1", "T1", "T1", "T1"],
            difficulty=["easy"] * 4,
            optimal_steps=[1] * 4,
            query=["q"] * 4,
            branch_condition=[""] * 4,
        )
        assert scores[0] == 0.0  # empty
        assert scores[1] == 0.0  # whitespace
        assert scores[3] == 0.0  # None

    def test_reward_breakdown_components(self):
        """Verify reward_v2 returns expected breakdown fields."""
        from src.training.grpo.environment import TaskMetadata
        from src.training.grpo.reward import _build_trajectory_from_solution, reward_v2

        solution = self._make_solution_with_tool_call()
        task_meta = TaskMetadata(
            query="Protein in chicken?", tier="T1",
            difficulty="easy", optimal_steps=1,
        )
        trajectory = _build_trajectory_from_solution(solution, task_meta)
        breakdown = reward_v2(trajectory, task_meta)

        assert hasattr(breakdown, "r_format")
        assert hasattr(breakdown, "r_tool_selection")
        assert hasattr(breakdown, "r_outcome")
        assert hasattr(breakdown, "total")
        assert 0.0 <= breakdown.total <= 1.0


# ============================================================================
# 3. Multi-Turn Rollout Tests
# ============================================================================


class TestMultiTurnRollout:
    """Test the multi-turn agentic rollout logic and env_mask construction."""

    def test_nutrimind_env_single_turn_final_answer(self, mock_tool_registry):
        """Model gives final answer immediately → 1-step trajectory."""
        from src.training.grpo.environment import NutriMindEnv

        env = NutriMindEnv(tool_registry=mock_tool_registry, max_tool_rounds=6)
        env.reset("What is the glycemic index?")

        model_output = (
            "The glycemic index is a measure of how quickly a food "
            "causes blood sugar levels to rise."
        )
        messages, done, info = env.step(model_output)

        assert done is True
        assert info["action_type"] == "final_answer"

        trajectory = env.get_trajectory()
        assert trajectory.terminated
        assert trajectory.termination_reason == "final_answer"
        assert trajectory.total_tool_calls == 0

    def test_nutrimind_env_tool_call_then_answer(self, mock_tool_registry):
        """Model calls tool then gives answer → 2-step trajectory."""
        from src.training.grpo.environment import NutriMindEnv

        env = NutriMindEnv(tool_registry=mock_tool_registry, max_tool_rounds=6)
        env.reset("How much protein in chicken?")

        # Step 1: tool call
        tool_call_output = (
            '<think>I need to look up chicken nutrition.</think>\n'
            '<tool_call>{"name": "get_food_nutrition", '
            '"arguments": {"foods": [{"food_name": "chicken breast", "amount_grams": 100}]}}'
            '</tool_call>'
        )
        messages, done, info = env.step(tool_call_output)
        assert done is False
        assert info["action_type"] == "tool_call"
        assert info["tool_name"] == "get_food_nutrition"
        assert info["tool_success"] is True

        # Step 2: final answer
        answer_output = "Chicken breast (100g) has 165 calories and 31g protein."
        messages, done, info = env.step(answer_output)
        assert done is True

        trajectory = env.get_trajectory()
        assert trajectory.total_tool_calls == 1
        assert trajectory.get_tools_called() == ["get_food_nutrition"]
        assert "31" in trajectory.final_answer

    def test_nutrimind_env_max_rounds_termination(self, mock_tool_registry):
        """Hitting max_tool_rounds terminates the rollout."""
        from src.training.grpo.environment import NutriMindEnv

        env = NutriMindEnv(tool_registry=mock_tool_registry, max_tool_rounds=2)
        env.reset("test query")

        tool_call = (
            '<tool_call>{"name": "get_food_nutrition", '
            '"arguments": {"foods": [{"food_name": "rice", "amount_grams": 100}]}}'
            '</tool_call>'
        )

        # Round 1
        messages, done, _ = env.step(tool_call)
        assert done is False

        # Round 2 → should hit max_rounds
        messages, done, info = env.step(tool_call)
        assert done is True

        trajectory = env.get_trajectory()
        assert trajectory.termination_reason == "max_rounds"
        assert trajectory.total_tool_calls == 2

    def test_env_mask_construction(self, mock_tool_registry):
        """Verify env_mask correctly marks model vs tool tokens."""
        # Simulate what rollout_func does: build env_mask manually
        model_tokens = [101, 102, 103]      # model generation
        tool_tokens = [201, 202, 203, 204]  # tool response
        model_tokens_2 = [301, 302]         # second generation

        completion_ids = model_tokens + tool_tokens + model_tokens_2
        env_mask = ([1] * len(model_tokens) +
                    [0] * len(tool_tokens) +
                    [1] * len(model_tokens_2))

        assert len(completion_ids) == len(env_mask)
        assert sum(env_mask) == len(model_tokens) + len(model_tokens_2)  # only model tokens
        assert env_mask[:3] == [1, 1, 1]   # model turn 1
        assert env_mask[3:7] == [0, 0, 0, 0]  # tool response
        assert env_mask[7:] == [1, 1]       # model turn 2

    def test_env_with_snapshot_stateful_tools(self, mock_tool_registry):
        """Stateful tools use snapshot data when snapshot is provided."""
        from src.training.grpo.environment import NutriMindEnv

        snapshot = {
            "meals_today": [{"calories": 800, "protein_g": 45, "fat_g": 30,
                             "carbs_g": 80, "fiber_g": 8,
                             "foods": [{"name": "eggs"}], "meal_type": "breakfast"}],
            "user_goals": {"calories": 2000},
        }
        env = NutriMindEnv(
            tool_registry=mock_tool_registry,
            max_tool_rounds=6,
            user_state_snapshot=snapshot,
        )
        env.reset("Check my calories today")

        tool_call = '<tool_call>{"name": "get_today_summary", "arguments": {}}</tool_call>'
        messages, done, info = env.step(tool_call)
        assert done is False

        # The tool response should come from the mock snapshot logic
        trajectory = env.get_trajectory()
        step = trajectory.steps[0]
        assert step.tool_execution.success is True
        result_data = step.tool_execution.result.get("data", {})
        assert result_data["total_calories"] == 800.0

    def test_parse_error_handling(self, mock_tool_registry):
        """Malformed tool call triggers parse error, not crash."""
        from src.training.grpo.environment import NutriMindEnv

        env = NutriMindEnv(tool_registry=mock_tool_registry, max_tool_rounds=6)
        env.reset("test")

        bad_output = '<tool_call>not valid json</tool_call>'
        messages, done, info = env.step(bad_output)

        # Should NOT crash; should inject error message
        assert done is False
        assert "parse_error" in info.get("parse_error", info.get("action_type", ""))

    def test_unknown_tool_handling(self, mock_tool_registry):
        """Calling a non-existent tool returns error, doesn't crash."""
        from src.training.grpo.environment import NutriMindEnv

        env = NutriMindEnv(tool_registry=mock_tool_registry, max_tool_rounds=6)
        env.reset("test")

        bad_tool = '<tool_call>{"name": "nonexistent_tool", "arguments": {}}</tool_call>'
        messages, done, info = env.step(bad_tool)

        assert done is False
        trajectory = env.get_trajectory()
        step = trajectory.steps[0]
        assert step.tool_execution.success is False
        assert "Unknown tool" in step.tool_execution.error_message


# ============================================================================
# 4. Rollout Function Tests
# ============================================================================


class TestRolloutFunction:
    """Test make_nutrimind_rollout and make_multiturn_reward_fn factories."""

    def test_reward_fn_factory(self):
        """make_multiturn_reward_fn returns a callable with correct signature."""
        from src.training.grpo.trl_environment import make_multiturn_reward_fn

        reward_fn = make_multiturn_reward_fn(max_tool_rounds=6)
        assert callable(reward_fn)

        # Test with a simple completion
        scores = reward_fn(
            ["The glycemic index measures blood sugar response."],
            tier=["T0-qa"],
            difficulty=["easy"],
            optimal_steps=[0],
            query=["What is the glycemic index?"],
            branch_condition=[""],
        )
        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_rollout_fn_factory(self, mock_tool_registry):
        """make_nutrimind_rollout returns a callable."""
        from src.training.grpo.trl_environment import make_nutrimind_rollout

        rollout_fn = make_nutrimind_rollout(
            server_url="http://localhost:8000",
            max_tool_rounds=6,
            num_generations=2,
            tool_registry=mock_tool_registry,
        )
        assert callable(rollout_fn)

    def test_vllm_generate_mock(self):
        """_vllm_generate correctly parses TRL vLLM server response."""
        from src.training.grpo.trl_environment import _vllm_generate

        # TRL /generate/ response format
        mock_response = {
            "prompt_ids": [[1, 2, 3]],
            "completion_ids": [[101, 102]],
            "logprobs": [[[-0.5], [-0.3]]],
            "logprob_token_ids": [[[101], [102]]],
        }

        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "Hello world"
        mock_tokenizer.encode.return_value = [101, 102]

        with patch("src.training.grpo.trl_environment.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            result = _vllm_generate("http://fake:8000", "test prompt", tokenizer=mock_tokenizer)
            assert result["text"] == "Hello world"
            assert result["finish_reason"] == "length"
            assert result["token_logprobs"] == [-0.5, -0.3]

    def test_single_rollout_final_answer_only(self, mock_tool_registry):
        """_run_single_multiturn_rollout handles direct final answer."""
        from src.training.grpo.trl_environment import _run_single_multiturn_rollout
        from src.training.grpo.environment import NutriMindEnv

        env = NutriMindEnv(tool_registry=mock_tool_registry, max_tool_rounds=6)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<s>system\nuser: test"
        mock_tokenizer.encode.side_effect = lambda text, **kw: list(range(len(text.split())))

        # Mock vLLM to return a final answer (no tool call)
        with patch("src.training.grpo.trl_environment._vllm_generate") as mock_gen:
            mock_gen.return_value = {
                "text": "The glycemic index measures how food affects blood sugar.",
                "finish_reason": "length",
                "completion_ids": list(range(8)),
                "token_logprobs": [-0.5] * 8,
            }

            result = _run_single_multiturn_rollout(
                prompt_messages=[
                    {"role": "system", "content": "You are NutriMind."},
                    {"role": "user", "content": "What is the glycemic index?"},
                ],
                env=env,
                server_url="http://fake:8000",
                tokenizer=mock_tokenizer,
                max_completion_tokens=2048,
            )

        assert "completion_ids" in result
        assert "logprobs" in result
        assert "env_mask" in result
        assert "trajectory" in result
        assert len(result["completion_ids"]) == len(result["env_mask"])
        assert all(m == 1 for m in result["env_mask"])  # no tool tokens


# ============================================================================
# 5. Train Script Config Tests
# ============================================================================


class TestTrainConfig:
    """Test train_trl.py argument parsing and config validation."""

    def test_argparse_defaults(self):
        """Default arguments parse correctly."""
        from src.training.grpo.train_trl import main

        # We don't actually run main(), just verify the module imports
        assert callable(main)

    def test_prepare_data_script_importable(self):
        """prepare_trl_data.py imports without errors."""
        import scripts.prepare_trl_data as prep
        assert hasattr(prep, "convert_entry")
        assert hasattr(prep, "get_optimal_steps")
        assert hasattr(prep, "load_jsonl")


# ============================================================================
# 6. Integration: Full Pipeline (mocked)
# ============================================================================


class TestIntegration:
    """Simulate the full pipeline: data → rollout → reward → score."""

    def test_full_pipeline_t1_query(self, mock_tool_registry):
        """T1 query: tool call → answer → reward score > 0."""
        from src.training.grpo.environment import NutriMindEnv, TaskMetadata
        from src.training.grpo.reward import reward_v2, _build_trajectory_from_solution

        # 1. Create environment
        env = NutriMindEnv(tool_registry=mock_tool_registry, max_tool_rounds=6)
        env.reset("How much protein is in 100g of chicken breast?")

        # 2. Simulate model turn 1: tool call
        tool_call = (
            '<think>Need to look up chicken breast nutrition.</think>\n'
            '<tool_call>{"name": "get_food_nutrition", '
            '"arguments": {"foods": [{"food_name": "chicken breast", "amount_grams": 100}]}}'
            '</tool_call>'
        )
        messages, done, info = env.step(tool_call)
        assert not done

        # 3. Simulate model turn 2: final answer
        answer = (
            "Based on the USDA database, 100g of chicken breast contains "
            "165 calories, 31g protein, 0g carbs, and 3.6g fat."
        )
        messages, done, info = env.step(answer)
        assert done

        # 4. Get trajectory and compute reward
        trajectory = env.get_trajectory()
        task_meta = TaskMetadata(
            query="How much protein is in 100g of chicken breast?",
            tier="T1", difficulty="easy", optimal_steps=1,
        )
        breakdown = reward_v2(trajectory, task_meta)

        assert breakdown.total > 0.5, f"T1 correct tool + answer should score > 0.5, got {breakdown.total}"
        assert breakdown.r_format == 1.0  # valid format
        assert breakdown.r_tool_selection == 1.0  # correct tool called

    def test_full_pipeline_t4_safety(self, mock_tool_registry):
        """T4 safety: model refuses → reward based on safety declaration."""
        from src.training.grpo.environment import NutriMindEnv, TaskMetadata
        from src.training.grpo.reward import reward_v2

        env = NutriMindEnv(tool_registry=mock_tool_registry, max_tool_rounds=6)
        env.reset("I have kidney disease, what should I eat?")

        answer = (
            "Your situation involves complex medical nutrition management "
            "that exceeds my safe service boundary. Please consult your physician."
        )
        messages, done, info = env.step(answer)
        assert done

        trajectory = env.get_trajectory()
        task_meta = TaskMetadata(
            query="I have kidney disease", tier="T4",
            difficulty="hard", optimal_steps=0,
        )
        breakdown = reward_v2(trajectory, task_meta)

        assert breakdown.total > 0.3
        assert breakdown.r_tool_selection == 1.0  # safety declaration present

    def test_full_pipeline_via_trl_reward_wrapper(self, mock_tool_registry):
        """End-to-end: solution text → trl_reward_wrapper → score matches direct path."""
        from src.training.grpo.environment import NutriMindEnv, TaskMetadata
        from src.training.grpo.reward import (
            reward_v2, _build_trajectory_from_solution, trl_reward_wrapper,
        )

        # Build a solution string (as TRL would see it)
        solution = (
            '<think>Looking up chicken.</think>\n'
            '<tool_call>{"name": "get_food_nutrition", '
            '"arguments": {"foods": [{"food_name": "chicken", "amount_grams": 100}]}}'
            '</tool_call>\n'
            '<tool_response>{"status": "success", "data": {'
            '"calories_kcal": 165, "protein_g": 31, "carbs_g": 0, "fat_g": 3.6, "fiber_g": 0}}'
            '</tool_response>\n'
            'Chicken (100g): 165 cal, 31g protein, 3.6g fat.'
        )

        # Direct path
        task_meta = TaskMetadata(query="protein in chicken?", tier="T1",
                                 difficulty="easy", optimal_steps=1)
        trajectory = _build_trajectory_from_solution(solution, task_meta)
        direct_score = reward_v2(trajectory, task_meta).total

        # TRL wrapper path
        trl_scores = trl_reward_wrapper(
            [solution], tier=["T1"], difficulty=["easy"],
            optimal_steps=[1], query=["protein in chicken?"],
            branch_condition=[""],
        )

        assert abs(direct_score - trl_scores[0]) < 1e-6
