"""
TRL Migration Smoke Tests — Real GPU + Real vLLM + Real Model.

These tests validate the FULL training pipeline on actual hardware before
committing to a long training run. Each test is designed to fail fast with
a clear error message.

Prerequisites:
    1. vLLM server running on GPU 1:
       CUDA_VISIBLE_DEVICES=1 trl vllm-serve \
           --model models/nutrimind-4b-sft-merged \
           --gpu-memory-utilization 0.85 --max-model-len 8192

    2. Training data prepared:
       python scripts/prepare_trl_data.py

Run:
    # Full smoke test suite (recommended order)
    CUDA_VISIBLE_DEVICES=0 pytest tests/test_trl_smoke.py -v --tb=short

    # Individual stages
    CUDA_VISIBLE_DEVICES=0 pytest tests/test_trl_smoke.py -v -k "stage1"  # vLLM health
    CUDA_VISIBLE_DEVICES=0 pytest tests/test_trl_smoke.py -v -k "stage2"  # generation
    CUDA_VISIBLE_DEVICES=0 pytest tests/test_trl_smoke.py -v -k "stage3"  # multi-turn
    CUDA_VISIBLE_DEVICES=0 pytest tests/test_trl_smoke.py -v -k "stage4"  # reward
    CUDA_VISIBLE_DEVICES=0 pytest tests/test_trl_smoke.py -v -k "stage5"  # training

Estimated time: ~3-5 minutes total.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

# ────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "nutrimind-4b-sft-merged"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "grpo" / "trl_train"
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")

sys.path.insert(0, str(PROJECT_ROOT))


# ────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def vllm_url():
    """Return vLLM server URL, skip all tests if unreachable."""
    try:
        r = requests.get(f"{VLLM_URL}/health", timeout=5)
        r.raise_for_status()
    except Exception:
        pytest.skip(
            f"vLLM server not running at {VLLM_URL}. Start it with:\n"
            f"  CUDA_VISIBLE_DEVICES=1 trl vllm-serve "
            f"--model models/nutrimind-4b-sft-merged "
            f"--gpu-memory-utilization 0.85 --max-model-len 8192"
        )
    return VLLM_URL


@pytest.fixture(scope="session")
def tokenizer():
    """Load tokenizer from model path."""
    from transformers import AutoTokenizer

    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")
    return AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)


@pytest.fixture(scope="session")
def train_dataset():
    """Load training dataset."""
    from datasets import load_from_disk

    if not TRAIN_DATA_PATH.exists():
        # Try to prepare it
        prep_script = PROJECT_ROOT / "scripts" / "prepare_trl_data.py"
        if (PROJECT_ROOT / "data" / "grpo" / "grpo_prompts.jsonl").exists():
            subprocess.run([sys.executable, str(prep_script)], check=True)
        else:
            pytest.skip(
                f"Training data not found at {TRAIN_DATA_PATH}. "
                f"Run: python scripts/prepare_trl_data.py"
            )
    return load_from_disk(str(TRAIN_DATA_PATH))


@pytest.fixture(scope="session")
def tool_registry():
    """Load real tool registry (with mock stateful tools for isolation)."""
    from src.training.grpo.environment import NutriMindEnv

    # Use mock snapshot so stateful tools don't touch real DB
    mock_snapshot = {
        "meals_today": [
            {"calories": 500, "protein_g": 30, "fat_g": 15,
             "carbs_g": 60, "fiber_g": 5,
             "foods": [{"name": "oatmeal"}], "meal_type": "breakfast"},
        ],
        "user_goals": {"calories": 2000, "protein_g": 120},
        "meal_history": [
            {"date": "2026-04-07", "calories": 1800, "protein_g": 95,
             "fat_g": 60, "carbs_g": 220, "fiber_g": 25},
        ],
    }
    return mock_snapshot


# ============================================================================
# Stage 1: Infrastructure Checks
# ============================================================================


class TestStage1Infrastructure:
    """Verify all prerequisites before testing the pipeline."""

    def test_stage1_model_exists(self):
        """Model directory exists and has config."""
        assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
        config_file = MODEL_PATH / "config.json"
        assert config_file.exists(), f"No config.json in {MODEL_PATH}"

    def test_stage1_vllm_health(self, vllm_url):
        """vLLM server is healthy."""
        r = requests.get(f"{vllm_url}/health", timeout=5)
        assert r.status_code == 200

    def test_stage1_vllm_models(self, vllm_url):
        """vLLM server has a model loaded (via trl vllm-serve /generate/ endpoint)."""
        probe = requests.post(
            f"{vllm_url}/generate/",
            json={"prompts": ["Hi"], "max_tokens": 1},
            timeout=30,
        )
        assert probe.status_code == 200, (
            f"/generate/ probe returned {probe.status_code}"
        )
        data = probe.json()
        assert "completion_ids" in data, "Response missing completion_ids"

    def test_stage1_gpu_available(self):
        """At least one GPU is visible."""
        import torch

        assert torch.cuda.is_available(), "No GPU available"
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        assert gpu_mem > 20, f"GPU 0 only has {gpu_mem:.0f}GB, need >20GB"

    def test_stage1_trl_importable(self):
        """TRL and required packages import correctly."""
        import trl
        import peft
        import transformers

        assert hasattr(trl, "GRPOTrainer")
        assert hasattr(trl, "GRPOConfig")

    def test_stage1_train_data_exists(self, train_dataset):
        """Training dataset is loaded and has expected columns."""
        required = {"prompt", "tier", "difficulty", "optimal_steps",
                    "env_state", "query"}
        actual = set(train_dataset.column_names)
        missing = required - actual
        assert not missing, f"Missing columns: {missing}"
        assert len(train_dataset) > 0, "Dataset is empty"


# ============================================================================
# Stage 2: vLLM Generation
# ============================================================================


class TestStage2Generation:
    """Test that vLLM can generate text for our prompts."""

    def test_stage2_simple_completion(self, vllm_url, tokenizer):
        """vLLM generates a non-empty completion."""
        from src.training.grpo.trl_environment import _vllm_generate

        result = _vllm_generate(
            server_url=vllm_url,
            prompt_text="What is protein?",
            tokenizer=tokenizer,
            max_tokens=100,
            temperature=0.7,
        )
        assert len(result["text"]) > 10, f"Completion too short: {result['text']!r}"
        assert result["finish_reason"] in ("stop", "length")

    def test_stage2_chat_template_completion(self, vllm_url, tokenizer):
        """Generate from a properly formatted chat prompt."""
        from src.training.grpo.trl_environment import _vllm_generate
        from src.training.grpo.environment import NutriMindEnv

        messages = [
            {"role": "system", "content": NutriMindEnv.SYSTEM_PROMPT},
            {"role": "user", "content": "How many calories in 100g of rice?"},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        result = _vllm_generate(
            server_url=vllm_url,
            prompt_text=prompt_text,
            tokenizer=tokenizer,
            max_tokens=512,
            temperature=0.7,
            stop=["</tool_call>"],
        )
        text = result["text"]
        assert len(text) > 20, f"Response too short: {text!r}"

        # Model should either call a tool or give a direct answer
        has_tool_call = "<tool_call>" in text
        has_answer = len(text) > 50
        assert has_tool_call or has_answer, (
            f"Model neither called a tool nor gave a substantive answer:\n{text[:200]}"
        )

    def test_stage2_logprobs_returned(self, vllm_url, tokenizer):
        """vLLM returns per-token logprobs."""
        from src.training.grpo.trl_environment import _vllm_generate

        result = _vllm_generate(
            server_url=vllm_url,
            prompt_text="Hello",
            tokenizer=tokenizer,
            max_tokens=20,
            temperature=0.7,
            logprobs=True,
        )
        assert len(result["token_logprobs"]) > 0, "No logprobs returned"
        assert all(isinstance(lp, (int, float)) for lp in result["token_logprobs"])

    def test_stage2_stop_at_tool_call(self, vllm_url, tokenizer):
        """Client-side stop at </tool_call> for multi-turn parsing."""
        from src.training.grpo.trl_environment import _vllm_generate
        from src.training.grpo.environment import NutriMindEnv

        messages = [
            {"role": "system", "content": NutriMindEnv.SYSTEM_PROMPT},
            {"role": "user", "content": "How much protein is in 100g of chicken breast?"},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        result = _vllm_generate(
            server_url=vllm_url,
            prompt_text=prompt_text,
            tokenizer=tokenizer,
            max_tokens=1024,
            temperature=0.3,  # lower temp to encourage tool use
            stop=["</tool_call>"],
        )

        text = result["text"]
        # Either stopped at tool_call boundary, or gave direct answer
        if "<tool_call>" in text:
            # Should have stopped BEFORE generating past the closing tag
            assert "</tool_call>" not in text, (
                "stop token should have prevented </tool_call> in output"
            )


# ============================================================================
# Stage 3: Multi-Turn Rollout
# ============================================================================


class TestStage3MultiTurn:
    """Test the full multi-turn agentic rollout with real vLLM."""

    def test_stage3_single_rollout(self, vllm_url, tokenizer, tool_registry):
        """Run one complete multi-turn rollout."""
        from src.training.grpo.trl_environment import _run_single_multiturn_rollout
        from src.training.grpo.environment import NutriMindEnv
        from src.orchestrator.orchestrator import TOOL_REGISTRY

        env = NutriMindEnv(
            tool_registry=TOOL_REGISTRY,
            max_tool_rounds=3,  # limit for speed
            user_state_snapshot=tool_registry,  # mock snapshot
        )

        result = _run_single_multiturn_rollout(
            prompt_messages=[
                {"role": "system", "content": NutriMindEnv.SYSTEM_PROMPT},
                {"role": "user", "content": "How much protein is in 100g of chicken breast?"},
            ],
            env=env,
            server_url=vllm_url,
            tokenizer=tokenizer,
            max_completion_tokens=2048,
            temperature=0.5,
        )

        # Verify structure
        assert "completion_ids" in result
        assert "logprobs" in result
        assert "env_mask" in result
        assert "trajectory" in result
        assert "prompt_ids" in result

        # Verify alignment
        assert len(result["completion_ids"]) == len(result["env_mask"]), (
            f"completion_ids ({len(result['completion_ids'])}) != "
            f"env_mask ({len(result['env_mask'])})"
        )
        assert len(result["completion_ids"]) == len(result["logprobs"]), (
            f"completion_ids ({len(result['completion_ids'])}) != "
            f"logprobs ({len(result['logprobs'])})"
        )

        # Verify env_mask has expected values
        assert all(m in (0, 1) for m in result["env_mask"])

        # Verify trajectory
        traj = result["trajectory"]
        assert traj.terminated, "Trajectory should be terminated"
        print(f"\n  Rollout complete: {traj.total_tool_calls} tool calls, "
              f"reason={traj.termination_reason}, "
              f"tokens={len(result['completion_ids'])}")

    def test_stage3_env_mask_tool_tokens_marked(self, vllm_url, tokenizer, tool_registry):
        """If model calls a tool, env_mask has 0s for tool response tokens."""
        from src.training.grpo.trl_environment import _run_single_multiturn_rollout
        from src.training.grpo.environment import NutriMindEnv
        from src.orchestrator.orchestrator import TOOL_REGISTRY

        env = NutriMindEnv(
            tool_registry=TOOL_REGISTRY,
            max_tool_rounds=2,
            user_state_snapshot=tool_registry,
        )

        result = _run_single_multiturn_rollout(
            prompt_messages=[
                {"role": "system", "content": NutriMindEnv.SYSTEM_PROMPT},
                {"role": "user", "content": "Check my calorie intake today and tell me how I'm doing."},
            ],
            env=env,
            server_url=vllm_url,
            tokenizer=tokenizer,
            max_completion_tokens=2048,
            temperature=0.3,
        )

        traj = result["trajectory"]
        env_mask = result["env_mask"]

        if traj.total_tool_calls > 0:
            # Must have some 0s (tool response tokens)
            num_tool_tokens = env_mask.count(0)
            num_model_tokens = env_mask.count(1)
            assert num_tool_tokens > 0, "Tool was called but no tool tokens (0s) in env_mask"
            assert num_model_tokens > 0, "No model tokens (1s) in env_mask"
            print(f"\n  env_mask: {num_model_tokens} model tokens, "
                  f"{num_tool_tokens} tool tokens")
        else:
            # All 1s (no tool called, direct answer)
            assert all(m == 1 for m in env_mask)
            print("\n  Model gave direct answer, no tool tokens")


# ============================================================================
# Stage 4: Reward Scoring
# ============================================================================


class TestStage4Reward:
    """Test reward scoring on real model outputs."""

    def test_stage4_reward_on_real_rollout(self, vllm_url, tokenizer, tool_registry):
        """Score a real rollout and verify reward is reasonable."""
        from src.training.grpo.trl_environment import (
            _run_single_multiturn_rollout,
            make_multiturn_reward_fn,
        )
        from src.training.grpo.environment import NutriMindEnv
        from src.orchestrator.orchestrator import TOOL_REGISTRY

        env = NutriMindEnv(
            tool_registry=TOOL_REGISTRY,
            max_tool_rounds=3,
            user_state_snapshot=tool_registry,
        )

        result = _run_single_multiturn_rollout(
            prompt_messages=[
                {"role": "system", "content": NutriMindEnv.SYSTEM_PROMPT},
                {"role": "user", "content": "How much protein is in 100g of chicken breast?"},
            ],
            env=env,
            server_url=vllm_url,
            tokenizer=tokenizer,
            max_completion_tokens=2048,
            temperature=0.5,
        )

        # Score with reward function
        reward_fn = make_multiturn_reward_fn(max_tool_rounds=6)
        scores = reward_fn(
            [result["completion_text"]],
            tier=["T1"],
            difficulty=["easy"],
            optimal_steps=[1],
            query=["How much protein is in 100g of chicken breast?"],
            branch_condition=[""],
            completion_text=[result["completion_text"]],
        )

        assert len(scores) == 1
        score = scores[0]
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        print(f"\n  T1 query reward: {score:.3f}")

    def test_stage4_batch_reward_diverse_tiers(self, vllm_url, tokenizer, tool_registry):
        """Score a batch of diverse queries to check reward distribution."""
        from src.training.grpo.trl_environment import (
            _run_single_multiturn_rollout,
            make_multiturn_reward_fn,
        )
        from src.training.grpo.environment import NutriMindEnv
        from src.orchestrator.orchestrator import TOOL_REGISTRY

        test_cases = [
            ("What is the glycemic index?", "T0-qa", 0),
            ("How much protein in chicken?", "T1", 1),
            ("I have stage 3 kidney disease. Design a diet.", "T4", 0),
        ]

        reward_fn = make_multiturn_reward_fn(max_tool_rounds=6)
        completions, tiers, steps, queries = [], [], [], []

        for query, tier, opt_steps in test_cases:
            env = NutriMindEnv(
                tool_registry=TOOL_REGISTRY,
                max_tool_rounds=2,
                user_state_snapshot=tool_registry if tier not in ("T0-qa", "T4") else None,
            )
            result = _run_single_multiturn_rollout(
                prompt_messages=[
                    {"role": "system", "content": NutriMindEnv.SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                env=env,
                server_url=vllm_url,
                tokenizer=tokenizer,
                max_completion_tokens=1024,
                temperature=0.5,
            )
            completions.append(result["completion_text"])
            tiers.append(tier)
            steps.append(opt_steps)
            queries.append(query)

        scores = reward_fn(
            completions,
            tier=tiers,
            difficulty=["easy", "easy", "hard"],
            optimal_steps=steps,
            query=queries,
            branch_condition=[""] * 3,
            completion_text=completions,
        )

        assert len(scores) == 3
        for i, (query, tier, score) in enumerate(zip(queries, tiers, scores)):
            print(f"\n  [{tier}] {query[:40]}... → reward={score:.3f}")
            assert 0.0 <= score <= 1.0


# ============================================================================
# Stage 5: GRPOTrainer 1-Step Training
# ============================================================================


class TestStage5Training:
    """The final test: run 1 actual training step with GRPOTrainer."""

    @pytest.mark.timeout(300)  # 5 min max
    def test_stage5_grpo_one_step(self, vllm_url, tokenizer, train_dataset, tmp_path):
        """Run GRPOTrainer for 1 step on 1 sample — the ultimate smoke test."""
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer

        from src.training.grpo.trl_environment import (
            make_multiturn_reward_fn,
            make_nutrimind_rollout,
        )

        # Tiny subset (2 samples)
        mini_dataset = train_dataset.select(range(min(2, len(train_dataset))))

        reward_fn = make_multiturn_reward_fn(max_tool_rounds=3)
        rollout_fn = make_nutrimind_rollout(
            server_url=vllm_url,
            max_tool_rounds=3,
            max_completion_tokens=1024,  # small for speed
            temperature=0.7,
            num_generations=2,  # minimal G
        )

        output_dir = tmp_path / "smoke_test_output"
        config = GRPOConfig(
            output_dir=str(output_dir),
            num_generations=2,
            max_completion_length=1024,
            temperature=0.7,
            learning_rate=5e-7,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            max_steps=1,  # JUST 1 STEP
            gradient_checkpointing=True,
            bf16=True,
            use_vllm=True,
            vllm_mode="server",
            vllm_server_url=vllm_url,
            vllm_gpu_memory_utilization=0.85,
            beta=0.001,
            loss_type="grpo",
            logging_steps=1,
            save_steps=999,  # don't save during smoke test
            report_to=[],  # no wandb
            seed=42,
        )

        lora_config = LoraConfig(
            r=8,  # smaller for smoke test
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

        print("\n  Initializing GRPOTrainer...")
        trainer = GRPOTrainer(
            model=str(MODEL_PATH),
            reward_funcs=[reward_fn],
            args=config,
            train_dataset=mini_dataset,
            peft_config=lora_config,
            rollout_func=rollout_fn,
        )

        print("  Running 1 training step...")
        train_result = trainer.train()

        # Verify training actually ran
        assert train_result is not None
        metrics = train_result.metrics
        print(f"  Training metrics: {json.dumps(metrics, indent=2, default=str)}")

        # Check loss is a real number
        loss = metrics.get("train_loss", metrics.get("loss"))
        if loss is not None:
            assert loss > 0, f"Loss should be > 0, got {loss}"
            assert loss < 100, f"Loss suspiciously high: {loss}"
            print(f"  Loss: {loss:.4f} ✓")

        print("  ✓ GRPOTrainer 1-step smoke test PASSED")
