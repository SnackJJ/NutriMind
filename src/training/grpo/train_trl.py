#!/usr/bin/env python3
"""
TRL-based GRPO Training Entry Point for NutriMind.

Replaces veRL's train_verl.py with TRL GRPOTrainer, supporting:
- vllm_mode="server" for GPU isolation (GPU 0: training, GPU 1: vLLM)
- LoRA fine-tuning via peft
- Custom reward function (reward_v2) with multi-turn trajectory scoring
- Qwen3-4B with <tool_call> XML format (ADR-001)

Usage:
    # Start vLLM server first (on GPU 1):
    CUDA_VISIBLE_DEVICES=1 trl vllm-serve \\
        --model models/nutrimind-4b-sft-merged \\
        --gpu-memory-utilization 0.85 --max-model-len 8192

    # Then run training (on GPU 0):
    CUDA_VISIBLE_DEVICES=0 python -m src.training.grpo.train_trl

    # Or with options:
    CUDA_VISIBLE_DEVICES=0 python -m src.training.grpo.train_trl \\
        --reward_version v2 \\
        --model_path models/nutrimind-4b-sft-merged \\
        --dry_run
"""

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="TRL GRPO Training for NutriMind",
    )
    parser.add_argument(
        "--model_path", type=str,
        default="models/nutrimind-4b-sft-merged",
        help="Path to base model",
    )
    parser.add_argument(
        "--reward_version", type=str, default="v2",
        choices=["v1", "v2"],
        help="Reward function version",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="models/trl_grpo_4090d",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--vllm_server_url", type=str,
        default="http://localhost:8000",
        help="URL of the vLLM server",
    )
    parser.add_argument(
        "--train_data", type=str,
        default="data/grpo/trl_train",
        help="Path to training dataset (HF format)",
    )
    parser.add_argument(
        "--val_data", type=str,
        default="data/grpo/trl_val",
        help="Path to validation dataset (HF format)",
    )
    parser.add_argument(
        "--num_generations", type=int, default=4,
        help="Number of completions per prompt (G in GRPO)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-7,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--max_completion_length", type=int, default=5120,
        help="Max tokens for model completion",
    )
    parser.add_argument(
        "--max_prompt_length", type=int, default=2048,
        help="Max tokens for prompt",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Validate config and print settings without training",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Resume from checkpoint path",
    )
    args = parser.parse_args()

    # Resolve paths
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    train_data_path = Path(args.train_data)
    if not train_data_path.is_absolute():
        train_data_path = PROJECT_ROOT / train_data_path

    logger.info(f"Model: {model_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Reward: {args.reward_version}")
    logger.info(f"vLLM server: {args.vllm_server_url}")

    # Check model exists
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1

    # Check dataset exists
    if not train_data_path.exists():
        logger.error(
            f"Training data not found: {train_data_path}\n"
            f"Run: python scripts/prepare_trl_data.py"
        )
        return 1

    # Late imports (heavy)
    logger.info("Loading libraries...")
    from datasets import load_from_disk
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from src.training.grpo.trl_environment import (
        make_multiturn_reward_fn,
        make_nutrimind_rollout,
    )

    # Load dataset
    logger.info(f"Loading dataset from {train_data_path}")
    train_dataset = load_from_disk(str(train_data_path))
    logger.info(f"Train samples: {len(train_dataset)}")

    # Reward function (scores full multi-turn trajectory)
    reward_fn = make_multiturn_reward_fn(max_tool_rounds=6)
    logger.info(f"Reward function: {args.reward_version}")

    # Rollout function (runs multi-turn agentic loop via vLLM server)
    rollout_fn = make_nutrimind_rollout(
        server_url=args.vllm_server_url,
        max_tool_rounds=6,
        max_completion_tokens=args.max_completion_length,
        temperature=0.7,
        num_generations=args.num_generations,
    )
    logger.info("Rollout function: nutrimind_rollout (multi-turn agentic)")

    # GRPOConfig
    config = GRPOConfig(
        output_dir=str(output_dir),
        # Generation
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        # Training
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        gradient_checkpointing=True,
        bf16=True,
        # vLLM
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url=args.vllm_server_url,
        vllm_gpu_memory_utilization=0.85,
        # GRPO
        beta=0.001,  # KL coefficient
        loss_type="grpo",
        # Logging & Saving
        logging_steps=1,
        save_steps=30,
        save_total_limit=3,
        report_to=["wandb"],
        run_name=f"nutrimind-grpo-trl-{args.reward_version}",
        # Misc
        seed=42,
        dataloader_num_workers=4,
    )

    # LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,  # alpha = rank (standard)
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=0.0,
    )

    if args.dry_run:
        logger.info("\n=== Dry Run ===")
        logger.info(f"GRPOConfig: {config}")
        logger.info(f"LoraConfig: {lora_config}")
        logger.info(f"Dataset columns: {train_dataset.column_names}")
        logger.info(f"Sample prompt: {train_dataset[0]['prompt'][:2]}")
        logger.info("Dry run complete.")
        return 0

    # Initialize trainer
    logger.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=str(model_path),
        reward_funcs=[reward_fn],
        args=config,
        train_dataset=train_dataset,
        peft_config=lora_config,
        rollout_func=rollout_fn,  # Multi-turn agentic rollout via vLLM server
    )

    # Train
    logger.info("Starting training...")
    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(str(output_dir / "final"))

    logger.info("Training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
