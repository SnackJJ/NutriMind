#!/usr/bin/env python3
"""
Single A800 GRPO training entry point for NutriMind (ADR-007).

Uses TRL's environment_factory with NutriMindToolEnv for online multi-turn
agent training. TRL handles the full rollout loop:
    generate → parse <tool_call> → call env method → inject response → loop

Usage:
    # Dry run — validate config without GPU
    python -m src.training.grpo.train_grpo --dry_run

    # Smoke test — one training step
    python -m src.training.grpo.train_grpo --max_steps 1

    # Full training on A800
    python -m src.training.grpo.train_grpo \
        --model_path models/nutrimind-4b-sft-merged \
        --output_dir models/grpo-a800

Requires: trl>=1.0, transformers>=5.2.0, peft
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train_grpo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NutriMind GRPO Training (A800)")

    # Model / data
    p.add_argument("--model_path", type=str, default="models/nutrimind-4b-sft-merged")
    p.add_argument("--train_data", type=str, default="data/grpo/trl_train")
    p.add_argument("--output_dir", type=str, default="models/grpo-a800")

    # GRPO
    p.add_argument("--num_generations", type=int, default=16, help="Number of generations per prompt (G)")
    p.add_argument("--max_completion_length", type=int, default=6144,
                   help="Max tokens per completion")

    # Training
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=-1, help="Override epochs; -1 = use epochs")
    p.add_argument("--beta", type=float, default=0.01,
                   help="KL penalty coefficient (0.01 for 4B; DeepSeek uses 0.04 for 70B)")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16,
                   help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32)

    # vLLM
    p.add_argument("--use_vllm", action="store_true", default=True)
    p.add_argument("--no_vllm", action="store_true", help="Disable vLLM (slow, for debugging)")
    p.add_argument("--vllm_gpu_memory", type=float, default=0.45,
                   help="GPU memory fraction for vLLM in colocate mode")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true", help="Print config and exit")
    p.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--use_gigpo", action="store_true", default=True,
                   help="Use GiGPOTrainer with step-level advantages (default: True)")
    p.add_argument("--no_gigpo", action="store_true",
                   help="Disable GiGPO, use standard GRPOTrainer")

    return p.parse_args()


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()

    model_path = resolve_path(args.model_path)
    train_data_path = resolve_path(args.train_data)
    output_dir = resolve_path(args.output_dir)

    log.info("Model:      %s", model_path)
    log.info("Train data: %s", train_data_path)
    log.info("Output:     %s", output_dir)

    if not model_path.exists():
        log.error("Model not found: %s", model_path)
        return 1
    if not train_data_path.exists():
        log.error("Training data not found: %s\nRun: python scripts/prepare_trl_data.py", train_data_path)
        return 1

    # Late imports (heavy)
    log.info("Loading libraries...")
    from datasets import load_from_disk
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from src.training.grpo.trl_env_factory import NutriMindToolEnv
    from src.training.grpo.reward import reward_from_env

    use_gigpo = args.use_gigpo and not args.no_gigpo
    if use_gigpo:
        from src.training.grpo.gigpo_trainer_v2 import GiGPOTrainerV2
        TrainerClass = GiGPOTrainerV2
        log.info("Using GiGPOTrainerV2 (step-level advantages with intercept hook)")
    else:
        TrainerClass = GRPOTrainer
        log.info("Using standard GRPOTrainer (trajectory-level advantages)")

    # Dataset
    log.info("Loading dataset...")
    train_dataset = load_from_disk(str(train_data_path))
    log.info("Train samples: %d", len(train_dataset))
    log.info("Columns: %s", train_dataset.column_names)

    # LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.0,
    )

    # GRPO config
    use_vllm = args.use_vllm and not args.no_vllm
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        # Generation
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        # Training
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        bf16=True,
        # LR schedule: constant with warmup (avoid cosine wasting LR budget)
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,
        # vLLM (colocate = single GPU, shared memory)
        use_vllm=use_vllm,
        vllm_mode="colocate" if use_vllm else None,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory if use_vllm else None,
        # Sleep mode disabled: keep vLLM weights on GPU (faster, uses more VRAM)
        vllm_enable_sleep_mode=False,
        # GRPO algorithm
        beta=args.beta,
        loss_type="grpo",
        # Qwen3: disable thinking mode during RL training
        chat_template_kwargs={"enable_thinking": False},
        # Logging / saving
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        report_to=[args.report_to] if args.report_to != "none" else [],
        run_name=args.run_name or f"nutrimind-grpo-a800",
        # Misc
        seed=args.seed,
    )

    if args.dry_run:
        log.info("\n=== DRY RUN ===")
        log.info("GRPOConfig:\n%s", grpo_config)
        log.info("LoraConfig:\n%s", lora_config)
        log.info("Environment: NutriMindToolEnv (6 tools)")
        log.info("Reward: reward_from_env → reward_v3 (hybrid rule + LLM judge)")
        log.info("Sample prompt: %s", str(train_dataset[0]["prompt"])[:200])
        log.info("Dry run complete — no training performed.")
        return 0

    # Initialize trainer
    log.info("Initializing GRPOTrainer...")
    from transformers import AutoTokenizer
    import trl.chat_template_utils

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # TRL demands a standard "inference" chat template to attach the response schema,
    # and then internally mapping it to the training counterpart template.
    # We assign the standard Qwen3 template so it seamlessly passes all internal TRL checks.
    tokenizer.chat_template = trl.chat_template_utils.qwen3_chat_template

    trainer = TrainerClass(
        model=str(model_path),
        processing_class=tokenizer,
        environment_factory=NutriMindToolEnv,
        reward_funcs=reward_from_env,
        train_dataset=train_dataset,
        peft_config=lora_config,
        args=grpo_config,
    )

    # Train
    log.info("Starting training...")
    trainer.train()

    # Save
    log.info("Saving final model...")
    trainer.save_model(str(output_dir / "final"))
    log.info("Training complete. Model saved to %s", output_dir / "final")

    return 0


if __name__ == "__main__":
    sys.exit(main())
