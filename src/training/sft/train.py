#!/usr/bin/env python3
"""
SFT Training Script for NutriMind (Qwen3-4B)

Usage:
    python train.py --data_path data/trajectories/sft_train_trajectory.jsonl
    python train.py --verify_labels  # Only verify loss mask labels, don't train
"""

import argparse
import json
import logging
import os
from pathlib import Path

# Must import unsloth before transformers/peft for optimizations
import unsloth  # noqa: F401
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_trajectory_data(data_path: str) -> Dataset:
    """Load trajectory JSONL and prepare for SFT."""
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            messages = d.get("messages", [])
            if not messages:
                continue
            records.append({"messages": messages, "tier": d.get("tier", "unknown")})

    logger.info(f"Loaded {len(records)} trajectories from {data_path}")
    return Dataset.from_list(records)


def formatting_func(tokenizer):
    """Return a formatting function for SFTTrainer."""

    def _format(example):
        # Use apply_chat_template with enable_thinking=True for Qwen3
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        return {"text": text}

    return _format


def verify_labels(trainer, tokenizer, num_samples: int = 3):
    """
    Verify that loss masking is correctly applied.
    Prints token-by-token label info for inspection.
    """
    logger.info("=" * 60)
    logger.info("VERIFYING LOSS MASK LABELS")
    logger.info("=" * 60)

    for i in range(min(num_samples, len(trainer.train_dataset))):
        sample = trainer.train_dataset[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        logger.info(f"\n--- Sample {i} ---")

        # Decode and check label distribution
        assistant_tokens = 0
        masked_tokens = 0

        for j, (tok, lbl) in enumerate(zip(input_ids, labels)):
            if lbl == -100:
                masked_tokens += 1
            else:
                assistant_tokens += 1

            # Print first/last few tokens for debugging
            if j < 20 or j > len(input_ids) - 20:
                tok_str = tokenizer.decode([tok])
                label_str = "MASKED" if lbl == -100 else f"label={lbl}"
                logger.info(f"  {j:4d}: {tok_str!r:20s} -> {label_str}")
            elif j == 20:
                logger.info("  ... (middle tokens omitted) ...")

        total = len(input_ids)
        logger.info(
            f"\nTotal: {total} tokens | "
            f"Masked: {masked_tokens} ({100*masked_tokens/total:.1f}%) | "
            f"Training: {assistant_tokens} ({100*assistant_tokens/total:.1f}%)"
        )

        # Sanity check: should have some training tokens
        if assistant_tokens == 0:
            logger.error("ALL TOKENS MASKED - loss masking is broken!")
            raise ValueError("Loss masking verification failed: no training tokens")

        if assistant_tokens < 0.1 * total:
            logger.warning(
                f"Only {100*assistant_tokens/total:.1f}% tokens for training - seems low"
            )

    logger.info("\n" + "=" * 60)
    logger.info("LOSS MASK VERIFICATION COMPLETE")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SFT training for NutriMind")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/trajectories/sft_train_trajectory.jsonl",
        help="Path to training data JSONL",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/nutrimind-4b-sft",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Base model name or path",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=8192, help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument(
        "--verify_labels",
        action="store_true",
        help="Only verify loss mask labels, don't train",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="nutrimind-sft",
        help="W&B project name",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Check if data exists
    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Training data not found: {args.data_path}")

    # Set up wandb
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        report_to = "wandb"
    else:
        report_to = "none"

    # Load model with Unsloth optimizations
    logger.info(f"Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype="bfloat16",  # Full precision LoRA (no quantization)
        load_in_4bit=False,
        local_files_only=True,  # Mandatory for restricted network (AutoDL)
    )

    # Add LoRA adapters
    logger.info(f"Adding LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,  # Unsloth is optimized with 0 dropout
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )

    # Load and prepare dataset
    logger.info(f"Loading data from: {args.data_path}")
    full_dataset = load_trajectory_data(args.data_path)

    # Format messages using chat template
    logger.info("Formatting dataset with chat template (enable_thinking=True)")
    format_fn = formatting_func(tokenizer)
    full_dataset = full_dataset.map(format_fn, remove_columns=["messages", "tier"])

    def custom_loss_masking(batch):
        # We manually mask everything to -100 except assistant turns
        # This completely replaces train_on_responses_only and DataCollator
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            add_special_tokens=False
        )
        input_ids_list = tokens["input_ids"]
        attention_mask_list = tokens["attention_mask"]
        
        labels_list = []
        for input_ids in input_ids_list:
            labels = [-100] * len(input_ids)
            # Find instances of <|im_start|>assistant (151644, 77091)
            # and <|im_start|>user (151644, 872)
            instr_pos = []
            resp_pos = []
            for i in range(len(input_ids) - 1):
                if input_ids[i] == 151644:
                    if input_ids[i+1] == 872:
                        instr_pos.append(i)
                    elif input_ids[i+1] == 77091:
                        resp_pos.append(i + 2) # add 2 to skip the "<|im_start|>assistant" tokens themselves
            
            # For each assistant start, find the next user start
            for ast_start in resp_pos:
                next_usr = len(input_ids)
                for usr_start in instr_pos:
                    if usr_start > ast_start:
                        next_usr = usr_start
                        break
                # Unmask tokens from assistant content up to next user
                for i in range(ast_start, next_usr):
                    labels[i] = input_ids[i]
            
            labels_list.append(labels)
            
        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list
        }

    logger.info("Applying custom loss masking...")
    full_dataset = full_dataset.map(
        custom_loss_masking, 
        batched=True, 
        num_proc=4, 
        remove_columns=["text"]
    )

    # Train/eval split (90/10)
    split = full_dataset.train_test_split(test_size=0.1, seed=3407)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info(f"Split: {len(train_dataset)} train / {len(eval_dataset)} eval")

    # Calculate training steps
    total_samples = len(train_dataset)
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    steps_per_epoch = total_samples // effective_batch
    total_steps = steps_per_epoch * args.num_epochs

    logger.info(f"Training samples: {total_samples}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    logger.info(f"Effective batch size: {effective_batch}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total training steps: {total_steps}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        report_to=report_to,
        run_name=f"nutrimind-sft-r{args.lora_r}",
        seed=3407,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Disabled to avoid chunking tool JSON logic
        dataset_kwargs={"skip_prepare_dataset": True},
        args=training_args,
    )

    # Verify labels if requested
    if args.verify_labels:
        verify_labels(trainer, tokenizer, num_samples=3)
        logger.info("Label verification complete. Exiting without training.")
        return

    # Verify labels before training (quick sanity check)
    logger.info("Running quick label verification...")
    try:
        verify_labels(trainer, tokenizer, num_samples=1)
    except ValueError as e:
        logger.error(f"Label verification failed: {e}")
        logger.error(
            "Aborting training. Check train_on_responses_only configuration."
        )
        raise

    # Show GPU memory usage
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU: {gpu_stats.name}")
        logger.info(f"Max VRAM: {max_memory} GB")
        logger.info(f"Reserved VRAM before training: {start_gpu_memory} GB")

    # Train
    logger.info("Starting training...")
    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Log final stats
    logger.info(f"Training complete! Stats: {trainer_stats}")

    # Save final model
    final_model_path = Path(args.output_dir) / "final"
    logger.info(f"Saving final adapter to: {final_model_path}")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    logger.info("=" * 60)
    logger.info("SFT TRAINING COMPLETE")
    logger.info(f"Adapter saved to: {final_model_path}")
    logger.info(f"To merge adapter, run: python merge_adapter.py --adapter_path {final_model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
