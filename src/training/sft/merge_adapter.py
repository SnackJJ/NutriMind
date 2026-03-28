#!/usr/bin/env python3
"""
Merge LoRA Adapter into Base Model

Usage:
    python merge_adapter.py --adapter_path models/nutrimind-4b-sft/final
    python merge_adapter.py --adapter_path models/nutrimind-4b-sft/final --verify
    python merge_adapter.py --adapter_path models/nutrimind-4b-sft/final --output_dir models/nutrimind-4b-sft-merged
"""

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def verify_merged_model(model, tokenizer):
    """Verify the merged model can generate sensible output."""
    logger.info("=" * 60)
    logger.info("VERIFYING MERGED MODEL")
    logger.info("=" * 60)

    # Simple nutrition query
    test_messages = [
        {
            "role": "system",
            "content": "You are NutriMind, a specialized AI nutrition assistant.",
        },
        {"role": "user", "content": "How much protein is in 100g chicken breast?"},
    ]

    # Apply chat template with thinking mode
    prompt = tokenizer.apply_chat_template(
        test_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    logger.info(f"Test prompt:\n{prompt[:200]}...")

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
    logger.info(f"Generated response:\n{response}")

    # Basic checks
    checks = {
        "has_think_tag": "<think>" in response,
        "has_tool_call": "<tool_call>" in response,
        "not_empty": len(response.strip()) > 10,
    }

    logger.info(f"\nVerification checks:")
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {check}: {status}")

    # At least not_empty should pass
    if not checks["not_empty"]:
        logger.error("Model generated empty or very short response!")
        return False

    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION COMPLETE")
    logger.info("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to trained LoRA adapter",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Base model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for merged model (default: adapter_path-merged)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification after merging",
    )
    parser.add_argument(
        "--save_4bit",
        action="store_true",
        help="Save in 4-bit precision instead of default 16-bit (smaller but less compatible with vLLM)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=8192,
        help="Maximum sequence length",
    )
    args = parser.parse_args()

    # Set output dir
    if args.output_dir is None:
        args.output_dir = str(Path(args.adapter_path).parent / "merged")

    # Check adapter exists
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    # Import Unsloth
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("Unsloth not installed. Install with: pip install unsloth")
        raise

    # Load base model with adapter
    logger.info(f"Loading base model: {args.base_model}")
    logger.info(f"Loading adapter from: {adapter_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Verify before merge if requested
    if args.verify:
        logger.info("Testing adapter model before merge...")
        FastLanguageModel.for_inference(model)
        verify_merged_model(model, tokenizer)

    # Merge and save
    if args.save_4bit:
        save_method = "merged_4bit_forced"
        logger.info(f"Merging adapter and saving 4-bit to: {args.output_dir}")
    else:
        save_method = "merged_16bit"
        logger.info(f"Merging adapter and saving 16-bit to: {args.output_dir}")

    model.save_pretrained_merged(
        args.output_dir,
        tokenizer,
        save_method=save_method,
    )

    logger.info(f"Merged model saved to: {args.output_dir} (method={save_method})")

    # Verify merged model if requested
    if args.verify:
        logger.info("Loading and verifying merged model...")
        # Load the merged model fresh
        merged_model, merged_tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.output_dir,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(merged_model)
        success = verify_merged_model(merged_model, merged_tokenizer)
        if not success:
            logger.warning("Verification had issues - check output manually")

    logger.info("=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info(f"Merged model: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
