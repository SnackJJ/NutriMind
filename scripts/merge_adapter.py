#!/usr/bin/env python3
import os
import argparse
from unsloth import FastLanguageModel

# Force any auto-downloads to use mainland mirror to prevent AutoDL network timeout
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA final dir")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for merged model")
    args = parser.parse_args()

    print(f"Loading adapter from {args.adapter_path}")
    print("This might take a minute...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_path,
        max_seq_length=6144,
        dtype="bfloat16",
        load_in_4bit=False,
        local_files_only=True, # Critical to avoid HF timeout
    )
    
    print(f"Merging weights and saving to {args.output_path} (This takes a few minutes, please wait)...")
    # This specifically bakes the LoRA weights into the base Bf16 weights permanently
    model.save_pretrained_merged(args.output_path, tokenizer, save_method="merged_16bit")
    print("\n✅ Successfully created standalone, vLLM-ready model! You can now use vLLM.")

if __name__ == "__main__":
    main()
