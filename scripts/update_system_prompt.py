"""
Update system prompts in existing SFT training data to match the new agent environment.

This script replaces the system message (first message with role="system") in each
trajectory with the updated SYSTEM_PROMPT that includes the set_goal tool.

Usage:
    python scripts/update_system_prompt.py
    python scripts/update_system_prompt.py --input data/training/old.jsonl --output data/training/new.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the canonical system prompt from collection script
from src.training.sft.collect_trajectories_v2 import SYSTEM_PROMPT

DEFAULT_INPUT = PROJECT_ROOT / "data/training/sft_train_trajectory.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/training/sft_train_trajectory_updated.jsonl"


def update_system_prompt(input_path: Path, output_path: Path, dry_run: bool = False) -> dict:
    """Update system prompts in training data.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        dry_run: If True, only count and report without writing

    Returns:
        Statistics dict
    """
    stats = {"total": 0, "updated": 0, "unchanged": 0, "errors": 0}

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    updated_lines = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                traj = json.loads(line)
                messages = traj.get("messages", [])

                if not messages:
                    print(f"WARNING: Line {line_num} has no messages, skipping")
                    stats["errors"] += 1
                    continue

                # Find system message (should be first)
                if messages[0].get("role") == "system":
                    old_prompt = messages[0].get("content", "")
                    new_prompt = SYSTEM_PROMPT.strip()

                    if old_prompt.strip() != new_prompt:
                        messages[0]["content"] = new_prompt
                        stats["updated"] += 1
                    else:
                        stats["unchanged"] += 1
                else:
                    # No system message - insert one
                    messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT.strip()})
                    stats["updated"] += 1

                traj["messages"] = messages
                updated_lines.append(json.dumps(traj, ensure_ascii=False))

            except json.JSONDecodeError as e:
                print(f"ERROR: Line {line_num} - Invalid JSON: {e}")
                stats["errors"] += 1
            except Exception as e:
                print(f"ERROR: Line {line_num} - {e}")
                stats["errors"] += 1

    if not dry_run and updated_lines:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(updated_lines) + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true", help="Only count, don't write")
    parser.add_argument("--in-place", action="store_true", help="Overwrite input file")
    parser.add_argument("--show-diff", action="store_true", help="Show first trajectory diff")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if not args.in_place else input_path

    if args.show_diff:
        # Show what the new system prompt looks like
        print("=" * 60)
        print("NEW SYSTEM PROMPT (excerpt):")
        print("=" * 60)
        lines = SYSTEM_PROMPT.strip().split("\n")
        for line in lines[:30]:
            print(line)
        print("... [truncated, full prompt has", len(lines), "lines]")
        print("=" * 60)

        # Show a sample from existing data
        if input_path.exists():
            with open(input_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    traj = json.loads(first_line)
                    old_prompt = traj.get("messages", [{}])[0].get("content", "")[:500]
                    print("\nOLD SYSTEM PROMPT (first 500 chars):")
                    print("=" * 60)
                    print(old_prompt)
                    print("..." if len(old_prompt) == 500 else "")
                    print("=" * 60)
        return

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    if args.dry_run:
        print("[DRY RUN - no files will be written]")

    stats = update_system_prompt(input_path, output_path, args.dry_run)

    print(f"\n✅ Processing complete:")
    print(f"   Total:     {stats['total']}")
    print(f"   Updated:   {stats['updated']}")
    print(f"   Unchanged: {stats['unchanged']}")
    print(f"   Errors:    {stats['errors']}")

    if not args.dry_run and stats['updated'] > 0:
        print(f"\n✅ Written to: {output_path}")


if __name__ == "__main__":
    main()
