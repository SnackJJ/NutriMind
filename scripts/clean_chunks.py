"""Clean chunks.jsonl by removing noise and short chunks.

Input:  data/knowledge/chunks.jsonl
Output: data/knowledge/chunks_cleaned.jsonl (or overwrites input with --inplace)

Cleaning rules:
1. Remove chunks with < min_tokens (default: 50)
2. Remove chunks containing navigation boilerplate patterns
3. Remove chunks that are primarily citations/references without content
"""

import argparse
import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Patterns indicating navigation/boilerplate content
NOISE_PATTERNS = [
    # Social sharing / navigation
    r"^Frequently Asked Questions\s*-\s*Share",
    r"Facebook\s*-\s*Email\s*-\s*Link",
    r"URL has been copied to the clipboard",
    r"^Share\s*-\s*(Facebook|Twitter|Email)",
    # PDF artifacts
    r"^Page \d+ of \d+$",
    r"^\d+\s*$",  # Just page numbers
    # Broken links / errors (but not page numbers like "391-404")
    r"404\s+(Page\s+)?[Nn]ot [Ff]ound",
    # Empty structural markers
    r"^(Expand|Collapse)\s*(All|Section)?\s*$",
    # Author conflict of interest / disclosure statements
    r"^[A-Z]{2,4}\s+(consults with|has received grants|serves on)",
    r"receives?\s+external funding from companies",
    r"served as an? expert witness",
]

# Patterns at start of content indicating low-value chunks
LOW_VALUE_START_PATTERNS = [
    r"^Data Source:\s*",  # Citation-only chunks
    r"^Note:\s*\d+\s*$",  # Footnote markers only
    r"^\[PubMed abstract\]",  # Reference-only
]

# Compile patterns for efficiency
NOISE_REGEXES = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]
LOW_VALUE_REGEXES = [re.compile(p, re.IGNORECASE) for p in LOW_VALUE_START_PATTERNS]


def is_noise_chunk(content: str) -> tuple[bool, str]:
    """Check if content matches noise patterns.

    Returns:
        (is_noise, reason)
    """
    # Check noise patterns
    for i, regex in enumerate(NOISE_REGEXES):
        if regex.search(content):
            return True, f"noise_pattern_{i}"

    # Check low-value start patterns (only if chunk is short)
    if len(content) < 200:
        for i, regex in enumerate(LOW_VALUE_REGEXES):
            if regex.match(content):
                return True, f"low_value_start_{i}"

    # Check if content is mostly non-alphabetic (tables of numbers, etc.)
    alpha_chars = sum(1 for c in content if c.isalpha())
    if len(content) > 50 and alpha_chars / len(content) < 0.3:
        return True, "low_alpha_ratio"

    return False, ""


def clean_chunks(input_path: Path, output_path: Path, min_tokens: int = 50, dry_run: bool = False):
    """Clean chunks by removing noise and short content."""

    # Load chunks
    chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    logger.info(f"Loaded {len(chunks)} chunks from {input_path}")

    # Track removal reasons
    removed = {
        "short": [],
        "noise": [],
    }
    kept = []

    for chunk in chunks:
        token_count = chunk.get("metadata", {}).get("token_count", 0)
        content = chunk.get("content", "")
        chunk_id = chunk.get("id", "unknown")

        # Check minimum tokens
        if token_count < min_tokens:
            removed["short"].append({
                "id": chunk_id,
                "tokens": token_count,
                "preview": content[:80],
            })
            continue

        # Check noise patterns
        is_noise, reason = is_noise_chunk(content)
        if is_noise:
            removed["noise"].append({
                "id": chunk_id,
                "reason": reason,
                "preview": content[:80],
            })
            continue

        kept.append(chunk)

    # Report
    logger.info(f"\n{'='*60}")
    logger.info(f"Cleaning Summary:")
    logger.info(f"  Original chunks: {len(chunks)}")
    logger.info(f"  Kept: {len(kept)}")
    logger.info(f"  Removed (short <{min_tokens} tokens): {len(removed['short'])}")
    logger.info(f"  Removed (noise patterns): {len(removed['noise'])}")
    logger.info(f"{'='*60}")

    # Show removed chunks
    if removed["short"]:
        logger.info(f"\nShort chunks removed ({len(removed['short'])}):")
        for item in removed["short"][:10]:
            logger.info(f"  [{item['tokens']}] {item['id'][:50]}")
            logger.info(f"       '{item['preview']}...'")
        if len(removed["short"]) > 10:
            logger.info(f"  ... and {len(removed['short']) - 10} more")

    if removed["noise"]:
        logger.info(f"\nNoise chunks removed ({len(removed['noise'])}):")
        for item in removed["noise"][:10]:
            logger.info(f"  [{item['reason']}] {item['id'][:50]}")
            logger.info(f"       '{item['preview']}...'")
        if len(removed["noise"]) > 10:
            logger.info(f"  ... and {len(removed['noise']) - 10} more")

    # Write output
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in kept:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        logger.info(f"\nWrote {len(kept)} cleaned chunks to {output_path}")
    else:
        logger.info(f"\n[DRY RUN] Would write {len(kept)} chunks to {output_path}")

    return kept, removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean chunks.jsonl")
    parser.add_argument("--input", type=str, default="data/knowledge/chunks.jsonl")
    parser.add_argument("--output", type=str, default="data/knowledge/chunks_cleaned.jsonl")
    parser.add_argument("--min-tokens", type=int, default=50, help="Minimum token count")
    parser.add_argument("--inplace", action="store_true", help="Overwrite input file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = input_path if args.inplace else Path(args.output)

    clean_chunks(input_path, output_path, args.min_tokens, args.dry_run)
