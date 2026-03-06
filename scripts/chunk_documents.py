"""Chunk parsed documents into knowledge chunks with domain tagging.

Input:  data/parsed/*.json
Output: data/knowledge/chunks.jsonl
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

# Ensure project root is on path for src.* imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Mapping from source_id to primary domain (must match collect_sources.py)
SOURCE_DOMAINS = {
    "nih_ods": "micronutrients",
    "dga_2020": "dietary_guidelines",
    "who_sugars": "dietary_guidelines",
    "who_sodium": "dietary_guidelines",
    "issn_protein": "sports_nutrition",
    "issn_nutrient_timing": "sports_nutrition",
    "myplate": "meal_planning",
    "acog_pregnancy": "life_stage",
    # Tier 3: GI disease diet management
    "niddk_gallstones": "medical_nutrition",
    "niddk_gerd": "medical_nutrition",
    "niddk_ibs": "medical_nutrition",
    "niddk_celiac": "medical_nutrition",
    # Tier 3: Food allergy / intolerance
    "medlineplus_food_allergy": "food_safety",
    "niaid_food_allergy": "food_safety",  # kept for reference; replaced by medlineplus
}


def chunk_documents(parsed_dir: Path, output_path: Path, config: dict):
    from src.retrieval.chunker import StructureAwareChunker
    from src.retrieval.domain_tagger import assign_domains

    chunker = StructureAwareChunker(
        max_tokens=config.get("chunk_max_tokens", 256),
        overlap_tokens=config.get("chunk_overlap_tokens", 48),
        min_tokens=config.get("chunk_min_tokens", 30),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    domain_counter = Counter()
    files = sorted(parsed_dir.glob("*.json"))

    logger.info(f"Found {len(files)} parsed documents")

    for json_path in files:
        with open(json_path, "r", encoding="utf-8") as f:
            parsed_doc = json.load(f)

        source_id = parsed_doc.get("source_id", "unknown")
        primary_domain = SOURCE_DOMAINS.get(source_id, "micronutrients")

        chunks = chunker.chunk_document(parsed_doc)

        for chunk in chunks:
            domains = assign_domains(chunk, primary_domain)
            chunk["domains"] = domains
            for d in domains:
                domain_counter[d] += 1

        all_chunks.extend(chunks)
        logger.info(f"  {json_path.name}: {len(chunks)} chunks")

    # Write chunks.jsonl
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"\nTotal chunks: {len(all_chunks)}")
    logger.info(f"Domain distribution:")
    for domain, count in domain_counter.most_common():
        pct = count / len(all_chunks) * 100 if all_chunks else 0
        logger.info(f"  {domain}: {count} ({pct:.1f}%)")

    # Validation
    small_chunks = [c for c in all_chunks if c["metadata"]["token_count"] < config.get("chunk_min_tokens", 30)]
    if small_chunks:
        logger.warning(f"{len(small_chunks)} chunks below min_tokens (should be 0 after merging)")


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description="Chunk parsed documents into knowledge chunks.")
    parser.add_argument("--parsed-dir", type=str, default="data/parsed", help="Parsed JSON directory")
    parser.add_argument("--output", type=str, default="data/knowledge/chunks.jsonl", help="Output JSONL path")
    parser.add_argument("--config", type=str, default="configs/tools.yaml", help="Config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f).get("rag", {})

    chunk_documents(Path(args.parsed_dir), Path(args.output), config)
