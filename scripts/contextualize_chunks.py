"""Add LLM-generated context to each chunk (Contextual Retrieval).

Input:  data/knowledge/chunks.jsonl
Output: data/knowledge/chunks_contextualized.jsonl
Cache:  data/knowledge/context_cache.json
"""

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONTEXT_PROMPT = """<document>
{document_title} — {source_description}
</document>

Here is a chunk from the section "{section_heading}":

<chunk>
{chunk_text}
</chunk>

Provide a brief context (1-2 sentences, under 50 tokens) that:
1. Identifies the specific nutrient/topic discussed
2. Explains what aspect this chunk covers (RDA, food sources, safety, etc.)
3. Notes the target population if specific (pregnant women, elderly, athletes)

Context:"""

# Source descriptions for the context prompt
SOURCE_DESCRIPTIONS = {
    "nih_ods": "NIH Office of Dietary Supplements Health Professional Fact Sheet",
    "dga_2020": "Dietary Guidelines for Americans 2020-2025",
    "who_sugars": "WHO Guideline on Sugars Intake for Adults and Children",
    "who_sodium": "WHO Guideline on Sodium Intake for Adults and Children",
    "issn_protein": "International Society of Sports Nutrition Position Stand on Protein and Exercise",
    "issn_nutrient_timing": "International Society of Sports Nutrition Position Stand on Nutrient Timing",
    "myplate": "USDA MyPlate Dietary Guidelines",
    "acog_pregnancy": "American College of Obstetricians and Gynecologists Nutrition During Pregnancy",
}


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def call_gemini_flash(prompt: str, max_tokens: int = 2048) -> str:
    """Call Gemini 2.5 Flash via OpenAI-compatible endpoint.

    max_tokens must be large enough to cover internal thinking tokens (~200-800)
    plus the actual output (~60 tokens). Only message.content (the final output)
    is returned — thinking is internal and never stored.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )

    content = response.choices[0].message.content
    if content is None:
        finish_reason = response.choices[0].finish_reason
        raise RuntimeError(f"Gemini returned None content (finish_reason={finish_reason})")
    return content.strip()


def contextualize_chunks(input_path: Path, output_path: Path, cache_path: Path, config: dict):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.get("embedding_model", "BAAI/bge-small-en-v1.5"))
    max_context_tokens = config.get("context_max_tokens", 50)

    # Load cache
    cache = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info(f"Loaded {len(cache)} cached contexts")

    # Load chunks
    chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    logger.info(f"Processing {len(chunks)} chunks")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    contextualized = []
    new_contexts = 0
    errors = 0

    for i, chunk in enumerate(chunks):
        c_hash = content_hash(chunk["content"])

        if c_hash in cache:
            context = cache[c_hash]
        else:
            source_id = chunk["metadata"].get("source_id", "unknown")
            prompt = CONTEXT_PROMPT.format(
                document_title=chunk["metadata"].get("document", ""),
                source_description=SOURCE_DESCRIPTIONS.get(source_id, ""),
                section_heading=chunk["metadata"].get("section", ""),
                chunk_text=chunk["content"],
            )

            try:
                context = call_gemini_flash(prompt)

                # Enforce token limit
                context_tokens = tokenizer.encode(context, add_special_tokens=False)
                if len(context_tokens) > max_context_tokens:
                    context = tokenizer.decode(context_tokens[:max_context_tokens])

                cache[c_hash] = context
                new_contexts += 1

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error contextualizing chunk {chunk['id']}: {e}")
                context = ""
                errors += 1
                time.sleep(2.0)  # Backoff on error

        contextualized_content = f"{context} | {chunk['content']}" if context else chunk["content"]

        ctx_chunk = {
            "id": chunk["id"],
            "original_content": chunk["content"],
            "context": context,
            "contextualized_content": contextualized_content,
            "metadata": chunk["metadata"],
            "domains": chunk.get("domains", []),
        }
        contextualized.append(ctx_chunk)

        if (i + 1) % 20 == 0:
            logger.info(f"  Progress: {i + 1}/{len(chunks)} ({new_contexts} new, {errors} errors)")
            # Save cache periodically
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in contextualized:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Final cache save
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    logger.info(f"\nDone: {len(contextualized)} chunks contextualized")
    logger.info(f"  New contexts generated: {new_contexts}")
    logger.info(f"  From cache: {len(contextualized) - new_contexts - errors}")
    logger.info(f"  Errors: {errors}")


if __name__ == "__main__":
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Add LLM context to chunks.")
    parser.add_argument("--input", type=str, default="data/knowledge/chunks.jsonl")
    parser.add_argument("--output", type=str, default="data/knowledge/chunks_contextualized.jsonl")
    parser.add_argument("--cache", type=str, default="data/knowledge/context_cache.json")
    parser.add_argument("--config", type=str, default="configs/tools.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f).get("rag", {})

    contextualize_chunks(Path(args.input), Path(args.output), Path(args.cache), config)
