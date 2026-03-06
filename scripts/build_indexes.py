"""Build ChromaDB + BM25 indexes from contextualized chunks.

Input:  data/knowledge/chunks_contextualized.jsonl
Output: data/knowledge_db/ (ChromaDB)
        data/knowledge_bm25/index.pkl (BM25)
"""

import argparse
import json
import logging
import pickle
import re
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Domain list must match domain_tagger.py
ALL_DOMAINS = [
    "micronutrients", "dietary_guidelines", "sports_nutrition",
    "medical_nutrition", "life_stage", "meal_planning",
    "food_safety", "supplements", "weight_management",
]

STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "to", "for",
    "with", "on", "at", "by", "from", "as", "into", "about", "between",
    "and", "or", "but", "not", "no", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "we", "our", "you", "your",
})


def simple_tokenize(text: str) -> list[str]:
    """BM25 tokenization. Keeps hyphens within words (omega-3, high-fiber)."""
    text = text.lower()
    text = re.sub(r'[^\w\s\-]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t and t not in STOPWORDS]


def build_chromadb_metadata(chunk: dict) -> dict:
    """Build ChromaDB-compatible metadata with boolean domain fields."""
    meta = chunk["metadata"]
    domains = set(chunk.get("domains", []))

    result = {
        "source_id": meta.get("source_id", ""),
        "document": meta.get("document", ""),
        "section": meta.get("section", ""),
        "heading_hierarchy": json.dumps(meta.get("heading_hierarchy", [])),
        "url": meta.get("url", ""),
        "page": meta.get("page") or 0,
        "source_type": meta.get("source_type", ""),
        "token_count": meta.get("token_count", 0),
        "is_table": meta.get("is_table", False),
    }

    for domain in ALL_DOMAINS:
        result[f"domain_{domain}"] = domain in domains

    return result


def build_indexes(input_path: Path, chroma_db_path: Path, bm25_path: Path, config: dict):
    # Load chunks
    chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    logger.info(f"Loaded {len(chunks)} contextualized chunks")

    if not chunks:
        logger.error("No chunks to index!")
        return

    # --- Embedding ---
    embedding_model_name = config.get("embedding_model", "BAAI/bge-small-en-v1.5")
    logger.info(f"Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)

    # Embed contextualized_content (documents don't use query prefix)
    logger.info("Embedding chunks...")
    ctx_contents = [c["contextualized_content"] for c in chunks]
    embeddings = embedding_model.encode(ctx_contents, normalize_embeddings=True, show_progress_bar=True)

    # --- ChromaDB ---
    logger.info(f"Building ChromaDB at {chroma_db_path}")
    chroma_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing DB
    import shutil
    if chroma_db_path.exists():
        shutil.rmtree(chroma_db_path)

    client = chromadb.PersistentClient(path=str(chroma_db_path))
    distance_metric = config.get("chroma_distance_metric", "cosine")

    collection = client.create_collection(
        name="nutrition_knowledge",
        metadata={"hnsw:space": distance_metric},
    )

    # Add in batches (ChromaDB limit)
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch_chunks = chunks[start:end]

        collection.add(
            ids=[c["id"] for c in batch_chunks],
            embeddings=[embeddings[i].tolist() for i in range(start, end)],
            documents=[c["original_content"] for c in batch_chunks],
            metadatas=[build_chromadb_metadata(c) for c in batch_chunks],
        )

    logger.info(f"  ChromaDB: {collection.count()} documents indexed")

    # --- BM25 ---
    logger.info(f"Building BM25 index at {bm25_path}")
    bm25_path.parent.mkdir(parents=True, exist_ok=True)

    # Tokenize contextualized content for BM25
    tokenized_corpus = [simple_tokenize(c["contextualized_content"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # Build metadata list for domain filtering during retrieval
    bm25_metadatas = [build_chromadb_metadata(c) for c in chunks]

    index_data = {
        "bm25": bm25,
        "chunk_ids": [c["id"] for c in chunks],
        "chunk_contents": [c["original_content"] for c in chunks],
        "chunk_contextualized": [c["contextualized_content"] for c in chunks],
        "chunk_metadatas": bm25_metadatas,
    }

    with open(bm25_path, "wb") as f:
        pickle.dump(index_data, f)

    logger.info(f"  BM25: {len(chunks)} documents indexed")

    # --- Validation ---
    logger.info("\nValidation:")
    logger.info(f"  ChromaDB count: {collection.count()}")
    logger.info(f"  BM25 count: {len(index_data['chunk_ids'])}")
    logger.info(f"  Chunks count: {len(chunks)}")

    assert collection.count() == len(chunks), "ChromaDB count mismatch!"
    assert len(index_data["chunk_ids"]) == len(chunks), "BM25 count mismatch!"

    # Test domain filter
    test_result = collection.query(
        query_embeddings=[embeddings[0].tolist()],
        n_results=5,
        where={"domain_micronutrients": True},
    )
    logger.info(f"  Domain filter test (micronutrients): {len(test_result['ids'][0])} results")

    logger.info("\nIndex build complete!")


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description="Build ChromaDB + BM25 indexes.")
    parser.add_argument("--input", type=str, default="data/knowledge/chunks_contextualized.jsonl")
    parser.add_argument("--config", type=str, default="configs/tools.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f).get("rag", {})

    build_indexes(
        input_path=Path(args.input),
        chroma_db_path=Path(config.get("chroma_db_path", "data/knowledge_db")),
        bm25_path=Path(config.get("bm25_index_path", "data/knowledge_bm25/index.pkl")),
        config=config,
    )
