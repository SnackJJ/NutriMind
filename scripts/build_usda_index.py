
import argparse
import logging
import pickle
import re
import sqlite3
import json
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "to", "for",
    "with", "on", "at", "by", "from", "as", "into", "about", "between",
    "and", "or", "but", "not", "no", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "we", "our", "you", "your",
})

def simple_tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s\-]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t and t not in STOPWORDS]

def build_usda_indexes(db_path: Path, chroma_db_path: Path, bm25_path: Path, config: dict):
    # Load data from SQLite
    logger.info(f"Loading data from {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, fdc_id, description, category FROM foods")
    rows = cursor.fetchall()
    conn.close()

    foods = []
    for row in rows:
        foods.append({
            "id": str(row[0]),
            "fdc_id": row[1],
            "description": row[2],
            "category": row[3] or ""
        })

    logger.info(f"Loaded {len(foods)} food items")

    if not foods:
        logger.error("No food items to index!")
        return

    # --- Embedding ---
    embedding_model_name = config.get("embedding_model", "BAAI/bge-small-en-v1.5")
    logger.info(f"Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)

    logger.info("Embedding food descriptions...")
    descriptions = [f["description"] for f in foods]
    embeddings = embedding_model.encode(descriptions, normalize_embeddings=True, show_progress_bar=True)

    # --- ChromaDB ---
    logger.info(f"Building ChromaDB at {chroma_db_path}")
    chroma_db_path.parent.mkdir(parents=True, exist_ok=True)

    import shutil
    if chroma_db_path.exists():
        shutil.rmtree(chroma_db_path)

    client = chromadb.PersistentClient(path=str(chroma_db_path))
    collection = client.create_collection(
        name="usda_foods",
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 500
    for start in range(0, len(foods), batch_size):
        end = min(start + batch_size, len(foods))
        batch_foods = foods[start:end]

        collection.add(
            ids=[f["id"] for f in batch_foods],
            embeddings=[embeddings[i].tolist() for i in range(start, end)],
            documents=[f["description"] for f in batch_foods],
            metadatas=[{"category": f["category"], "fdc_id": f["fdc_id"]} for f in batch_foods],
        )

    logger.info(f"  ChromaDB: {collection.count()} foods indexed")

    # --- BM25 ---
    logger.info(f"Building BM25 index at {bm25_path}")
    bm25_path.parent.mkdir(parents=True, exist_ok=True)

    tokenized_corpus = [simple_tokenize(f["description"]) for f in foods]
    bm25 = BM25Okapi(tokenized_corpus)

    index_data = {
        "bm25": bm25,
        "chunk_ids": [f["id"] for f in foods],
        "chunk_contents": [f["description"] for f in foods],
        "chunk_metadatas": [{"category": f["category"], "fdc_id": f["fdc_id"]} for f in foods],
    }

    with open(bm25_path, "wb") as f:
        pickle.dump(index_data, f)

    logger.info(f"  BM25: {len(foods)} foods indexed")
    logger.info("\nIndex build complete!")

if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser(description="Build ChromaDB + BM25 indexes for USDA foods.")
    parser.add_argument("--config", type=str, default="configs/tools.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        full_config = yaml.safe_load(f)
        rag_config = full_config.get("rag", {})
        food_config = full_config.get("food_search", {})
        db_path = Path(full_config.get("database_path", "data/usda.db"))

    # Merge rag config for model names
    food_config.update({
        "embedding_model": rag_config.get("embedding_model"),
        "reranker_model": rag_config.get("reranker_model")
    })

    build_usda_indexes(
        db_path=db_path,
        chroma_db_path=Path(food_config.get("chroma_db_path")),
        bm25_path=Path(food_config.get("bm25_index_path")),
        config=food_config
    )
