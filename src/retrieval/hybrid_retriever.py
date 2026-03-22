"""Hybrid retriever combining semantic search (ChromaDB) + BM25 + reranking.

Features:
- ChromaDB semantic search (bge-small-en-v1.5)
- BM25 keyword search
- Reciprocal Rank Fusion (RRF) merge
- Lazy-loaded BGE reranker with graceful fallback
- Query preprocessing with abbreviation expansion
- Threshold only when reranker is available (RRF scores are not comparable)

Note: Domain filtering was removed in 2026-03. With ~1700 chunks, the four-stage
pipeline provides sufficient precision without hard filtering.
"""

import json
import logging
import pickle
import re
import threading
from typing import Dict, List, Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# --- BM25 tokenization (shared with build_indexes.py) ---

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


# --- Query preprocessing ---

ABBREVIATIONS = {
    "t2d": "type 2 diabetes",
    "t1d": "type 1 diabetes",
    "bp": "blood pressure",
    "ckd": "chronic kidney disease",
    "cvd": "cardiovascular disease",
    "gi": "glycemic index",
    "rda": "recommended dietary allowance",
    "dri": "dietary reference intake",
    "ul": "tolerable upper intake level",
}


def preprocess_query(query: str) -> str:
    """Normalize abbreviations and expand medical shorthand."""
    normalized = query.lower()
    for abbrev, full in ABBREVIATIONS.items():
        normalized = re.sub(rf'\b{abbrev}\b', full, normalized)
    return normalized


class HybridRetriever:
    def __init__(self, config: dict, collection_name: str = "nutrition_knowledge", instruction: str = "Represent this sentence for searching relevant passages: "):
        # --- Semantic index ---
        chroma_path = config.get("chroma_db_path", "data/knowledge_db")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_collection(collection_name)
        self.instruction = instruction

        # --- Embedding model ---
        embedding_model_name = config.get("embedding_model", "BAAI/bge-small-en-v1.5")
        self._embedding_model = SentenceTransformer(embedding_model_name)

        # --- Keyword index ---
        bm25_path = config.get("bm25_index_path", "data/knowledge_bm25/index.pkl")
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_chunk_ids = bm25_data["chunk_ids"]
        self.bm25_chunk_contents = bm25_data["chunk_contents"]
        self.bm25_chunk_metadatas = bm25_data["chunk_metadatas"]

        # --- Reranker (lazy load with thread lock) ---
        self._reranker = None
        self._reranker_failed = False
        self._reranker_lock = threading.Lock()
        self._reranker_model_name = config.get("reranker_model", "BAAI/bge-reranker-base")

        # --- Parameters ---
        self.semantic_top_k = config.get("semantic_top_k", 20)
        self.bm25_top_k = config.get("bm25_top_k", 20)
        self.rerank_top_k = config.get("rerank_top_k", 10)
        self.final_top_k = config.get("final_top_k", 3)
        self.relevance_threshold = config.get("relevance_threshold", 0.3)
        self.rrf_k = config.get("rrf_k", 60)

    @property
    def reranker(self):
        if self._reranker is None and not self._reranker_failed:
            with self._reranker_lock:
                # Double-check after acquiring lock
                if self._reranker is None and not self._reranker_failed:
                    try:
                        from sentence_transformers import CrossEncoder
                        self._reranker = CrossEncoder(self._reranker_model_name)
                        logger.info("Reranker loaded successfully")
                    except Exception as e:
                        self._reranker_failed = True
                        logger.warning(
                            f"Reranker failed to load: {e}. Falling back to RRF scores. "
                            f"If using HuggingFace models, ensure HF_ENDPOINT or HF_HOME is set correctly."
                        )
        return self._reranker

    def retrieve(self, query: str, mode: str = "hybrid", allow_fallback: bool = True) -> list[dict]:
        """Run the hybrid retrieval pipeline or specialized search.

        Args:
            query: Natural language question.
            mode: "hybrid" (default), "semantic" (ChromaDB only), or "keyword" (BM25 only).
            allow_fallback: Whether to return the best result with low_confidence=True
                            if no results pass the threshold.

        Returns:
            List of result dicts with content, metadata, and scores.
        """
        query = preprocess_query(query)

        # 1. Semantic search (ChromaDB)
        sem_results = []
        if mode in ("hybrid", "semantic"):
            query_embedding = self._embed_query(query)
            sem_results = self._parse_chroma_results(
                self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=self.semantic_top_k,
                )
            )

        # 2. BM25 keyword search
        bm25_results = []
        if mode in ("hybrid", "keyword"):
            bm25_tokens = simple_tokenize(query)
            bm25_scores = self.bm25.get_scores(bm25_tokens)
            bm25_top_idx = np.argsort(bm25_scores)[-self.bm25_top_k:][::-1]

            for i in bm25_top_idx:
                if bm25_scores[i] <= 0:
                    continue
                bm25_results.append({
                    "id": self.bm25_chunk_ids[i],
                    "content": self.bm25_chunk_contents[i],
                    "metadata": self.bm25_chunk_metadatas[i],
                    "score": float(bm25_scores[i]),
                })

        # 3. Merge / Candidates selection
        if mode == "hybrid":
            candidates = self._rrf_merge(sem_results, bm25_results)
        elif mode == "semantic":
            candidates = sem_results
        else:  # keyword
            candidates = bm25_results

        # 4. Rerank
        rerank_candidates = candidates[:self.rerank_top_k]
        if rerank_candidates:
            rerank_candidates = self._rerank(query, rerank_candidates)

        # 5. Threshold + low_confidence fallback
        if self.reranker is not None:
            final = [
                c for c in rerank_candidates
                if c.get("rerank_score", 0) >= self.relevance_threshold
            ][:self.final_top_k]

            # Low confidence fallback
            if not final and rerank_candidates and allow_fallback:
                best = rerank_candidates[0]
                best["low_confidence"] = True
                final = [best]
                logger.debug(f"No results above threshold ({self.relevance_threshold}), mode={mode}")
        else:
            # Reranker unavailable: trust ranking, take top_k directly
            final = rerank_candidates[:self.final_top_k]

        return final

    def _embed_query(self, query: str) -> list[float]:
        """BGE models use instruction prefix for queries."""
        return self._embedding_model.encode(
            self.instruction + query,
            normalize_embeddings=True,
        ).tolist()

    def _parse_chroma_results(self, results: dict) -> list[dict]:
        """Convert ChromaDB query results to standard format."""
        parsed = []
        if not results["ids"] or not results["ids"][0]:
            return parsed

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)

        for i in range(len(ids)):
            parsed.append({
                "id": ids[i],
                "content": documents[i],
                "metadata": metadatas[i],
                "score": 1.0 - distances[i],  # cosine distance -> similarity
            })

        return parsed

    def _rrf_merge(self, list_a: list[dict], list_b: list[dict]) -> list[dict]:
        """Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank))."""
        scores: Dict[str, float] = {}
        candidates: Dict[str, dict] = {}

        for rank, item in enumerate(list_a):
            scores[item["id"]] = scores.get(item["id"], 0) + 1.0 / (self.rrf_k + rank + 1)
            candidates[item["id"]] = item

        for rank, item in enumerate(list_b):
            scores[item["id"]] = scores.get(item["id"], 0) + 1.0 / (self.rrf_k + rank + 1)
            if item["id"] not in candidates:
                candidates[item["id"]] = item

        for cid in candidates:
            candidates[cid]["rrf_score"] = scores[cid]

        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        return [candidates[cid] for cid in sorted_ids]

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Rerank candidates using cross-encoder. Falls back to RRF order."""
        if self.reranker is None:
            logger.warning("Reranker unavailable, using RRF scores for ranking")
            return candidates

        pairs = [(query, c["content"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates
