"""retrieve_knowledge tool — search the nutrition knowledge base.

Singleton HybridRetriever loaded on first call.
"""

import threading
import yaml
from src.utils.logger import logger
from src.retrieval.hybrid_retriever import HybridRetriever

_retriever: HybridRetriever | None = None
_init_lock = threading.Lock()

def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        with _init_lock:
            if _retriever is None:
                with open("configs/tools.yaml", "r") as f:
                    config = yaml.safe_load(f).get("rag", {})
                _retriever = HybridRetriever(config)
                logger.info("HybridRetriever initialized safely")
    return _retriever



def retrieve_knowledge(query: str, mode: str = "hybrid", top_k: int = 3) -> dict:
    """Search the nutrition knowledge base.

    Args:
        query: Natural language question or topic.
        mode: Retrieval strategy: "hybrid" (default), "semantic", or "keyword".
        top_k: Number of passages to return (default 3).

    Returns:
        dict with status, retrieval_quality, top_relevance_score, and passages.
    """
    if not query or not query.strip():
        return {"status": "error", "error_type": "empty_query", "message": "Query cannot be empty"}

    try:
        retriever = _get_retriever()
        results = retriever.retrieve(query=query, mode=mode)

        if not results:
            return {
                "status": "success",
                "retrieval_quality": "none",
                "top_relevance_score": 0.0,
                "data": {"passages": []}
            }

        passages = []
        max_score = 0.0
        for r in results[:top_k]:
            meta = r["metadata"]
            score = r.get("rerank_score", r.get("rrf_score", 0.0))
            max_score = max(max_score, score)

            passage = {
                "content": r["content"],
                "source": meta.get("document", "unknown"),
                "source_id": meta.get("source_id", "unknown"),
                "section": meta.get("section", ""),
                "url": meta.get("url", ""),
                "relevance_score": round(score, 4),
            }
            passages.append(passage)

        # No retrieval_quality classification - model judges relevance
        # by examining top_relevance_score + passage source/section/content
        return {
            "status": "success",
            "top_relevance_score": round(max_score, 4),
            "data": {"passages": passages},
        }

    except Exception as e:
        logger.error(f"retrieve_knowledge error: {e}")
        return {"status": "error", "error_type": "internal_error", "message": str(e)}
