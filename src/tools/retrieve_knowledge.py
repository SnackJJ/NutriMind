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



def retrieve_knowledge(query: str, domain: str = None, top_k: int = 3) -> dict:
    """Search the nutrition knowledge base.

    Args:
        query: Natural language question or topic.
        domain: Optional domain filter (micronutrients, sports_nutrition, etc.).
        top_k: Number of passages to return (default 3).

    Returns:
        dict with status + passages including source attribution.
    """
    if not query or not query.strip():
        return {"status": "error", "error_type": "empty_query", "message": "Query cannot be empty"}

    try:
        retriever = _get_retriever()
        results = retriever.retrieve(query=query, domain_filter=domain)

        if not results:
            return {"status": "error", "error_type": "no_relevant_results", "message": "No relevant results found"}

        passages = []
        for r in results[:top_k]:
            meta = r["metadata"]
            passage = {
                "content": r["content"],
                "source": meta.get("document", "unknown"),
                "source_id": meta.get("source_id", "unknown"),
                "section": meta.get("section", ""),
                "url": meta.get("url", ""),
                "relevance_score": r.get("rerank_score", r.get("rrf_score", 0.0)),
            }
            if r.get("low_confidence"):
                passage["low_confidence"] = True
            passages.append(passage)

        return {
            "status": "success",
            "data": {"passages": passages},
        }

    except Exception as e:
        logger.error(f"retrieve_knowledge error: {e}")
        return {"status": "error", "error_type": "internal_error", "message": str(e)}
