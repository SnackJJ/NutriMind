"""RAG retrieval quality evaluation tests.

Covers all 9 domains with must_contain / must_not_contain assertions.

Note: Domain filtering was removed (2026-03-14 decision) - the four-stage pipeline
(ChromaDB + BM25 + RRF + reranker) provides sufficient precision without hard filtering.
"""

import pytest

EVAL_QUERIES = [
    # --- micronutrients ---
    {
        "query": "how much vitamin B12 do pregnant women need",
        "expected_domains": ["micronutrients", "life_stage"],
        "must_contain": ["b12"],
        # Note: must_not_contain "vitamin b6" removed — ACOG pregnancy nutrition tables
        # legitimately list B6 alongside B12; co-mention does not indicate wrong retrieval.
    },
    {
        "query": "vitamin D recommended daily intake for adults",
        "expected_domains": ["micronutrients"],
        "must_contain": ["vitamin d"],
    },
    {
        "query": "iron deficiency symptoms and dietary sources",
        "expected_domains": ["micronutrients"],
        "must_contain": ["iron"],
    },
    {
        "query": "zinc food sources and recommended intake",
        "expected_domains": ["micronutrients"],
        "must_contain": ["zinc"],
    },
    # --- dietary_guidelines ---
    {
        "query": "daily recommended sugar intake guidelines",
        "expected_domains": ["dietary_guidelines"],
        "must_contain": ["sugar"],
    },
    {
        "query": "sodium intake recommendations for adults",
        "expected_domains": ["dietary_guidelines"],
        "must_contain": ["sodium"],
    },
    # --- sports_nutrition ---
    {
        "query": "protein intake for strength training athletes",
        "expected_domains": ["sports_nutrition"],
        "must_contain": ["protein"],
    },
    {
        "query": "nutrient timing around exercise for muscle recovery",
        "expected_domains": ["sports_nutrition"],
        "must_contain": ["exercise"],
    },
    # --- life_stage ---
    {
        "query": "folate requirements during pregnancy",
        "expected_domains": ["life_stage"],
        "must_contain": ["folate"],
    },
    {
        "query": "calcium needs for elderly adults",
        "expected_domains": ["micronutrients"],
        "must_contain": ["calcium"],
    },
    # --- meal_planning ---
    {
        "query": "food sources rich in omega-3 fatty acids",
        "expected_domains": ["micronutrients", "meal_planning"],
        "must_contain": ["omega-3"],
    },
    # --- food_safety ---
    {
        "query": "vitamin A toxicity and upper intake levels",
        "expected_domains": ["micronutrients", "food_safety"],
        "must_contain": ["vitamin a"],
    },
    # --- supplements ---
    {
        "query": "vitamin B12 supplement forms and absorption",
        "expected_domains": ["micronutrients", "supplements"],
        "must_contain": ["b12"],
    },
    # --- medical_nutrition ---
    {
        "query": "potassium intake for blood pressure management",
        "expected_domains": ["micronutrients"],
        "must_contain": ["potassium"],
    },
    # --- disambiguation tests ---
    {
        "query": "high fiber foods for digestive health",
        "expected_domains": ["dietary_guidelines", "meal_planning"],
        "must_contain": ["fiber"],
        "must_not_contain": ["muscle fiber"],
    },
    # --- abbreviation expansion ---
    {
        "query": "rda for vitamin C",
        "expected_domains": ["micronutrients"],
        "must_contain": ["vitamin c"],
    },
    # --- protein in sports context ---
    {
        "query": "protein recommendations for athletes",
        "expected_domains": ["sports_nutrition"],
        "must_contain": ["protein"],
    },
    # --- selenium (less common nutrient) ---
    {
        "query": "selenium health benefits and food sources",
        "expected_domains": ["micronutrients"],
        "must_contain": ["selenium"],
    },
    # --- magnesium ---
    {
        "query": "magnesium daily requirements",
        "expected_domains": ["micronutrients"],
        "must_contain": ["magnesium"],
    },
    # --- choline ---
    {
        "query": "choline intake recommendations",
        "expected_domains": ["micronutrients"],
        "must_contain": ["choline"],
    },
]


@pytest.fixture(scope="module")
def retriever():
    """Load retriever once for all tests."""
    from src.tools.retrieve_knowledge import retrieve_knowledge
    return retrieve_knowledge


class TestRetrievalQuality:
    """Test retrieval quality across all domains."""

    @pytest.mark.parametrize("case", EVAL_QUERIES, ids=[c["query"][:50] for c in EVAL_QUERIES])
    def test_query(self, retriever, case):
        # Domain filtering removed (2026-03-14) - call without domain param
        result = retriever(case["query"])

        assert result["status"] == "success", f"Failed: {case['query']} -> {result}"

        passages = result["data"]["passages"]
        assert len(passages) > 0, f"No passages for: {case['query']}"

        full_text = " ".join(p["content"].lower() for p in passages)

        for term in case.get("must_contain", []):
            assert term.lower() in full_text, (
                f"Missing '{term}' for query: {case['query']}\n"
                f"Got passages from: {[p['source'] for p in passages]}"
            )

        for term in case.get("must_not_contain", []):
            assert term.lower() not in full_text, (
                f"Unexpected '{term}' for query: {case['query']}\n"
                f"Got: {full_text[:200]}"
            )

    def test_empty_query(self, retriever):
        result = retriever("")
        assert result["status"] == "error"
        assert result["error_type"] == "empty_query"

    def test_source_attribution(self, retriever):
        """All results should have source attribution."""
        result = retriever("vitamin B12 recommended intake")
        assert result["status"] == "success"

        for passage in result["data"]["passages"]:
            assert passage["source"], "Missing source"
            assert passage["source_id"], "Missing source_id"
            assert passage["url"], "Missing URL"
