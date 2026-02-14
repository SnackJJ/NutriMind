# Phase 1.2: RAG Knowledge Base — v3 Rewrite

- Status: done
- Spec: docs/specs/rag.md (v3)

## Background

v1 implementation had critical bugs (B1-B4), data quality issues (D1-D5), and design flaws (S1-S4).
After a comprehensive design review, the decision was made to adopt **Contextual Retrieval**
(Anthropic, Sept 2024) which reduces retrieval failure rate by 67%.

**Key changes from v2 plan**:
1. Add Contextual Retrieval step (LLM-generated context prepended to chunks)
2. Use docling for PDF parsing (replaces pdfplumber)
3. Boolean domain metadata for ChromaDB native filtering
4. Two-level domain tagging to reduce false positives
5. Download all 8 sources (not just NIH + ISSN)
6. Remove safety_boundaries domain (T4 handled at orchestrator layer)
7. Keep hyphens in BM25 tokens (preserve "omega-3", "high-fiber")

---

## Completed (from v1)

- [x] Source collection script structure (NIH ODS HTML, ISSN PDFs)
- [x] NIH fact sheet HTML parser (verified working, 61/616 sections have tables)
- [x] Basic project structure

---

## Tasks (v3 Rewrite)

### Phase A: Clean Slate (DONE)

- [x] **A.1** Delete all RAG code files
- [x] **A.2** Delete all RAG data files (ChromaDB, BM25, chunks.jsonl)
- [x] **A.3** Keep `data/parsed/` (25 files from NIH + ISSN)
- [x] **A.4** Keep `scripts/collect_sources.py` and `scripts/process_documents.py`

### Phase B: Expand Data Sources

**Goal**: Download and parse all 8 sources defined in `collect_sources.py`

- [x] **B.1** Run `scripts/collect_sources.py` to download available sources:
  - ~~DGA 2020-2025 (PDF)~~ ✅ Downloaded
  - WHO Sugar Guidelines (PDF) — ⏸ Deferred (DGA 2020 covers same scope)
  - WHO Sodium Guidelines (PDF) — ⏸ Deferred
  - ~~USDA MyPlate (PDF)~~ — ⏸ Deferred
  - ~~ACOG Pregnancy Nutrition (PDF)~~ ✅ Downloaded
- [x] **B.2** Add docling dependency to project
- [x] **B.3** Rewrite `scripts/process_documents.py`:
  - Keep BeautifulSoup for HTML (NIH, etc.)
  - Add `DoclingPDFParser` for PDFs (DGA, WHO, ISSN, ACOG, MyPlate)
  - Ensure table extraction works for PDFs
- [x] **B.4** Run `scripts/process_documents.py` on all downloaded raw files
- [x] **B.5** Validate: `data/parsed/` has 26 files (5 sources; WHO + MyPlate deferred)

### Phase C: Chunking with Domain Tagging

**Goal**: Structure-aware chunking with per-chunk domain assignment

- [x] **C.1** Create `src/retrieval/domain_tagger.py`:
  - Implement two-level matching (heading + content)
  - Require 2+ keywords for content-based tagging
  - Return list of domains per chunk
- [x] **C.2** Rewrite `scripts/chunk_documents.py`:
  - Max 450 tokens, overlap 48 tokens, min 30 tokens (450 + 50 context prefix ≤ 512 BGE limit)
  - Table handling: atomic if ≤450 tokens, else split by row groups with header
  - No heading prefix (context added in next step)
  - Call domain_tagger for each chunk
  - Output `data/knowledge/chunks.jsonl`
- [x] **C.3** Validate:
  - No chunk < 30 tokens (enforced by post-merge filter in chunker.py)
  - All table chunks have accurate token_count
  - Domain distribution is reasonable (micronutrients 68%, no single domain > 70%)

### Phase D: Contextual Retrieval

**Goal**: Add LLM-generated context to each chunk (Anthropic's Contextual Retrieval)

- [x] **D.1** Create `scripts/contextualize_chunks.py`:
  - Load chunks.jsonl
  - For each chunk, call Gemini 2.5 Flash (OpenAI-compatible endpoint) with context prompt
  - Enforce ≤50 tokens for context
  - Cache by content hash to avoid regeneration
  - Output `chunks_contextualized.jsonl`
- [x] **D.2** Run contextualization (one-time cost; cache cleared, all chunks need fresh contexts)
- [x] **D.3** Validate:
  - All chunks have context field
  - Context is ≤50 tokens
  - Context is English and relevant

### Phase E: Build Indexes

**Goal**: ChromaDB + BM25 with proper metadata handling

- [x] **E.1** Create `scripts/build_indexes.py` (merged from init_vectorstore + init_bm25):
  - Build ChromaDB:
    - Embed `contextualized_content` (not original)
    - Store `original_content` in documents field
    - Boolean domain fields in metadata
    - heading_hierarchy as JSON string
    - Cosine distance metric
  - Build BM25:
    - Tokenize `contextualized_content`
    - Store original_content for retrieval
    - No hyphen splitting (keep "omega-3" intact)
    - Stopword removal
- [x] **E.2** Run index build
- [x] **E.3** Validate:
  - ChromaDB count == BM25 count == chunks_contextualized.jsonl lines
  - `domain_micronutrients: true` filter returns results
  - heading_hierarchy is valid JSON

### Phase F: Retrieval Implementation

**Goal**: HybridRetriever with native domain filtering and lazy reranker

- [x] **F.1** Create `src/retrieval/hybrid_retriever.py`:
  - ChromaDB query with `where={f"domain_{domain}": True}`
  - BM25 with domain post-filter (check boolean metadata)
  - RRF merge (k=60)
  - `preprocess_query()` with abbreviation expansion (t2d, bp, ckd, etc.)
  - Lazy reranker loading with warning log on failure
  - Threshold only when reranker available; skip threshold for RRF-only path
  - Low confidence fallback (return best with flag) only when reranker active
  - Return original_content to user (not contextualized)
- [x] **F.2** Create `src/tools/retrieve_knowledge.py`:
  - Pass domain parameter through (fix B1)
  - Singleton retriever
  - Structured response with source attribution
- [x] **F.3** Validate:
  - Domain filtering works end-to-end
  - Reranker fallback logs warning
  - Results contain original_content (no context prefix visible)

### Phase G: Evaluation & Verification

**Goal**: Automated test suite for retrieval quality

- [x] **G.1** Create `tests/test_rag_quality.py`:
  - 20+ evaluation queries covering all domains
  - must_contain / must_not_contain assertions
  - Domain filter correctness checks
- [x] **G.2** Run full pipeline end-to-end
- [x] **G.3** Run evaluation tests
- [x] **G.4** Spot checks:
  - "vitamin B12 pregnant" returns B12 content (not B6)
  - "high-fiber foods" returns dietary fiber (not muscle fiber)
  - "protein for athletes" returns sports_nutrition content
  - Domain filter "sports_nutrition" excludes NIH micronutrient chunks

---

## Dependencies

- `docling` — PDF parsing (new dependency)
- `openai` — Gemini 2.5 Flash via OpenAI-compatible endpoint (GEMINI_API_KEY in .env)
- `data/parsed/*.json` — 26 parsed files (5 sources downloaded)
- `configs/tools.yaml` — RAG configuration

---

## Acceptance Criteria

| Metric | Target |
|--------|--------|
| Source coverage | 5/8 sources downloaded and parsed (WHO + MyPlate deferred) |
| Chunk count | 1300-1600 chunks (≥30 tokens each, max 450 tokens) |
| Context coverage | 100% chunks have ≤50 token context |
| Domain distribution | No single domain > 70% |
| B12/B6 disambiguation | Query "vitamin B12" returns B12 only |
| Fiber disambiguation | Query "dietary fiber" excludes "muscle fiber" |
| Domain filter | Filter reduces results to target domain |
| Reranker fallback | Graceful degradation with log warning |
| Evaluation pass rate | ≥90% of test queries pass |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| docling parsing errors | Fallback to pdfplumber for problematic PDFs |
| Qwen-Max API failures | Retry with exponential backoff, cache successful results |
| Context quality variance | Manual review of 10% sample, regenerate if needed |
| Domain tagging false positives | Adjust keyword lists based on validation results |

---

## Estimated Timeline

| Phase | Effort |
|-------|--------|
| B: Data sources | 2-3 hours |
| C: Chunking | 2-3 hours |
| D: Contextualization | 1-2 hours (mostly waiting for API) |
| E: Indexes | 1-2 hours |
| F: Retrieval | 2-3 hours |
| G: Evaluation | 1-2 hours |
| **Total** | **9-15 hours** |
