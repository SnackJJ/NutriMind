# RAG Specification (v3)

> **Version**: 3.0 (complete rewrite)
> **Date**: 2026-03-08
> **Status**: approved

## Architecture Overview

```
                           OFFLINE PIPELINE
┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌────────────────┐   ┌─────────────┐
│ collect  │──▶│ process  │──▶│ chunk        │──▶│ contextualize  │──▶│ build       │
│ sources  │   │ documents│   │ documents    │   │ (Gemini Flash) │   │ indexes     │
└──────────┘   └──────────┘   └──────────────┘   └────────────────┘   └─────────────┘
  Download       HTML: BS4      Structure-        Generate 50-tok      ChromaDB +
  all sources    PDF: docling   aware split       context prefix       BM25

                           ONLINE PIPELINE
┌───────────────────────────────────────────────────────────────────────────────────┐
│ retrieve_knowledge(query, domain=None)                                            │
│                                                                                   │
│  preprocess(query)                                                                │
│       │                                                                           │
│       ├──▶ embed_query() ──▶ ChromaDB (where=domain) ──▶ top 20                  │
│       │                                                       │                   │
│       ├──▶ BM25 tokenize ──▶ BM25 score + domain filter ──▶ top 20               │
│       │                                                       │                   │
│       └──▶ RRF Merge ──────────────────────────────────▶ top 10                  │
│                                                               │                   │
│            BGE-reranker-base (lazy load) ──────────────▶ top 3                   │
│                                                               │                   │
│            Threshold (0.3) + low_confidence fallback ──▶ final                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## Core Principles

1. **Authoritative Sources Only**: All factual knowledge originates from external authoritative sources (NIH, WHO, DGA, ISSN, etc.). Every chunk is traceable to document + section + page/URL.

2. **Contextual Retrieval**: Each chunk is augmented with LLM-generated context (≤50 tokens) before embedding/indexing, following Anthropic's Contextual Retrieval methodology (67% failure rate reduction).

3. **No Safety Boundary Chunks**: T4 safety handling is done at the orchestrator layer. Source documents contain their own safety information (toxicity, interactions, contraindications) which gets chunked naturally.

4. **100% English**: All documents, chunks, queries, and responses are in English.

---

## Knowledge Sources

### Tier 1: Primary Sources (must have)

| Source | Format | Primary Domain | Actual Chunks |
|--------|--------|----------------|---------------|
| NIH Office of Dietary Supplements — Health Professional Fact Sheets (23) | HTML | micronutrients | ~1853 |
| Dietary Guidelines for Americans 2020-2025 (DGA) | PDF | dietary_guidelines | ~519 |
| WHO Guidelines (sugar, sodium) | PDF | dietary_guidelines | ~40 (deferred) |
| USDA MyPlate | PDF | meal_planning | ~30 (deferred) |

### Tier 2: Specialized Sources (high priority)

| Source | Format | Primary Domain | Actual Chunks |
|--------|--------|----------------|---------------|
| ISSN Position Stands (protein, nutrient timing) | PDF | sports_nutrition | ~343 |
| ACOG Nutrition During Pregnancy | PDF | life_stage | ~324 |

### Tier 3: Clinical Diet Management (added 2026-03-09)

| Source | Format | Primary Domain | Motivation |
|--------|--------|----------------|------------|
| NIH NIDDK: Gallstones eating/diet | HTML | medical_nutrition | Cover GI condition diet queries |
| NIH NIDDK: GERD eating/diet | HTML | medical_nutrition | Cover acid reflux diet queries |
| NIH NIDDK: IBS eating/diet | HTML | medical_nutrition | Cover IBS diet queries |
| NIH NIDDK: Celiac eating/diet | HTML | medical_nutrition | Cover gluten-free diet queries |
| MedlinePlus (NIH NLM): Food Allergy | HTML | food_safety | Cover food allergy/intolerance queries; NIAID 405 blocks scraping |

**Current total: 1730 chunks (was 1635, +95 from Tier 3 sources).**

### Current Status

| Source | Status | Parsed Files |
|--------|--------|--------------|
| NIH ODS (23 nutrients) | ✅ Downloaded + Parsed | 23 |
| DGA 2020-2025 | ✅ Downloaded + Parsed | 1 |
| ISSN (protein + timing) | ✅ Downloaded + Parsed | 2 |
| ACOG Pregnancy | ✅ Downloaded + Parsed | 1 |
| NIDDK (gallstones/gerd/ibs/celiac) | ✅ Downloaded + Indexed | 4 (19 chunks) |
| MedlinePlus Food Allergy | ✅ Downloaded + Indexed | 1 (17 chunks) |
| WHO Sugar/Sodium | ⏸ Deferred | — |
| MyPlate | ⏸ Deferred | — |

---

## Document Processing Pipeline

### Data Flow

```
data/raw/                       ← scripts/collect_sources.py
  ├── nih_ods/*.html
  ├── dga_2020/*.pdf
  ├── who/*.pdf
  └── ...
         ↓
data/parsed/                    ← scripts/process_documents.py
  ├── nih_ods_vitamin_b12.json
  ├── dga_chapter_1.json
  └── ...
         ↓
data/knowledge/                 ← scripts/chunk_documents.py
  └── chunks.jsonl
         ↓
data/knowledge/                 ← scripts/contextualize_chunks.py [NEW]
  └── chunks_contextualized.jsonl
         ↓
data/knowledge_db/              ← scripts/build_indexes.py
data/knowledge_bm25/
  └── index.pkl
```

### Source Collection

```python
# scripts/collect_sources.py

SOURCE_MANIFEST = [
    {
        "id": "nih_ods",
        "base_url": "https://ods.od.nih.gov/factsheets/",
        "pages": ["VitaminA-HealthProfessional", "VitaminB12-HealthProfessional", ...],
        "format": "html",
        "primary_domain": "micronutrients",
        "license": "public_domain"
    },
    {
        "id": "dga_2020",
        "url": "https://www.dietaryguidelines.gov/.../Dietary_Guidelines_for_Americans-2020-2025.pdf",
        "format": "pdf",
        "primary_domain": "dietary_guidelines",
        "license": "public_domain"
    },
    # ... all 8 sources
]
```

- Download with `httpx`, rate limited (1 req/sec)
- Save raw files to `data/raw/{source_id}/`
- Record SHA-256 hash in manifest for idempotency

### Document Parsing

**HTML Sources (NIH ODS, etc.)**: BeautifulSoup with source-specific logic

```python
class NIHFactSheetParser:
    """
    NIH ODS fact sheets have consistent HTML structure:
    - <h2> section headings
    - Tables with RDA/AI values
    - Bulleted lists of food sources
    """
    def parse(self, html_path: str) -> List[Section]: ...
```

**PDF Sources (DGA, WHO, ISSN, ACOG)**: docling

```python
from docling.document_converter import DocumentConverter

class DoclingPDFParser:
    """
    Use docling for PDF parsing. Advantages over pdfplumber:
    - Layout-aware heading detection (font size, bold)
    - Structured table extraction
    - Multi-column support
    """
    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, pdf_path: str) -> List[Section]:
        result = self.converter.convert(pdf_path)
        doc = result.document

        sections = []
        for item in doc.iterate_items():
            if item.label == "section_header":
                # New section
                ...
            elif item.label == "table":
                # Extract table with caption
                ...
            elif item.label == "text":
                # Paragraph content
                ...

        return sections
```

**Parsed Output Format** (`data/parsed/*.json`):

```json
{
  "source_id": "nih_ods",
  "document": "Vitamin B12",
  "url": "https://ods.od.nih.gov/factsheets/VitaminB12-HealthProfessional/",
  "sections": [
    {
      "heading": "Recommended Intakes",
      "heading_hierarchy": ["Vitamin B12", "Recommended Intakes"],
      "content": "The amount of vitamin B12 you need depends on your age...",
      "page": null,
      "tables": [
        {
          "caption": "Recommended Dietary Allowances for Vitamin B12",
          "rows": [["Life Stage", "Amount"], ["Birth to 6 months", "0.4 mcg"]]
        }
      ]
    }
  ]
}
```

---

## Chunking Strategy

**Method**: Structure-aware, token-based, sentence boundary.

```python
class StructureAwareChunker:
    def __init__(self, max_tokens: int = 450, overlap_tokens: int = 48, min_tokens: int = 30):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens
```

### Chunking Rules

1. **Sentence boundaries**: Never cut mid-sentence
2. **Tables**: Prefer atomic. If table > max_tokens, split by row groups, each sub-chunk retains caption + column header
3. **Minimum length**: Chunks < 30 tokens are merged with adjacent chunk in same section; if no mergeable neighbor, the chunk is discarded (prevents figure labels, boilerplate from entering the index)
4. **Overlap**: 48 tokens at sentence boundary for continuity
5. **No heading prepend**: Context is added in the next step (contextualization)

### Table Handling

```python
def chunk_table(self, table: dict, section: Section) -> list[Chunk]:
    """
    Tables are preferably atomic. If too large, split by row groups.
    Each sub-chunk retains caption + column header.
    """
    table_text = self._table_to_text(table)
    table_tokens = len(self.tokenizer.encode(table_text, add_special_tokens=False))

    if table_tokens <= self.max_tokens:
        # Atomic chunk
        return [self._build_chunk(table_text, section, is_table=True)]

    # Split by row groups
    header_row = table["rows"][0]
    data_rows = table["rows"][1:]

    chunks = []
    current_rows = []
    current_len = self._header_tokens(table["caption"], header_row)

    for row in data_rows:
        row_len = len(self.tokenizer.encode(" | ".join(row), add_special_tokens=False))
        if current_len + row_len > self.max_tokens and current_rows:
            chunk_text = self._format_table_chunk(table["caption"], header_row, current_rows)
            chunks.append(self._build_chunk(chunk_text, section, is_table=True))
            current_rows = []
            current_len = self._header_tokens(table["caption"], header_row)
        current_rows.append(row)
        current_len += row_len

    if current_rows:
        chunk_text = self._format_table_chunk(table["caption"], header_row, current_rows)
        chunks.append(self._build_chunk(chunk_text, section, is_table=True))

    return chunks
```

### Chunk Schema (Pre-Contextualization)

```json
{
  "id": "nih_ods__vitamin_b12__recommended_intakes__001",
  "content": "The amount of vitamin B12 you need depends on your age...",
  "metadata": {
    "source_id": "nih_ods",
    "document": "Vitamin B12",
    "section": "Recommended Intakes",
    "heading_hierarchy": ["Vitamin B12", "Recommended Intakes"],
    "url": "https://ods.od.nih.gov/factsheets/VitaminB12-HealthProfessional/",
    "page": null,
    "source_type": "government",
    "is_table": false,
    "token_count": 187
  }
}
```

---

## Contextual Retrieval (Key Innovation)

Based on Anthropic's research (Sept 2024): prepending LLM-generated context to each chunk before embedding reduces retrieval failure rate by **67%** when combined with BM25 and reranking.

### Context Generation

```python
# scripts/contextualize_chunks.py

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

def contextualize_chunk(chunk: dict, source_meta: dict) -> str:
    """Generate context prefix using Gemini 2.5 Flash (OpenAI-compatible endpoint)."""
    prompt = CONTEXT_PROMPT.format(
        document_title=chunk["metadata"]["document"],
        source_description=source_meta.get("description", ""),
        section_heading=chunk["metadata"]["section"],
        chunk_text=chunk["content"]
    )

    context = call_gemini_flash(prompt, max_tokens=60)

    # Enforce 50 token limit
    context_tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(context_tokens) > 50:
        context = tokenizer.decode(context_tokens[:50])

    return context
```

### Contextualized Chunk Format

```json
{
  "id": "nih_ods__vitamin_b12__recommended_intakes__001",
  "original_content": "The amount of vitamin B12 you need depends on your age...",
  "context": "This chunk from the NIH Vitamin B12 fact sheet provides the Recommended Dietary Allowance (RDA) values for different age groups and life stages.",
  "contextualized_content": "This chunk from the NIH Vitamin B12 fact sheet provides the Recommended Dietary Allowance (RDA) values for different age groups and life stages. | The amount of vitamin B12 you need depends on your age...",
  "metadata": { ... }
}
```

**Important**:
- `contextualized_content` is used for embedding and BM25 indexing
- `original_content` is returned to the user (no synthetic prefix exposed)

### Caching

Context generation is cached by content hash:

```python
def get_or_generate_context(chunk: dict, cache: dict) -> str:
    content_hash = hashlib.sha256(chunk["content"].encode()).hexdigest()
    if content_hash in cache:
        return cache[content_hash]

    context = contextualize_chunk(chunk)
    cache[content_hash] = context
    return context
```

---

## Domain Tagging

### Domain Categories

| Category | Description |
|----------|-------------|
| `micronutrients` | Vitamins, minerals |
| `dietary_guidelines` | DGA, DRI, RDA reference values |
| `sports_nutrition` | Exercise fueling, recovery, protein timing |
| `medical_nutrition` | Diabetes, CVD, kidney disease dietary management |
| `life_stage` | Pregnancy, lactation, elderly, children |
| `meal_planning` | Food groups, portions, practical guidance |
| `food_safety` | Allergies, toxicity, drug interactions |
| `supplements` | Supplement forms, dosages, interactions |
| `weight_management` | Energy balance, calorie control |

### Two-Level Domain Assignment

Domain tags are assigned per-chunk using a two-level matching system to reduce false positives:

```python
DOMAIN_RULES = {
    "sports_nutrition": {
        "heading_keywords": ["exercise", "athlete", "training", "sport", "performance"],
        "content_keywords": ["exercise", "athlete", "training", "ergogenic", "recovery",
                            "muscle protein", "post-workout", "pre-workout", "endurance"],
        "min_content_hits": 2  # Require 2+ different keywords in content
    },
    "life_stage": {
        "heading_keywords": ["pregnancy", "lactation", "infant", "elderly", "pediatric", "aging"],
        "content_keywords": ["pregnant", "pregnancy", "lactation", "breastfeeding", "infant",
                            "elderly", "older adult", "children", "prenatal", "postnatal"],
        "min_content_hits": 2
    },
    "food_safety": {
        "heading_keywords": ["safety", "toxicity", "interaction", "adverse", "allergen"],
        "content_keywords": ["toxicity", "overdose", "upper limit", "drug interaction",
                            "contraindicated", "adverse effect", "allergen", "intolerance"],
        "min_content_hits": 2
    },
    "medical_nutrition": {
        "heading_keywords": ["diabetes", "cardiovascular", "kidney", "disease", "clinical"],
        "content_keywords": ["diabetes", "cardiovascular", "hypertension", "kidney disease",
                            "renal", "heart disease", "cholesterol", "blood pressure"],
        "min_content_hits": 2
    },
    "meal_planning": {
        "heading_keywords": ["food source", "dietary source", "intake"],
        "content_keywords": ["food source", "rich in", "good source", "dietary intake",
                            "servings", "portion", "meal", "daily intake"],
        "min_content_hits": 2
    },
    "supplements": {
        "heading_keywords": ["supplement", "supplementation", "fortification"],
        "content_keywords": ["supplement", "supplementation", "fortified",
                            "capsule", "tablet", "dosage", "milligrams",
                            "micrograms", "iu", "dietary supplement"],
        "min_content_hits": 2
    },
    "weight_management": {
        "heading_keywords": ["weight", "obesity", "calorie", "energy balance", "bmi"],
        "content_keywords": ["weight loss", "weight gain", "obesity", "overweight",
                            "caloric deficit", "energy balance", "bmi",
                            "body weight", "calorie restriction"],
        "min_content_hits": 2
    },
}

def assign_domains(chunk: dict, source_primary_domain: str) -> list[str]:
    """
    Assign domains to a chunk using two-level matching.

    Level 1 (heading): Single keyword match in section heading
    Level 2 (content): Requires min_content_hits different keywords
    """
    domains = {source_primary_domain}  # Always include source's primary domain

    heading = chunk["metadata"]["section"].lower()
    content = chunk["content"].lower()

    for domain, rules in DOMAIN_RULES.items():
        # Level 1: Heading match (high signal, single keyword sufficient)
        if any(kw in heading for kw in rules["heading_keywords"]):
            domains.add(domain)
            continue

        # Level 2: Content match (require multiple keywords to reduce false positives)
        content_hits = [kw for kw in rules["content_keywords"] if kw in content]
        unique_hits = len(set(content_hits))
        if unique_hits >= rules["min_content_hits"]:
            domains.add(domain)

    return list(domains)
```

### Boolean Metadata Storage

Domains are stored as boolean fields in ChromaDB for native filtering:

```python
def build_metadata(chunk: dict, domains: list[str]) -> dict:
    """Build ChromaDB-compatible metadata with boolean domain fields."""
    meta = {
        "source_id": chunk["metadata"]["source_id"],
        "document": chunk["metadata"]["document"],
        "section": chunk["metadata"]["section"],
        "heading_hierarchy": json.dumps(chunk["metadata"]["heading_hierarchy"]),
        "url": chunk["metadata"].get("url", ""),
        "page": chunk["metadata"].get("page") or 0,
        "source_type": chunk["metadata"].get("source_type", ""),
        "token_count": chunk["metadata"]["token_count"],
        # Boolean domain fields
        "domain_micronutrients": "micronutrients" in domains,
        "domain_dietary_guidelines": "dietary_guidelines" in domains,
        "domain_sports_nutrition": "sports_nutrition" in domains,
        "domain_medical_nutrition": "medical_nutrition" in domains,
        "domain_life_stage": "life_stage" in domains,
        "domain_meal_planning": "meal_planning" in domains,
        "domain_food_safety": "food_safety" in domains,
        "domain_supplements": "supplements" in domains,
        "domain_weight_management": "weight_management" in domains,
    }
    return meta
```

ChromaDB query with domain filter:

```python
where_clause = {f"domain_{domain_filter}": True} if domain_filter else None
results = collection.query(
    query_embeddings=[query_emb],
    n_results=20,
    where=where_clause
)
```

---

## Embedding Model

| Model | Dimensions | Max Tokens | Notes |
|-------|------------|------------|-------|
| `BAAI/bge-small-en-v1.5` | 384 | 512 | Asymmetric (query prefix required) |

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_query(query: str) -> list[float]:
    """BGE models use instruction prefix for queries."""
    return embedding_model.encode(
        "Represent this sentence for searching relevant passages: " + query,
        normalize_embeddings=True
    ).tolist()

def embed_document(text: str) -> list[float]:
    """Documents are embedded without prefix."""
    return embedding_model.encode(text, normalize_embeddings=True).tolist()
```

**Critical**: BGE is asymmetric. Queries use instruction prefix, documents do not. We embed manually and pass vectors to ChromaDB (not using ChromaDB's built-in embedding).

---

## Reranker Model

| Model | Parameters | Latency |
|-------|------------|---------|
| `BAAI/bge-reranker-base` | 278M | ~50ms for 10 candidates |

### Lazy Loading with Fallback

```python
class HybridRetriever:
    def __init__(self, config: dict):
        self._reranker = None
        self._reranker_failed = False
        # ... other init

    @property
    def reranker(self):
        if self._reranker is None and not self._reranker_failed:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder("BAAI/bge-reranker-base")
                logger.info("Reranker loaded successfully")
            except Exception as e:
                self._reranker_failed = True
                logger.warning(f"Reranker failed to load: {e}. Falling back to RRF scores.")
        return self._reranker

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if self.reranker is None:
            logger.warning("Reranker unavailable, using RRF scores for ranking")
            return candidates

        pairs = [(query, c["content"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates
```

---

## BM25 Index

### Tokenization

```python
import re

STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "to", "for",
    "with", "on", "at", "by", "from", "as", "into", "about", "between",
    "and", "or", "but", "not", "no", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "we", "our", "you", "your",
})

def simple_tokenize(text: str) -> list[str]:
    """
    BM25 tokenization.
    - Lowercase
    - Remove punctuation (preserve hyphens within words)
    - Split on whitespace only (keep "omega-3", "high-fiber" intact)
    - Remove stopwords
    """
    text = text.lower()
    # Remove punctuation except hyphens
    text = re.sub(r'[^\w\s\-]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t and t not in STOPWORDS]
```

**Key decision**: Do NOT split on hyphens. "omega-3" stays as one token. This preserves nutrition compound terms.

### Index Structure

```python
index_data = {
    "bm25": bm25,  # BM25Okapi instance
    "chunk_ids": [...],
    "chunk_contents": [...],  # original_content (not contextualized)
    "chunk_contextualized": [...],  # for BM25 scoring
    "chunk_metadatas": [...],
}
```

BM25 scores against `contextualized_content`, but returns `original_content` to user.

---

## Hybrid Retrieval Pipeline

### Full Flow

```
Query
  │
  ├──▶ preprocess_query() ──▶ normalized query
  │
  ├──▶ embed_query() ──▶ ChromaDB (where=domain) ──▶ top 20
  │                                                    │
  ├──▶ BM25 tokenize ──▶ BM25 score ──▶ domain filter ──▶ top 20
  │                                                    │
  └──▶ RRF Merge ────────────────────────────────▶ top 10
                                                       │
       BGE-reranker-base (or RRF fallback) ──────▶ top 3
                                                       │
       Threshold (0.3) + low_confidence fallback ──▶ final
```

### Implementation

```python
class HybridRetriever:
    def __init__(self, config: dict):
        # Semantic index
        self.chroma_client = chromadb.PersistentClient(path=config["chroma_db_path"])
        self.collection = self.chroma_client.get_collection("nutrition_knowledge")

        # Keyword index
        with open(config["bm25_index_path"], "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_chunk_ids = bm25_data["chunk_ids"]
        self.bm25_chunk_contents = bm25_data["chunk_contents"]
        self.bm25_chunk_metadatas = bm25_data["chunk_metadatas"]

        # Reranker (lazy load)
        self._reranker = None
        self._reranker_failed = False

        # Parameters
        self.semantic_top_k = config.get("semantic_top_k", 20)
        self.bm25_top_k = config.get("bm25_top_k", 20)
        self.rerank_top_k = config.get("rerank_top_k", 10)
        self.final_top_k = config.get("final_top_k", 3)
        self.relevance_threshold = config.get("relevance_threshold", 0.3)
        self.rrf_k = config.get("rrf_k", 60)

    def retrieve(self, query: str, domain_filter: str = None) -> list[dict]:
        query = preprocess_query(query)

        # 1. Semantic search (ChromaDB native domain filtering)
        query_embedding = embed_query(query)
        where_clause = {f"domain_{domain_filter}": True} if domain_filter else None

        sem_results = self._parse_chroma_results(
            self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.semantic_top_k,
                where=where_clause
            )
        )

        # 2. BM25 keyword search + domain filter
        bm25_scores = self.bm25.get_scores(simple_tokenize(query))
        bm25_top_idx = np.argsort(bm25_scores)[-self.bm25_top_k * 2:][::-1]  # Get extra for filtering

        bm25_results = []
        for i in bm25_top_idx:
            if bm25_scores[i] <= 0:
                continue
            meta = self.bm25_chunk_metadatas[i]
            # Domain filter
            if domain_filter and not meta.get(f"domain_{domain_filter}", False):
                continue
            bm25_results.append({
                "id": self.bm25_chunk_ids[i],
                "content": self.bm25_chunk_contents[i],
                "metadata": meta,
                "score": float(bm25_scores[i]),
            })
            if len(bm25_results) >= self.bm25_top_k:
                break

        # 3. RRF Merge
        fused = self._rrf_merge(sem_results, bm25_results)

        # 4. Rerank
        rerank_candidates = fused[:self.rerank_top_k]
        if rerank_candidates:
            rerank_candidates = self._rerank(query, rerank_candidates)

        # 5. Threshold + low_confidence fallback
        #    RRF scores (~0.01-0.03) and reranker scores are on different scales.
        #    Only apply threshold when reranker is available; otherwise take top_k directly.
        if self.reranker is not None:
            final = [c for c in rerank_candidates
                     if c.get("rerank_score", 0) >= self.relevance_threshold
                    ][:self.final_top_k]

            # Low confidence fallback: return best candidate with flag
            if not final and rerank_candidates:
                best = rerank_candidates[0]
                best["low_confidence"] = True
                final = [best]
                logger.info("No results above threshold, returning best candidate with low_confidence flag")
        else:
            # Reranker unavailable: RRF scores are not comparable to threshold.
            # Trust RRF ranking and return top_k directly.
            final = rerank_candidates[:self.final_top_k]

        return final

    def _rrf_merge(self, list_a: list[dict], list_b: list[dict]) -> list[dict]:
        """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank)"""
        scores = {}
        candidates = {}

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
```

---

## retrieve_knowledge Tool Interface

```python
# src/tools/retrieve_knowledge.py

from loguru import logger

_retriever: HybridRetriever = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        config = load_config("configs/tools.yaml")["rag"]
        _retriever = HybridRetriever(config)
    return _retriever

def retrieve_knowledge(query: str, domain: str = None, top_k: int = 3) -> dict:
    """
    Search the nutrition knowledge base.

    Args:
        query: Natural language question or topic
        domain: Optional filter (micronutrients, sports_nutrition, etc.)
        top_k: Number of passages (default 3)

    Returns:
        dict with status + passages including source attribution
    """
    if not query or not query.strip():
        return {"status": "error", "error_type": "empty_query"}

    retriever = get_retriever()
    results = retriever.retrieve(query=query, domain_filter=domain)

    if not results:
        return {"status": "error", "error_type": "no_relevant_results"}

    passages = []
    for r in results[:top_k]:
        meta = r["metadata"]
        passage = {
            "content": r["content"],  # original_content, not contextualized
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
        "data": {"passages": passages}
    }
```

---

## Query Preprocessing

```python
def preprocess_query(query: str) -> str:
    """Normalize abbreviations and expand medical shorthand."""
    replacements = {
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
    normalized = query.lower()
    for abbrev, full in replacements.items():
        normalized = re.sub(rf'\b{abbrev}\b', full, normalized)
    return normalized
```

---

## Configuration

```yaml
# configs/tools.yaml — rag section

rag:
  # Embedding
  embedding_model: "BAAI/bge-small-en-v1.5"
  embedding_dim: 384

  # Reranker
  reranker_model: "BAAI/bge-reranker-base"

  # Chunking
  chunk_max_tokens: 450
  chunk_overlap_tokens: 48
  chunk_min_tokens: 30
  context_max_tokens: 50

  # Index paths
  chroma_db_path: "data/knowledge_db"
  chroma_distance_metric: "cosine"
  bm25_index_path: "data/knowledge_bm25/index.pkl"

  # Retrieval parameters
  semantic_top_k: 20
  bm25_top_k: 20
  rerank_top_k: 10
  final_top_k: 3
  relevance_threshold: 0.3
  rrf_k: 60

  # Contextualization
  context_model: "gemini-2.5-flash"   # OpenAI-compatible endpoint; requires GEMINI_API_KEY
  context_cache_path: "data/knowledge/context_cache.json"
```

---

## File Structure

```
scripts/
  ├── collect_sources.py         # Download raw documents
  ├── process_documents.py       # Parse → structured JSON
  ├── chunk_documents.py         # Sections → chunks.jsonl
  ├── contextualize_chunks.py    # Add Gemini Flash context (one-time)
  └── build_indexes.py           # ChromaDB + BM25 [merged]

src/retrieval/
  ├── __init__.py
  ├── hybrid_retriever.py        # HybridRetriever
  ├── chunker.py                 # StructureAwareChunker
  ├── domain_tagger.py           # Two-level domain assignment
  └── parsers/
      ├── __init__.py
      ├── nih_parser.py          # NIH HTML parser
      ├── docling_parser.py      # PDF parser via docling
      └── generic_html_parser.py # Generic HTML fallback

src/tools/
  └── retrieve_knowledge.py      # Tool entry point

data/
  ├── raw/                       # Downloaded documents (git-ignored)
  ├── parsed/                    # Structured JSON
  ├── knowledge/
  │   ├── chunks.jsonl
  │   ├── chunks_contextualized.jsonl
  │   └── context_cache.json
  ├── knowledge_db/              # ChromaDB
  └── knowledge_bm25/
      └── index.pkl
```

---

## Pipeline Execution

```bash
# Current state: parsed docs ready (26 files), data/knowledge/ cleared
# Ready to rebuild from 450-token chunks

# Remaining steps:
python scripts/chunk_documents.py          # Rebuild chunks at 450 tokens
python scripts/contextualize_chunks.py     # Add Gemini Flash context (~45 min, ~$1 one-time)
python scripts/build_indexes.py            # Build ChromaDB + BM25

# Full rebuild from scratch:
python scripts/collect_sources.py          # Download available 5 sources
python scripts/process_documents.py        # Parse HTML + PDF
python scripts/chunk_documents.py          # Create chunks.jsonl
python scripts/contextualize_chunks.py     # Add LLM context
python scripts/build_indexes.py            # Build ChromaDB + BM25

# Verify
python -m pytest tests/test_rag_quality.py
```

---

## Evaluation Framework

```python
# tests/test_rag_quality.py

EVAL_QUERIES = [
    {
        "query": "how much vitamin B12 do pregnant women need",
        "expected_domains": ["micronutrients", "life_stage"],
        "must_contain": ["2.6", "mcg", "pregnant"],
        "must_not_contain": ["vitamin B6"],
    },
    {
        "query": "high fiber foods for digestive health",
        "expected_domains": ["dietary_guidelines", "meal_planning"],
        "must_contain": ["fiber"],
        "must_not_contain": ["muscle fiber"],
    },
    {
        "query": "protein intake for strength training",
        "expected_domains": ["sports_nutrition"],
        "must_contain": ["protein", "exercise"],
    },
    # ... 20+ queries covering all domains
]

def test_retrieval_quality():
    for case in EVAL_QUERIES:
        result = retrieve_knowledge(case["query"])
        assert result["status"] == "success", f"Failed: {case['query']}"

        passages = result["data"]["passages"]
        full_text = " ".join(p["content"].lower() for p in passages)

        for term in case["must_contain"]:
            assert term.lower() in full_text, f"Missing '{term}' for: {case['query']}"

        for term in case.get("must_not_contain", []):
            assert term.lower() not in full_text, f"Unexpected '{term}' for: {case['query']}"
```

---

## Dependencies

```
# Core retrieval
chromadb >= 0.4.0
sentence-transformers >= 2.2.0
rank-bm25 >= 0.2.2

# Document processing
docling >= 0.1.0                  # PDF parsing (replaces pdfplumber)
beautifulsoup4 >= 4.12.0          # HTML parsing
lxml >= 4.9.0
nltk >= 3.8.0
httpx >= 0.25.0

# Contextualization
openai >= 1.0.0                   # Gemini 2.5 Flash via OpenAI-compatible endpoint
                                  # Requires GEMINI_API_KEY in .env

# Already in project
transformers, torch, loguru, pydantic
```

---

## Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| Contextual Retrieval | Anthropic research shows 67% failure rate reduction |
| docling for PDF | Better heading detection, table extraction vs pdfplumber |
| Boolean domain metadata | ChromaDB native `where` filter (no post-retrieval loss) |
| Two-level domain tagging | Heading match (single kw) + content match (2+ kw) reduces false positives |
| No hyphen splitting in BM25 | Preserves "omega-3", "high-fiber" as single tokens |
| No safety_boundaries domain | T4 handled at orchestrator layer; source docs have natural safety content |
| Context ≤ 50 tokens | Balances information vs embedding space |
| Lazy reranker + logging | Graceful degradation with visibility |
| Low confidence fallback | Returns best result with flag instead of empty error |
| Threshold only with reranker | RRF scores (~0.03) are not comparable to reranker logits; skip threshold in fallback path |
| All 9 domains have rules | `supplements` and `weight_management` have DOMAIN_RULES entries so they can be tagged beyond source primary_domain |

---

## Audit History

### v2 Audit (2026-03-08)

Critical bugs found in v1:
- B1: domain_filter=None hardcoded
- B2: ChromaDB $contains broken on string metadata
- B3: Table token_count not recalculated after truncation
- B4: heading_hierarchy serialized via str() not json.dumps()

Data quality issues:
- D1: 79% micronutrients, 0% fiber/glucose/meal-planning
- D2: "fiber" false positives (muscle fibers)
- D3: No postprandial content

**Resolution**: Complete rewrite as v3 with Contextual Retrieval architecture.
