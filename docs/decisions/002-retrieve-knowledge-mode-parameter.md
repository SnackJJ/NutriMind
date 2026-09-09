# ADR-002: Parameterized Retrieval Mode for retrieve_knowledge

- **Status**: accepted
- **Date**: 2026-03-17
- **Deciders**: @jzq

## Context

During trajectory collection, we observed that the teacher model (qwen3.5-plus) would repeatedly call `retrieve_knowledge` with different query phrasings when initial results had low relevance scores. This "blind reformulation" pattern is inefficient:

- 4+ consecutive calls with different keywords
- No strategy switching (always using hybrid retrieval)
- Final answer often came from model's internal knowledge, not RAG

Example problematic trajectory:
```
retrieve_knowledge("potato only diet risks")     → low quality
retrieve_knowledge("single food diet dangers")  → low quality
retrieve_knowledge("potato nutrition deficiency") → low quality
retrieve_knowledge("mono diet health effects")   → low quality
→ Model answers from internal knowledge anyway
```

### Options Considered

**Option A: Split into multiple tools**
- `search_semantic` (embedding-based)
- `search_keyword` (BM25-based)
- `search_hybrid` (both + RRF)

**Pros**: Atomic tools, clear separation
**Cons**: Increases tool count from 5 to 7; harder tool selection for 3B model

**Option B: Add `mode` parameter**
- Single tool with `mode: "hybrid" | "semantic" | "keyword"`

**Pros**: Tool count unchanged; strategy selection via argument filling
**Cons**: Tool is no longer "atomic" (contains strategy choice)

## Decision

We adopt **Option B**: Add a `mode` parameter to `retrieve_knowledge`.

```json
{
  "name": "retrieve_knowledge",
  "arguments": {
    "query": "vitamin B12 RDA",
    "mode": "keyword",  // hybrid | semantic | keyword
    "top_k": 3
  }
}
```

Additionally, we add `retrieval_quality` to the return value:

```json
{
  "status": "success",
  "retrieval_quality": "low",  // high | medium | low | none
  "top_relevance_score": 0.34,
  "data": { "passages": [...] }
}
```

### Strategy Pattern to Teach in SFT

```
retrieval_quality: high   → Cite results directly
retrieval_quality: medium → Cite with caveats, or reformulate once
retrieval_quality: low    → Switch mode (hybrid→keyword) or rephrase; max 2 retries
retrieval_quality: none   → Use internal knowledge with disclaimer
```

## Consequences

### Positive

- **Tool count unchanged** (5 tools) — reduces tool selection burden on 3B model
- **Explicit strategy switching** — model can learn "hybrid failed → try keyword"
- **Graceful degradation** — `retrieval_quality` enables model to decide when to stop retrying

### Negative

- **Not atomic** — tool contains internal strategy choice, violating purist tool design
- **Parameter learning required** — model must learn when to use each mode

### Trade-off Rationale

For a 3B model, **parameter filling is easier than tool selection**. The same decision (which retrieval strategy?) happens in either approach, but:
- Option A: Decision at tool selection stage (harder for small models)
- Option B: Decision at argument filling stage (easier, same schema)

## Related

- [specs/tools.md](../specs/tools.md) — Tool 5: retrieve_knowledge
- [src/tools/retrieve_knowledge.py](../../src/tools/retrieve_knowledge.py) — Implementation
- [src/retrieval/hybrid_retriever.py](../../src/retrieval/hybrid_retriever.py) — Mode handling
