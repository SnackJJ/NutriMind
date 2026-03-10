# Trajectory Collection Quality Fixes

**Date**: 2026-03-06
**Module**: training/sft, tools, retrieval
**Action**: Bug fixes for trajectory data quality

## Summary

Fixed critical issues discovered in `data/trajectories/real_trajectories.jsonl` that would have produced unusable training data for the 3B model.

## Issues Fixed

### 1. Missing `<think>` blocks (collect_trajectories.py)

**Problem**: All assistant messages had empty `content` fields. Qwen's reasoning was returned in a separate `reasoning_content` field that wasn't being extracted.

**Fix**: Extract `reasoning_content` from Qwen API response and wrap it as `<think>...</think>` block.

```python
# Before
msg_dict = msg.model_dump(exclude_none=True)

# After
msg_dict = msg.model_dump(exclude_none=True)
reasoning = getattr(msg, 'reasoning_content', None)
if reasoning:
    original_content = msg_dict.get("content", "") or ""
    msg_dict["content"] = f"<think>{reasoning}</think>\n{original_content}".strip()
```

### 2. SafetensorError in retrieve_knowledge (hybrid_retriever.py)

**Problem**: CrossEncoder reranker loading failed with SafetensorError, causing all knowledge retrieval to fail. This led to:
- Incorrect T4 safety refusals (e.g., walking advice wrongly classified as medical)
- Missing knowledge context for nutrition questions

**Fix**:
- Lazy-load CrossEncoder with error caching
- Graceful degradation: fall back to RRF scores if reranker fails
- Better exception handling in `retrieve_knowledge.py`

### 3. Data loss in calculate_meal (calculate_meal.py)

**Problem**: When `search_food` failed for a food item, it was silently dropped from the breakdown. Example: 3 foods in, 1 food out.

**Fix**:
- Track failed items in `failed_items` array with error details
- Return `status: "partial_success"` when some foods found, some failed
- Return `status: "error"` with `error_type: "all_foods_not_found"` when all fail

### 4. Poor fuzzy matching in search_food (search_food.py)

**Problem**: "grilled chicken breast" couldn't match "Chicken, broiler or fryers, breast, skinless, boneless, meat only, cooked, grilled" in USDA database.

**Fix**:
- Added `FOOD_ALIASES` mapping for common query→USDA term translations
- Added `normalize_food_name()` that generates word permutations
- Case-insensitive LIKE queries

## Files Modified

| File | Lines Changed |
|------|--------------|
| `src/training/sft/collect_trajectories.py` | +7 |
| `src/retrieval/hybrid_retriever.py` | +35 |
| `src/tools/retrieve_knowledge.py` | +12 |
| `src/tools/calculate_meal.py` | +18 |
| `src/tools/search_food.py` | +55 |

## Testing Required

1. **Re-run trajectory collection** with a few test queries to verify `<think>` blocks appear
2. **Test retrieve_knowledge** - should gracefully degrade if reranker fails
3. **Test calculate_meal** with mixed valid/invalid foods - verify `failed_items` populated
4. **Test search_food** with aliases like "grilled chicken breast", "steamed broccoli"

## Next Steps

- Rebuild BM25 index and Chroma DB if they are corrupted
- Re-collect all trajectories with fixed scripts
- Consider adding more food aliases based on common failures
