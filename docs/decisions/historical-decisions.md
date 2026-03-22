# Historical Decisions (Pre-ADR)

> These decisions were made before the ADR process was adopted. They are preserved here for reference.
> For new decisions, create an ADR file using the [template](000-template.md).

## 2025

- Removed `call_expert_nutritionist` — T4 cases handled by safety boundary declarations instead
- Hybrid retrieval over pure semantic — cosine similarity fails for similar nutrient names (B12 vs B6 ≈ 0.92)
- Sequential forcing via history truncation — 3B model cannot reliably do parallel tool calls
- Safety layers: allergen detection, extreme calorie checks (<800/>5000), T4 clinical boundary refusal

## 2026-03

### 2026-03-08: RAG v3 Rewrite

- RAG v2 rewrite: deleted all RAG code/data due to audit findings (B1-B4, D1-D5, S1-S4)
- RAG v3 design approved: Contextual Retrieval (Anthropic 2024) + docling PDF parsing + boolean domain metadata + two-level domain tagging. No safety_boundaries chunks (T4 at orchestrator). Context prefix ≤50 tokens. See specs/rag.md v3
- RAG v3 spec review fixes: (1) threshold only applies when reranker available — RRF scores ~0.03 incompatible with 0.3 threshold; (2) added DOMAIN_RULES for supplements & weight_management — without rules these domains were unreachable; (3) synced tools.yaml (top_k 15→20, removed prepend_heading, added context_* fields)
- Contextualization LLM switched from Qwen-Max to Gemini 2.5 Flash — uses OpenAI-compatible endpoint (no new dependency); .env needs GEMINI_API_KEY
- Chunker fix: filter residual <min_tokens chunks after merge — removes ~68 noise chunks (figure labels, 404 pages, boilerplate) from DGA/ACOG PDFs
- WHO sugars/sodium and MyPlate sources deferred — DGA 2020 covers the same scope; chunk count target updated to 2500-2700
- Chunk max_tokens updated 256→450 — 450 + 50 context prefix ≤ 512 BGE-small limit; estimated ~1300-1600 total chunks; chunks.jsonl needs regeneration
- RAG v3 fully complete — 1635 chunks indexed, ChromaDB + BM25 built, HybridRetriever integrated, 23/23 eval tests pass; bge-reranker-base corrupted cache deleted and re-downloaded via hf-mirror.com; HF_ENDPOINT=https://hf-mirror.com required for downloads
- Phase 1 infrastructure complete — all 8 tools backed by real SQLite DB; user tables created via scripts/init_user_tables.py; calculate_meal.py bug fixed (amount→amount_grams in failed_items); 27/27 unit tests pass in tests/test_tools.py
- Removed `set_goal` from SFT schema (7 tools remain) — query pool has 0/2495 set_goal queries; `src/tools/set_goal.py` retained for GRPO phase

### 2026-03-09: Tool Consolidation

- Merged `search_food` + `calculate_meal` → `get_food_nutrition` — eliminates tool overlap, reduces 3B model decision burden (7→6 tools); single tool handles both single-food and multi-food queries; T1 tier now uses get_food_nutrition for all food lookups
- Trajectory collector improvements — per-tier MAX_TURNS (T0:3, T1:5, T2:8, T3:12, T4:3), tenacity retry with exponential backoff, failed queries saved to *_failed.jsonl, tool result truncation (4000 chars max), JSON error detection via parsing instead of string match
- Knowledge base gap fix — Tier 3 sources added (NIH NIDDK: gallstones/GERD/IBS/celiac diet pages + MedlinePlus food allergy page); root cause: medical_nutrition domain had 0 GI disease content, food_safety had 0 food allergy content; fix also expands DOMAIN_RULES keywords for both domains; collect_sources.py PDF validation bug fixed (was rejecting HTML downloads); NIAID blocked scraping (405) → replaced with MedlinePlus (nlm.nih.gov); pipeline rebuild required after this change

### 2026-03-11: Data Rebuild & Model Changes

- data/ directory accidentally deleted — full rebuild required; all three tracks (RAG, USDA, query pool) must be regenerated; candidate_seeds.json (57 seeds) already committed to data/queries/
- Teacher model changed qwen-plus → qwen3.5-plus in TEACHER_MODELS dict (src/training/sft/collect_trajectories.py); also qwen3.5-plus-2026-02-15 is a valid alias
- Query pool target reduced 5800 → 5000: new TARGETS T0=200/T1=1050/T2=1300/T3=1500/T4=700/Error=250; SFT/GRPO split changed from 43/57 → 50/50 (2500+2500); expected ~1000-1300 validated trajectories after validation passes
- Added scripts/export_usda_foods.py — exports food names from usda.db to data/usda_foods.json; required by expand_query_pool.py before T1 query expansion; run after B3 (init_user_tables.py)
- DGA 2020-2025 manually downloaded as compressed version; actual filename: Dietary_Guidelines_for_Americans_2020-2025-compressed_1.pdf (not dga_2020_2025.pdf); registered in manifest.json manually; WHO/MyPlate/ACOG sources skipped — DGA covers the same scope

### 2026-03-13: Base Model Switch

- Base model switched Qwen2.5-3B-Instruct → Qwen3-4B: native single-token <think>/<tool_call>/<tool_response> (all 6 tags in vocabulary); Qwen3-4B ≈ Qwen2.5-7B performance; pre-trained agentic capability; 4bit LoRA on 4090 feasible (~15GB VRAM); output dir changed models/nutrimind-3b-sft → models/nutrimind-4b-sft; SFT loss masking: only compute loss on assistant turns (mask system/user/tool_response)
- Merged `get_goal_adherence` → `get_history(compare_to_goal=True)` — 200条 trajectory 仅调用1次，teacher 模型无法区分两者使用场景；工具数量 6→5；`src/tools/get_goal_adherence.py` 标记 deprecated

### 2026-03-14: Tool Quality Fixes

- think 退化根因确认：是 `get_food_nutrition` LIKE 检索质量差（"cola"→无关条目），迫使 teacher model 重试 3-4 轮但无新信息可分析，导致模板 think。修复方案：alias 表 (data/food_aliases.json, 100-200条) + `match_confidence` 字段 (high/medium/low)，作为 Phase 2.6 Step 0 前置。tool 修好、dry-run 确认 low confidence rate <10% 前不收集数据。spec: docs/specs/tools.md, plan: docs/plans/phase2.6_trajectory_collection.md § 0
- **移除 `retrieve_knowledge` 的 domain 硬过滤**：分析 1988 条 trajectory 发现 sports_nutrition/supplements/micronutrients 三个 domain 空结果率 40-50%；根因是 teacher 选 domain 的语义逻辑与 chunk 打标签的关键词规则不对齐；~1700 chunks 规模下四级 pipeline (ChromaDB+BM25+RRF+reranker) 本身足以保证精度；改为全库搜索，tool schema 移除 domain 参数；如未来 chunk 数量增长到 5000+ 可考虑 soft boost 而非硬过滤

### 2026-03-18: Trajectory Collection Fixes

- **修复 trajectory 收集丢失 `<think>` 块问题**：DashScope API 返回字段名是 `reasoning_content` 而非 `thinking_content`；修复 collect_trajectories.py 中两处 getattr 调用
- **System prompt v2**：删除重复段落，合并 BEHAVIOR GUIDELINES + CRITICAL CONSTRAINTS 为更清晰的结构；增加语义不匹配检查规则（high confidence but wrong food）；移除对 `<think>` 输出格式的引用（API reasoning_content 是独立通道）
