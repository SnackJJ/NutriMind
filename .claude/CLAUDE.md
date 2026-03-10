# Project Memory

## Tech Stack
- Python 3.10+, uv (package management)
- Base Model: Qwen2.5-3B-Instruct (student), Qwen-Max/Plus (teacher)
- Training: transformers + trl (SFT + GRPO) + peft (LoRA)
- Inference: vLLM (prod) / Transformers (dev)
- Database: SQLite (USDA SR Legacy + Foundation Foods)
- RAG: Contextual Retrieval + ChromaDB + BM25 + bge-small-en-v1.5 + bge-reranker-base
- Document Processing: BeautifulSoup (HTML) + docling (PDF)
- Config: YAML (configs/) + .env

## Project Structure
- `src/orchestrator/` — Agentic loop state machine (observe→think→act→observe) ✅ COMPLETE
- `src/tools/` — 6 tools for SFT (get_food_nutrition, log_meal, get_today_summary, get_history, retrieve_knowledge, get_goal_adherence) ✅ COMPLETE
- `src/retrieval/` — Contextual Retrieval RAG v3 ✅ COMPLETE (1635 chunks, ChromaDB + BM25 indexed, 23/23 eval tests pass)
- `src/training/sft/` — Teacher trajectory collection & validation
- `configs/` — model.yaml, orchestrator.yaml, tools.yaml
- `data/` — USDA DB (8109 foods + user tables), parsed docs (26 files), data/knowledge/ (1635 chunks + contextualized + cache), knowledge_db/ (ChromaDB), knowledge_bm25/index.pkl, trajectories, SFT/GRPO data
- `docs/plans/` — 7-phase execution plans + phase1.2 RAG rewrite
- `docs/specs/` — Technical specifications
- `scripts/` — Data pipeline (USDA download, document processing, init_user_tables.py)

## File Map

| Spec | Related Plans | Code |
|------|--------------|------|
| specs/database.md | phase1_infrastructure | scripts/, data/ |
| specs/rag.md | phase1_infrastructure, phase1.2_rag_knowledge_base | src/retrieval/ |
| specs/tools.md | phase2_sft_data, phase2.5_sft_data_pipeline | src/tools/ |
| specs/orchestrator.md | phase2.6_trajectory_collection | src/orchestrator/ |
| specs/training.md | phase3_sft_training, phase4_grpo | src/training/ |
| specs/deployment.md | phase7_deployment | — |

Cross-cutting plans (no single spec):
- master_plan.md → overview of all phases
- phase5_ablation.md → specs/training.md + specs/rag.md
- phase6_evaluation.md → all specs

## Conventions
- 100% English policy: all queries, <think> blocks, tool params, answers
- Tier system: T1 (single tool) / T2 (multi-step) / T3 (conditional) / T4 (safety boundary)
- Sequential tool execution only (no parallel tool calls)
- Max 6 tool rounds per conversation

## Key Decisions
- [2025] Removed `call_expert_nutritionist` — T4 cases handled by safety boundary declarations instead
- [2025] Hybrid retrieval over pure semantic — cosine similarity fails for similar nutrient names (B12 vs B6 ≈ 0.92)
- [2025] Sequential forcing via history truncation — 3B model cannot reliably do parallel tool calls
- [2025] Safety layers: allergen detection, extreme calorie checks (<800/>5000), T4 clinical boundary refusal
- [2026-03-08] RAG v2 rewrite: deleted all RAG code/data due to audit findings (B1-B4, D1-D5, S1-S4)
- [2026-03-08] RAG v3 design approved: Contextual Retrieval (Anthropic 2024) + docling PDF parsing + boolean domain metadata + two-level domain tagging. No safety_boundaries chunks (T4 at orchestrator). Context prefix ≤50 tokens. See specs/rag.md v3
- [2026-03-08] RAG v3 spec review fixes: (1) threshold only applies when reranker available — RRF scores ~0.03 incompatible with 0.3 threshold; (2) added DOMAIN_RULES for supplements & weight_management — without rules these domains were unreachable; (3) synced tools.yaml (top_k 15→20, removed prepend_heading, added context_* fields)
- [2026-03-08] Contextualization LLM switched from Qwen-Max to Gemini 2.5 Flash — uses OpenAI-compatible endpoint (no new dependency); .env needs GEMINI_API_KEY
- [2026-03-08] Chunker fix: filter residual <min_tokens chunks after merge — removes ~68 noise chunks (figure labels, 404 pages, boilerplate) from DGA/ACOG PDFs
- [2026-03-08] WHO sugars/sodium and MyPlate sources deferred — DGA 2020 covers the same scope; chunk count target updated to 2500-2700
- [2026-03-08] Chunk max_tokens updated 256→450 — 450 + 50 context prefix ≤ 512 BGE-small limit; estimated ~1300-1600 total chunks; chunks.jsonl needs regeneration
- [2026-03-08] RAG v3 fully complete — 1635 chunks indexed, ChromaDB + BM25 built, HybridRetriever integrated, 23/23 eval tests pass; bge-reranker-base corrupted cache deleted and re-downloaded via hf-mirror.com; HF_ENDPOINT=https://hf-mirror.com required for downloads
- [2026-03-08] Phase 1 infrastructure complete — all 8 tools backed by real SQLite DB; user tables created via scripts/init_user_tables.py; calculate_meal.py bug fixed (amount→amount_grams in failed_items); 27/27 unit tests pass in tests/test_tools.py
- [2026-03-08] Removed `set_goal` from SFT schema (7 tools remain) — query pool has 0/2495 set_goal queries; `src/tools/set_goal.py` retained for GRPO phase
- [2026-03-09] Merged `search_food` + `calculate_meal` → `get_food_nutrition` — eliminates tool overlap, reduces 3B model decision burden (7→6 tools); single tool handles both single-food and multi-food queries; T1 tier now uses get_food_nutrition for all food lookups
- [2026-03-09] Trajectory collector improvements — per-tier MAX_TURNS (T0:3, T1:5, T2:8, T3:12, T4:3), tenacity retry with exponential backoff, failed queries saved to *_failed.jsonl, tool result truncation (4000 chars max), JSON error detection via parsing instead of string match
- [2026-03-09] Knowledge base gap fix — Tier 3 sources added (NIH NIDDK: gallstones/GERD/IBS/celiac diet pages + MedlinePlus food allergy page); root cause: medical_nutrition domain had 0 GI disease content, food_safety had 0 food allergy content; fix also expands DOMAIN_RULES keywords for both domains; collect_sources.py PDF validation bug fixed (was rejecting HTML downloads); NIAID blocked scraping (405) → replaced with MedlinePlus (nlm.nih.gov); pipeline rebuild required after this change