# Project Memory

## Tech Stack
- Python 3.10+, uv (package management)
- Base Model: Qwen3-4B (student), qwen3.5-plus (teacher)
- Training: Unsloth (SFT) + trl (SFTTrainer/GRPO) + peft (LoRA)
- Inference: vLLM (prod) / Transformers (dev)
- Database: SQLite (USDA SR Legacy + Foundation Foods)
- RAG: Contextual Retrieval + ChromaDB + BM25 + bge-small-en-v1.5 + bge-reranker-base
- Document Processing: BeautifulSoup (HTML) + docling (PDF)
- Config: YAML (configs/) + .env

## Project Structure
- `src/orchestrator/` — Agentic loop state machine (observe→think→act→observe) ✅ COMPLETE
- `src/tools/` — 5 tools for SFT (get_food_nutrition, log_meal, get_today_summary, get_history, retrieve_knowledge) ✅ COMPLETE
- `src/retrieval/` — Contextual Retrieval RAG v3 ✅ COMPLETE (1635 chunks, ChromaDB + BM25 indexed, 23/23 eval tests pass)
- `src/training/sft/` — Teacher trajectory collection & validation
- `configs/` — model.yaml, orchestrator.yaml, tools.yaml
- `data/` — **NEEDS FULL REBUILD** (accidentally deleted 2026-03-11); rebuild order: Track A (RAG) + Track B (USDA) + Track C (query pool) → Track D (trajectories)
- `docs/plans/` — 7-phase execution plans + phase1.2 RAG rewrite
- `docs/specs/` — Technical specifications
- `scripts/` — Data pipeline (USDA download, document processing, init_user_tables.py, export_usda_foods.py)

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

**ADR (Architecture Decision Records)**: `docs/decisions/`
- Template: [000-template.md](docs/decisions/000-template.md)
- New decisions → create ADR file (ADR-NNN-title.md)
- ADR status: proposed → accepted → [deprecated | superseded]

**Active ADRs**:
- ADR-001: Pure Text Tool Calling — **Never created as file**; format is Qwen3 native `<tool_call>` JSON, not a custom protocol
- [ADR-002: veRL to TRL Migration](docs/decisions/002-verl-to-trl-migration.md) — **Partially superseded** by ADR-007; veRL→TRL decision valid, `rollout_func` choice superseded
- [ADR-007: environment_factory Migration](docs/decisions/007-environment-factory-migration.md) — **Proposed**; migrate from `rollout_func` to TRL native `environment_factory` for single A800

**Obsolete Files** (per ADR-007):
- `src/training/grpo/trl_environment.py` — rollout_func, replaced by environment_factory
- `src/training/grpo/train_trl.py` — dual-GPU entry point, replaced by single A800 setup
- `src/training/grpo/train_verl.py` — veRL entry point (ADR-002)
- `src/training/grpo/verl_interaction.py` — veRL interaction (ADR-002)
- `src/training/grpo/verl_agent_env.py` — veRL env (ADR-002)
- `scripts/run_trl_grpo_4090d.sh` — dual-GPU launch script
- `scripts/run_verl_grpo_4090d.sh` — veRL launch script

### Historical Decisions (pre-ADR)

See [docs/decisions/historical-decisions.md](../docs/decisions/historical-decisions.md) for decisions made before ADR process adoption.