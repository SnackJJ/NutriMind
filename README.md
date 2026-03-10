# NutriMind

A nutrition-focused AI agent powered by a fine-tuned 3B parameter language model.

## Overview

NutriMind is an intelligent nutrition assistant that helps users:
- Track daily food intake and nutritional information
- Get personalized dietary recommendations
- Access evidence-based nutrition knowledge

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   User      │────▶│  Orchestrator │────▶│   Tools     │
│   Query     │     │  (Agentic)   │     │  (6 tools)  │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              ┌─────────┐   ┌─────────┐
              │   RAG   │   │  USDA   │
              │ (Hybrid)│   │   DB    │
              └─────────┘   └─────────┘
```

## Tech Stack

- **Model**: Qwen2.5-3B-Instruct (fine-tuned with SFT + GRPO)
- **RAG**: Contextual Retrieval + ChromaDB + BM25 + Reranker
- **Database**: SQLite (USDA SR Legacy + Foundation Foods)
- **Training**: transformers + trl + peft (LoRA)

## Available Tools

| Tool | Description |
|------|-------------|
| `get_food_nutrition` | Look up nutritional info for foods |
| `log_meal` | Record meals with portions |
| `get_today_summary` | View daily intake summary |
| `get_history` | Query historical meal data |
| `get_goal_adherence` | Check progress toward goals |
| `retrieve_knowledge` | Search nutrition knowledge base |

## Project Structure

```
├── configs/          # YAML configuration files
├── docs/             # Specifications and plans
├── scripts/          # Data processing pipelines
├── src/
│   ├── orchestrator/ # Agentic loop (observe-think-act)
│   ├── retrieval/    # RAG system (hybrid search)
│   ├── tools/        # 6 nutrition tools
│   └── training/     # SFT data collection
└── tests/            # Unit tests
```

## Getting Started

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Initialize database
python scripts/download_usda.py
python scripts/init_user_tables.py

# Build knowledge index
python scripts/build_indexes.py
```

## Development Status

- [x] Phase 1: Infrastructure (USDA DB, RAG system)
- [x] Phase 2: Tools implementation
- [x] Phase 2.5: SFT data pipeline
- [ ] Phase 3: SFT training
- [ ] Phase 4: GRPO alignment
- [ ] Phase 5: Ablation studies
- [ ] Phase 6: Evaluation
- [ ] Phase 7: Deployment

## License

MIT
