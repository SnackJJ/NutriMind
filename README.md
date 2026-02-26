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

- **Model**: Qwen2.5-3B-Instruct (fine-tuned)
- **RAG**: Contextual Retrieval + ChromaDB + BM25
- **Database**: SQLite (USDA FoodData)
- **Training**: SFT + GRPO

## Available Tools

| Tool | Description |
|------|-------------|
| `get_food_nutrition` | Look up nutritional info for foods |
| `log_meal` | Record meals with portions |
| `get_today_summary` | View daily intake summary |
| `get_history` | Query historical meal data |
| `get_goal_adherence` | Check progress toward goals |
| `retrieve_knowledge` | Search nutrition knowledge base |

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
```

## Status

🚧 Under active development

## License

MIT
