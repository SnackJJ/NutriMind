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

## Status

🚧 Under active development

## License

MIT
