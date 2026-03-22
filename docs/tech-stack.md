# NutriMind Agent: Tech Stack

## Runtime Environment

```
Python 3.10+
```

## Core Dependencies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Base Model** | Qwen3-4B | Fine-tuned as planner/router |
| **Expert Model** | Qwen-Max (primary), GPT-4o (fallback) | Complex planning tasks |
| **Serving** | vLLM | High-throughput 4B inference |
| **Food Database** | SQLite | USDA SR Legacy + Foundation Foods |
| **Vector Store** | Chroma (dev) / Supabase pgvector (prod) | RAG knowledge retrieval |
| **Embedding** | bge-small-en-v1.5 | Local embedding model |
| **Training** | transformers + trl | SFT + GRPO |

## Python Packages

```txt
# Inference & Serving
vllm>=0.4.0
transformers>=4.40.0
torch>=2.2.0

# Training
trl>=0.8.0
peft>=0.10.0
datasets>=2.18.0

# Database & RAG
chromadb>=0.4.0
sentence-transformers>=2.5.0
sqlite3  # stdlib

# API Clients
openai>=1.0.0  # for Qwen-Max and GPT-4o
httpx>=0.27.0

# Utilities
pydantic>=2.0.0
loguru>=0.7.0
```

## Directory Structure

```
NutriMind/
├── agent/
│   ├── PRD.md                 # Main product requirements
│   ├── tech-stack.md          # This file
│   └── specs/                 # Detailed specifications
│       ├── tools.md
│       ├── orchestrator.md
│       ├── database.md
│       ├── rag.md
│       └── training.md
├── serving/
│   ├── orchestrator.py        # Main agentic loop
│   ├── tools/                 # Tool implementations
│   │   ├── search_food.py
│   │   ├── calculate_meal.py
│   │   ├── log_user_data.py
│   │   ├── retrieve_knowledge.py
│   │   └── call_expert.py
│   └── inference.py           # vLLM client
├── training/
│   ├── sft/                   # Supervised fine-tuning
│   │   ├── data_generation.py
│   │   ├── train.py
│   │   └── data/
│   └── grpo/                  # GRPO alignment
│       ├── reward.py
│       └── train.py
├── data/
│   ├── usda.db               # SQLite database
│   └── knowledge/            # RAG source documents
└── configs/
    └── model_config.yaml
```

## Environment Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Download USDA data (see specs/database.md)
python scripts/download_usda.py

# Initialize vector store (see specs/rag.md)
python scripts/init_vectorstore.py
```

## API Keys Required

```bash
# .env file
QWEN_API_KEY=xxx          # For Qwen-Max expert calls
OPENAI_API_KEY=xxx        # For GPT-4o (fallback/cross-validation)
```
