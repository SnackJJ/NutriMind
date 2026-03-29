#!/usr/bin/env python3
"""Test agent environment with prompts from trajectory data."""

import json
import random
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.orchestrator.orchestrator import orchestrate, get_backend, TOOL_REGISTRY

def load_test_prompts(trajectory_path: str, n_samples: int = 5) -> list:
    """Load diverse prompts from trajectory data."""
    prompts_by_tier = {}

    with open(trajectory_path, "r") as f:
        for line in f:
            data = json.loads(line)
            tier = data.get("tier", "unknown")
            if tier not in prompts_by_tier:
                prompts_by_tier[tier] = []
            prompts_by_tier[tier].append(data["query"])

    # Sample from each tier
    selected = []
    for tier in sorted(prompts_by_tier.keys()):
        queries = prompts_by_tier[tier]
        sample_size = min(1, len(queries))
        sampled = random.sample(queries, sample_size)
        for q in sampled:
            selected.append({"tier": tier, "query": q})

    return selected[:n_samples]


def test_tools():
    """Test each tool individually."""
    print("=" * 60)
    print("TOOL SMOKE TEST")
    print("=" * 60)

    tests = [
        ("get_food_nutrition", {"foods": [{"food_name": "apple", "amount_grams": 100}]}),
        ("get_today_summary", {}),
        ("get_history", {"days": 3, "compare_to_goal": True}),
        ("retrieve_knowledge", {"query": "protein intake for athletes"}),
        ("set_goal", {"metric": "calories", "target_value": 2500, "goal_type": "maintain"}),
    ]

    for tool_name, args in tests:
        try:
            func = TOOL_REGISTRY[tool_name]
            if tool_name == "get_today_summary":
                result = func()
            else:
                result = func(**args)
            status = result.get("status", "unknown")
            print(f"✓ {tool_name}: {status}")
            if status == "success" and "results" in result:
                print(f"  → {len(result['results'])} results")
        except Exception as e:
            print(f"✗ {tool_name}: {e}")

    print()


def test_backend():
    """Test backend initialization."""
    print("=" * 60)
    print("BACKEND TEST")
    print("=" * 60)

    backend = get_backend()
    print(f"Backend type: {type(backend).__name__}")

    # Quick generation test
    try:
        response = backend.generate([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one word."}
        ])
        print(f"Generation test: {response[:100]}...")
    except Exception as e:
        print(f"Generation error: {e}")

    print()


def test_orchestrator(prompts: list):
    """Test full orchestration with prompts."""
    print("=" * 60)
    print("ORCHESTRATOR TEST")
    print("=" * 60)

    for i, item in enumerate(prompts, 1):
        tier = item["tier"]
        query = item["query"]
        print(f"\n[{i}] Tier: {tier}")
        print(f"Query: {query[:80]}{'...' if len(query) > 80 else ''}")
        print("-" * 40)

        try:
            response = orchestrate(query)
            print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        except Exception as e:
            print(f"Error: {e}")


def main():
    trajectory_path = project_root / "data" / "trajectories" / "sft_train_trajectory.jsonl"

    print("\n" + "=" * 60)
    print("NUTRIAGENT ENVIRONMENT TEST")
    print("=" * 60)
    print(f"Trajectory: {trajectory_path}")
    print()

    # 1. Test tools
    test_tools()

    # 2. Test backend
    test_backend()

    # 3. Load prompts and test orchestrator
    if trajectory_path.exists():
        prompts = load_test_prompts(str(trajectory_path), n_samples=5)
        print(f"Loaded {len(prompts)} test prompts")
        test_orchestrator(prompts)
    else:
        print(f"Trajectory file not found: {trajectory_path}")
        # Use fallback prompts
        fallback = [
            {"tier": "T1", "query": "How much protein is in 100g of chicken breast?"},
            {"tier": "T2", "query": "What did I eat today?"},
        ]
        test_orchestrator(fallback)


if __name__ == "__main__":
    main()
