#!/usr/bin/env python3
"""Simple one-shot multi-turn dry run for NutriMind GRPO environment.

This script validates that a single rollout can complete two turns:
1) assistant emits a tool call
2) environment injects tool response
3) assistant emits final answer

It is intended as a fast sanity check after config/code changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator.orchestrator import TOOL_REGISTRY
from src.training.grpo.environment import NutriMindEnv


def _build_snapshot() -> Dict[str, Any]:
    """Build a deterministic mock user state for stateful tool dry run."""
    return {
        "user_goals": {"calories": 2000, "protein": 130, "fat": 65, "carbs": 240},
        "meals_today": [
            {
                "meal_type": "breakfast",
                "foods": [{"name": "oatmeal"}],
                "calories": 420,
                "protein_g": 18,
                "fat_g": 12,
                "carbs_g": 60,
                "fiber_g": 8,
            },
            {
                "meal_type": "lunch",
                "foods": [{"name": "chicken salad"}],
                "calories": 560,
                "protein_g": 48,
                "fat_g": 22,
                "carbs_g": 34,
                "fiber_g": 7,
            },
        ],
        "meal_history": [
            {"date": "2026-04-01", "calories": 1950, "protein_g": 120, "fat_g": 64, "carbs_g": 230, "fiber_g": 26},
            {"date": "2026-04-02", "calories": 2100, "protein_g": 128, "fat_g": 71, "carbs_g": 245, "fiber_g": 24},
        ],
    }


def run_dry_once(max_rounds: int) -> int:
    query = "Am I over my calorie budget today? If yes, suggest a light dinner."
    env = NutriMindEnv(
        tool_registry=TOOL_REGISTRY,
        max_tool_rounds=max_rounds,
        user_state_snapshot=_build_snapshot(),
    )

    messages = env.reset(query)
    if len(messages) < 2:
        print("FAIL: reset did not initialize conversation messages")
        return 1

    # Turn 1: tool call
    turn1 = (
        "<think>I should check today's totals first.</think>\n"
        "<tool_call>{\"name\":\"get_today_summary\",\"arguments\":{}}</tool_call>"
    )
    next_messages, done1, info1 = env.step(turn1)
    if done1:
        print("FAIL: rollout terminated unexpectedly after first tool call")
        return 1

    if not next_messages or not next_messages[-1]["content"].startswith("<tool_response>"):
        print("FAIL: environment did not inject a tool response after tool call")
        return 1

    # Turn 2: final answer
    turn2 = (
        "Based on your current intake, you are still under budget. "
        "A light dinner option is grilled fish with steamed vegetables."
    )
    _, done2, info2 = env.step(turn2)
    if not done2:
        print("FAIL: rollout did not terminate after final answer")
        return 1

    traj = env.get_trajectory()
    if traj.total_tool_calls != 1:
        print(f"FAIL: expected 1 tool call, got {traj.total_tool_calls}")
        return 1
    if traj.termination_reason != "final_answer":
        print(f"FAIL: expected termination_reason=final_answer, got {traj.termination_reason}")
        return 1

    summary = {
        "status": "ok",
        "query": query,
        "steps": len(traj.steps),
        "tool_calls": traj.total_tool_calls,
        "termination_reason": traj.termination_reason,
        "turn1_action": info1.get("action_type"),
        "turn2_action": info2.get("action_type"),
        "tools_called": traj.get_tools_called(),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run one multi-turn interaction")
    parser.add_argument("--max_rounds", type=int, default=6, help="Maximum tool rounds for the environment")
    args = parser.parse_args()
    return run_dry_once(max_rounds=args.max_rounds)


if __name__ == "__main__":
    sys.exit(main())
