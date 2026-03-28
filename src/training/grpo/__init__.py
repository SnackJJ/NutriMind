"""
GRPO/GiGPO Training Module for NutriMind.

This module implements reinforcement learning training for the NutriMind
nutrition assistant, using GRPO (Group Relative Policy Optimization) and
GiGPO (Group-in-Group Policy Optimization) algorithms.

Key components:
- environment.py: Multi-turn rollout environment (NutriMindEnv)
- reward.py: Iterative reward functions (v1/v2/v3)
- gigpo.py: Step-level credit assignment for GiGPO
- monitor.py: Training monitoring and reward hacking detection
- train.py: Main training script

See docs/plans/phase4_grpo.md for architecture and experiment design.
"""

from src.training.grpo.environment import (
    NutriMindEnv,
    RolloutGroup,
    RolloutTrajectory,
    RolloutStep,
    TaskMetadata,
    DeterministicToolCache,
    compute_state_key,
)

from src.training.grpo.reward import (
    RewardBreakdown,
    reward_v1,
    reward_v2,
    reward_v3,
    LLMJudge,
    detect_reward_hacking,
    RewardHackingAlert,
)

from src.training.grpo.gigpo import (
    GiGPOComputer,
    GiGPOResult,
    StepAdvantage,
    AnchorState,
    compute_gigpo_advantages,
    compute_group_advantages,
)

from src.training.grpo.monitor import (
    TrainingMonitor,
    MonitorConfig,
    EvalMetrics,
    WandbMonitor,
)

__all__ = [
    # Environment
    "NutriMindEnv",
    "RolloutGroup",
    "RolloutTrajectory",
    "RolloutStep",
    "TaskMetadata",
    "DeterministicToolCache",
    "compute_state_key",
    # Reward
    "RewardBreakdown",
    "reward_v1",
    "reward_v2",
    "reward_v3",
    "LLMJudge",
    "detect_reward_hacking",
    "RewardHackingAlert",
    # GiGPO
    "GiGPOComputer",
    "GiGPOResult",
    "StepAdvantage",
    "AnchorState",
    "compute_gigpo_advantages",
    "compute_group_advantages",
    # Monitor
    "TrainingMonitor",
    "MonitorConfig",
    "EvalMetrics",
    "WandbMonitor",
]
