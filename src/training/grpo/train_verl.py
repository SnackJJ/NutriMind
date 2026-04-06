#!/usr/bin/env python3
"""
veRL-based GRPO/GiGPO Training Entry Point for NutriMind.

This script provides a convenient entry point for launching veRL training
with NutriMind-specific configurations.

Supported algorithms:
- GRPO: Uses veRL's main_ppo with grpo advantage estimator
- GiGPO: Uses veRL-agent for step-level credit assignment (requires verl-agent package)

Usage:
    # GRPO training (2x A100)
    python -m src.training.grpo.train_verl \
        --algorithm grpo \
        --config configs/verl_grpo.yaml \
        --reward_version v2

    # GiGPO training (requires verl-agent)
    python -m src.training.grpo.train_verl \
        --algorithm gigpo \
        --config configs/verl_agent_gigpo.yaml \
        --reward_version v2

    # Consumer GPU training
    python -m src.training.grpo.train_verl \
        --config configs/verl_grpo_consumer.yaml \
        --reward_version v2

    # Dry run (validate config)
    python -m src.training.grpo.train_verl \
        --config configs/verl_grpo.yaml \
        --dry_run

    # Resume from checkpoint
    python -m src.training.grpo.train_verl \
        --config configs/verl_grpo.yaml \
        trainer.resume_mode=auto

Features:
- Automatically prepares data if parquet files don't exist
- Updates reward version in interaction config
- Validates configuration before training
- Supports passing additional veRL overrides via command line
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_data_exists(config_path: str) -> bool:
    """Check if training data exists."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_file = config.get("data", {}).get("train_files", "")
    train_path = PROJECT_ROOT / train_file

    return train_path.exists()


def prepare_data() -> bool:
    """Run data preparation script."""
    logger.info("Preparing veRL training data...")

    script_path = PROJECT_ROOT / "scripts" / "prepare_verl_data.py"
    if not script_path.exists():
        logger.error(f"Data preparation script not found: {script_path}")
        return False

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Data preparation failed:\n{result.stderr}")
        return False

    logger.info("Data preparation complete")
    return True


def update_reward_version(config_path: str, reward_version: str) -> None:
    """Update reward version in interaction config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    interaction_path = config.get("actor_rollout_ref", {}).get(
        "rollout", {}
    ).get("multi_turn", {}).get("interaction_config_path")

    if not interaction_path:
        logger.warning("No interaction_config_path found in config")
        return

    interaction_path = PROJECT_ROOT / interaction_path
    if not interaction_path.exists():
        logger.warning(f"Interaction config not found: {interaction_path}")
        return

    with open(interaction_path, "r") as f:
        interaction_config = yaml.safe_load(f)

    # Update reward version
    updated = False
    for interaction in interaction_config.get("interaction", []):
        if interaction.get("name") == "nutrimind":
            old_version = interaction.get("config", {}).get("reward_version")
            interaction["config"]["reward_version"] = reward_version
            if old_version != reward_version:
                updated = True
                logger.info(f"Updated reward version: {old_version} -> {reward_version}")

    if updated:
        with open(interaction_path, "w") as f:
            yaml.dump(interaction_config, f, default_flow_style=False)


def validate_config(config_path: str) -> bool:
    """Validate veRL configuration."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check required sections
        required_sections = ["data", "actor_rollout_ref", "algorithm", "trainer"]
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required config section: {section}")
                return False

        # Check multi-turn is enabled
        rollout_cfg = config.get("actor_rollout_ref", {}).get("rollout", {})
        multi_turn = rollout_cfg.get("multi_turn", {})

        if not multi_turn.get("enable", False):
            logger.warning("Multi-turn is not enabled in config")

        # Warn about loop mismatch that causes num_turns to stay constant (typically 2)
        agent_loop = rollout_cfg.get("agent", {}).get("default_agent_loop")
        if multi_turn.get("enable", False) and agent_loop == "single_turn_agent":
            logger.warning(
                "multi_turn.enable=true but agent.default_agent_loop=single_turn_agent. "
                "This keeps num_turns metrics near a fixed value; use tool_agent for true multi-turn stats."
            )

        # Check return_raw_chat
        if not config.get("data", {}).get("return_raw_chat", False):
            logger.error("data.return_raw_chat must be true for multi-turn")
            return False

        # Guard against response budget exhaustion in tool_agent loop.
        data_cfg = config.get("data", {})
        max_response_len = int(data_cfg.get("max_response_length", 0) or 0)
        max_tool_response_len = int(multi_turn.get("max_tool_response_length", 0) or 0)
        max_assistant_turns = int(multi_turn.get("max_assistant_turns", 0) or 0)
        if max_response_len < 1:
            logger.error("data.max_response_length must be >= 1")
            return False
        if multi_turn.get("enable", False) and max_tool_response_len >= max_response_len:
            logger.error(
                "Invalid token budget: rollout.multi_turn.max_tool_response_length "
                f"({max_tool_response_len}) must be smaller than data.max_response_length ({max_response_len}). "
                "Otherwise vLLM may receive max_tokens=0 in later turns."
            )
            return False
        if (
            multi_turn.get("enable", False)
            and max_assistant_turns > 0
            and (max_assistant_turns * max_tool_response_len) >= max_response_len
        ):
            logger.error(
                "Invalid token budget: max_assistant_turns * max_tool_response_length "
                f"({max_assistant_turns} * {max_tool_response_len} = {max_assistant_turns * max_tool_response_len}) "
                f"must be smaller than data.max_response_length ({max_response_len}). "
                "Otherwise repeated tool rounds can exhaust generation budget and cause max_tokens=0."
            )
            return False

        logger.info("Configuration validation passed")
        return True

    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="veRL GRPO/GiGPO Training for NutriMind",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # GRPO training
    python -m src.training.grpo.train_verl --algorithm grpo --config configs/verl_grpo.yaml

    # GiGPO training (requires verl-agent package)
    python -m src.training.grpo.train_verl --algorithm gigpo --config configs/verl_agent_gigpo.yaml

    # With reward version override
    python -m src.training.grpo.train_verl --config configs/verl_grpo.yaml --reward_version v1

    # Dry run
    python -m src.training.grpo.train_verl --config configs/verl_grpo.yaml --dry_run

    # Pass additional veRL overrides
    python -m src.training.grpo.train_verl --config configs/verl_grpo.yaml \
        trainer.experiment_name=my_experiment \
        data.train_batch_size=16
        """,
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="grpo",
        choices=["grpo", "gigpo"],
        help="RL algorithm (grpo uses veRL, gigpo uses veRL-agent)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/verl_grpo.yaml",
        help="Path to veRL config file",
    )
    parser.add_argument(
        "--reward_version",
        type=str,
        default="v2",
        choices=["v1", "v2", "v3"],
        help="Reward function version",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate config and print command without executing",
    )
    parser.add_argument(
        "--skip_data_check",
        action="store_true",
        help="Skip data existence check",
    )

    args, unknown_args = parser.parse_known_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    logger.info(f"Using config: {config_path}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Reward version: {args.reward_version}")

    # Make reward version visible to custom reward function running inside veRL workers.
    os.environ["NUTRIMIND_REWARD_VERSION"] = args.reward_version

    # Validate configuration
    if not validate_config(str(config_path)):
        return 1

    # Check/prepare data
    if not args.skip_data_check:
        if not check_data_exists(str(config_path)):
            logger.info("Training data not found, preparing...")
            if not prepare_data():
                return 1
        else:
            logger.info("Training data exists")

    # Update reward version in interaction config
    update_reward_version(str(config_path), args.reward_version)

    # Build veRL command based on algorithm
    config_dir = config_path.parent
    config_name = config_path.stem

    if args.algorithm == "gigpo":
        # GiGPO uses verl-agent entry point
        try:
            import verl_agent  # noqa: F401
        except ImportError:
            logger.error(
                "GiGPO requires verl-agent package. Install with:\n"
                "  pip install verl-agent\n"
                "Or:\n"
                "  pip install git+https://github.com/langfengQ/verl-agent.git"
            )
            return 1

        cmd = [
            sys.executable, "-m", "verl_agent.trainer.main_ppo",
            f"--config-path={config_dir}",
            f"--config-name={config_name}",
            "algorithm.adv_estimator=gigpo",
        ]
        logger.info("Using veRL-agent for GiGPO training")
    else:
        # GRPO uses standard veRL entry point
        cmd = [
            sys.executable, "-m", "verl.trainer.main_ppo",
            f"--config-path={config_dir}",
            f"--config-name={config_name}",
            "algorithm.adv_estimator=grpo",
        ]
        logger.info("Using veRL for GRPO training")

    # Add any additional overrides from command line
    cmd.extend(unknown_args)

    logger.info(f"Command: {' '.join(cmd)}")

    if args.dry_run:
        logger.info("Dry run complete. Command would be executed above.")
        return 0

    # Execute veRL training
    os.chdir(PROJECT_ROOT)
    result = subprocess.run(cmd)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
