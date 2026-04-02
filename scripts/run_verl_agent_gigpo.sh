#!/bin/bash
# GiGPO Training Script using veRL-agent
#
# This script runs GiGPO (Group-in-Group Policy Optimization) training
# using the official verl-agent implementation.
#
# Prerequisites:
#   pip install verl-agent
#   # Or: pip install git+https://github.com/langfengQ/verl-agent.git
#
# Usage:
#   ./scripts/run_verl_agent_gigpo.sh          # Default v2 reward
#   ./scripts/run_verl_agent_gigpo.sh v1       # Use v1 reward
#   ./scripts/run_verl_agent_gigpo.sh v2       # Use v2 reward

set -e

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Parse reward version argument (default: v2)
REWARD_VERSION="${1:-v2}"

echo "=========================================="
echo "NutriMind GiGPO Training (veRL-agent)"
echo "=========================================="
echo "Reward version: $REWARD_VERSION"
echo "Config: configs/verl_agent_gigpo.yaml"
echo "=========================================="

# Check if verl-agent is installed
if ! python -c "import verl_agent" 2>/dev/null; then
    echo "ERROR: verl-agent not installed."
    echo "Install with:"
    echo "  pip install verl-agent"
    echo "Or:"
    echo "  pip install git+https://github.com/langfengQ/verl-agent.git"
    exit 1
fi

# Run training via train_verl.py
python -m src.training.grpo.train_verl \
    --algorithm gigpo \
    --config configs/verl_agent_gigpo.yaml \
    --reward_version "$REWARD_VERSION" \
    "$@"
