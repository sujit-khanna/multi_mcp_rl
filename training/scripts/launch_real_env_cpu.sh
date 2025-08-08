#!/bin/bash
# Launch script for REAL ENVIRONMENT GRPO Training on CPU
# ========================================================
# This script forces CPU usage to avoid MPS memory issues

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the skyrl_tool_agent root directory (two levels up from scripts)
ROOT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to root directory
cd "$ROOT_DIR"
echo "Working directory: $(pwd)"

# Activate virtual environment if it exists
if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
    source "$ROOT_DIR/.venv/bin/activate"
    echo "Virtual environment activated"
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/.."
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export DEVICE_TYPE="cpu"  # FORCE CPU to avoid MPS issues
export DISABLE_BITSANDBYTES=1  # Force disable BitsAndBytes

# Load environment variables from .env
if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    source "$ROOT_DIR/.env"
    set +a
    echo ".env file loaded"
elif [ -f "$ROOT_DIR/../.env" ]; then
    set -a
    source "$ROOT_DIR/../.env"
    set +a
    echo "Parent .env file loaded"
fi

# WandB configuration (optional)
export WANDB_PROJECT="skyrl-grpo-real-env"
export WANDB_MODE="online"  # Set to "offline" to disable wandb

# Create output directory with timestamp
OUTPUT_DIR="outputs/real-env-grpo-cpu-$(date +%Y%m%d-%H%M%S)"
mkdir -p $OUTPUT_DIR

# Copy configuration files to output directory
cp training/configs/training_config_qwen3_0.6b.yaml $OUTPUT_DIR/ 2>/dev/null || echo "Warning: Could not copy training config"
cp training/configs/grpo_config_fixed.yaml $OUTPUT_DIR/ 2>/dev/null || echo "Warning: Could not copy grpo config"

echo "================================================"
echo "Starting REAL ENVIRONMENT GRPO Training (CPU)"
echo "================================================"
echo "Output directory: $OUTPUT_DIR"
echo "Device: CPU (forced to avoid MPS issues)"
echo "Python: $(which python)"
echo "================================================"

# Run training with real environment rollouts
python training/scripts/train_qwen3_grpo_real_env.py \
    --config training/configs/training_config_qwen3_0.6b.yaml \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "================================================"
echo "Training completed!"
echo "Logs saved to: $OUTPUT_DIR/training.log"
echo "================================================"