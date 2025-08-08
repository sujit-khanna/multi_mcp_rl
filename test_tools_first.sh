#!/bin/bash
# Quick test to verify tool execution before training

echo "================================================"
echo "Testing Real Tool Execution"
echo "================================================"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# This script is in the root directory
ROOT_DIR="$SCRIPT_DIR"

# Change to root directory
cd "$ROOT_DIR"
echo "Working directory: $(pwd)"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/.."
export PYTHONDONTWRITEBYTECODE=1

# Load environment variables from .env
if [ -f "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/.env" ]; then
    export $(cat /Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/.env | grep -v '^#' | xargs)
    echo "✅ Loaded API keys from .env"
else
    echo "❌ .env file not found!"
    exit 1
fi

# Run the test
python training/scripts/test_real_tool_execution.py

echo "================================================"
echo "If you see tool execution logs above, the system is working!"
echo "You can now run: ./training/scripts/launch_real_env_training.sh"
echo "================================================"