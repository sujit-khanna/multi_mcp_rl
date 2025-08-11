#!/bin/bash

# Activate Python 3.12 virtual environment
source venv312/bin/activate

# Set up Python path for SkyRL
export PYTHONPATH="/home/ubuntu/multi_mcp_rl/SkyRL/skyrl-train:/home/ubuntu/multi_mcp_rl:/home/ubuntu/multi_mcp_rl/SkyRL/skyrl-gym:$PYTHONPATH"

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# GPU mode with CUDA 12.8
export DEVICE_TYPE="cuda"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.0"  # For A100

echo "Environment setup complete!"
echo "Python version: $(python --version)"
echo "PyTorch device: GPU mode (NVIDIA A100)"
echo "CUDA version: $(nvcc --version 2>/dev/null || echo 'CUDA 12.8')"
echo "PYTHONPATH configured for SkyRL"