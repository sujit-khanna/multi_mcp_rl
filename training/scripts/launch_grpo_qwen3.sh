#!/bin/bash
# Launch script for GRPO training with Qwen3-0.6B
# This runs the actual RL training with environments and rewards

set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "======================================"
echo "Qwen3-0.6B GRPO (RL) Training Launcher"
echo "======================================"
echo "Project root: $PROJECT_ROOT"
echo "Script dir: $SCRIPT_DIR"

# Check Python environment
echo -e "\n🐍 Checking Python environment..."
python --version

# Install required packages if missing
echo -e "\n📦 Checking required packages..."
pip install -q transformers torch peft datasets accelerate bitsandbytes tqdm

# Check for optional packages
echo -e "\n📊 Checking optional packages..."
pip install -q wandb weave || echo "Warning: WandB/Weave not installed, logging will be limited"

# Detect available hardware
echo -e "\n🖥️  Detecting hardware..."
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✅ CUDA GPU detected"
    export DEVICE_TYPE="cuda"
elif python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    echo "✅ Apple Silicon MPS detected"
    export DEVICE_TYPE="mps"
    # Clear any existing MPS env vars that might cause issues
    unset PYTORCH_MPS_HIGH_WATERMARK_RATIO
    unset PYTORCH_MPS_MEMORY_FRACTION
    unset PYTORCH_MPS_LOW_WATERMARK_RATIO
    export PYTORCH_ENABLE_MPS_FALLBACK=1
else
    echo "⚠️  No GPU detected, using CPU (will be slow)"
    export DEVICE_TYPE="cpu"
fi

# Set memory limits to prevent OOM
if [[ "$DEVICE_TYPE" == "cuda" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
elif [[ "$DEVICE_TYPE" == "mps" ]]; then
    # Ensure no MPS memory settings interfere
    unset PYTORCH_MPS_MEMORY_FRACTION
    unset PYTORCH_MPS_HIGH_WATERMARK_RATIO
    unset PYTORCH_MPS_LOW_WATERMARK_RATIO
fi

# Create output directory
OUTPUT_DIR="$PROJECT_ROOT/training/outputs/qwen3-0.6b-grpo-rl"
mkdir -p "$OUTPUT_DIR"

# Set logging
export WANDB_PROJECT="skyrl-qwen3-0.6b"
export WEAVE_PROJECT="synergia_Agents/skyrl-qwen3-0.6b"
export HF_DATASETS_OFFLINE=0

# Make sure environment paths are correct
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/environments:$PROJECT_ROOT/training:$PYTHONPATH"

echo -e "\n🚀 Starting GRPO (RL) training..."
echo "Output directory: $OUTPUT_DIR"
echo "Device type: $DEVICE_TYPE"
echo "Using actual RL with environments and rewards!"

# Change to project root for proper imports
cd "$PROJECT_ROOT"

# Run GRPO training with proper error handling
# Use LoRA mode for memory efficiency on MPS
if [[ "$DEVICE_TYPE" == "mps" ]]; then
    echo "Using LoRA mode for MPS device"
    MODE="lora"
else
    MODE="lora"  # Default to LoRA for testing
fi

python -u "$SCRIPT_DIR/train_grpo.py" \
    --config "$SCRIPT_DIR/../configs/training_config_qwen3_0.6b.yaml" \
    --mode "$MODE" \
    --num_gpus 1 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "\n✅ GRPO training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    
    # Show best model location if exists
    if [ -d "$OUTPUT_DIR/best_model" ]; then
        echo "Best model saved at: $OUTPUT_DIR/best_model"
    fi
else
    echo -e "\n❌ GRPO training failed. Check logs at: $OUTPUT_DIR/training.log"
    exit 1
fi