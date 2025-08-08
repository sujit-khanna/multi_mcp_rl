#!/bin/bash
# Launch script for Qwen3-0.6B training
# Supports both CUDA and MPS (macOS) backends

set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "======================================"
echo "Qwen3-0.6B GRPO Training Launcher"
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
OUTPUT_DIR="$PROJECT_ROOT/training/outputs/qwen3-0.6b-grpo"
mkdir -p "$OUTPUT_DIR"

# Set logging
export WANDB_PROJECT="skyrl-qwen3-0.6b"
export WEAVE_PROJECT="synergia_Agents/skyrl-qwen3-0.6b"
export HF_DATASETS_OFFLINE=0

echo -e "\n🚀 Starting training..."
echo "Output directory: $OUTPUT_DIR"
echo "Device type: $DEVICE_TYPE"

# Change to project root for proper imports
cd "$PROJECT_ROOT"

# Run training with proper error handling
python -u "$SCRIPT_DIR/train_qwen3_0.6b_fixed.py" \
    --config "configs/training_config_qwen3_0.6b.yaml" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "\n✅ Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    
    # Show best model location if exists
    if [ -d "$OUTPUT_DIR/best_model" ]; then
        echo "Best model saved at: $OUTPUT_DIR/best_model"
    fi
else
    echo -e "\n❌ Training failed. Check logs at: $OUTPUT_DIR/training.log"
    exit 1
fi