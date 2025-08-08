#!/bin/bash
# Launch script for Enhanced GRPO training with all Phase 1 fixes
# Includes: Value function, Reference policy updates, Gradient clipping fix

set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "======================================"
echo "Enhanced GRPO Training Launcher"
echo "======================================"
echo "Improvements included:"
echo "  ✓ Value function training (Fix 1.1)"
echo "  ✓ Reference policy updates (Fix 1.2)"
echo "  ✓ Gradient clipping for mixed precision (Fix 3.1)"
echo "======================================"
echo "Project root: $PROJECT_ROOT"
echo "Script dir: $SCRIPT_DIR"

# Check Python environment
echo -e "\n🐍 Checking Python environment..."
python --version

# Install required packages if missing
echo -e "\n📦 Checking required packages..."
pip install -q transformers torch peft datasets accelerate bitsandbytes tqdm pyyaml

# Check for optional packages
echo -e "\n📊 Checking optional packages..."
pip install -q wandb weave || echo "Warning: WandB/Weave not installed, logging will be limited"

# Detect available hardware
echo -e "\n🖥️  Detecting hardware..."
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✅ CUDA GPU detected"
    export DEVICE_TYPE="cuda"
    # Enable mixed precision for CUDA
    export USE_MIXED_PRECISION=1
elif python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    echo "✅ Apple Silicon MPS detected"
    export DEVICE_TYPE="mps"
    # Clear any existing MPS env vars that might cause issues (like the working script)
    unset PYTORCH_MPS_HIGH_WATERMARK_RATIO
    unset PYTORCH_MPS_MEMORY_FRACTION
    unset PYTORCH_MPS_LOW_WATERMARK_RATIO
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    # Disable mixed precision for MPS (not supported)
    export USE_MIXED_PRECISION=0
else
    echo "⚠️  No GPU detected, using CPU (will be slow)"
    export DEVICE_TYPE="cpu"
    export USE_MIXED_PRECISION=0
fi

# Set memory limits to prevent OOM
if [[ "$DEVICE_TYPE" == "cuda" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
elif [[ "$DEVICE_TYPE" == "mps" ]]; then
    # Ensure no MPS memory settings interfere (like the working script)
    unset PYTORCH_MPS_MEMORY_FRACTION
    unset PYTORCH_MPS_HIGH_WATERMARK_RATIO
    unset PYTORCH_MPS_LOW_WATERMARK_RATIO
fi

# Create output directory
OUTPUT_DIR="$PROJECT_ROOT/training/outputs/enhanced-grpo-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Copy configs to output directory for reproducibility
cp "$PROJECT_ROOT/training/configs/grpo_config_fixed.yaml" "$OUTPUT_DIR/"
cp "$PROJECT_ROOT/training/configs/training_config_qwen3_0.6b.yaml" "$OUTPUT_DIR/" 2>/dev/null || true

# Set logging
export WANDB_PROJECT="skyrl-grpo-enhanced"
export WEAVE_PROJECT="synergia_Agents/skyrl-grpo-enhanced"
export HF_DATASETS_OFFLINE=0

# Enhanced logging
export GRPO_LOG_LEVEL=INFO
export GRPO_LOG_GRADIENT_NORMS=1
export GRPO_LOG_REFERENCE_UPDATES=1

# Suppress tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

echo -e "\n🚀 Starting enhanced GRPO training..."
echo "Output directory: $OUTPUT_DIR"
echo "Device type: $DEVICE_TYPE"
echo "Mixed precision: $USE_MIXED_PRECISION"

# Change to project root for proper imports
cd "$PROJECT_ROOT"

# Run enhanced training with proper error handling
python -u "$SCRIPT_DIR/train_qwen3_grpo_enhanced.py" \
    --config "training/configs/training_config_qwen3_0.6b.yaml" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "\n✅ Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    
    # Show best model location if exists
    if [ -d "$OUTPUT_DIR/best_model" ]; then
        echo "Best model saved at: $OUTPUT_DIR/best_model"
        
        # Show final metrics
        if [ -f "$OUTPUT_DIR/best_model/metadata.json" ]; then
            echo -e "\nBest model metrics:"
            python -c "import json; print(json.dumps(json.load(open('$OUTPUT_DIR/best_model/metadata.json')), indent=2))"
        fi
    fi
    
    # Show gradient statistics if available
    echo -e "\n📊 Training statistics:"
    grep -E "(grad_norm|reference_policy_updated|value_loss)" "$OUTPUT_DIR/training.log" | tail -20 || true
    
else
    echo -e "\n❌ Training failed. Check logs at: $OUTPUT_DIR/training.log"
    
    # Show last few error lines
    echo -e "\nLast 20 lines of log:"
    tail -20 "$OUTPUT_DIR/training.log"
    
    exit 1
fi