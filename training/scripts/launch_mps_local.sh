#!/bin/bash
# MPS (GPU) launch script for Apple Silicon

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "========================================"
echo "Qwen-0.5B MPS Training (Apple Silicon)"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Using MPS (Metal Performance Shaders) GPU"
echo ""

# Check if MPS is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"

if ! python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    echo "❌ MPS not available. Falling back to CPU..."
    exec "$SCRIPT_DIR/launch_cpu_local.sh"
    exit 1
fi

# Limit tokenizer parallelism
export TOKENIZERS_PARALLELISM=false

# Set paths
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/environments:$PROJECT_ROOT/training:$PYTHONPATH"

# Create output directory
OUTPUT_DIR="$PROJECT_ROOT/training/outputs/qwen-mps-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Set logging
export WANDB_PROJECT="skyrl-qwen-mps"
export WEAVE_PROJECT="synergia_Agents/skyrl-qwen-mps"
export HF_DATASETS_OFFLINE=0

# MPS optimization
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

echo "Output directory: $OUTPUT_DIR"
echo ""
echo "🚀 Starting GRPO training on MPS GPU..."

# Run with MPS-specific configs
cd "$PROJECT_ROOT"

python -u training/scripts/train_grpo.py \
    --config training/configs/training_config_mps.yaml \
    --mode lora \
    --num_gpus 1 \
    --device mps \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ MPS Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    
    # Show best model location if exists
    if [ -d "$OUTPUT_DIR/best_model" ]; then
        echo "Best model saved at: $OUTPUT_DIR/best_model"
    fi
    
    # Show performance comparison
    echo ""
    echo "📊 Performance Summary:"
    if [ -f "$OUTPUT_DIR/training.log" ]; then
        echo "Training time per episode:"
        grep "Batch collection completed" "$OUTPUT_DIR/training.log" | tail -3
    fi
else
    echo ""
    echo "❌ MPS Training failed. Check logs at: $OUTPUT_DIR/training.log"
    exit 1
fi