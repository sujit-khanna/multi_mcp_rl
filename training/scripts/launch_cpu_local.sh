#!/bin/bash
# CPU-only launch script for testing

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "========================================"
echo "Qwen-0.5B CPU Training (Testing)"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Using CPU for all operations"
echo ""

# Limit tokenizer parallelism
export TOKENIZERS_PARALLELISM=false

# Set paths
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/environments:$PROJECT_ROOT/training:$PYTHONPATH"

# Create output directory
OUTPUT_DIR="$PROJECT_ROOT/training/outputs/qwen-cpu-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Set logging
export WANDB_PROJECT="skyrl-qwen-cpu"
export WEAVE_PROJECT="synergia_Agents/skyrl-qwen-cpu"
export HF_DATASETS_OFFLINE=0

echo "Output directory: $OUTPUT_DIR"
echo ""
echo "🚀 Starting GRPO training on CPU..."

# Run with CPU-specific configs
cd "$PROJECT_ROOT"

python -u training/scripts/train_grpo.py \
    --config training/configs/training_config_mps.yaml \
    --mode lora \
    --num_gpus 0 \
    --device cpu \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    
    # Show best model location if exists
    if [ -d "$OUTPUT_DIR/best_model" ]; then
        echo "Best model saved at: $OUTPUT_DIR/best_model"
    fi
else
    echo ""
    echo "❌ Training failed. Check logs at: $OUTPUT_DIR/training.log"
    exit 1
fi