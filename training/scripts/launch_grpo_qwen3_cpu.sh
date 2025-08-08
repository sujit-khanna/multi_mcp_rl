#!/bin/bash
# Launch script for GRPO training with Qwen3-0.6B on CPU
# Fallback when MPS has memory issues

set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "======================================"
echo "Qwen3-0.6B GRPO (RL) Training - CPU Mode"
echo "======================================"
echo "Project root: $PROJECT_ROOT"
echo "Script dir: $SCRIPT_DIR"
echo ""
echo "⚠️  Running on CPU due to MPS memory limitations"
echo "⚠️  This will be slower but should complete successfully"
echo ""

# Force CPU mode
export PYTORCH_ENABLE_MPS_FALLBACK=0
export CUDA_VISIBLE_DEVICES=""

# Create output directory
OUTPUT_DIR="$PROJECT_ROOT/training/outputs/qwen3-0.6b-grpo-rl-cpu"
mkdir -p "$OUTPUT_DIR"

# Set logging
export WANDB_PROJECT="skyrl-qwen3-0.6b"
export WEAVE_PROJECT="synergia_Agents/skyrl-qwen3-0.6b"
export HF_DATASETS_OFFLINE=0

# Make sure environment paths are correct
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/environments:$PROJECT_ROOT/training:$PYTHONPATH"

echo -e "\n🚀 Starting GRPO (RL) training on CPU..."
echo "Output directory: $OUTPUT_DIR"

# Change to project root for proper imports
cd "$PROJECT_ROOT"

# Create a modified config that forces CPU
cat > "$SCRIPT_DIR/../configs/training_config_qwen3_0.6b_cpu.yaml" << EOF
# Training Configuration for Qwen3-0.6B - CPU Mode
# Based on training_config_qwen3_0.6b.yaml but with CPU-specific settings

# Data paths (relative to skyrl_tool_agent directory)
data_path: "data/inputs/train.json"
validation_data_path: "data/inputs/validation.json"
output_dir: "./outputs/qwen3-0.6b-grpo-cpu"

# Training hyperparameters (reduced for CPU)
num_epochs: 1  # Very reduced for CPU testing
batch_size: 1  
learning_rate: 5e-5  
weight_decay: 0.01
warmup_steps: 10  # Reduced
gradient_accumulation_steps: 2  # Reduced

# Force CPU device
device_config:
  use_mps: false
  use_cuda: false
  device_map: null

# Reduced settings for CPU
lora_mode:
  per_device_train_batch_size: 1
  gradient_checkpointing: false  # Disable for CPU
  fp16: false
  
full_finetune_mode:
  per_device_train_batch_size: 1
  gradient_checkpointing: false
  bf16: false
  deepspeed_config: null

# Evaluation settings (reduced)
eval_steps: 10
eval_batch_size: 1
save_steps: 20
save_total_limit: 2
load_best_model_at_end: false

# Disable early stopping for quick test
early_stopping:
  enabled: false
  
# Logging configuration
logging:
  logging_steps: 5
  logging_first_step: true
  report_to: ["wandb", "weave"]
  wandb_project: "skyrl-qwen3-0.6b"
  weave_project: "synergia_Agents/skyrl-qwen3-0.6b"
  
# Disable curriculum learning for simplicity
curriculum_learning:
  enabled: false
  
# Data loader settings
cache_size: 100
num_workers: 0  # Single thread for CPU
shuffle: true
seed: 42

# Memory optimization (CPU specific)
memory_optimization:
  gradient_checkpointing: false
  clear_cache_steps: 50
  max_memory_mb: 8000  # 8GB limit
EOF

# Run GRPO training on CPU
python -u "$SCRIPT_DIR/train_grpo.py" \
    --config "$SCRIPT_DIR/../configs/training_config_qwen3_0.6b_cpu.yaml" \
    --mode "lora" \
    --num_gpus 0 \
    --output_dir "$OUTPUT_DIR" \
    --device "cpu" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "\n✅ GRPO training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo -e "\n❌ GRPO training failed. Check logs at: $OUTPUT_DIR/training.log"
    exit 1
fi