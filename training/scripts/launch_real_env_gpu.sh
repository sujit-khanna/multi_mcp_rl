#!/bin/bash
# Launch script for REAL ENVIRONMENT GRPO Training on GPU (CUDA)
# ==============================================================
# This script is optimized for NVIDIA GPU training with maximum utilization

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the multi_mcp_rl root directory (two levels up from scripts)
ROOT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to root directory
cd "$ROOT_DIR"
echo "Working directory: $(pwd)"

# Activate virtual environment if it exists
if [ -d "/home/ubuntu/skyrl_env" ]; then
    source "/home/ubuntu/skyrl_env/bin/activate"
    echo "SkyRL virtual environment activated"
elif [ -d "$ROOT_DIR/venv312" ]; then
    source "$ROOT_DIR/venv312/bin/activate"
    echo "Python 3.12 virtual environment activated"
elif [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
    source "$ROOT_DIR/.venv/bin/activate"
    echo "Virtual environment activated"
fi

# Set environment variables for GPU optimization
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/.."
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

# CRITICAL FIXES: Environment variables for RL training improvements
export FORCE_RATE="0.0"                    # Disable forcing during RL updates
export ASSIST_WARMUP="0"                   # No warmup steps  
export RL_MODE="true"                      # Enable RL mode (critical fix)
export RL_DISABLE_FORCED="1"               # Disable forced actions in RL
export PPO_RECORD_AT_SAMPLE="1"            # Record log-probs at sampling time
export STRICT_TRAJ_KEYS="1"                # Strict trajectory key validation

# Force CUDA device (no MPS, no CPU fallback)
export DEVICE_TYPE="cuda"
export CUDA_VISIBLE_DEVICES=0  # Use first GPU, change if needed

# CUDA optimization settings
export CUDA_LAUNCH_BLOCKING=0  # Async execution for better performance
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"  # Support various GPU architectures
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # For deterministic operations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,backend:cudaMallocAsync"  # Memory optimization

# Enable TF32 for Ampere GPUs (3090, A100, etc.)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NVIDIA_TF32_OVERRIDE=1

# Mixed precision training settings
export MIXED_PRECISION="fp16"  # or "bf16" for newer GPUs
export ACCELERATE_MIXED_PRECISION="fp16"

export CUDA_MODULE_LOADING=LAZY  # Lazy loading for faster startup

# Disable debugging features for performance
export TORCH_USE_CUDA_DSA=1  # Device-side assertions

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

# WandB configuration - FIXED PROJECT
export WANDB_PROJECT="multi-mcp-rl-fixed"  # Fixed training system
export WANDB_MODE="online"  # Set to "offline" to disable wandb
export WANDB_TAGS="gpu,cuda,grpo,real-env,fixed,critical-fixes"
export WANDB_NAME="grpo-fixed-gpu-$(date +%Y%m%d-%H%M%S)"  # Descriptive run name
export WEAVE_PROJECT="synergia_Agents/multi-mcp-rl-fixed"  # Match WandB project

# Enable comprehensive logging
export WANDB_LOG_MODEL="true"  # Log model checkpoints
export WANDB_WATCH="all"  # Watch gradients and parameters

# Create output directory with timestamp
OUTPUT_DIR="outputs/real-env-grpo-gpu-$(date +%Y%m%d-%H%M%S)"
mkdir -p $OUTPUT_DIR

# Copy configuration files to output directory
cp training/configs/training_config_qwen3_0.6b.yaml $OUTPUT_DIR/ 2>/dev/null || echo "Warning: Could not copy training config"
cp training/configs/grpo_config_fixed.yaml $OUTPUT_DIR/ 2>/dev/null || echo "Warning: Could not copy grpo config"

# Check GPU availability
echo "================================================"
echo "GPU Environment Check"
echo "================================================"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader || echo "nvidia-smi not available"
echo ""
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}')
        print(f'  Memory: {props.total_memory / (1024**3):.1f} GB')
        print(f'  Compute Capability: {props.major}.{props.minor}')
        print(f'  Multi-processors: {props.multi_processor_count}')
"
echo "================================================"

echo ""
echo "================================================"
echo "Starting REAL ENVIRONMENT GRPO Training (GPU)"
echo "================================================"
echo "Output directory: $OUTPUT_DIR"
echo "Device: CUDA GPU"
echo "Mixed Precision: $MIXED_PRECISION"
echo "Python: $(which python)"
echo "================================================"

# Create a temporary config with GPU-specific settings
cat > $OUTPUT_DIR/gpu_training_config.yaml << EOF
# GPU-Optimized Training Configuration
# Auto-generated for GPU training

# Model configuration
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  use_lora: true
  value_head_hidden_dim: 1024
  load_in_4bit: true  # Enable 4-bit quantization for memory efficiency
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: true

# Data paths  
data_path: "data/inputs/train.json"
validation_data_path: "data/inputs/validation.json"
output_dir: "$OUTPUT_DIR"

# Training hyperparameters optimized for GPU (A100 40GB)
num_epochs: 3
batch_size: 16  # Maximized for A100 40GB
learning_rate: 5e-5
weight_decay: 0.01
warmup_steps: 100
gradient_accumulation_steps: 1  # No accumulation needed with large batch
max_grad_norm: 1.0

# GPU-specific settings
training_mode:
  per_device_train_batch_size: 16  # Maximize A100 GPU utilization
  gradient_checkpointing: true
  fp16: true  # Enable mixed precision
  fp16_opt_level: "O2"  # Aggressive mixed precision
  dataloader_num_workers: 8  # More workers for faster data loading
  dataloader_pin_memory: true
  prefetch_factor: 4  # Prefetch more batches
  
# Optimizer settings
optimizer:
  type: "adamw_torch"  # Use PyTorch AdamW
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1e-8

# Learning rate scheduler
lr_scheduler:
  type: "cosine"
  warmup_ratio: 0.1

# Evaluation settings
eval_steps: 50
eval_batch_size: 32  # Larger batch for faster evaluation on A100
save_steps: 100
save_total_limit: 3
load_best_model_at_end: true
metric_for_best_model: "eval_success_rate"
greater_is_better: true

# Early stopping
early_stopping:
  enabled: true
  patience: 5
  threshold: 0.001

# Logging configuration
logging:
  logging_steps: 10
  logging_first_step: true
  report_to: ["wandb", "weave", "tensorboard"]
  wandb_project: "skyrl-qwen3-gpu"
  weave_project: "synergia_Agents/skyrl-qwen3-gpu"
  
# Curriculum learning settings
curriculum_learning:
  enabled: true
  warmup_epochs: 1
  difficulty_schedule: "linear"
  min_complexity: "easy"
  max_complexity: "hard"
  
# Data loader settings
cache_size: 5000  # Larger cache for A100
num_workers: 8  # More workers for parallel data loading
shuffle: true
seed: 42
prefetch_factor: 4  # Prefetch more batches
persistent_workers: true
pin_memory: true  # Pin memory for faster GPU transfer

# Device settings
device_config:
  use_mps: false  # Disable MPS
  use_cuda: true  # Force CUDA
  device_map: "cuda:0"
  
# Memory optimization for GPU
memory_optimization:
  gradient_checkpointing: true
  clear_cache_steps: 50  # More frequent cache clearing
  max_memory_mb: null  # No limit, use all available GPU memory
  empty_cache_after_validation: true
  
# GPU-specific optimizations
gpu_optimization:
  use_amp: true  # Automatic Mixed Precision
  cudnn_benchmark: true  # Enable cuDNN autotuner
  cudnn_deterministic: false  # Trade reproducibility for speed
  matmul_precision: "high"  # Options: highest, high, medium
EOF

# Run training with GPU-optimized configuration
echo ""
echo "Starting training with GPU optimizations..."
echo "================================================"

# Set ulimit for better performance
ulimit -n 65536  # Increase file descriptor limit

# Launch training with GPU monitoring (using existing config with our critical fixes)
python training/scripts/train_qwen3_grpo_real_env.py \
    --config training/configs/training_config_qwen3_0.6b.yaml \
    --device cuda \
    --mixed-precision $MIXED_PRECISION \
    --enable-profiling \
    2>&1 | tee $OUTPUT_DIR/training.log &

# Store the PID
TRAINING_PID=$!
echo "Training PID: $TRAINING_PID"

# Monitor GPU usage in background
(
    while kill -0 $TRAINING_PID 2>/dev/null; do
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw --format=csv,noheader >> $OUTPUT_DIR/gpu_usage.csv
        sleep 10
    done
) &
MONITOR_PID=$!
echo "GPU Monitor PID: $MONITOR_PID"

# Wait for training to complete
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

# Stop GPU monitoring
kill $MONITOR_PID 2>/dev/null

echo ""
echo "================================================"
echo "Training completed with exit code: $TRAINING_EXIT_CODE"
echo "Logs saved to: $OUTPUT_DIR/training.log"
echo "GPU usage data: $OUTPUT_DIR/gpu_usage.csv"
echo "================================================"

# Generate GPU usage summary
if [ -f "$OUTPUT_DIR/gpu_usage.csv" ]; then
    echo ""
    echo "GPU Usage Summary:"
    python -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('$OUTPUT_DIR/gpu_usage.csv', header=None)
    df.columns = ['timestamp', 'gpu_name', 'gpu_util', 'mem_util', 'mem_used', 'mem_free', 'temperature', 'power']
    
    # Clean percentage values
    df['gpu_util'] = df['gpu_util'].str.rstrip(' %').astype(float)
    df['mem_util'] = df['mem_util'].str.rstrip(' %').astype(float)
    
    print(f'Average GPU Utilization: {df[\"gpu_util\"].mean():.1f}%')
    print(f'Peak GPU Utilization: {df[\"gpu_util\"].max():.1f}%')
    print(f'Average Memory Utilization: {df[\"mem_util\"].mean():.1f}%')
    print(f'Peak Memory Utilization: {df[\"mem_util\"].max():.1f}%')
    
    # Parse memory values
    mem_used = df['mem_used'].str.rstrip(' MiB').astype(float) / 1024
    print(f'Average Memory Used: {mem_used.mean():.1f} GB')
    print(f'Peak Memory Used: {mem_used.max():.1f} GB')
    
    # Temperature
    temp = df['temperature'].str.rstrip(' C').astype(float)
    print(f'Average GPU Temperature: {temp.mean():.1f}°C')
    print(f'Peak GPU Temperature: {temp.max():.1f}°C')
    
    # Power
    if not df['power'].str.contains('N/A').all():
        power = df['power'].str.rstrip(' W').replace('[Not Supported]', '0').astype(float)
        print(f'Average Power Draw: {power.mean():.1f} W')
        print(f'Peak Power Draw: {power.max():.1f} W')
except Exception as e:
    print(f'Could not generate summary: {e}', file=sys.stderr)
"
fi

exit $TRAINING_EXIT_CODE