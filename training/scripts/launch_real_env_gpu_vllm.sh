#!/bin/bash
# Launch script for REAL ENVIRONMENT GRPO Training on GPU with vLLM (CUDA)
# ========================================================================
# This is a copy of launch_real_env_gpu.sh with vLLM integration added
# Your original script remains unchanged and working

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

# Install vLLM if needed in the SkyRL environment
echo "🔍 Checking and installing vLLM if needed..."
if ! python -c "import vllm" 2>/dev/null; then
    echo "📦 Installing vLLM in SkyRL environment..."
    pip install --no-deps --force-reinstall vllm==0.10.0
    echo "✅ vLLM installation completed"
fi

if python -c "import vllm; print(f'vLLM {vllm.__version__} ready')" 2>/dev/null; then
    echo "✅ vLLM is available and ready"
    export ENABLE_VLLM="true"
else
    echo "❌ vLLM installation failed - falling back to HuggingFace"
    export ENABLE_VLLM="false"
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

# vLLM-specific settings for memory optimization
# Respect existing env vars, else use defaults
export ENABLE_VLLM="${ENABLE_VLLM:-true}"                           # Enable/disable vLLM
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.6}"                             # Default 0.6, override by exporting before calling
: "${VLLM_MAX_MODEL_LEN:=8192}"                                     # Default 8192, override by exporting before calling
export VLLM_GPU_MEMORY_UTILIZATION
export VLLM_MAX_MODEL_LEN
export VLLM_USE_TRITON_FLASH_ATTN="true"   # Enable FlashAttention
export VLLM_ATTENTION_BACKEND="FLASH_ATTN" # Use FlashAttention backend
export VLLM_WORKER_MULTIPROC_METHOD="spawn" # Multiprocessing method

# Force CUDA device (no MPS, no CPU fallback)
export DEVICE_TYPE="cuda"
export CUDA_VISIBLE_DEVICES=0  # Use first GPU, change if needed

# CUDA optimization settings
export CUDA_LAUNCH_BLOCKING=0  # Async execution for better performance
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"  # Support various GPU architectures
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # For deterministic operations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,backend:cudaMallocAsync"  # Optimized for vLLM sharing

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

# ================================================
# MCP Server Health Check and Restart
# ================================================
echo ""
echo "================================================"
echo "MCP Server Health Check"
echo "================================================"

# Test MCP servers and restart if needed
python "$ROOT_DIR/test_mcp_restart.py" || {
    echo "⚠️  MCP server check failed, but continuing with training..."
    echo "   Some tools may not be available during training"
}

echo "================================================"
echo ""

# WandB configuration - vLLM VERSION
export WANDB_PROJECT="multi-mcp-rl-vllm"  # Different project for vLLM runs
export WANDB_MODE="online"  # Set to "offline" to disable wandb
export WANDB_TAGS="gpu,cuda,grpo,real-env,vllm,fast-inference"
export WANDB_NAME="grpo-vllm-gpu-$(date +%Y%m%d-%H%M%S)"  # Descriptive run name
export WEAVE_PROJECT="synergia_Agents/multi-mcp-rl-vllm"  # Match WandB project

# Enable comprehensive logging
export WANDB_LOG_MODEL="true"  # Log model checkpoints
export WANDB_WATCH="all"  # Watch gradients and parameters

# Create output directory with timestamp
OUTPUT_DIR="outputs/real-env-grpo-vllm-$(date +%Y%m%d-%H%M%S)"
mkdir -p $OUTPUT_DIR

# Copy configuration files to output directory
cp training/configs/training_config_qwen3_0.6b.yaml $OUTPUT_DIR/ 2>/dev/null || echo "Warning: Could not copy training config"
cp training/configs/grpo_config_fixed.yaml $OUTPUT_DIR/ 2>/dev/null || echo "Warning: Could not copy grpo config"

# Check GPU availability
echo "================================================"
echo "GPU Environment Check (with vLLM)"
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

# Check vLLM
try:
    import vllm
    print(f'vLLM version: {vllm.__version__}')
    print('✅ vLLM integration ready for 10x faster inference!')
except ImportError:
    print('⚠️  vLLM not available - using HuggingFace (slower)')
"
echo "================================================"

echo ""
echo "================================================"
echo "Starting REAL ENVIRONMENT GRPO Training (GPU + vLLM)"
echo "================================================"
echo "Output directory: $OUTPUT_DIR"
echo "Device: CUDA GPU"
echo "Mixed Precision: $MIXED_PRECISION"
echo "vLLM Enabled: $ENABLE_VLLM"
echo "Python: $(which python)"
echo "================================================"

# Set ulimit for better performance
ulimit -n 65536  # Increase file descriptor limit

# Launch training with existing script but vLLM environment variables
echo ""
echo "Starting training with vLLM optimizations..."
echo "================================================"

# Verify dataset exists before starting
if [ ! -f "data/inputs/train.json" ] && [ ! -f "data/processed/train.json" ]; then
    echo "❌ Training dataset not found!"
    echo "   Expected: data/inputs/train.json or data/processed/train.json"
    exit 1
fi

echo "📊 Dataset check:"
if [ -f "data/inputs/train.json" ]; then
    echo "   Found: data/inputs/train.json"
    python -c "import json; data=json.load(open('data/inputs/train.json')); print(f'   Samples: {len(data)}')"
fi
if [ -f "data/processed/train.json" ]; then
    echo "   Found: data/processed/train.json" 
    python -c "import json; data=json.load(open('data/processed/train.json')); print(f'   Samples: {len(data)}')"
fi

echo ""
echo "🚀 Launching vLLM-enhanced training on your dataset..."
echo "Expected improvements:"
echo "   🚀 Generation speed: ~47s → ~4-5s (10x faster)"
echo "   📈 GPU utilization: 11% → 70-90%"
echo "   💾 Memory efficiency: OOM-prone → Stable"

# Use the vLLM-INTEGRATED training script for 10x speedup
python training/scripts/train_qwen3_grpo_real_env_vllm.py \
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

# Generate GPU usage summary (same as original)
if [ -f "$OUTPUT_DIR/gpu_usage.csv" ]; then
    echo ""
    echo "GPU Usage Summary:"
    python -c "
import pandas as pd
import sys

try:
    import pandas as pd
    df = pd.read_csv('$OUTPUT_DIR/gpu_usage.csv', header=None)
    # Handle potential extra columns gracefully
    cols = ['timestamp', 'gpu_name', 'gpu_util', 'mem_util', 'mem_used', 'mem_free', 'temperature', 'power']
    df = df.iloc[:, :len(cols)]
    df.columns = cols[:df.shape[1]]
    
    # Clean percentage values (cast to str first)
    if 'gpu_util' in df:
        df['gpu_util'] = df['gpu_util'].astype(str).str.rstrip(' %').astype(float)
    if 'mem_util' in df:
        df['mem_util'] = df['mem_util'].astype(str).str.rstrip(' %').astype(float)
    
    if 'gpu_util' in df:
        print(f'Average GPU Utilization: {df["gpu_util"].mean():.1f}%')
        print(f'Peak GPU Utilization: {df["gpu_util"].max():.1f}%')
    if 'mem_util' in df:
        print(f'Average Memory Utilization: {df["mem_util"].mean():.1f}%')
        print(f'Peak Memory Utilization: {df["mem_util"].max():.1f}%')
    
    # Parse memory values
    if 'mem_used' in df:
        mem_used = df['mem_used'].astype(str).str.rstrip(' MiB').astype(float) / 1024
        print(f'Average Memory Used: {mem_used.mean():.1f} GB')
        print(f'Peak Memory Used: {mem_used.max():.1f} GB')
    
    # Temperature
    if 'temperature' in df:
        temp = df['temperature'].astype(str).str.rstrip(' C').astype(float)
        print(f'Average GPU Temperature: {temp.mean():.1f}°C')
        print(f'Peak GPU Temperature: {temp.max():.1f}°C')
    
    # Power
    if 'power' in df:
        power_str = df['power'].astype(str)
        # If every value is N/A or Not Supported, skip stats
        if not power_str.str.contains('N/A').all():
            power = power_str.str.replace('[Not Supported]', '0', regex=False).str.rstrip(' W').astype(float)
            print(f'Average Power Draw: {power.mean():.1f} W')
            print(f'Peak Power Draw: {power.max():.1f} W')
        
    # vLLM specific metrics
    if df['gpu_util'].mean() > 70:
        print('\\n✅ Excellent GPU utilization! vLLM integration working well.')
    elif df['gpu_util'].mean() > 40:
        print('\\n✅ Good GPU utilization with vLLM acceleration.')
    else:
        print('\\n⚠️  Low GPU utilization - vLLM may not be active.')
        
except Exception as e:
    print(f'Could not generate summary: {e}', file=sys.stderr)
"
fi

exit $TRAINING_EXIT_CODE
