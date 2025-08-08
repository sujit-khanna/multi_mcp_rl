#!/bin/bash
# GRPO Training - Full Fine-tuning Multi-GPU Launch Script
# ========================================================
# Supports: Multi-GPU CUDA, DeepSpeed ZeRO, Gradient Checkpointing
# Optimized for: Production training on A100/H100 clusters

set -euo pipefail

# Script metadata
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default configuration
DEFAULT_CONFIG="$PROJECT_ROOT/config/training_config.yaml"
DEFAULT_MODEL_CONFIG="$PROJECT_ROOT/config/model_config.yaml"
DEFAULT_DEEPSPEED_CONFIG="$PROJECT_ROOT/config/deepspeed_config.json"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/outputs/full_ft_training_$TIMESTAMP"
DEFAULT_DATA_PATH="$PROJECT_ROOT/data/processed/train.json"

# Parse command line arguments
USAGE="Usage: $0 [OPTIONS]
Options:
  -c, --config PATH           Training config file (default: $DEFAULT_CONFIG)
  -m, --model-config PATH     Model config file (default: $DEFAULT_MODEL_CONFIG)
  -ds, --deepspeed-config PATH DeepSpeed config file (default: $DEFAULT_DEEPSPEED_CONFIG)
  -o, --output-dir PATH       Output directory (default: $DEFAULT_OUTPUT_DIR)
  -d, --data PATH            Training data path (default: $DEFAULT_DATA_PATH)
  -n, --num-gpus NUM         Number of GPUs to use (default: auto-detect)
  -g, --gpu-ids IDS          Comma-separated GPU IDs (e.g., 0,1,2,3)
  -e, --env-name NAME        Conda environment name (optional)
  -r, --resume PATH          Resume from checkpoint (optional)
  --master-port PORT         Master port for distributed training (default: 29500)
  --zero-stage STAGE         DeepSpeed ZeRO stage (1,2,3) (default: 3)
  --gradient-checkpointing   Enable gradient checkpointing
  --offload-optimizer        Offload optimizer to CPU
  --offload-params           Offload parameters to CPU
  --monitor-memory           Enable memory monitoring
  --dry-run                  Show commands without executing
  --debug                    Enable debug logging
  -h, --help                 Show this help message

Examples:
  # Basic multi-GPU training (auto-detect GPUs)
  $0
  
  # Specify 4 GPUs with gradient checkpointing
  $0 --num-gpus 4 --gradient-checkpointing
  
  # Full memory optimization on 8 GPUs
  $0 --num-gpus 8 --zero-stage 3 --offload-optimizer --offload-params
  
  # Resume training with monitoring
  $0 --resume ./outputs/full_ft_training_20250802_120000/checkpoint-1000 --monitor-memory
"

# Initialize variables
CONFIG_FILE="$DEFAULT_CONFIG"
MODEL_CONFIG="$DEFAULT_MODEL_CONFIG"
DEEPSPEED_CONFIG="$DEFAULT_DEEPSPEED_CONFIG"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
DATA_PATH="$DEFAULT_DATA_PATH"
NUM_GPUS=""
GPU_IDS=""
ENV_NAME=""
RESUME_PATH=""
MASTER_PORT="29500"
ZERO_STAGE="3"
GRADIENT_CHECKPOINTING=false
OFFLOAD_OPTIMIZER=false
OFFLOAD_PARAMS=false
MONITOR_MEMORY=false
DRY_RUN=false
DEBUG=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -m|--model-config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        -ds|--deepspeed-config)
            DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -d|--data)
            DATA_PATH="$2"
            shift 2
            ;;
        -n|--num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -g|--gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        -e|--env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME_PATH="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --zero-stage)
            ZERO_STAGE="$2"
            shift 2
            ;;
        --gradient-checkpointing)
            GRADIENT_CHECKPOINTING=true
            shift
            ;;
        --offload-optimizer)
            OFFLOAD_OPTIMIZER=true
            shift
            ;;
        --offload-params)
            OFFLOAD_PARAMS=true
            shift
            ;;
        --monitor-memory)
            MONITOR_MEMORY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        -h|--help)
            echo "$USAGE"
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "$USAGE"
            exit 1
            ;;
    esac
done

# Logging setup
LOG_DIR="$OUTPUT_DIR/logs"
LOG_FILE="$LOG_DIR/training_$TIMESTAMP.log"
MONITOR_LOG="$LOG_DIR/monitor_$TIMESTAMP.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

log_debug() {
    if [ "$DEBUG" = true ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: $*" | tee -a "$LOG_FILE"
    fi
}

# Create output directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

log "🚀 GRPO Full Fine-tuning Multi-GPU Launch"
log "=========================================="
log "Project Root: $PROJECT_ROOT"
log "Output Directory: $OUTPUT_DIR"
log "Timestamp: $TIMESTAMP"

# Validate files exist
validate_file() {
    local file_path="$1"
    local description="$2"
    
    if [ ! -f "$file_path" ]; then
        log_error "$description not found: $file_path"
        exit 1
    fi
    log "✓ Found $description: $file_path"
}

log "📁 Validating input files..."
validate_file "$CONFIG_FILE" "Training config"
validate_file "$MODEL_CONFIG" "Model config"
validate_file "$DATA_PATH" "Training data"

# GPU detection and validation
detect_gpus() {
    log "🔍 Detecting GPUs..."
    
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_error "CUDA not found. This script requires NVIDIA GPUs."
        exit 1
    fi
    
    local available_gpus=$(nvidia-smi --list-gpus | wc -l)
    log "✓ Found $available_gpus GPU(s)"
    
    # Display GPU information
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu --format=csv,noheader | while IFS=',' read idx name memory_total memory_free temp; do
        log "  GPU $idx: $name, ${memory_total// /} total, ${memory_free// /} free, ${temp// /}°C"
    done
    
    echo "$available_gpus"
}

AVAILABLE_GPUS=$(detect_gpus)

# Set up GPU configuration
setup_gpu_config() {
    if [ -n "$GPU_IDS" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        local gpu_count=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
        NUM_GPUS="$gpu_count"
        log "🎯 Using specified GPUs: $GPU_IDS ($NUM_GPUS GPUs)"
    elif [ -n "$NUM_GPUS" ]; then
        if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
            log_error "Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
            exit 1
        fi
        local gpu_list=$(seq -s, 0 $((NUM_GPUS - 1)))
        export CUDA_VISIBLE_DEVICES="$gpu_list"
        log "🎯 Using first $NUM_GPUS GPUs: $gpu_list"
    else
        NUM_GPUS="$AVAILABLE_GPUS"
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
        log "🎯 Using all $NUM_GPUS GPUs"
    fi
    
    # Validate minimum requirements
    if [ "$NUM_GPUS" -lt 2 ]; then
        log_error "Multi-GPU training requires at least 2 GPUs. Found: $NUM_GPUS"
        log_error "Use launch_lora_single_gpu.sh for single GPU training."
        exit 1
    fi
}

setup_gpu_config

# Set up distributed training environment
setup_distributed_environment() {
    log "⚙️  Setting up distributed training environment..."
    
    export MASTER_ADDR="127.0.0.1"
    export MASTER_PORT="$MASTER_PORT"
    export WORLD_SIZE="$NUM_GPUS"
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=1  # Disable InfiniBand for local training
    export NCCL_P2P_DISABLE=1  # Disable P2P for stability
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
    
    # Memory optimization settings
    export CUDA_LAUNCH_BLOCKING=0
    export TORCH_NCCL_BLOCKING_WAIT=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    
    log "  Master: $MASTER_ADDR:$MASTER_PORT"
    log "  World Size: $WORLD_SIZE"
    log "  NCCL Debug: $NCCL_DEBUG"
}

setup_distributed_environment

# Activate conda environment if specified
if [ -n "$ENV_NAME" ]; then
    log "🐍 Activating conda environment: $ENV_NAME"
    
    # Initialize conda
    if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        log_error "Conda not found. Please install conda or specify correct path."
        exit 1
    fi
    
    conda activate "$ENV_NAME" || {
        log_error "Failed to activate conda environment: $ENV_NAME"
        exit 1
    }
    log "✓ Activated environment: $(conda info --envs | grep '*' | awk '{print $1}')"
fi

# Check dependencies
log "🔍 Checking dependencies..."
python3 -c "
import sys, torch, transformers, deepspeed
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'DeepSpeed: {deepspeed.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'NCCL: Available' if torch.distributed.is_nccl_available() else 'NCCL: Not available')
" | while read line; do log "  $line"; done

# Create or modify DeepSpeed config
create_deepspeed_config() {
    local ds_config="$LOG_DIR/deepspeed_config_generated.json"
    
    log "📝 Creating DeepSpeed configuration..."
    
    cat > "$ds_config" << EOF
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    
    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true,
        "cpu_offload": $OFFLOAD_OPTIMIZER
    },
    
    "activation_checkpointing": {
        "partition_activations": $([ "$GRADIENT_CHECKPOINTING" = true ] && echo "true" || echo "false"),
        "cpu_checkpointing": $([ "$OFFLOAD_PARAMS" = true ] && echo "true" || echo "false"),
        "contiguous_memory_optimization": false,
        "number_checkpoints": 4,
        "synchronize_checkpoint_boundary": true,
        "profile": false
    },
    
    "gradient_clipping": "auto",
    "steps_per_print": 10,
    "wall_clock_breakdown": false,
    "dump_state": false
}
EOF
    
    log "✓ Generated DeepSpeed config: $ds_config"
    echo "$ds_config"
}

FINAL_DEEPSPEED_CONFIG=$(create_deepspeed_config)

# Memory monitoring setup
setup_memory_monitoring() {
    if [ "$MONITOR_MEMORY" = true ]; then
        log "📊 Setting up memory monitoring..."
        
        local monitor_script="$LOG_DIR/memory_monitor.sh"
        cat > "$monitor_script" << 'EOF'
#!/bin/bash
LOG_FILE="$1"

echo "Starting GPU memory monitoring..." >> "$LOG_FILE"
while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read idx used total util; do
        echo "[$timestamp] GPU$idx: ${used}MB/${total}MB (${util}%)" >> "$LOG_FILE"
    done
    sleep 30
done
EOF
        
        chmod +x "$monitor_script"
        
        # Start monitoring in background
        "$monitor_script" "$MONITOR_LOG" &
        MONITOR_PID=$!
        log "✓ Memory monitoring started (PID: $MONITOR_PID)"
        log "  Monitor log: $MONITOR_LOG"
    fi
}

setup_memory_monitoring

# Prepare training arguments
prepare_training_args() {
    local args=(
        "--config" "$CONFIG_FILE"
        "--model-config" "$MODEL_CONFIG"
        "--output-dir" "$OUTPUT_DIR"
        "--data-path" "$DATA_PATH"
        "--training-mode" "full"
        "--device-type" "cuda"
        "--logging-dir" "$LOG_DIR"
        "--deepspeed" "$FINAL_DEEPSPEED_CONFIG"
        "--bf16"  # Use BF16 for better numerical stability
    )
    
    # Add resume path if specified
    if [ -n "$RESUME_PATH" ]; then
        args+=("--resume-from-checkpoint" "$RESUME_PATH")
        log "📂 Resuming from checkpoint: $RESUME_PATH"
    fi
    
    # Add gradient checkpointing if enabled
    if [ "$GRADIENT_CHECKPOINTING" = true ]; then
        args+=("--gradient-checkpointing")
        log "💾 Gradient checkpointing enabled"
    fi
    
    # Add debug flag if enabled
    if [ "$DEBUG" = true ]; then
        args+=("--debug")
    fi
    
    echo "${args[@]}"
}

TRAINING_ARGS=($(prepare_training_args))

# Display training configuration
log "📋 Training Configuration:"
log "  Mode: Full Fine-tuning"
log "  GPUs: $NUM_GPUS"
log "  DeepSpeed ZeRO Stage: $ZERO_STAGE"
log "  Gradient Checkpointing: $GRADIENT_CHECKPOINTING"
log "  Optimizer Offload: $OFFLOAD_OPTIMIZER"
log "  Parameter Offload: $OFFLOAD_PARAMS"
log "  Memory Monitoring: $MONITOR_MEMORY"
log "  Master Port: $MASTER_PORT"
log "  Config: $CONFIG_FILE"
log "  Output: $OUTPUT_DIR"

# Pre-training GPU check
check_gpu_memory() {
    log "🔍 Pre-training GPU memory check..."
    
    local min_memory_gb=40  # Minimum memory for full fine-tuning
    local warnings=0
    
    nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits | while IFS=',' read idx total free; do
        local total_gb=$((total / 1024))
        local free_gb=$((free / 1024))
        
        log "  GPU $idx: ${total_gb}GB total, ${free_gb}GB free"
        
        if [ "$total_gb" -lt "$min_memory_gb" ]; then
            log "  ⚠️  GPU $idx has only ${total_gb}GB memory (recommended: ${min_memory_gb}GB+)"
            warnings=$((warnings + 1))
        fi
        
        if [ "$free_gb" -lt 30 ]; then
            log "  ⚠️  GPU $idx has only ${free_gb}GB free memory"
            warnings=$((warnings + 1))
        fi
    done
    
    if [ "$warnings" -gt 0 ]; then
        log "⚠️  Memory warnings detected. Consider:"
        log "  - Using DeepSpeed ZeRO Stage 3 (--zero-stage 3)"
        log "  - Enabling optimizer offload (--offload-optimizer)"
        log "  - Reducing batch size in config"
        log "  - Using LoRA training instead"
    fi
}

check_gpu_memory

# Execute training
execute_training() {
    log "🎯 Starting GRPO full fine-tuning..."
    
    local cmd="python3 -m torch.distributed.launch"
    cmd+=" --nproc_per_node=$NUM_GPUS"
    cmd+=" --master_port=$MASTER_PORT"
    cmd+=" train_grpo.py"
    cmd+=" ${TRAINING_ARGS[*]}"
    
    log "Command: $cmd"
    
    if [ "$DRY_RUN" = true ]; then
        log "🔍 DRY RUN - Command would be executed:"
        echo "cd '$PROJECT_ROOT/training'"
        echo "$cmd"
        return 0
    fi
    
    # Change to training directory
    cd "$PROJECT_ROOT/training"
    
    # Execute training with proper error handling
    set +e  # Don't exit on error for training script
    eval "$cmd" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=$?
    set -e
    
    if [ $exit_code -eq 0 ]; then
        log "✅ Training completed successfully!"
        log "📁 Output directory: $OUTPUT_DIR"
        log "📄 Training log: $LOG_FILE"
        if [ "$MONITOR_MEMORY" = true ]; then
            log "📄 Memory log: $MONITOR_LOG"
        fi
        
        # Show final model info
        if [ -d "$OUTPUT_DIR" ]; then
            log "📊 Final outputs:"
            find "$OUTPUT_DIR" -name "*.bin" -o -name "*.safetensors" -o -name "pytorch_model*" | head -5 | while read file; do
                log "  - $(basename "$file")"
            done
        fi
    else
        log_error "Training failed with exit code: $exit_code"
        log_error "Check log file for details: $LOG_FILE"
        exit $exit_code
    fi
}

# Cleanup function
cleanup() {
    log "🧹 Cleaning up..."
    
    # Stop memory monitoring
    if [ -n "${MONITOR_PID:-}" ]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        log "✓ Stopped memory monitoring"
    fi
    
    # Kill any remaining background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Clear GPU memory
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    log "👋 Cleanup completed"
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    log "🎬 Executing multi-GPU training..."
    execute_training
    
    log "🎉 Multi-GPU training session completed!"
    log "📁 All outputs saved to: $OUTPUT_DIR"
    log "📊 Logs available in: $LOG_DIR"
}

# Run main function
main "$@"