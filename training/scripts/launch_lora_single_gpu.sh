#!/bin/bash
# GRPO Training - LoRA Single GPU Launch Script
# ==================================================
# Supports: CUDA, Apple Silicon MPS, CPU fallback
# Optimized for: Development and LoRA fine-tuning

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script metadata
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default configuration
DEFAULT_CONFIG="$PROJECT_ROOT/config/training_config.yaml"
DEFAULT_MODEL_CONFIG="$PROJECT_ROOT/config/model_config.yaml"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/outputs/lora_training_$TIMESTAMP"
DEFAULT_DATA_PATH="$PROJECT_ROOT/data/processed/train.json"

# Parse command line arguments
USAGE="Usage: $0 [OPTIONS]
Options:
  -c, --config PATH         Training config file (default: $DEFAULT_CONFIG)
  -m, --model-config PATH   Model config file (default: $DEFAULT_MODEL_CONFIG)
  -o, --output-dir PATH     Output directory (default: $DEFAULT_OUTPUT_DIR)
  -d, --data PATH          Training data path (default: $DEFAULT_DATA_PATH)
  -g, --gpu-id ID          GPU ID to use (default: auto-detect)
  -e, --env-name NAME      Conda environment name (optional)
  -r, --resume PATH        Resume from checkpoint (optional)
  --dry-run                Show commands without executing
  --debug                  Enable debug logging
  -h, --help               Show this help message

Examples:
  # Basic LoRA training
  $0
  
  # With custom config
  $0 --config custom_config.yaml --output-dir ./my_outputs
  
  # Resume training
  $0 --resume ./outputs/lora_training_20250802_120000/checkpoint-1000
  
  # Debug mode
  $0 --debug --dry-run
"

# Initialize variables
CONFIG_FILE="$DEFAULT_CONFIG"
MODEL_CONFIG="$DEFAULT_MODEL_CONFIG"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
DATA_PATH="$DEFAULT_DATA_PATH"
GPU_ID=""
ENV_NAME=""
RESUME_PATH=""
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
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -d|--data)
            DATA_PATH="$2"
            shift 2
            ;;
        -g|--gpu-id)
            GPU_ID="$2"
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

log "🚀 GRPO LoRA Training Launch"
log "=============================="
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

# Detect and setup device
detect_device() {
    log "🔍 Detecting compute device..."
    
    # Check for MLX (Apple Silicon optimized)
    if command -v python3 >/dev/null 2>&1 && python3 -c "import mlx.core" >/dev/null 2>&1; then
        log "✓ MLX detected (Apple Silicon optimized)"
        echo "mlx"
        return
    fi
    
    # Check for CUDA
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_count=$(nvidia-smi --list-gpus | wc -l)
        log "✓ CUDA detected with $gpu_count GPU(s)"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | while read line; do
            log "  GPU: $line"
        done
        echo "cuda"
        return
    fi
    
    # Check for MPS (Apple Silicon)
    if python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
        log "✓ Apple Silicon MPS detected"
        echo "mps"
        return
    fi
    
    # Fallback to CPU
    log "⚠️  No GPU detected, using CPU"
    echo "cpu"
}

DEVICE_TYPE=$(detect_device)

# Set device-specific environment variables
setup_device_environment() {
    log "⚙️  Setting up device environment for $DEVICE_TYPE..."
    
    case $DEVICE_TYPE in
        "mlx")
            export MLX_METAL_DEBUG=1
            export PYTORCH_ENABLE_MPS_FALLBACK=1
            log "  MLX environment configured"
            ;;
        "cuda")
            if [ -n "$GPU_ID" ]; then
                export CUDA_VISIBLE_DEVICES="$GPU_ID"
                log "  Using CUDA GPU: $GPU_ID"
            else
                export CUDA_VISIBLE_DEVICES="0"
                log "  Using CUDA GPU: 0 (default)"
            fi
            export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
            export NCCL_DEBUG=INFO
            ;;
        "mps")
            export PYTORCH_ENABLE_MPS_FALLBACK=1
            export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
            log "  MPS environment configured"
            ;;
        "cpu")
            export OMP_NUM_THREADS=4
            export MKL_NUM_THREADS=4
            log "  CPU environment configured"
            ;;
    esac
}

setup_device_environment

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

# Check Python environment
log "🐍 Checking Python environment..."
python3 -c "
import sys
import torch
import transformers
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
if torch.cuda.is_available():
    print(f'CUDA: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS: Available')
else:
    print('Device: CPU')
" | while read line; do log "  $line"; done

# Prepare training arguments
prepare_training_args() {
    local args=(
        "--config" "$CONFIG_FILE"
        "--model-config" "$MODEL_CONFIG"
        "--output-dir" "$OUTPUT_DIR"
        "--data-path" "$DATA_PATH"
        "--training-mode" "lora"
        "--device-type" "$DEVICE_TYPE"
        "--logging-dir" "$LOG_DIR"
    )
    
    # Add resume path if specified
    if [ -n "$RESUME_PATH" ]; then
        args+=("--resume-from-checkpoint" "$RESUME_PATH")
        log "📂 Resuming from checkpoint: $RESUME_PATH"
    fi
    
    # Add debug flag if enabled
    if [ "$DEBUG" = true ]; then
        args+=("--debug")
    fi
    
    # Device-specific arguments
    case $DEVICE_TYPE in
        "cuda")
            args+=("--fp16")
            if [ -n "$GPU_ID" ]; then
                args+=("--gpu-id" "$GPU_ID")
            fi
            ;;
        "mps"|"mlx")
            # Use FP16 on Apple Silicon for better performance
            args+=("--fp16")
            ;;
        "cpu")
            # Use FP32 on CPU
            args+=("--fp32")
            ;;
    esac
    
    echo "${args[@]}"
}

TRAINING_ARGS=($(prepare_training_args))

# Display training configuration
log "📋 Training Configuration:"
log "  Mode: LoRA Fine-tuning"
log "  Device: $DEVICE_TYPE"
log "  Config: $CONFIG_FILE"
log "  Model Config: $MODEL_CONFIG"
log "  Data: $DATA_PATH"
log "  Output: $OUTPUT_DIR"
if [ -n "$RESUME_PATH" ]; then
    log "  Resume: $RESUME_PATH"
fi

# Create monitoring script
create_monitoring_script() {
    local monitor_script="$LOG_DIR/monitor_training.sh"
    cat > "$monitor_script" << 'EOF'
#!/bin/bash
# Training monitoring script

LOG_FILE="$1"
OUTPUT_DIR="$2"

echo "🔍 Monitoring training progress..."
echo "Log file: $LOG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Press Ctrl+C to stop monitoring"
echo "=========================="

# Monitor log file
if [ -f "$LOG_FILE" ]; then
    tail -f "$LOG_FILE" | while read line; do
        echo "[$(date '+%H:%M:%S')] $line"
    done
else
    echo "Waiting for log file to be created..."
    while [ ! -f "$LOG_FILE" ]; do
        sleep 1
    done
    tail -f "$LOG_FILE"
fi
EOF
    
    chmod +x "$monitor_script"
    log "📊 Created monitoring script: $monitor_script"
    log "   Run: $monitor_script \"$LOG_FILE\" \"$OUTPUT_DIR\""
}

create_monitoring_script

# Resource monitoring
monitor_resources() {
    if [ "$DEVICE_TYPE" = "cuda" ]; then
        log "📊 GPU Memory Usage:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
            log "  $line"
        done
    elif [ "$DEVICE_TYPE" = "mps" ]; then
        log "📊 System Memory Usage:"
        vm_stat | grep -E "Pages (free|active|inactive|speculative|wired down)" | while read line; do
            log "  $line"
        done
    fi
}

monitor_resources

# Execute training
execute_training() {
    log "🎯 Starting GRPO LoRA training..."
    log "Command: python3 train_grpo.py ${TRAINING_ARGS[*]}"
    
    if [ "$DRY_RUN" = true ]; then
        log "🔍 DRY RUN - Command would be executed:"
        echo "cd '$PROJECT_ROOT/training'"
        echo "python3 train_grpo.py ${TRAINING_ARGS[*]}"
        return 0
    fi
    
    # Change to training directory
    cd "$PROJECT_ROOT/training"
    
    # Execute training with proper error handling
    set +e  # Don't exit on error for training script
    python3 train_grpo.py "${TRAINING_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=$?
    set -e
    
    if [ $exit_code -eq 0 ]; then
        log "✅ Training completed successfully!"
        log "📁 Output directory: $OUTPUT_DIR"
        log "📄 Log file: $LOG_FILE"
        
        # Show final model info
        if [ -d "$OUTPUT_DIR" ]; then
            log "📊 Final outputs:"
            find "$OUTPUT_DIR" -name "*.bin" -o -name "*.safetensors" -o -name "adapter_*" | head -5 | while read file; do
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
    
    # Kill any remaining background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Clear GPU memory if CUDA
    if [ "$DEVICE_TYPE" = "cuda" ]; then
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    fi
    
    log "👋 Cleanup completed"
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    log "🎬 Executing training..."
    execute_training
    
    log "🎉 LoRA training session completed!"
    log "📁 All outputs saved to: $OUTPUT_DIR"
    log "📊 Monitor training: $LOG_DIR/monitor_training.sh"
}

# Run main function
main "$@"