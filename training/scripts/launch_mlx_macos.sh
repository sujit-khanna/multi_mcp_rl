#!/bin/bash
# GRPO Training - macOS MLX Optimized Launch Script
# ================================================
# Supports: Apple Silicon with MLX optimization, unified memory management
# Optimized for: macOS development with Metal Performance Shaders

set -euo pipefail

# Script metadata
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default configuration
DEFAULT_CONFIG="$PROJECT_ROOT/config/training_config.yaml"
DEFAULT_MODEL_CONFIG="$PROJECT_ROOT/config/model_config.yaml"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/outputs/mlx_training_$TIMESTAMP"
DEFAULT_DATA_PATH="$PROJECT_ROOT/data/processed/train.json"

# Parse command line arguments
USAGE="Usage: $0 [OPTIONS]
macOS MLX-Optimized Training Options:
  -c, --config PATH         Training config file (default: $DEFAULT_CONFIG)
  -m, --model-config PATH   Model config file (default: $DEFAULT_MODEL_CONFIG)
  -o, --output-dir PATH     Output directory (default: $DEFAULT_OUTPUT_DIR)
  -d, --data PATH          Training data path (default: $DEFAULT_DATA_PATH)
  
  # MLX Configuration
  --mlx-memory-limit GB     Memory limit for MLX (default: auto)
  --mlx-precision PREC      Precision: float16, bfloat16, float32 (default: float16)
  --unified-memory-opt      Enable unified memory optimization
  --metal-debug            Enable Metal debugging
  
  # Training Configuration
  --training-mode MODE      Training mode: lora, qlora, full (default: lora)
  --lora-rank RANK         LoRA rank (default: 8)
  --lora-alpha ALPHA       LoRA alpha (default: 32)
  --batch-size SIZE        Batch size (default: auto)
  --gradient-accumulation N Gradient accumulation steps (default: 4)
  
  # Environment
  -e, --env-name NAME      Conda environment name (optional)
  --use-mps-fallback       Enable MPS fallback for unsupported ops
  
  # Monitoring
  --monitor-memory         Enable memory monitoring
  --profile-performance    Enable performance profiling
  --debug                  Enable debug logging
  --dry-run               Show commands without executing
  -r, --resume PATH        Resume from checkpoint (optional)
  -h, --help              Show this help message

Examples:
  # Basic MLX LoRA training
  $0
  
  # QLoRA with memory optimization
  $0 --training-mode qlora --unified-memory-opt --monitor-memory
  
  # Full precision with performance profiling
  $0 --mlx-precision float32 --profile-performance
  
  # Custom LoRA configuration
  $0 --lora-rank 16 --lora-alpha 64 --batch-size 8
"

# Initialize variables
CONFIG_FILE="$DEFAULT_CONFIG"
MODEL_CONFIG="$DEFAULT_MODEL_CONFIG"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
DATA_PATH="$DEFAULT_DATA_PATH"

# MLX configuration
MLX_MEMORY_LIMIT=""
MLX_PRECISION="float16"
UNIFIED_MEMORY_OPT=false
METAL_DEBUG=false

# Training configuration
TRAINING_MODE="lora"
LORA_RANK="8"
LORA_ALPHA="32"
BATCH_SIZE=""
GRADIENT_ACCUMULATION="4"

# Environment
ENV_NAME=""
USE_MPS_FALLBACK=false

# Monitoring
MONITOR_MEMORY=false
PROFILE_PERFORMANCE=false
DEBUG=false
DRY_RUN=false
RESUME_PATH=""

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
        --mlx-memory-limit)
            MLX_MEMORY_LIMIT="$2"
            shift 2
            ;;
        --mlx-precision)
            MLX_PRECISION="$2"
            shift 2
            ;;
        --unified-memory-opt)
            UNIFIED_MEMORY_OPT=true
            shift
            ;;
        --metal-debug)
            METAL_DEBUG=true
            shift
            ;;
        --training-mode)
            TRAINING_MODE="$2"
            shift 2
            ;;
        --lora-rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --lora-alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient-accumulation)
            GRADIENT_ACCUMULATION="$2"
            shift 2
            ;;
        -e|--env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --use-mps-fallback)
            USE_MPS_FALLBACK=true
            shift
            ;;
        --monitor-memory)
            MONITOR_MEMORY=true
            shift
            ;;
        --profile-performance)
            PROFILE_PERFORMANCE=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -r|--resume)
            RESUME_PATH="$2"
            shift 2
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
LOG_FILE="$LOG_DIR/mlx_training_$TIMESTAMP.log"

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

log "🍎 GRPO MLX Training Launch (macOS Optimized)"
log "=============================================="
log "Project Root: $PROJECT_ROOT"
log "Output Directory: $OUTPUT_DIR"
log "Timestamp: $TIMESTAMP"

# Validate macOS and Apple Silicon
validate_macos_environment() {
    log "🔍 Validating macOS environment..."
    
    # Check if running on macOS
    if [[ "$(uname)" != "Darwin" ]]; then
        log_error "This script is designed for macOS only"
        exit 1
    fi
    
    # Check for Apple Silicon
    local arch=$(uname -m)
    if [[ "$arch" != "arm64" ]]; then
        log "⚠️  Warning: Not running on Apple Silicon (detected: $arch)"
        log "   MLX optimizations may not be available"
    else
        log "✓ Apple Silicon detected: $arch"
    fi
    
    # Check macOS version
    local macos_version=$(sw_vers -productVersion)
    log "✓ macOS version: $macos_version"
    
    # Check available memory
    local total_memory_gb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
    log "✓ Total system memory: ${total_memory_gb}GB unified memory"
    
    # Set memory limit if not specified
    if [ -z "$MLX_MEMORY_LIMIT" ]; then
        # Use 70% of available memory for MLX
        MLX_MEMORY_LIMIT=$((total_memory_gb * 70 / 100))
        log "✓ Auto-set MLX memory limit: ${MLX_MEMORY_LIMIT}GB"
    fi
}

validate_macos_environment

# Check MLX availability
check_mlx_availability() {
    log "🔍 Checking MLX availability..."
    
    # Check if MLX is installed
    if ! python3 -c "import mlx.core" >/dev/null 2>&1; then
        log "⚠️  MLX not found, attempting installation..."
        
        # Try to install MLX
        if pip install mlx >/dev/null 2>&1; then
            log "✓ MLX installed successfully"
        else
            log_error "Failed to install MLX. Please install manually:"
            log_error "  pip install mlx"
            exit 1
        fi
    else
        log "✓ MLX is available"
    fi
    
    # Check MLX version and capabilities
    python3 -c "
import mlx.core as mx
import mlx.nn as nn
print(f'MLX version: {mx.__version__}')
print(f'Metal device: {mx.default_device()}')
print(f'Available memory: {mx.metal.get_peak_memory() / 1024**3:.1f} GB')
" | while read line; do log "  $line"; done
    
    # Check for PyTorch MPS fallback
    if [ "$USE_MPS_FALLBACK" = true ]; then
        if python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
            log "✓ PyTorch MPS fallback available"
        else
            log "⚠️  PyTorch MPS not available for fallback"
        fi
    fi
}

check_mlx_availability

# Setup MLX environment
setup_mlx_environment() {
    log "⚙️  Setting up MLX environment..."
    
    # MLX memory management
    export MLX_MEMORY_LIMIT="${MLX_MEMORY_LIMIT}GB"
    export MLX_GPU_MEMORY_FRACTION=0.8
    
    # Metal debugging
    if [ "$METAL_DEBUG" = true ]; then
        export MLX_METAL_DEBUG=1
        export METAL_DEVICE_WRAPPER_TYPE=1
        log "  Metal debugging enabled"
    fi
    
    # Unified memory optimization
    if [ "$UNIFIED_MEMORY_OPT" = true ]; then
        export MLX_UNIFIED_MEMORY=1
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
        log "  Unified memory optimization enabled"
    fi
    
    # MPS fallback
    if [ "$USE_MPS_FALLBACK" = true ]; then
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        log "  MPS fallback enabled"
    fi
    
    # Performance optimization
    export MLX_OPTIMIZE_MEMORY=1
    export MLX_LAZY_EVAL=1
    
    log "  Memory limit: $MLX_MEMORY_LIMIT"
    log "  Precision: $MLX_PRECISION"
}

setup_mlx_environment

# Activate conda environment if specified
if [ -n "$ENV_NAME" ]; then
    log "🐍 Activating conda environment: $ENV_NAME"
    
    # Initialize conda
    if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh" ]; then
        source "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh"
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
import platform
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.architecture()}')

try:
    import mlx.core as mx
    print(f'MLX: {mx.__version__}')
except ImportError:
    print('MLX: Not available')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    if torch.backends.mps.is_available():
        print('MPS: Available')
    else:
        print('MPS: Not available')
except ImportError:
    print('PyTorch: Not available')

try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except ImportError:
    print('Transformers: Not available')
" | while read line; do log "  $line"; done

# Auto-configure batch size based on memory
auto_configure_batch_size() {
    if [ -z "$BATCH_SIZE" ]; then
        local memory_gb=$(echo "$MLX_MEMORY_LIMIT" | sed 's/GB//')
        
        case "$TRAINING_MODE" in
            "lora")
                # LoRA is memory efficient
                if [ "$memory_gb" -ge 32 ]; then
                    BATCH_SIZE=8
                elif [ "$memory_gb" -ge 16 ]; then
                    BATCH_SIZE=4
                else
                    BATCH_SIZE=2
                fi
                ;;
            "qlora")
                # QLoRA is even more memory efficient
                if [ "$memory_gb" -ge 32 ]; then
                    BATCH_SIZE=16
                elif [ "$memory_gb" -ge 16 ]; then
                    BATCH_SIZE=8
                else
                    BATCH_SIZE=4
                fi
                ;;
            "full")
                # Full fine-tuning requires more memory
                if [ "$memory_gb" -ge 64 ]; then
                    BATCH_SIZE=4
                elif [ "$memory_gb" -ge 32 ]; then
                    BATCH_SIZE=2
                else
                    BATCH_SIZE=1
                fi
                ;;
        esac
        
        log "✓ Auto-configured batch size: $BATCH_SIZE (based on ${memory_gb}GB memory)"
    fi
}

auto_configure_batch_size

# Setup memory monitoring
setup_memory_monitoring() {
    if [ "$MONITOR_MEMORY" = true ]; then
        log "📊 Setting up memory monitoring..."
        
        local monitor_script="$LOG_DIR/memory_monitor_mlx.sh"
        cat > "$monitor_script" << 'EOF'
#!/bin/bash
LOG_FILE="$1"

echo "Starting MLX memory monitoring..." >> "$LOG_FILE"
while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # System memory
    memory_info=$(vm_stat | grep -E "Pages (free|active|inactive|speculative|wired down)" | tr '\n' ' ')
    echo "[$timestamp] System Memory: $memory_info" >> "$LOG_FILE"
    
    # MLX memory if available
    if python3 -c "import mlx.core as mx; print(f'MLX Memory: {mx.metal.get_peak_memory() / 1024**3:.2f}GB peak')" 2>/dev/null; then
        mlx_memory=$(python3 -c "import mlx.core as mx; print(f'{mx.metal.get_peak_memory() / 1024**3:.2f}GB')" 2>/dev/null)
        echo "[$timestamp] MLX Memory: $mlx_memory" >> "$LOG_FILE"
    fi
    
    sleep 30
done
EOF
        
        chmod +x "$monitor_script"
        
        # Start monitoring in background
        "$monitor_script" "$LOG_DIR/memory_monitor.log" &
        MONITOR_PID=$!
        log "✓ Memory monitoring started (PID: $MONITOR_PID)"
    fi
}

setup_memory_monitoring

# Setup performance profiling
setup_performance_profiling() {
    if [ "$PROFILE_PERFORMANCE" = true ]; then
        log "📈 Setting up performance profiling..."
        
        # Create profiling configuration
        local profile_config="$LOG_DIR/profile_config.json"
        cat > "$profile_config" << EOF
{
    "enable_profiling": true,
    "profile_memory": true,
    "profile_compute": true,
    "profile_communication": false,
    "output_dir": "$LOG_DIR/profiles",
    "trace_format": "json"
}
EOF
        
        mkdir -p "$LOG_DIR/profiles"
        log "✓ Performance profiling configured: $profile_config"
        export MLX_PROFILE_CONFIG="$profile_config"
    fi
}

setup_performance_profiling

# Prepare training arguments
prepare_training_args() {
    local args=(
        "--config" "$CONFIG_FILE"
        "--model-config" "$MODEL_CONFIG"
        "--output-dir" "$OUTPUT_DIR"
        "--data-path" "$DATA_PATH"
        "--training-mode" "$TRAINING_MODE"
        "--device-type" "mlx"
        "--logging-dir" "$LOG_DIR"
        "--precision" "$MLX_PRECISION"
        "--batch-size" "$BATCH_SIZE"
        "--gradient-accumulation-steps" "$GRADIENT_ACCUMULATION"
    )
    
    # Add LoRA-specific arguments
    if [[ "$TRAINING_MODE" == "lora" || "$TRAINING_MODE" == "qlora" ]]; then
        args+=("--lora-rank" "$LORA_RANK")
        args+=("--lora-alpha" "$LORA_ALPHA")
    fi
    
    # Add resume path if specified
    if [ -n "$RESUME_PATH" ]; then
        args+=("--resume-from-checkpoint" "$RESUME_PATH")
        log "📂 Resuming from checkpoint: $RESUME_PATH"
    fi
    
    # Add debug flag if enabled
    if [ "$DEBUG" = true ]; then
        args+=("--debug")
    fi
    
    # Add profiling if enabled
    if [ "$PROFILE_PERFORMANCE" = true ]; then
        args+=("--profile")
    fi
    
    echo "${args[@]}"
}

TRAINING_ARGS=($(prepare_training_args))

# Display training configuration
log "📋 MLX Training Configuration:"
log "=============================="
log "  Mode: $TRAINING_MODE"
log "  Device: MLX (Apple Silicon optimized)"
log "  Precision: $MLX_PRECISION"
log "  Memory limit: $MLX_MEMORY_LIMIT"
log "  Batch size: $BATCH_SIZE"
log "  Gradient accumulation: $GRADIENT_ACCUMULATION"
if [[ "$TRAINING_MODE" == "lora" || "$TRAINING_MODE" == "qlora" ]]; then
    log "  LoRA rank: $LORA_RANK"
    log "  LoRA alpha: $LORA_ALPHA"
fi
log "  Unified memory opt: $UNIFIED_MEMORY_OPT"
log "  Memory monitoring: $MONITOR_MEMORY"
log "  Performance profiling: $PROFILE_PERFORMANCE"

# Execute training
execute_training() {
    log "🎯 Starting MLX-optimized GRPO training..."
    log "Command: python3 train_grpo_mlx.py ${TRAINING_ARGS[*]}"
    
    if [ "$DRY_RUN" = true ]; then
        log "🔍 DRY RUN - Command would be executed:"
        echo "cd '$PROJECT_ROOT/training'"
        echo "python3 train_grpo_mlx.py ${TRAINING_ARGS[*]}"
        return 0
    fi
    
    # Change to training directory
    cd "$PROJECT_ROOT/training"
    
    # Check if MLX training script exists, fallback to regular script
    if [ -f "train_grpo_mlx.py" ]; then
        local training_script="train_grpo_mlx.py"
    else
        local training_script="train_grpo.py"
        log "⚠️  MLX-specific script not found, using regular training script"
    fi
    
    # Execute training with proper error handling
    set +e  # Don't exit on error for training script
    python3 "$training_script" "${TRAINING_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=$?
    set -e
    
    if [ $exit_code -eq 0 ]; then
        log "✅ MLX training completed successfully!"
        log "📁 Output directory: $OUTPUT_DIR"
        log "📄 Log file: $LOG_FILE"
        
        # Show performance summary if profiling was enabled
        if [ "$PROFILE_PERFORMANCE" = true ] && [ -d "$LOG_DIR/profiles" ]; then
            log "📈 Performance profiles saved to: $LOG_DIR/profiles"
            find "$LOG_DIR/profiles" -name "*.json" | head -3 | while read file; do
                log "  - $(basename "$file")"
            done
        fi
        
        # Show final model info
        if [ -d "$OUTPUT_DIR" ]; then
            log "📊 Final outputs:"
            find "$OUTPUT_DIR" -name "*.bin" -o -name "*.safetensors" -o -name "adapter_*" | head -5 | while read file; do
                log "  - $(basename "$file")"
            done
        fi
    else
        log_error "MLX training failed with exit code: $exit_code"
        log_error "Check log file for details: $LOG_FILE"
        exit $exit_code
    fi
}

# Cleanup function
cleanup() {
    log "🧹 Cleaning up MLX training..."
    
    # Stop memory monitoring
    if [ -n "${MONITOR_PID:-}" ]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        log "✓ Stopped memory monitoring"
    fi
    
    # Kill any remaining background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Clear MLX memory cache
    python3 -c "
try:
    import mlx.core as mx
    mx.metal.clear_cache()
    print('MLX memory cache cleared')
except:
    pass
" 2>/dev/null | while read line; do log "  $line"; done
    
    log "👋 Cleanup completed"
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    log "🎬 Executing MLX-optimized training..."
    execute_training
    
    log "🎉 MLX training session completed!"
    log "📁 All outputs saved to: $OUTPUT_DIR"
    
    # Show MLX-specific tips
    log ""
    log "💡 MLX Training Tips:"
    log "  - Use unified memory optimization for large models"
    log "  - Monitor memory usage to avoid swapping"
    log "  - Enable profiling to identify bottlenecks"
    log "  - Consider QLoRA for memory-constrained training"
}

# Run main function
main "$@"