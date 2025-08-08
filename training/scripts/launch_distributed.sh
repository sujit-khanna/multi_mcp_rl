#!/bin/bash
# GRPO Training - Distributed Training Launch Script
# =================================================
# Supports: SLURM clusters, manual multi-node setup, fault tolerance
# Optimized for: Large-scale distributed training across multiple nodes

set -euo pipefail

# Script metadata
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default configuration
DEFAULT_CONFIG="$PROJECT_ROOT/config/training_config.yaml"
DEFAULT_MODEL_CONFIG="$PROJECT_ROOT/config/model_config.yaml"
DEFAULT_DEEPSPEED_CONFIG="$PROJECT_ROOT/config/deepspeed_config.json"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/outputs/distributed_training_$TIMESTAMP"
DEFAULT_DATA_PATH="$PROJECT_ROOT/data/processed/train.json"

# Parse command line arguments
USAGE="Usage: $0 [OPTIONS]
Options:
  -c, --config PATH           Training config file (default: $DEFAULT_CONFIG)
  -m, --model-config PATH     Model config file (default: $DEFAULT_MODEL_CONFIG)
  -ds, --deepspeed-config PATH DeepSpeed config file (default: $DEFAULT_DEEPSPEED_CONFIG)
  -o, --output-dir PATH       Output directory (default: $DEFAULT_OUTPUT_DIR)
  -d, --data PATH            Training data path (default: $DEFAULT_DATA_PATH)
  
  # Distributed Configuration
  --mode MODE                Distribution mode: slurm, manual, auto (default: auto)
  --nodes NUM                Number of nodes (default: 1)
  --gpus-per-node NUM        GPUs per node (default: auto-detect)
  --node-rank RANK           Node rank for manual mode (default: 0)
  --master-addr ADDR         Master node address (default: auto-detect)
  --master-port PORT         Master port (default: 29500)
  
  # SLURM Configuration
  --slurm-job-name NAME      SLURM job name (default: grpo-training)
  --slurm-partition PART     SLURM partition (default: gpu)
  --slurm-time TIME          Time limit (default: 24:00:00)
  --slurm-mem MEM           Memory per node (default: 100G)
  --slurm-account ACCOUNT    SLURM account (optional)
  --slurm-qos QOS           Quality of Service (optional)
  
  # Training Configuration
  --training-mode MODE       Training mode: lora, full (default: full)
  --zero-stage STAGE         DeepSpeed ZeRO stage (1,2,3) (default: 3)
  --gradient-checkpointing   Enable gradient checkpointing
  --offload-optimizer        Offload optimizer to CPU
  --offload-params           Offload parameters to CPU
  
  # Environment
  -e, --env-name NAME        Conda environment name (optional)
  --container-image IMAGE    Singularity/Docker container image (optional)
  
  # Monitoring & Debugging
  --monitor-nodes            Enable per-node monitoring
  --fault-tolerance          Enable fault tolerance features
  --debug                    Enable debug logging
  --dry-run                  Show commands without executing
  -r, --resume PATH          Resume from checkpoint (optional)
  -h, --help                 Show this help message

Examples:
  # Auto-detect single node training
  $0
  
  # SLURM 4-node training
  $0 --mode slurm --nodes 4 --slurm-partition gpu-a100
  
  # Manual 2-node setup (run on each node)
  $0 --mode manual --nodes 2 --node-rank 0 --master-addr 192.168.1.100
  $0 --mode manual --nodes 2 --node-rank 1 --master-addr 192.168.1.100
  
  # Large-scale SLURM training with fault tolerance
  $0 --mode slurm --nodes 8 --fault-tolerance --monitor-nodes
"

# Initialize variables
CONFIG_FILE="$DEFAULT_CONFIG"
MODEL_CONFIG="$DEFAULT_MODEL_CONFIG"
DEEPSPEED_CONFIG="$DEFAULT_DEEPSPEED_CONFIG"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
DATA_PATH="$DEFAULT_DATA_PATH"

# Distributed configuration
MODE="auto"
NODES=1
GPUS_PER_NODE=""
NODE_RANK=0
MASTER_ADDR=""
MASTER_PORT="29500"

# SLURM configuration
SLURM_JOB_NAME="grpo-training"
SLURM_PARTITION="gpu"
SLURM_TIME="24:00:00"
SLURM_MEM="100G"
SLURM_ACCOUNT=""
SLURM_QOS=""

# Training configuration
TRAINING_MODE="full"
ZERO_STAGE="3"
GRADIENT_CHECKPOINTING=false
OFFLOAD_OPTIMIZER=false
OFFLOAD_PARAMS=false

# Environment
ENV_NAME=""
CONTAINER_IMAGE=""

# Monitoring & debugging
MONITOR_NODES=false
FAULT_TOLERANCE=false
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
        --mode)
            MODE="$2"
            shift 2
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --gpus-per-node)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --slurm-job-name)
            SLURM_JOB_NAME="$2"
            shift 2
            ;;
        --slurm-partition)
            SLURM_PARTITION="$2"
            shift 2
            ;;
        --slurm-time)
            SLURM_TIME="$2"
            shift 2
            ;;
        --slurm-mem)
            SLURM_MEM="$2"
            shift 2
            ;;
        --slurm-account)
            SLURM_ACCOUNT="$2"
            shift 2
            ;;
        --slurm-qos)
            SLURM_QOS="$2"
            shift 2
            ;;
        --training-mode)
            TRAINING_MODE="$2"
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
        -e|--env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --container-image)
            CONTAINER_IMAGE="$2"
            shift 2
            ;;
        --monitor-nodes)
            MONITOR_NODES=true
            shift
            ;;
        --fault-tolerance)
            FAULT_TOLERANCE=true
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
LOG_FILE="$LOG_DIR/distributed_training_$TIMESTAMP.log"

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

log "🚀 GRPO Distributed Training Launch"
log "==================================="
log "Project Root: $PROJECT_ROOT"
log "Output Directory: $OUTPUT_DIR"
log "Mode: $MODE"
log "Timestamp: $TIMESTAMP"

# Auto-detect mode if not specified
auto_detect_mode() {
    if [ "$MODE" = "auto" ]; then
        if command -v sbatch >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
            MODE="slurm"
            log "🔍 Auto-detected mode: SLURM"
        elif [ "$NODES" -gt 1 ]; then
            MODE="manual"
            log "🔍 Auto-detected mode: Manual (multi-node)"
        else
            MODE="single"
            log "🔍 Auto-detected mode: Single node"
        fi
    fi
}

auto_detect_mode

# Detect GPU configuration
detect_gpu_config() {
    log "🔍 Detecting GPU configuration..."
    
    if [ -z "$GPUS_PER_NODE" ]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
            log "✓ Auto-detected $GPUS_PER_NODE GPUs per node"
        else
            log_error "No NVIDIA GPUs detected"
            exit 1
        fi
    fi
    
    local total_gpus=$((NODES * GPUS_PER_NODE))
    log "📊 Total training configuration:"
    log "  Nodes: $NODES"
    log "  GPUs per node: $GPUS_PER_NODE"
    log "  Total GPUs: $total_gpus"
}

detect_gpu_config

# Setup distributed environment variables
setup_distributed_env() {
    log "⚙️  Setting up distributed environment..."
    
    # Auto-detect master address for single node
    if [ -z "$MASTER_ADDR" ]; then
        if [ "$NODES" -eq 1 ]; then
            MASTER_ADDR="127.0.0.1"
        elif [ -n "${SLURM_JOB_ID:-}" ]; then
            MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
        else
            log_error "Master address not specified for multi-node setup"
            exit 1
        fi
    fi
    
    export MASTER_ADDR="$MASTER_ADDR"
    export MASTER_PORT="$MASTER_PORT"
    export WORLD_SIZE=$((NODES * GPUS_PER_NODE))
    export NCCL_DEBUG=INFO
    
    # NCCL optimization for distributed training
    export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
    export NCCL_NET_GDR_LEVEL=3
    export NCCL_SOCKET_IFNAME=^docker0,lo
    
    # Fault tolerance settings
    if [ "$FAULT_TOLERANCE" = true ]; then
        export NCCL_ASYNC_ERROR_HANDLING=1
        export TORCH_DISTRIBUTED_DEBUG=DETAIL
    fi
    
    log "  Master: $MASTER_ADDR:$MASTER_PORT"
    log "  World Size: $WORLD_SIZE"
    log "  Node Rank: $NODE_RANK"
}

setup_distributed_env

# Create SLURM batch script
create_slurm_script() {
    local slurm_script="$LOG_DIR/slurm_job.sh"
    
    log "📝 Creating SLURM batch script..."
    
    cat > "$slurm_script" << EOF
#!/bin/bash
#SBATCH --job-name=$SLURM_JOB_NAME
#SBATCH --partition=$SLURM_PARTITION
#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=$GPUS_PER_NODE
#SBATCH --gpus-per-node=$GPUS_PER_NODE
#SBATCH --time=$SLURM_TIME
#SBATCH --mem=$SLURM_MEM
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --error=$LOG_DIR/slurm-%j.err
EOF

    # Add optional SLURM parameters
    if [ -n "$SLURM_ACCOUNT" ]; then
        echo "#SBATCH --account=$SLURM_ACCOUNT" >> "$slurm_script"
    fi
    
    if [ -n "$SLURM_QOS" ]; then
        echo "#SBATCH --qos=$SLURM_QOS" >> "$slurm_script"
    fi
    
    # Add environment setup
    cat >> "$slurm_script" << EOF

# Environment setup
export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=\$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export NODE_RANK=\$SLURM_NODEID
export LOCAL_RANK=\$SLURM_LOCALID

# Load modules (customize as needed)
# module load cuda/11.8
# module load python/3.9

EOF

    # Add conda activation if specified
    if [ -n "$ENV_NAME" ]; then
        cat >> "$slurm_script" << EOF
# Activate conda environment
source ~/.bashrc
conda activate $ENV_NAME
EOF
    fi
    
    # Add container support if specified
    if [ -n "$CONTAINER_IMAGE" ]; then
        cat >> "$slurm_script" << EOF
# Run with Singularity container
srun singularity exec --nv $CONTAINER_IMAGE \\
EOF
    else
        cat >> "$slurm_script" << EOF
# Run training
srun \\
EOF
    fi
    
    # Add training command
    cat >> "$slurm_script" << EOF
python3 -m torch.distributed.launch \\
    --nproc_per_node=\$SLURM_NTASKS_PER_NODE \\
    --nnodes=\$SLURM_NNODES \\
    --node_rank=\$SLURM_NODEID \\
    --master_addr=\$MASTER_ADDR \\
    --master_port=\$MASTER_PORT \\
    $PROJECT_ROOT/training/train_grpo.py \\
    --config "$CONFIG_FILE" \\
    --model-config "$MODEL_CONFIG" \\
    --deepspeed "$DEEPSPEED_CONFIG" \\
    --output-dir "$OUTPUT_DIR" \\
    --data-path "$DATA_PATH" \\
    --training-mode "$TRAINING_MODE" \\
    --device-type cuda \\
    --logging-dir "$LOG_DIR"
EOF

    # Add optional training arguments
    if [ -n "$RESUME_PATH" ]; then
        echo "    --resume-from-checkpoint \"$RESUME_PATH\" \\" >> "$slurm_script"
    fi
    
    if [ "$GRADIENT_CHECKPOINTING" = true ]; then
        echo "    --gradient-checkpointing \\" >> "$slurm_script"
    fi
    
    if [ "$DEBUG" = true ]; then
        echo "    --debug \\" >> "$slurm_script"
    fi
    
    # Remove trailing backslash
    sed -i '$ s/ \\$//' "$slurm_script"
    
    chmod +x "$slurm_script"
    log "✓ Created SLURM script: $slurm_script"
    
    echo "$slurm_script"
}

# Create manual launch script for each node
create_manual_script() {
    local manual_script="$LOG_DIR/manual_launch_node_${NODE_RANK}.sh"
    
    log "📝 Creating manual launch script for node $NODE_RANK..."
    
    cat > "$manual_script" << EOF
#!/bin/bash
# Manual distributed training script for node $NODE_RANK

set -euo pipefail

# Environment setup
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"
export WORLD_SIZE=$((NODES * GPUS_PER_NODE))
export NODE_RANK="$NODE_RANK"
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=\$(seq -s, 0 \$((${GPUS_PER_NODE} - 1)))

echo "Starting training on node $NODE_RANK..."
echo "Master: \$MASTER_ADDR:\$MASTER_PORT"
echo "World Size: \$WORLD_SIZE"
echo "Node Rank: \$NODE_RANK"
echo "GPUs: \$CUDA_VISIBLE_DEVICES"

EOF

    # Add conda activation if specified
    if [ -n "$ENV_NAME" ]; then
        cat >> "$manual_script" << EOF
# Activate conda environment
source ~/.bashrc
conda activate $ENV_NAME
EOF
    fi
    
    # Add training command
    cat >> "$manual_script" << EOF
# Change to training directory
cd "$PROJECT_ROOT/training"

# Launch distributed training
python3 -m torch.distributed.launch \\
    --nproc_per_node=$GPUS_PER_NODE \\
    --nnodes=$NODES \\
    --node_rank=$NODE_RANK \\
    --master_addr=\$MASTER_ADDR \\
    --master_port=\$MASTER_PORT \\
    train_grpo.py \\
    --config "$CONFIG_FILE" \\
    --model-config "$MODEL_CONFIG" \\
    --deepspeed "$DEEPSPEED_CONFIG" \\
    --output-dir "$OUTPUT_DIR" \\
    --data-path "$DATA_PATH" \\
    --training-mode "$TRAINING_MODE" \\
    --device-type cuda \\
    --logging-dir "$LOG_DIR"
EOF

    # Add optional arguments
    if [ -n "$RESUME_PATH" ]; then
        echo "    --resume-from-checkpoint \"$RESUME_PATH\" \\" >> "$manual_script"
    fi
    
    if [ "$GRADIENT_CHECKPOINTING" = true ]; then
        echo "    --gradient-checkpointing \\" >> "$manual_script"
    fi
    
    if [ "$DEBUG" = true ]; then
        echo "    --debug \\" >> "$manual_script"
    fi
    
    # Remove trailing backslash and add log redirection
    sed -i '$ s/ \\$//' "$manual_script"
    echo "    2>&1 | tee \"$LOG_DIR/node_${NODE_RANK}_training.log\"" >> "$manual_script"
    
    chmod +x "$manual_script"
    log "✓ Created manual script: $manual_script"
    
    echo "$manual_script"
}

# Setup node monitoring
setup_node_monitoring() {
    if [ "$MONITOR_NODES" = true ]; then
        log "📊 Setting up node monitoring..."
        
        local monitor_script="$LOG_DIR/node_monitor.sh"
        cat > "$monitor_script" << 'EOF'
#!/bin/bash
# Node monitoring script

LOG_DIR="$1"
NODE_RANK="${2:-0}"

echo "Starting node monitoring for rank $NODE_RANK..."
monitor_log="$LOG_DIR/monitor_node_${NODE_RANK}.log"

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # GPU monitoring
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "[$timestamp] GPU Status:" >> "$monitor_log"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader >> "$monitor_log"
    fi
    
    # System monitoring
    echo "[$timestamp] System Status:" >> "$monitor_log"
    echo "  CPU: $(cat /proc/loadavg)" >> "$monitor_log"
    echo "  Memory: $(free -h | grep Mem | awk '{print $3"/"$2}')" >> "$monitor_log"
    echo "  Disk: $(df -h . | tail -1 | awk '{print $3"/"$2" ("$5" used)"}')" >> "$monitor_log"
    
    sleep 60
done
EOF
        
        chmod +x "$monitor_script"
        
        # Start monitoring in background
        "$monitor_script" "$LOG_DIR" "$NODE_RANK" &
        MONITOR_PID=$!
        log "✓ Node monitoring started (PID: $MONITOR_PID)"
    fi
}

setup_node_monitoring

# Display configuration summary
display_config_summary() {
    log "📋 Distributed Training Configuration:"
    log "======================================"
    log "  Mode: $MODE"
    log "  Nodes: $NODES"
    log "  GPUs per node: $GPUS_PER_NODE"
    log "  Total GPUs: $((NODES * GPUS_PER_NODE))"
    log "  Training mode: $TRAINING_MODE"
    log "  DeepSpeed ZeRO: Stage $ZERO_STAGE"
    log "  Master: $MASTER_ADDR:$MASTER_PORT"
    
    if [ "$MODE" = "slurm" ]; then
        log "  SLURM partition: $SLURM_PARTITION"
        log "  SLURM time limit: $SLURM_TIME"
    fi
    
    log "  Config: $CONFIG_FILE"
    log "  Output: $OUTPUT_DIR"
    log "  Fault tolerance: $FAULT_TOLERANCE"
    log "  Monitoring: $MONITOR_NODES"
}

display_config_summary

# Execute based on mode
execute_training() {
    log "🎯 Launching distributed training..."
    
    case $MODE in
        "slurm")
            local slurm_script=$(create_slurm_script)
            
            if [ "$DRY_RUN" = true ]; then
                log "🔍 DRY RUN - SLURM script would be submitted:"
                cat "$slurm_script"
                return 0
            fi
            
            log "🚀 Submitting SLURM job..."
            local job_id=$(sbatch "$slurm_script" | grep -o '[0-9]\+')
            log "✓ SLURM job submitted: $job_id"
            log "📊 Monitor with: squeue -j $job_id"
            log "📄 Logs: $LOG_DIR/slurm-${job_id}.out"
            ;;
            
        "manual")
            local manual_script=$(create_manual_script)
            
            if [ "$DRY_RUN" = true ]; then
                log "🔍 DRY RUN - Manual script would be executed:"
                cat "$manual_script"
                return 0
            fi
            
            log "🚀 Starting manual distributed training on node $NODE_RANK..."
            log "📄 Execute on each node: $manual_script"
            
            if [ "$NODE_RANK" -eq 0 ]; then
                log "⚠️  Remember to run this script on ALL $NODES nodes!"
                log "   Node 0: $manual_script (current)"
                for ((i=1; i<NODES; i++)); do
                    log "   Node $i: Set --node-rank $i and run script"
                done
            fi
            
            # Execute the script
            exec "$manual_script"
            ;;
            
        "single")
            log "🚀 Running single-node distributed training..."
            
            # Use the multi-GPU script for single node
            exec "$SCRIPT_DIR/launch_full_ft_multi_gpu.sh" \
                --config "$CONFIG_FILE" \
                --model-config "$MODEL_CONFIG" \
                --deepspeed-config "$DEEPSPEED_CONFIG" \
                --output-dir "$OUTPUT_DIR" \
                --data-path "$DATA_PATH" \
                --num-gpus "$GPUS_PER_NODE" \
                $([ "$GRADIENT_CHECKPOINTING" = true ] && echo "--gradient-checkpointing") \
                $([ "$OFFLOAD_OPTIMIZER" = true ] && echo "--offload-optimizer") \
                $([ "$OFFLOAD_PARAMS" = true ] && echo "--offload-params") \
                $([ "$MONITOR_NODES" = true ] && echo "--monitor-memory") \
                $([ "$DEBUG" = true ] && echo "--debug") \
                $([ "$DRY_RUN" = true ] && echo "--dry-run") \
                $([ -n "$RESUME_PATH" ] && echo "--resume \"$RESUME_PATH\"")
            ;;
    esac
}

# Cleanup function
cleanup() {
    log "🧹 Cleaning up distributed training..."
    
    # Stop monitoring if running
    if [ -n "${MONITOR_PID:-}" ]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        log "✓ Stopped node monitoring"
    fi
    
    # Kill any remaining background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    log "👋 Cleanup completed"
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    log "🎬 Starting distributed training execution..."
    execute_training
    
    log "🎉 Distributed training launch completed!"
    log "📁 All outputs will be saved to: $OUTPUT_DIR"
    log "📊 Monitor logs in: $LOG_DIR"
}

# Run main function
main "$@"