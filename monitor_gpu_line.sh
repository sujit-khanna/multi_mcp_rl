#!/bin/bash
# Ultra-compact single-line GPU monitor
# ======================================

# Check nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found"
    exit 1
fi

# Refresh interval
INTERVAL=${1:-0.5}

echo "GPU Monitor (updates every ${INTERVAL}s, Ctrl+C to stop)"
echo ""

while true; do
    # Get GPU stats
    stats=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits)
    
    # Parse values
    IFS=',' read -r gpu_util mem_used mem_total temp <<< "$stats"
    
    # Calculate memory percentage
    mem_percent=$(echo "scale=0; $mem_used * 100 / $mem_total" | bc)
    mem_gb=$(echo "scale=1; $mem_used / 1024" | bc)
    total_gb=$(echo "scale=0; $mem_total / 1024" | bc)
    
    # Get process count
    proc_count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    
    # Create status line
    printf "\r[%s] GPU: %3d%% | MEM: %3d%% (%.1f/%dGB) | TEMP: %2d°C | PROCS: %d     " \
           "$(date '+%H:%M:%S')" \
           $gpu_util \
           $mem_percent \
           $mem_gb \
           $total_gb \
           $temp \
           $proc_count
    
    sleep $INTERVAL
done