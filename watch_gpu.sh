#!/bin/bash
# GPU Monitor using watch and nvidia-smi
# =======================================
# Best option for real-time monitoring

echo "Starting GPU Monitor (Ctrl+C to stop)"
echo "====================================="

# Use watch to update nvidia-smi output every second
watch -n 1 -c '
echo "GPU STATUS - $(date +"%Y-%m-%d %H:%M:%S")"
echo "=========================================="
nvidia-smi --query-gpu=gpu_name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader | \
while IFS="," read -r name gpu mem_used mem_total temp power; do
    # Clean up values
    gpu=$(echo $gpu | sed "s/ %//")
    mem_used=$(echo $mem_used | sed "s/ MiB//")
    mem_total=$(echo $mem_total | sed "s/ MiB//")
    power=$(echo $power | sed "s/ W//")
    
    # Calculate memory percentage
    if [ "$mem_total" != "0" ]; then
        mem_percent=$((mem_used * 100 / mem_total))
    else
        mem_percent=0
    fi
    
    # Display
    echo "GPU: $name"
    echo "  Utilization: $gpu%"
    echo "  Memory:      $mem_percent% ($mem_used MB / $mem_total MB)"
    echo "  Temperature: $temp°C"
    echo "  Power:       $power W"
done
echo ""
echo "=========================================="
echo "PROCESSES:"
nvidia-smi pmon -c 1 | tail -n +3 | head -10
'