#!/bin/bash
# Simple GPU Monitor using nvidia-smi
# ====================================
# Real-time GPU usage monitoring

echo "GPU Monitor - Updates every 1 second (Ctrl+C to stop)"
echo "======================================================"
echo ""

# Run nvidia-smi in continuous mode
nvidia-smi --loop=1 --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv