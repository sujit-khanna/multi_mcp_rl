#!/bin/bash
# Simple GPU Monitor - Compact real-time display
# ===============================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'

# Check nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found"
    exit 1
fi

# Refresh interval (default 1 second)
INTERVAL=${1:-1}

echo -e "${CYAN}GPU Monitor (Refresh: ${INTERVAL}s, Ctrl+C to stop)${NC}"
echo ""

# Main loop
while true; do
    # Move cursor to top (after header)
    tput cup 2 0
    
    # Clear from cursor to end of screen
    tput ed
    
    # Get timestamp
    echo -e "${WHITE}[$(date '+%H:%M:%S')]${NC}"
    echo ""
    
    # Query GPU stats
    nvidia-smi --query-gpu=gpu_name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
               --format=csv,noheader,nounits | while IFS=',' read -r name gpu_util mem_used mem_total temp power; do
        
        # Clean up values
        name=$(echo $name | xargs | cut -c1-20)
        gpu_util=$(echo $gpu_util | xargs)
        mem_used=$(echo $mem_used | xargs)
        mem_total=$(echo $mem_total | xargs)
        temp=$(echo $temp | xargs)
        power=$(echo $power | xargs | sed 's/\[Not Supported\]/N\/A/')
        
        # Calculate memory percentage and GB
        mem_percent=$(echo "scale=0; $mem_used * 100 / $mem_total" | bc)
        mem_gb=$(echo "scale=1; $mem_used / 1024" | bc)
        total_gb=$(echo "scale=1; $mem_total / 1024" | bc)
        
        # Color coding
        if [ $gpu_util -lt 50 ]; then gpu_color=$GREEN
        elif [ $gpu_util -lt 80 ]; then gpu_color=$YELLOW
        else gpu_color=$RED; fi
        
        if [ $mem_percent -lt 60 ]; then mem_color=$GREEN
        elif [ $mem_percent -lt 85 ]; then mem_color=$YELLOW
        else mem_color=$RED; fi
        
        if [ $temp -lt 60 ]; then temp_color=$GREEN
        elif [ $temp -lt 80 ]; then temp_color=$YELLOW
        else temp_color=$RED; fi
        
        # Display stats in one line
        printf "${WHITE}%-20s${NC} │ " "$name"
        printf "GPU: ${gpu_color}%3d%%${NC} │ " $gpu_util
        printf "MEM: ${mem_color}%3d%%${NC} (${mem_color}%.1f${NC}/%.1fGB) │ " $mem_percent $mem_gb $total_gb
        printf "TEMP: ${temp_color}%2d°C${NC} │ " $temp
        
        if [ "$power" != "N/A" ]; then
            printf "PWR: ${YELLOW}%.0fW${NC}" $power
        else
            printf "PWR: ${WHITE}N/A${NC}"
        fi
        echo ""
    done
    
    echo ""
    echo -e "${CYAN}──────────────────────────────────────────────────────────────${NC}"
    
    # Show processes
    echo -e "${WHITE}Processes:${NC}"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader,nounits 2>/dev/null | \
    head -5 | while IFS=',' read -r pid name mem; do
        if [ ! -z "$pid" ]; then
            mem_gb=$(echo "scale=2; $mem / 1024" | bc)
            printf "  PID %-7s: ${YELLOW}%6.2f GB${NC} - %s\n" "$pid" "$mem_gb" "$(echo $name | cut -c1-30)"
        fi
    done
    
    # If no processes
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
        echo -e "  ${YELLOW}No active GPU processes${NC}"
    fi
    
    sleep $INTERVAL
done