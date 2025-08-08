#!/bin/bash
# Real-time GPU Monitoring Script
# ================================
# Streams GPU usage statistics to terminal with color formatting

# Colors for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. NVIDIA drivers not installed?${NC}"
    exit 1
fi

# Parse command line arguments
INTERVAL=${1:-1}  # Default 1 second refresh
COMPACT=${2:-false}  # Default to detailed view

clear
echo -e "${CYAN}${BOLD}========================================${NC}"
echo -e "${CYAN}${BOLD}    GPU Real-Time Monitor Started      ${NC}"
echo -e "${CYAN}${BOLD}========================================${NC}"
echo -e "${WHITE}Refresh Interval: ${INTERVAL}s${NC}"
echo -e "${WHITE}Press Ctrl+C to stop${NC}"
echo ""

# Function to get color based on utilization percentage
get_color() {
    local value=$1
    local threshold_low=$2
    local threshold_high=$3
    
    if (( $(echo "$value < $threshold_low" | bc -l) )); then
        echo "${GREEN}"
    elif (( $(echo "$value < $threshold_high" | bc -l) )); then
        echo "${YELLOW}"
    else
        echo "${RED}"
    fi
}

# Function to draw a progress bar
draw_bar() {
    local percent=$1
    local width=30
    local filled=$(echo "$percent * $width / 100" | bc)
    local empty=$((width - filled))
    
    printf "["
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' '-'
    printf "]"
}

# Main monitoring loop
while true; do
    # Clear screen for clean output
    clear
    
    # Header with timestamp
    echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║               GPU MONITORING - $(date '+%Y-%m-%d %H:%M:%S')              ║${NC}"
    echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Get GPU information
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    
    for gpu_id in $(seq 0 $((gpu_count - 1))); do
        # Query GPU metrics
        gpu_info=$(nvidia-smi --id=$gpu_id --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits)
        
        # Parse the values
        IFS=',' read -r gpu_name gpu_util mem_util mem_used mem_total temp power_draw power_limit <<< "$gpu_info"
        
        # Remove leading/trailing spaces
        gpu_name=$(echo $gpu_name | xargs)
        gpu_util=$(echo $gpu_util | xargs)
        mem_util=$(echo $mem_util | xargs)
        mem_used=$(echo $mem_used | xargs)
        mem_total=$(echo $mem_total | xargs)
        temp=$(echo $temp | xargs)
        power_draw=$(echo $power_draw | xargs | sed 's/\[Not Supported\]/0/')
        power_limit=$(echo $power_limit | xargs | sed 's/\[Not Supported\]/0/')
        
        # Calculate memory percentage if not provided
        if [ "$mem_util" == "[Not Supported]" ] || [ -z "$mem_util" ]; then
            mem_util=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc)
        fi
        
        # Get colors based on thresholds
        gpu_color=$(get_color $gpu_util 50 80)
        mem_color=$(get_color $mem_util 60 85)
        temp_color=$(get_color $temp 60 80)
        
        # Display GPU information
        echo -e "${WHITE}${BOLD}GPU $gpu_id: $gpu_name${NC}"
        echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        # GPU Utilization
        printf "${WHITE}GPU Usage:    ${NC}"
        printf "${gpu_color}%3d%%${NC} " $gpu_util
        draw_bar $gpu_util
        echo ""
        
        # Memory Usage
        printf "${WHITE}Memory Usage: ${NC}"
        printf "${mem_color}%3d%%${NC} " ${mem_util%.*}
        draw_bar ${mem_util%.*}
        printf " ${WHITE}[${mem_color}%.1f GB${NC} / ${WHITE}%.1f GB${NC}]\n" \
            $(echo "scale=1; $mem_used / 1024" | bc) \
            $(echo "scale=1; $mem_total / 1024" | bc)
        
        # Temperature
        printf "${WHITE}Temperature:  ${NC}"
        printf "${temp_color}%3d°C${NC} " $temp
        
        # Temperature bar (0-100°C scale)
        temp_percent=$(echo "$temp * 100 / 100" | bc)
        draw_bar $temp_percent
        echo ""
        
        # Power if supported
        if [ "$power_draw" != "0" ] && [ "$power_limit" != "0" ]; then
            power_percent=$(echo "scale=0; $power_draw * 100 / $power_limit" | bc)
            power_color=$(get_color $power_percent 60 85)
            printf "${WHITE}Power Draw:   ${NC}"
            printf "${power_color}%3d%%${NC} " $power_percent
            draw_bar $power_percent
            printf " ${WHITE}[${power_color}%.0f W${NC} / ${WHITE}%.0f W${NC}]\n" $power_draw $power_limit
        fi
        
        echo ""
    done
    
    # Show processes using GPU
    echo -e "${WHITE}${BOLD}Active GPU Processes:${NC}"
    echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Get process information
    processes=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader,nounits 2>/dev/null)
    
    if [ -z "$processes" ]; then
        echo -e "${YELLOW}No active GPU processes${NC}"
    else
        echo -e "${WHITE}PID     Memory    Process${NC}"
        while IFS=',' read -r pid name mem; do
            pid=$(echo $pid | xargs)
            name=$(echo $name | xargs | cut -c1-30)
            mem=$(echo $mem | xargs)
            
            # Convert memory to GB
            mem_gb=$(echo "scale=2; $mem / 1024" | bc)
            
            # Get process command if possible
            if [ -f "/proc/$pid/cmdline" ]; then
                cmd=$(tr '\0' ' ' < /proc/$pid/cmdline | cut -c1-40)
                printf "${GREEN}%-7s ${YELLOW}%6.2f GB ${WHITE}%s${NC}\n" "$pid" "$mem_gb" "$cmd"
            else
                printf "${GREEN}%-7s ${YELLOW}%6.2f GB ${WHITE}%s${NC}\n" "$pid" "$mem_gb" "$name"
            fi
        done <<< "$processes"
    fi
    
    echo ""
    echo -e "${CYAN}────────────────────────────────────────────────────${NC}"
    
    # Quick stats summary
    echo -e "${WHITE}${BOLD}Summary Statistics:${NC}"
    
    # Get overall GPU utilization
    avg_gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum/NR}')
    avg_mem=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum/NR}')
    
    # Handle cases where values might be empty
    avg_gpu=${avg_gpu:-0}
    avg_mem=${avg_mem:-0}
    
    echo -e "${WHITE}Average GPU Utilization:  ${GREEN}${avg_gpu%.*}%${NC}"
    echo -e "${WHITE}Average Memory Usage:     ${GREEN}${avg_mem%.*}%${NC}"
    
    # Show CUDA version
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo -e "${WHITE}CUDA Version:            ${CYAN}$cuda_version${NC}"
    
    echo ""
    echo -e "${CYAN}────────────────────────────────────────────────────${NC}"
    echo -e "${WHITE}Refreshing every ${INTERVAL}s... Press ${RED}Ctrl+C${WHITE} to exit${NC}"
    
    # Wait before next update
    sleep $INTERVAL
done