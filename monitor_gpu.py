#!/usr/bin/env python3
"""
Real-time GPU Monitor
Displays GPU utilization, memory usage, temperature, and processes
"""

import subprocess
import time
import sys
import os
from datetime import datetime

# ANSI color codes
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def clear_screen():
    """Clear the terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def get_color(value, low_threshold, high_threshold):
    """Get color based on value and thresholds"""
    if value < low_threshold:
        return Colors.GREEN
    elif value < high_threshold:
        return Colors.YELLOW
    else:
        return Colors.RED

def draw_bar(percentage, width=30):
    """Draw a progress bar"""
    filled = int(percentage * width / 100)
    bar = '=' * filled + '-' * (width - filled)
    return f"[{bar}]"

def get_gpu_info():
    """Get GPU information from nvidia-smi"""
    try:
        # Query GPU stats
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            
            # Handle missing values
            gpu_info = {
                'name': parts[0] if len(parts) > 0 else 'Unknown',
                'gpu_util': float(parts[1]) if len(parts) > 1 and parts[1] else 0,
                'mem_util': float(parts[2]) if len(parts) > 2 and parts[2] else 0,
                'mem_used': float(parts[3]) if len(parts) > 3 and parts[3] else 0,
                'mem_total': float(parts[4]) if len(parts) > 4 and parts[4] else 1,
                'temperature': float(parts[5]) if len(parts) > 5 and parts[5] else 0,
                'power_draw': parts[6] if len(parts) > 6 else 'N/A',
                'power_limit': parts[7] if len(parts) > 7 else 'N/A'
            }
            
            # Calculate memory percentage if not provided
            if gpu_info['mem_util'] == 0 and gpu_info['mem_total'] > 0:
                gpu_info['mem_util'] = (gpu_info['mem_used'] / gpu_info['mem_total']) * 100
            
            # Clean power values
            if gpu_info['power_draw'] == '[Not Supported]':
                gpu_info['power_draw'] = 'N/A'
            else:
                try:
                    gpu_info['power_draw'] = float(gpu_info['power_draw'])
                except:
                    gpu_info['power_draw'] = 'N/A'
                    
            if gpu_info['power_limit'] == '[Not Supported]':
                gpu_info['power_limit'] = 'N/A'
            else:
                try:
                    gpu_info['power_limit'] = float(gpu_info['power_limit'])
                except:
                    gpu_info['power_limit'] = 'N/A'
            
            gpus.append(gpu_info)
        
        return gpus
    except subprocess.CalledProcessError:
        return []
    except FileNotFoundError:
        print(f"{Colors.RED}Error: nvidia-smi not found. NVIDIA drivers not installed?{Colors.RESET}")
        sys.exit(1)

def get_gpu_processes():
    """Get processes using GPU"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-compute-apps=pid,name,used_memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    processes.append({
                        'pid': parts[0],
                        'name': parts[1],
                        'memory': float(parts[2]) / 1024  # Convert to GB
                    })
        return processes
    except:
        return []

def display_compact(refresh_interval=1):
    """Display GPU stats in compact format"""
    print(f"{Colors.CYAN}{Colors.BOLD}GPU Monitor (Compact Mode){Colors.RESET}")
    print(f"Refresh: {refresh_interval}s | Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Move cursor to line 3
            print('\033[3;0H', end='')
            # Clear from cursor to end of screen
            print('\033[J', end='')
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"{Colors.WHITE}[{timestamp}]{Colors.RESET}\n")
            
            gpus = get_gpu_info()
            for i, gpu in enumerate(gpus):
                gpu_color = get_color(gpu['gpu_util'], 50, 80)
                mem_color = get_color(gpu['mem_util'], 60, 85)
                temp_color = get_color(gpu['temperature'], 60, 80)
                
                # Format GPU name (truncate if needed)
                name = gpu['name'][:20]
                
                # Create status line
                line = f"{Colors.WHITE}{name:20}{Colors.RESET} │ "
                line += f"GPU: {gpu_color}{gpu['gpu_util']:3.0f}%{Colors.RESET} │ "
                line += f"MEM: {mem_color}{gpu['mem_util']:3.0f}%{Colors.RESET} "
                line += f"({mem_color}{gpu['mem_used']/1024:4.1f}{Colors.RESET}/"
                line += f"{gpu['mem_total']/1024:4.1f}GB) │ "
                line += f"TEMP: {temp_color}{gpu['temperature']:2.0f}°C{Colors.RESET}"
                
                if gpu['power_draw'] != 'N/A':
                    line += f" │ PWR: {Colors.YELLOW}{gpu['power_draw']:.0f}W{Colors.RESET}"
                
                print(line)
            
            # Show processes
            processes = get_gpu_processes()
            print(f"\n{Colors.CYAN}{'─' * 70}{Colors.RESET}")
            
            if processes:
                print(f"{Colors.WHITE}Active Processes:{Colors.RESET}")
                for proc in processes[:5]:  # Show top 5
                    print(f"  PID {proc['pid']:7}: {Colors.YELLOW}{proc['memory']:6.2f} GB{Colors.RESET} - {proc['name'][:30]}")
            else:
                print(f"  {Colors.YELLOW}No active GPU processes{Colors.RESET}")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.CYAN}Monitor stopped.{Colors.RESET}")

def display_detailed(refresh_interval=1):
    """Display GPU stats in detailed format with bars"""
    try:
        while True:
            clear_screen()
            
            # Header
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"{Colors.CYAN}{Colors.BOLD}{'═' * 70}{Colors.RESET}")
            print(f"{Colors.CYAN}{Colors.BOLD}{'GPU MONITORING':^70}{Colors.RESET}")
            print(f"{Colors.CYAN}{Colors.BOLD}{timestamp:^70}{Colors.RESET}")
            print(f"{Colors.CYAN}{Colors.BOLD}{'═' * 70}{Colors.RESET}\n")
            
            gpus = get_gpu_info()
            for i, gpu in enumerate(gpus):
                print(f"{Colors.WHITE}{Colors.BOLD}GPU {i}: {gpu['name']}{Colors.RESET}")
                print(f"{Colors.WHITE}{'─' * 50}{Colors.RESET}")
                
                # GPU Utilization
                gpu_color = get_color(gpu['gpu_util'], 50, 80)
                print(f"GPU Usage:    {gpu_color}{gpu['gpu_util']:3.0f}%{Colors.RESET} {draw_bar(gpu['gpu_util'])}")
                
                # Memory Usage
                mem_color = get_color(gpu['mem_util'], 60, 85)
                mem_gb = gpu['mem_used'] / 1024
                total_gb = gpu['mem_total'] / 1024
                print(f"Memory Usage: {mem_color}{gpu['mem_util']:3.0f}%{Colors.RESET} {draw_bar(gpu['mem_util'])} "
                      f"[{mem_color}{mem_gb:.1f}GB{Colors.RESET}/{total_gb:.1f}GB]")
                
                # Temperature
                temp_color = get_color(gpu['temperature'], 60, 80)
                print(f"Temperature:  {temp_color}{gpu['temperature']:3.0f}°C{Colors.RESET} {draw_bar(gpu['temperature'])}")
                
                # Power
                if gpu['power_draw'] != 'N/A' and gpu['power_limit'] != 'N/A':
                    power_percent = (gpu['power_draw'] / gpu['power_limit']) * 100
                    power_color = get_color(power_percent, 60, 85)
                    print(f"Power Draw:   {power_color}{power_percent:3.0f}%{Colors.RESET} {draw_bar(power_percent)} "
                          f"[{power_color}{gpu['power_draw']:.0f}W{Colors.RESET}/{gpu['power_limit']:.0f}W]")
                
                print()
            
            # Processes
            processes = get_gpu_processes()
            print(f"{Colors.WHITE}{Colors.BOLD}Active GPU Processes:{Colors.RESET}")
            print(f"{Colors.WHITE}{'─' * 50}{Colors.RESET}")
            
            if processes:
                print(f"{'PID':>7}  {'Memory':>10}  {'Process'}")
                for proc in processes:
                    print(f"{proc['pid']:>7}  {Colors.YELLOW}{proc['memory']:>8.2f}GB{Colors.RESET}  {proc['name'][:40]}")
            else:
                print(f"{Colors.YELLOW}No active GPU processes{Colors.RESET}")
            
            print(f"\n{Colors.CYAN}{'─' * 70}{Colors.RESET}")
            print(f"{Colors.WHITE}Refreshing every {refresh_interval}s... Press {Colors.RED}Ctrl+C{Colors.WHITE} to exit{Colors.RESET}")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.CYAN}Monitor stopped.{Colors.RESET}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time GPU Monitor')
    parser.add_argument('-i', '--interval', type=float, default=1, 
                       help='Refresh interval in seconds (default: 1)')
    parser.add_argument('-c', '--compact', action='store_true',
                       help='Use compact display mode')
    
    args = parser.parse_args()
    
    try:
        if args.compact:
            display_compact(args.interval)
        else:
            display_detailed(args.interval)
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        sys.exit(1)

if __name__ == '__main__':
    main()