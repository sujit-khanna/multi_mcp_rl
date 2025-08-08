#!/usr/bin/env python3
"""
Quick script to verify which training script is being used
and whether real tool execution is happening
"""

import os
import sys
import subprocess

print("=" * 80)
print("VERIFYING REAL ENVIRONMENT TRAINING SETUP")
print("=" * 80)

# Check which training scripts exist
scripts_dir = os.path.dirname(os.path.abspath(__file__))
print(f"\n1. Training scripts in {scripts_dir}:")

for script in sorted(os.listdir(scripts_dir)):
    if script.startswith("train_") and script.endswith(".py"):
        full_path = os.path.join(scripts_dir, script)
        size = os.path.getsize(full_path)
        print(f"   - {script} ({size:,} bytes)")
        
        # Check if it has real environment imports
        with open(full_path, 'r') as f:
            content = f.read()
            if "MCPToolEnvironmentWithLogging" in content:
                print(f"     ✅ Uses MCPToolEnvironmentWithLogging (real tools with logging)")
            elif "MCPToolEnvironment" in content and "TrajectoryCollector" in content:
                print(f"     ✅ Uses MCPToolEnvironment + TrajectoryCollector (real tools)")
            elif "_create_trajectory_from_task" in content:
                print(f"     ⚠️  Uses mock trajectory creation")
            else:
                print(f"     ❓ Unknown trajectory collection method")

# Check launch scripts
print(f"\n2. Launch scripts:")
for script in sorted(os.listdir(scripts_dir)):
    if script.startswith("launch_") and script.endswith(".sh"):
        full_path = os.path.join(scripts_dir, script)
        print(f"\n   {script}:")
        
        # Find which Python script it launches
        with open(full_path, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'python' in line and '.py' in line:
                    print(f"     Launches: {line.strip()}")
                    break

# Check environment
print(f"\n3. Environment check:")
print(f"   - Python: {sys.executable}")
print(f"   - Working directory: {os.getcwd()}")

# Check if .env exists
env_path = "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/.env"
if os.path.exists(env_path):
    print(f"   - ✅ .env file found at {env_path}")
    
    # Count API keys (without showing them)
    with open(env_path, 'r') as f:
        api_keys = 0
        for line in f:
            if '_API_KEY' in line or '_TOKEN' in line:
                api_keys += 1
        print(f"   - Found {api_keys} API keys/tokens")
else:
    print(f"   - ❌ .env file NOT found at {env_path}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
print("\nTo run training with REAL tool execution:")
print("1. Use: ./training/scripts/launch_real_env_training.sh")
print("2. Or directly: python training/scripts/train_qwen3_grpo_real_env.py")
print("\nTo test tool execution:")
print("   python training/scripts/test_real_tool_execution.py")
print("\n" + "=" * 80)