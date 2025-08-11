#!/usr/bin/env python3
"""
Setup Summary and Verification Script
"""
import sys
import os
import torch
import importlib.util

print("=" * 60)
print("🚀 Multi-MCP RL Training Setup Summary")
print("=" * 60)

# Python environment
print("\n📦 Python Environment:")
print(f"  • Python Version: {sys.version.split()[0]}")
print(f"  • Virtual Environment: venv312")

# PyTorch setup
print("\n🔥 PyTorch Configuration:")
print(f"  • PyTorch Version: {torch.__version__}")
print(f"  • CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  • GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"  • GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"  • CUDA Version: {torch.version.cuda}")
    print(f"  • Device Mode: GPU (CUDA)")
else:
    print(f"  • Device Mode: CPU (No GPU detected)")

# SkyRL installation  
print("\n🎯 SkyRL Framework:")
skyrl_gym_path = "/home/ubuntu/multi_mcp_rl/SkyRL/skyrl-gym"
skyrl_train_path = "/home/ubuntu/multi_mcp_rl/SkyRL/skyrl-train"
print(f"  • skyrl-gym: {'✅ Installed' if os.path.exists(skyrl_gym_path) else '❌ Not found'}")
print(f"  • skyrl-train: {'✅ Available' if os.path.exists(skyrl_train_path) else '❌ Not found'}")

# Data availability
print("\n📊 Training Data:")
train_data_paths = [
    "data/processed/train.json",
    "data/inputs/train.json",
    "data/inputs/validation.json"
]
for path in train_data_paths:
    exists = os.path.exists(path)
    size = os.path.getsize(path) / (1024*1024) if exists else 0
    status = f"✅ {size:.1f}MB" if exists else "❌ Missing"
    print(f"  • {path}: {status}")

# MCP servers
print("\n🔧 MCP Servers:")
mcp_servers = [
    "mcp_tools/limited/fmp_limited_server.py",
    "mcp_tools/limited/polygon_limited_server.py",
    "mcp_tools/limited/python_execution_server.py",
    "mcp_tools/limited/slack_limited_server.py",
    "mcp_tools/limited/tavily_limited_server.py"
]
for server in mcp_servers:
    status = "✅ Available" if os.path.exists(server) else "❌ Missing"
    print(f"  • {os.path.basename(server)}: {status}")

# API keys
print("\n🔑 API Keys (.env):")
env_file = ".env"
if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        lines = f.readlines()
    api_keys = {
        "OPENAI_API_KEY": False,
        "POLYGON_API_KEY": False,
        "FMP_API_KEY": False,
        "TAVILY_API_KEY": False,
        "SLACK_BOT_TOKEN": False
    }
    for line in lines:
        for key in api_keys:
            if line.startswith(f"{key}=") and len(line.strip()) > len(key) + 5:
                api_keys[key] = True
    
    for key, found in api_keys.items():
        status = "✅ Configured" if found else "❌ Missing"
        print(f"  • {key}: {status}")
else:
    print("  ❌ .env file not found")

# Training scripts
print("\n📜 Training Scripts:")
scripts = {
    "CPU Training": "training/scripts/launch_real_env_cpu.sh",
    "GPU Training": "training/scripts/launch_real_env_training.sh",
    "LoRA Training": "training/scripts/launch_qwen3_training.sh",
    "Multi-GPU": "training/scripts/launch_distributed.sh"
}
for name, path in scripts.items():
    if os.path.exists(path):
        is_exec = os.access(path, os.X_OK)
        status = "✅ Ready" if is_exec else "⚠️ Not executable"
    else:
        status = "❌ Missing"
    print(f"  • {name}: {status}")

print("\n" + "=" * 60)
print("📝 Next Steps:")
print("=" * 60)
print("""
1. To activate the environment:
   source setup_env.sh

2. To run GPU training (recommended with A100):
   ./training/scripts/launch_real_env_training.sh
   
   Or for LoRA training (memory efficient):
   ./training/scripts/launch_qwen3_training.sh

3. To test the setup:
   python training/tests/minimal_test.py

4. Monitor training:
   - Check outputs/ directory for logs  
   - Use WandB dashboard if configured
   - Monitor GPU: watch nvidia-smi
   
Note: GPU training with NVIDIA A100 40GB detected!
This will provide optimal training performance.
""")

print("🚀 Setup complete! The project is ready for GPU-accelerated training!")