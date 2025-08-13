#!/usr/bin/env python3
"""
Copy Essential Training Files Script
===================================

This script copies all the essential files needed for the training pipeline
to a limited folder, creating a minimal self-contained version.

Based on the execution tree analysis from EXECUTION_TREE.md
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict

# Source and destination paths
SOURCE_ROOT = Path(__file__).parent
DEST_ROOT = Path("/Users/sujitkhanna/Desktop/ongoing_projects/rl_projects/limited_folder")

def ensure_dir(path: Path):
    """Ensure directory exists, create if it doesn't"""
    path.mkdir(parents=True, exist_ok=True)

def copy_file_with_structure(src_file: Path, dest_root: Path, relative_path: str = None):
    """Copy file maintaining directory structure"""
    if not src_file.exists():
        print(f"⚠️  WARNING: Source file does not exist: {src_file}")
        return False
    
    if relative_path:
        dest_file = dest_root / relative_path
    else:
        dest_file = dest_root / src_file.relative_to(SOURCE_ROOT)
    
    ensure_dir(dest_file.parent)
    shutil.copy2(src_file, dest_file)
    print(f"✅ Copied: {src_file.name} -> {dest_file.relative_to(dest_root)}")
    return True

def copy_essential_files():
    """Copy all essential files for the training pipeline"""
    
    print("🚀 Starting Essential Files Copy...")
    print(f"📂 Source: {SOURCE_ROOT}")
    print(f"📁 Destination: {DEST_ROOT}")
    print("=" * 60)
    
    # Ensure destination exists
    ensure_dir(DEST_ROOT)
    
    # Define essential files to copy
    essential_files = {
        # Entry points
        "launch_scripts": [
            "training/scripts/launch_real_env_cpu.sh",
            "training/scripts/launch_real_env_training.sh",
            "training/scripts/train_qwen3_grpo_real_env.py",
        ],
        
        # Core training components
        "core_training": [
            "training/core/__init__.py",
            "training/core/qwen_policy.py",
            "training/core/qwen_policy_with_prompting.py", 
            "training/core/qwen_policy_with_value_prompting.py",
            "training/core/grpo_trainer.py",
            "training/core/grpo_trainer_with_value.py",
            "training/core/grpo_trainer_fixed_ref_policy.py",
            "training/core/grpo_trainer_gradient_fix.py",
        ],
        
        # Data collection
        "data_collection": [
            "training/data/__init__.py",
            "training/data/trajectory_collector.py",
            "training/data/data_loader.py",
        ],
        
        # Environment and tool management
        "environment": [
            "environments/__init__.py",
            "environments/mcp_tool_environment.py",
            "environments/mcp_tool_environment_with_logging.py",
            "environments/simple_shared_manager.py",
            "environments/retry_utils.py",
        ],
        
        # MCP tool servers
        "mcp_servers": [
            "mcp_tools/limited/fmp_limited_server.py",
            "mcp_tools/limited/polygon_limited_server.py", 
            "mcp_tools/limited/python_execution_server.py",
            "mcp_tools/limited/slack_limited_server.py",
            "mcp_tools/limited/tavily_limited_server.py",
        ],
        
        # Configuration files
        "configs": [
            "training/configs/__init__.py",
            "training/configs/training_config_qwen3_0.6b.yaml",
            "training/configs/grpo_config_fixed.yaml",
            "training/configs/model_config_qwen3_0.6b.yaml",
            "configs/model_config_temp.yaml",
            "configs/training_config_temp.yaml",
        ],
        
        # Utilities
        "utils": [
            "training/utils/__init__.py",
            "training/utils/logging_utils.py",
            "training/utils/checkpoint_utils.py",
            "training/utils/monitoring.py",
        ],
        
        # Data files
        "data": [
            "data/processed/train.json",
            "data/inputs/train.json",
            "data/inputs/validation.json",
        ],
        
        # Root files
        "root_files": [
            "requirements.txt",
            "README.md", 
            "CLAUDE.md",
            "SETUP.md",
            "EXECUTION_TREE.md",
            ".gitignore",
        ],
        
        # Training module init files
        "init_files": [
            "training/__init__.py",
        ]
    }
    
    # Copy files by category
    total_files = 0
    copied_files = 0
    
    for category, files in essential_files.items():
        print(f"\n📁 {category.upper().replace('_', ' ')}")
        print("-" * 40)
        
        for file_path in files:
            src_file = SOURCE_ROOT / file_path
            total_files += 1
            
            if copy_file_with_structure(src_file, DEST_ROOT):
                copied_files += 1
    
    # Copy SkyRL components (if they exist)
    print(f"\n📁 SKYRL COMPONENTS")
    print("-" * 40)
    
    skyrl_files = [
        "SkyRL/skyrl-gym/skyrl_gym/__init__.py",
        "SkyRL/skyrl-gym/skyrl_gym/envs/__init__.py", 
        "SkyRL/skyrl-gym/skyrl_gym/envs/base_text_env.py",
        "SkyRL/skyrl-gym/pyproject.toml",
    ]
    
    for file_path in skyrl_files:
        src_file = SOURCE_ROOT / file_path
        total_files += 1
        
        if copy_file_with_structure(src_file, DEST_ROOT):
            copied_files += 1
    
    # Create template .env file
    print(f"\n📁 ENVIRONMENT SETUP")
    print("-" * 40)
    
    env_template_content = """# Required API Keys - Replace with your actual keys
OPENAI_API_KEY=your-openai-key-here
POLYGON_API_KEY=your-polygon-api-key
FMP_API_KEY=your-fmp-api-key
TAVILY_API_KEY=tvly-your-tavily-key
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token

# WandB Configuration (optional)
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=skyrl-tool-training

# Weave Configuration (optional) 
WEAVE_PROJECT=your-org/skyrl-tool-training
"""
    
    env_file = DEST_ROOT / ".env.template"
    with open(env_file, "w") as f:
        f.write(env_template_content)
    print("✅ Created: .env.template")
    
    # Create setup script for the limited folder
    setup_script_content = '''#!/bin/bash
# Setup script for limited training environment

echo "🚀 Setting up Limited Multi-MCP RL Training Environment..."

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install SkyRL components
cd SkyRL/skyrl-gym && pip install -e . && cd ../..

# Copy template env file if .env doesn't exist
if [[ ! -f ".env" ]]; then
    cp .env.template .env
    echo "📝 Please update .env file with your API keys"
fi

echo "✅ Setup complete! Don't forget to:"
echo "1. Update .env with your API keys" 
echo "2. Run: source .venv/bin/activate"
echo "3. Run: ./training/scripts/launch_real_env_cpu.sh"
'''
    
    setup_script = DEST_ROOT / "setup.sh"
    with open(setup_script, "w") as f:
        f.write(setup_script_content)
    setup_script.chmod(0o755)
    print("✅ Created: setup.sh (executable)")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 COPY SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully copied: {copied_files}/{total_files} files")
    print(f"📁 Destination: {DEST_ROOT}")
    
    if copied_files < total_files:
        print(f"⚠️  {total_files - copied_files} files were missing from source")
    
    print(f"\n🚀 To use the limited environment:")
    print(f"   cd {DEST_ROOT}")
    print(f"   ./setup.sh")
    print(f"   # Update .env file with your API keys")
    print(f"   ./training/scripts/launch_real_env_cpu.sh")
    
    return copied_files, total_files

if __name__ == "__main__":
    try:
        copied, total = copy_essential_files()
        print(f"\n✅ Copy operation completed: {copied}/{total} files")
        sys.exit(0 if copied == total else 1)
    except Exception as e:
        print(f"❌ Error during copy operation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)