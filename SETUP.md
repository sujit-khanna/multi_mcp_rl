# Setup Guide for Multi-MCP RL Training

This guide provides step-by-step instructions for setting up the Multi-MCP RL Training environment on a CUDA-enabled Linux machine using `uv`.

## Prerequisites

- **CUDA-enabled Linux machine** with NVIDIA GPU(s)
- **CUDA 12.8** (recommended for SkyRL compatibility)
- **Python 3.12**
- **Git** with LFS support
- **uv package manager** (recommended) or pip

## Quick Setup Script

For automated setup, you can run:

```bash
# Download and run the setup script
curl -sSL https://raw.githubusercontent.com/sujit-khanna/multi_mcp_rl/main/scripts/setup.sh | bash
```

## Manual Setup Instructions

### 1. Install System Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential libnuma-dev git git-lfs curl

# Install NVIDIA drivers if not already installed
# (Skip if drivers are already installed)
sudo apt install -y nvidia-driver-535
```

### 2. Install uv Package Manager

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (restart shell or source)
source ~/.bashrc
```

### 3. Clone Repository

```bash
# Clone the repository
git clone https://github.com/sujit-khanna/multi_mcp_rl.git
cd multi_mcp_rl
```

### 4. Set Up Python Environment

```bash
# Create Python 3.12 virtual environment
uv venv --python 3.12

# Activate virtual environment
source .venv/bin/activate
```

### 5. Install SkyRL Framework

```bash
# Clone SkyRL repository
git clone https://github.com/novasky-ai/SkyRL.git

# Install SkyRL core training framework
cd SkyRL/skyrl-train
uv pip install -e .

# Install SkyRL gym (optional but recommended)
cd ../skyrl-gym  
uv pip install -e .

# Return to project root
cd ../..
```

### 6. Install Project Dependencies

```bash
# Install all project requirements
uv pip install -r requirements.txt

# Install additional CUDA-specific packages (if needed)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 7. Environment Variables Setup

Create a `.env` file in the project root with your API keys:

```bash
# Create .env file
cat > .env << 'EOF'
# Required API Keys
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
EOF
```

### 8. Configure Ray (Required for SkyRL)

```bash
# Set Ray runtime environment hook
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# Add to shell profile for persistence
echo "export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook" >> ~/.bashrc
```

### 9. Verify Installation

```bash
# Run the comprehensive setup validation
python training/tests/run_comprehensive_tests.py

# Or run a quick verification
python -c "
import torch
import transformers
import peft
import accelerate
import skyrl_gym
print('✅ Core packages installed successfully')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"
```

### 10. Initialize Ray Cluster (For Multi-GPU Training)

```bash
# Start Ray head node
ray start --head

# Verify Ray is running
ray status
```

## Training Data Setup

Ensure training data is in the correct location:

```bash
# Primary locations (choose one)
# Option 1: Processed data
data/processed/train.json
data/processed/validation.json

# Option 2: Input data  
data/inputs/train.json
data/inputs/validation.json
```

## MCP Server Setup

The project uses MCP servers located in `mcp_tools/limited/`. Ensure they're executable:

```bash
chmod +x mcp_tools/limited/*.py
```

## Quick Start Training

### GPU Training (Recommended)

```bash
# Single GPU training
./training/scripts/launch_real_env_training.sh

# Multi-GPU training (DeepSpeed)
./training/scripts/launch_distributed.sh
```

### CPU Training (Development/Testing)

```bash
./training/scripts/launch_real_env_cpu.sh
```

## Configuration Files

Key configuration files to customize:

- `training/configs/training_config_qwen3_0.6b.yaml` - Training hyperparameters
- `training/configs/grpo_config_fixed.yaml` - GRPO algorithm settings
- `training/configs/model_config_qwen3_0.6b.yaml` - Model configuration
- `training/configs/deepspeed_config.json` - Multi-GPU training settings

## Hardware Requirements

### Minimum Requirements

- **RAM**: 16GB system RAM
- **GPU**: 8GB VRAM (for LoRA training)
- **Storage**: 50GB free space

### Recommended Requirements

- **RAM**: 32GB+ system RAM
- **GPU**: 2x A100 40GB or equivalent (for full fine-tuning)
- **Storage**: 200GB+ SSD storage

## Memory Optimization Tips

### For Limited GPU Memory

```bash
# Use LoRA training instead of full fine-tuning
./training/scripts/launch_qwen3_training.sh

# Reduce batch size in config files
# Edit training/configs/training_config_qwen3_0.6b.yaml:
# batch_size: 1
# gradient_accumulation_steps: 8
```

### For CPU-Only Training

```bash
# Force CPU usage
export DEVICE_TYPE="cpu"
export DISABLE_BITSANDBYTES=1
./training/scripts/launch_real_env_cpu.sh
```

## Monitoring and Logging

Training progress can be monitored via:

- **WandB Dashboard**: Automatically logged if WANDB_API_KEY is set
- **Weave UI**: For detailed trajectory logging
- **Local logs**: Saved in `outputs/` directory
- **TensorBoard**: `tensorboard --logdir outputs/`

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size and enable gradient checkpointing
# Edit training configs:
# - batch_size: 1
# - gradient_checkpointing: true
```

#### SkyRL Import Errors
```bash
# Ensure SkyRL is properly installed
cd SkyRL/skyrl-train && uv pip install -e . --force-reinstall
```

#### Ray Connection Issues
```bash
# Restart Ray cluster
ray stop
ray start --head
```

#### MCP Server Connection Failures
```bash
# Check API keys in .env file
# Test individual servers:
python mcp_tools/limited/fmp_limited_server.py
```

### Performance Optimization

```bash
# Enable mixed precision training
export ACCELERATE_MIXED_PRECISION="bf16"

# Use multiple GPUs with DeepSpeed
./training/scripts/launch_distributed.sh

# Enable gradient checkpointing for memory efficiency
# (configured in training configs)
```

## Development Commands

### Testing
```bash
# Run comprehensive tests
python training/tests/run_comprehensive_tests.py

# Run specific component tests
python training/tests/test_mcp_integration.py
python training/tests/test_environment_adapter.py
```

### Code Formatting
```bash
# Format code using SkyRL standards
cd SkyRL && ./format.sh
```

### Adding New Tools
1. Create MCP server in `mcp_tools/limited/`
2. Add to `environments/simple_shared_manager.py`
3. Update tool definitions in training data
4. Test integration with `tests/test_mcp_integration.py`

## Support and Documentation

- **Project Documentation**: See `README.md` and `CLAUDE.md`
- **SkyRL Documentation**: https://skyrl.readthedocs.io/
- **Training Logs**: Check `outputs/` directory
- **Debug Mode**: Add `--debug` flag to training scripts

## Next Steps

After successful setup:

1. **Verify training data format** matches expected structure
2. **Test MCP server connectivity** with your API keys
3. **Run a small training job** to validate the setup
4. **Monitor resource usage** and adjust batch sizes accordingly
5. **Set up monitoring dashboards** (WandB/Weave) for production runs

For any issues, check the logs in `outputs/` directory or run the comprehensive test suite for detailed diagnostics.