# Multi-Turn Multi-MCP RL Training

This repository contains the SkyRL-based training implementation for multi-turn tool use with MCP (Model Context Protocol) servers.

## Overview

This project implements Group Relative Policy Optimization (GRPO) training for fine-tuning language models on multi-turn tool use tasks. It features:

- Real environment rollouts with actual MCP tool execution
- Parallel trajectory collection for efficient training
- Support for multiple MCP servers (financial data, web search, Python execution, etc.)
- Comprehensive logging and evaluation framework

## Key Components

- **Training Pipeline**: GRPO implementation with value function and reference policy updates
- **Environment**: MCPToolEnvironment with real tool execution via MCP servers
- **Data Collection**: Parallel trajectory collector with retry mechanisms
- **Evaluation**: Integration with Weave and WandB for tracking

## Requirements

- Python 3.12+
- PyTorch 2.0+
- See `requirements.txt` for full dependencies

## Setup

### Prerequisites

- Python 3.12+ (recommended)
- CUDA 11.8+ (for GPU training)
- Git with LFS support
- uv package manager (recommended) or pip

### 1. SkyRL Framework Installation

This project requires the SkyRL framework. Follow these steps to install it:

#### Option A: Install from Source (Recommended)
```bash
# Clone the SkyRL repository
git clone https://github.com/Sky-T/SkyRL.git
cd SkyRL

# Install SkyRL core
uv pip install -e .
# OR with pip: pip install -e .

# Install skyagent (VERL implementation) - for advanced multi-GPU training
cd skyagent
uv pip install -e .
# OR with pip: pip install -e .

# Format code (optional, for development)
cd .. && ./format.sh
```

#### Option B: Install from PyPI (if available)
```bash
pip install skyrl
pip install skyrl[skyagent]  # For VERL support
```

### 2. Install Project Dependencies

```bash
# Install project-specific requirements
pip install -r requirements.txt

# Key dependencies installed:
# - torch>=2.0.0 (PyTorch)
# - transformers>=4.40.0 (Hugging Face models)
# - peft>=0.10.0 (LoRA/Parameter Efficient Fine-tuning)
# - accelerate>=0.27.0 (Distributed training)
# - wandb>=0.15.0 (Experiment tracking)
# - weave>=0.50.0 (Evaluation tracking)
```

### 3. Environment Variables Setup

Create a `.env` file in the parent directory with your API keys:

```bash
# Required API Keys
OPENAI_API_KEY=sk-your-openai-key-here
POLYGON_API_KEY=your-polygon-api-key
FMP_API_KEY=your-fmp-api-key
TAVILY_API_KEY=tvly-your-tavily-key
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token

# WandB Configuration (optional)
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=skyrl-tool-training

# Weave Configuration (optional)
WEAVE_PROJECT=your-org/skyrl-tool-training
```

### 4. Training Data Preparation

Ensure you have training data in one of these locations:
```bash
# Primary location
data/processed/train.json
data/processed/validation.json

# Alternative location
data/inputs/train.json
data/inputs/validation.json
```

### 5. Verify Installation

Run the setup validation script:
```bash
python training/validate_setup.py
```

This will check:
- SkyRL installation
- Required dependencies
- API key availability
- MCP server connectivity
- Training data format

## Training

### Quick Start

For GPU training:
```bash
./training/scripts/launch_real_env_training.sh
```

For CPU training (recommended for debugging):
```bash
./training/scripts/launch_real_env_cpu.sh
```

### SkyRL Training Modes

This project supports multiple training configurations optimized for different hardware:

#### 1. LoRA Mode (Single GPU)
```bash
# Uses 4-bit quantization + LoRA adapters
# Memory efficient: ~8-16GB GPU memory
./training/scripts/launch_qwen3_training.sh
```

#### 2. Full Fine-tuning (Multi-GPU)
```bash
# BF16 precision with DeepSpeed ZeRO-3
# Requires: 2x A100 40GB minimum
./training/scripts/launch_distributed.sh
```

#### 3. CPU Mode (Development)
```bash
# For development and debugging
# No GPU required, slower training
./training/scripts/launch_real_env_cpu.sh
```

### Training Configuration

The training uses SkyRL's GRPO (Group Relative Policy Optimization) algorithm with:

- **Value Function Training**: Learns state value estimates
- **Reference Policy Updates**: EMA updates every 100 steps  
- **KL Divergence Penalty**: Prevents policy collapse
- **Advantage Estimation**: GAE with γ=0.99, λ=0.95
- **Gradient Clipping**: Prevents training instabilities

Key hyperparameters in `training/configs/training_config_qwen3_0.6b.yaml`:
```yaml
# GRPO Settings
grpo:
  gamma: 0.99          # Discount factor
  lambda: 0.95         # GAE lambda
  clip_ratio: 0.2      # Policy clipping
  kl_coef: 0.01        # KL penalty coefficient
  value_loss_coef: 0.5 # Value function loss weight

# Training Settings
training:
  learning_rate: 5e-6   # Conservative for stability
  batch_size: 1         # Effective batch via gradient accumulation
  gradient_accumulation_steps: 8
  max_epochs: 3
  warmup_steps: 100
```

## MCP Tool Servers

The training uses real MCP servers for tool execution. Servers are located in `../mcp_tools/limited/` relative to this directory.

## Configuration

Main configuration files:
- `training/configs/training_config_qwen3_0.6b.yaml` - Training hyperparameters
- `training/configs/grpo_config_fixed.yaml` - GRPO algorithm settings

## Monitoring

Training progress can be monitored via:
- WandB dashboard
- Weave UI
- Local logs in `outputs/`

## SkyRL Integration Details

### Environment Integration

This project uses SkyRL's policy-environment integration pattern:

```python
# Real environment rollouts instead of mock data
from environments.mcp_tool_environment import MCPToolEnvironment
from training.data.trajectory_collector import TrajectoryCollector

# Collect real trajectories
collector = TrajectoryCollector(
    policy=policy,
    shared_tool_manager=tool_manager,
    num_parallel_envs=4
)

trajectories = await collector.collect_batch(tasks)
```

### Key SkyRL Components Used

1. **GRPO Trainer**: `training/core/grpo_trainer_gradient_fix.py`
2. **Policy with Value Head**: `training/core/qwen_policy_with_value.py`
3. **Environment Adapter**: Integrates MCP tools with SkyRL
4. **Trajectory Collection**: Parallel rollout collection
5. **Reference Policy Management**: EMA-based updates

### Memory Management

SkyRL integration includes memory optimizations:

```python
# Gradient checkpointing
model.gradient_checkpointing_enable()

# Memory-efficient LoRA
from peft import LoraConfig, get_peft_model
peft_config = LoraConfig(r=16, lora_alpha=32)

# Mixed precision training  
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

## Troubleshooting

### Common Issues

#### 1. SkyRL Import Errors
```bash
# Error: No module named 'skyrl'
# Solution: Install SkyRL properly
cd /path/to/SkyRL && pip install -e .
```

#### 2. MPS Memory Issues (macOS)
```bash
# Error: MPSNDArray total bytes > 2^32
# Solution: Use CPU mode
export DEVICE_TYPE="cpu"
./training/scripts/launch_real_env_cpu.sh
```

#### 3. CUDA Out of Memory
```bash
# Reduce batch size and enable gradient checkpointing
# Edit training/configs/training_config_qwen3_0.6b.yaml:
batch_size: 1
gradient_checkpointing: true
```

#### 4. MCP Server Connection Failures
```bash
# Check API keys in .env file
# Verify internet connectivity
# Test individual servers:
cd mcp_tools/limited && python fmp_limited_server.py
```

#### 5. BitsAndBytes Issues
```bash
# Disable if causing problems
export DISABLE_BITSANDBYTES=1
```

### Debug Mode

Enable detailed logging for debugging:

```bash
export PYTHONPATH="$(pwd):$(pwd)/.."
export CUDA_LAUNCH_BLOCKING=1  # For CUDA debugging
python training/scripts/train_qwen3_grpo_real_env.py --debug
```

### Performance Optimization

1. **Use multiple GPUs**: Configure DeepSpeed in `training/configs/deepspeed_config.json`
2. **Optimize batch size**: Balance memory usage vs. training speed
3. **Enable gradient checkpointing**: Reduces memory at cost of compute
4. **Use mixed precision**: FP16/BF16 for faster training

## Development

### Code Formatting

Follow SkyRL's formatting standards:
```bash
cd /path/to/SkyRL
./format.sh  # Formats entire SkyRL codebase
```

### Testing

Run comprehensive tests:
```bash
python training/tests/run_comprehensive_tests.py
```

### Adding New Tools

1. Create MCP server in `mcp_tools/limited/`
2. Add to `environments/simple_shared_manager.py`
3. Update tool definitions in training data
4. Test integration with `tests/test_mcp_integration.py`

## Contributing

1. Follow SkyRL's coding standards
2. Add comprehensive tests for new features
3. Update documentation
4. Ensure backward compatibility

## License

MIT License - See LICENSE file for details