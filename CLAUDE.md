# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a SkyRL-based reinforcement learning implementation for training language models on multi-turn tool use with MCP (Model Context Protocol) servers. The project uses Group Relative Policy Optimization (GRPO) to fine-tune models for real environment rollouts with actual tool execution.

## Key Commands

### Training Commands

**GPU Training (primary):**
```bash
./training/scripts/launch_real_env_training.sh
```

**CPU Training (for debugging/development):**
```bash
./training/scripts/launch_real_env_cpu.sh
```

**LoRA Training (memory-efficient single GPU):**
```bash
./training/scripts/launch_qwen3_training.sh
```

**Multi-GPU Training (DeepSpeed):**
```bash
./training/scripts/launch_distributed.sh
```

### Testing Commands

**Run comprehensive tests:**
```bash
python training/tests/run_comprehensive_tests.py
```

**Test real tool execution:**
```bash
python training/scripts/test_real_tool_execution.py
```

**Test MCP integration:**
```bash
python training/tests/test_mcp_integration.py
```

## Architecture Overview

### Core Training Pipeline

The training system follows this flow:
1. **Policy** (`training/core/qwen_policy_with_value.py`) - Manages the Qwen model with value head
2. **Environment** (`environments/mcp_tool_environment.py`) - Handles real MCP tool execution
3. **Trajectory Collector** (`training/data/trajectory_collector.py`) - Collects parallel rollouts
4. **GRPO Trainer** (`training/core/grpo_trainer_gradient_fix.py`) - Implements the RL training loop
5. **Tool Manager** (`environments/simple_shared_manager.py`) - Manages MCP server connections

### Key Integration Points

- **SkyRL Framework**: Core RL algorithms are implemented via SkyRL (must be installed separately)
- **MCP Servers**: Located in `../mcp_tools/limited/` relative to this directory
- **Environment Variables**: API keys loaded from parent directory's `.env` file
- **Configuration**: YAML configs in `training/configs/` control training hyperparameters

### Data Flow

1. Training data loaded from `data/processed/train.json` or `data/inputs/train.json`
2. Tasks sent to parallel environments for trajectory collection
3. Real MCP tools execute via shared tool manager
4. Trajectories used to update policy via GRPO algorithm
5. Reference policy updated via EMA every 100 steps

## Important Configuration Files

- `training/configs/training_config_qwen3_0.6b.yaml` - Main training hyperparameters
- `training/configs/grpo_config_fixed.yaml` - GRPO algorithm settings
- `training/configs/deepspeed_config.json` - Multi-GPU training configuration

## Environment Setup Requirements

Before running training:
1. SkyRL must be installed (from https://github.com/Sky-T/SkyRL)
2. API keys must be in parent directory's `.env` file
3. MCP servers must be accessible in `../mcp_tools/limited/`
4. Training data must be in proper JSON format

## Common Debugging Approaches

For MPS memory issues on macOS:
```bash
export DEVICE_TYPE="cpu"
export DISABLE_BITSANDBYTES=1
```

For CUDA out of memory:
- Reduce batch_size in training config
- Enable gradient_checkpointing in model config
- Use LoRA instead of full fine-tuning

For detailed debugging:
```bash
export PYTHONPATH="$(pwd):$(pwd)/.."
export CUDA_LAUNCH_BLOCKING=1
python training/scripts/train_qwen3_grpo_real_env.py --debug
```

## Critical Implementation Details

- The system uses real environment rollouts, not mock data
- Tool execution happens via MCP protocol with retry mechanisms
- Parallel trajectory collection uses asyncio for efficiency
- Value function and reference policy are crucial for training stability
- Gradient clipping and KL penalties prevent policy collapse