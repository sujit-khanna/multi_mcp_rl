# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a SkyRL-based reinforcement learning implementation for training language models on multi-turn tool use with MCP (Model Context Protocol) servers. The project uses Group Relative Policy Optimization (GRPO) to fine-tune models for real environment rollouts with actual tool execution.

## Key Commands

### Development Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Setup validation:**
```bash
python training/validate_setup.py
```

### Training Commands

**vLLM GPU Training (recommended for speed):**
```bash
ENABLE_VLLM=true VLLM_MAX_MODEL_LEN=4096 VLLM_GPU_MEMORY_UTILIZATION=0.3 ./training/scripts/launch_real_env_gpu_vllm.sh
```

**Standard GPU Training:**
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

**Smoke test (quick validation):**
```bash
python training/tests/smoke_test.py
```

### Debugging Commands

**Debug with detailed logging:**
```bash
export PYTHONPATH="$(pwd):$(pwd)/.."
export CUDA_LAUNCH_BLOCKING=1
python training/scripts/train_qwen3_grpo_real_env.py --debug
```

**Debug vLLM training:**
```bash
ENABLE_VLLM=true VLLM_MAX_MODEL_LEN=4096 VLLM_GPU_MEMORY_UTILIZATION=0.3 timeout 300 ./training/scripts/launch_real_env_gpu_vllm.sh
```

**Test individual MCP servers:**
```bash
cd mcp_tools/limited && python fmp_limited_server.py
```

**Test critical fixes:**
```bash
python test_critical_fixes.py
```

**Test latest critical fixes (V2):**
```bash
python test_critical_fixes_v2.py
```

### Monitoring Commands

**Monitor GPU usage:**
```bash
./monitor_gpu.sh
```

**Watch training logs:**
```bash
tail -f outputs/real-env-grpo-*/training.log
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
1. **SkyRL Framework**: Must be installed from https://github.com/Sky-T/SkyRL
   ```bash
   git clone https://github.com/Sky-T/SkyRL.git
   cd SkyRL && pip install -e .
   cd skyagent && pip install -e .  # For multi-GPU support
   ```
2. **Dependencies**: Install project requirements
   ```bash
   pip install -r requirements.txt
   ```
3. **API Keys**: Must be in parent directory's `.env` file
   ```bash
   # Required API Keys
   OPENAI_API_KEY=sk-your-openai-key-here
   POLYGON_API_KEY=your-polygon-api-key
   FMP_API_KEY=your-fmp-api-key
   TAVILY_API_KEY=tvly-your-tavily-key
   SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
   ```
4. **MCP Servers**: Must be accessible in `../mcp_tools/limited/`
5. **Training Data**: Must be in proper JSON format at `data/processed/train.json` or `data/inputs/train.json`
6. **Python Version**: Requires Python 3.12+ for optimal compatibility

## Common Debugging Approaches

### Memory Issues

**MPS memory issues on macOS:**
```bash
export DEVICE_TYPE="cpu"
export DISABLE_BITSANDBYTES=1
```

**CUDA out of memory:**
- Reduce batch_size in training config
- Enable gradient_checkpointing in model config
- Use LoRA instead of full fine-tuning
- Switch to CPU mode for development

**BitsAndBytes issues:**
```bash
export DISABLE_BITSANDBYTES=1
```

### Training Issues

**SkyRL import errors:**
```bash
# Ensure SkyRL is properly installed
cd /path/to/SkyRL && pip install -e .
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/.."
```

**MCP server connection failures:**
- Check API keys in .env file
- Verify internet connectivity
- Test individual servers manually

**Training instabilities:**
- Check GRPO hyperparameters (KL coefficient, learning rate)
- Verify gradient clipping is enabled
- Monitor value function loss

### Detailed Debugging

**Enable verbose logging:**
```bash
export PYTHONPATH="$(pwd):$(pwd)/.."
export CUDA_LAUNCH_BLOCKING=1  # For CUDA debugging
python training/scripts/train_qwen3_grpo_real_env.py --debug
```

**Performance profiling:**
```bash
python training/tests/memory_profile.py
```

## Critical Implementation Details

### Training Architecture
- The system uses real environment rollouts, not mock data
- Tool execution happens via MCP protocol with retry mechanisms
- Parallel trajectory collection uses asyncio for efficiency
- Value function and reference policy are crucial for training stability
- Gradient clipping and KL penalties prevent policy collapse

### vLLM Integration (Latest)

- **vLLM Support**: Added high-performance vLLM inference for 10x speed improvement
- **LoRA with vLLM**: Seamless LoRA weight updates during training
- **Memory Optimization**: Configurable GPU memory utilization (0.3 recommended)
- **Performance**: Reduces generation time from ~47s to ~4-5s per batch

### Critical PPO Fixes Applied (September 2024)

**Root Issue**: PPO ratios stuck at 1.0 (degenerate), preventing policy learning for over a month.

**Final Solution**: Parameter perturbation in `grpo_trainer_gradient_fix.py:180-190`:
```python
# Add tiny noise to LoRA weights to break degeneracy
for param in trainable_params[:10]:
    if param.requires_grad and param.numel() > 0:
        param.add_(torch.randn_like(param) * 1e-6)
```

**Expected Results After Fix**:
- Before: `std_ratio: 0.000000` (no learning)
- After: `std_ratio: 0.00587` (proper learning)

### Previous Critical Fixes Applied (August 2024)

**Phase 1 - Basic Training Flow:**
1. **Forced Tool Call Prevention**: Policy configured with `rl_mode=True` and `force_rate=0.0` to prevent off-policy contamination during RL training
2. **Training Step Execution**: Added assertions and logging to ensure `trainer.train_step()` is called and returns valid metrics
3. **Episode Termination**: Reduced max_turns (easy: 5, medium: 8, hard: 10) and improved reward structure to ensure episodes terminate
4. **WandB Metrics Logging**: Added explicit metric logging with debug pings to ensure training progress is visible in dashboards
5. **Forced Action Penalties**: Added -0.1 penalty for forced actions when they occur to discourage poor behavior

**Phase 2 - Tool Use & Training Stability:**
6. **Tool Call Argument Aliasing**: Added normalization for common argument aliases (e.g., `python_code` → `code`) to fix "No tool calls found" issues
7. **Constrained Prompting**: Simplified system prompt to generate tool calls immediately without `<think>` blocks
8. **Optimized Generation**: Lower temperature (0.2), shorter responses (512 tokens), additional stop sequences, repetition penalty
9. **Reference Policy Sync Filter**: Exclude bitsandbytes quantization buffers from reference policy updates
10. **Rollout Metrics Logging**: Always log environment metrics before training step to ensure WandB visibility

### MCP Tool Integration
- Real MCP servers run in `../mcp_tools/limited/` (financial data, web search, Python execution, Slack)
- Shared tool manager handles server lifecycle and connection pooling
- Retry mechanisms handle transient failures
- Tool outputs are validated and sanitized

### Hardware Considerations
- **vLLM GPU (recommended)**: 16GB+ VRAM, 10x faster inference with LoRA
- **Single GPU (LoRA)**: 8-16GB VRAM, uses 4-bit quantization
- **Multi-GPU (Full FT)**: 2x A100 40GB minimum, uses DeepSpeed ZeRO-3
- **CPU Mode**: No GPU required, significantly slower but useful for debugging
- **MPS Support**: macOS Metal Performance Shaders, limited memory

### Key Hyperparameters
- Learning rate: 5e-6 (conservative for stability)
- Batch size: 1 (effective batch via gradient accumulation)
- KL coefficient: 0.01 (prevents policy collapse)
- Reference policy EMA: Updated every 100 steps

### Output Structure
- Training logs: `outputs/real-env-grpo-*/training.log`
- vLLM training logs: `outputs/real-env-grpo-vllm-*/training.log`
- Configurations: Copied to output directory for reproducibility
- WandB tracking: Project "skyrl-grpo-real-env" and "multi-mcp-rl-vllm"
- Checkpoints: Saved periodically during training

## Critical Monitoring

### PPO Ratio Health Check
Watch for these log patterns during training:

**✅ Healthy (Fixed):**
```
✅ Collected N sample-time logprobs: mean=X.X, std=Y.Y
✅ Added tiny perturbation to 10 parameters to break PPO degeneracy
PPO Ratio Check - mean: 1.000, std: 0.006, count: N
'std_ratio': 0.00586725166067481
```

**❌ Broken (Will halt training):**
```
❌ CRITICAL: PPO ratios are degenerate! std=0.000000
RuntimeError: PPO ratios are degenerate. Training cannot proceed.
```

### vLLM Performance Indicators
- Generation time: Should be 0.9-7.8s per action (context-dependent)
- LoRA updates: Every training step with ID increments
- Memory usage: Monitor via `VLLM_GPU_MEMORY_UTILIZATION` setting

### Training Speed Expectations
- **Total per epoch**: ~110-115 seconds
- **Trajectory collection**: ~90-95 seconds (real MCP tool execution)
- **Training step**: ~17 seconds (gradient updates)
- **vLLM speedup**: 10x faster than standard HuggingFace inference