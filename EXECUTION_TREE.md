# Training Execution Tree

This document shows the complete file dependency tree for the training script execution starting from `launch_real_env_cpu.sh`.

## Execution Flow

```
launch_real_env_cpu.sh
└── train_qwen3_grpo_real_env.py (Main Training Script)
    │
    ├── Core Training Components
    │   ├── training/core/qwen_policy_with_value_prompting.py
    │   │   └── training/core/qwen_policy_with_prompting.py
    │   │       └── training/core/qwen_policy.py (Base Policy)
    │   │           ├── configs/model_config_temp.yaml
    │   │           └── configs/training_config_temp.yaml
    │   │
    │   ├── training/core/grpo_trainer_gradient_fix.py (Main Trainer)
    │   │   ├── training/core/grpo_trainer_fixed_ref_policy.py
    │   │   └── training/core/grpo_trainer.py (Base Trainer + Trajectory class)
    │   │
    │   └── training/configs/grpo_config_fixed.yaml (GRPO Algorithm Config)
    │
    ├── Data Collection Components
    │   ├── training/data/trajectory_collector.py (Parallel Rollout Collection)
    │   └── data/processed/train.json (Training Data)
    │
    ├── Environment & Tool Management
    │   ├── environments/mcp_tool_environment.py (Main RL Environment)
    │   │   └── SkyRL/skyrl-gym/skyrl_gym/envs/base_text_env.py (Base Environment)
    │   │
    │   ├── environments/mcp_tool_environment_with_logging.py (Enhanced Environment)
    │   │
    │   ├── environments/simple_shared_manager.py (MCP Tool Manager)
    │   │   ├── environments/retry_utils.py (Retry Logic)
    │   │   └── .env (API Keys)
    │   │
    │   └── MCP Tool Servers (Real Tool Execution)
    │       ├── mcp_tools/limited/fmp_limited_server.py (Financial Data)
    │       ├── mcp_tools/limited/polygon_limited_server.py (Stock Data)
    │       ├── mcp_tools/limited/python_execution_server.py (Code Execution)
    │       ├── mcp_tools/limited/slack_limited_server.py (Communication)
    │       └── mcp_tools/limited/tavily_limited_server.py (Web Search)
    │
    ├── Configuration Files
    │   ├── training/configs/training_config_qwen3_0.6b.yaml (Main Training Config)
    │   ├── training/configs/grpo_config_fixed.yaml (GRPO Settings)
    │   ├── configs/model_config_temp.yaml (Model Settings)
    │   └── configs/training_config_temp.yaml (Training Settings)
    │
    ├── Optional Monitoring
    │   ├── wandb (Experiment Tracking)
    │   └── weave (Trajectory Logging)
    │
    └── Utility Components
        ├── training/utils/logging_utils.py (Logging)
        └── training/utils/tool_validator.py (Tool Validation)
```

## Key File Descriptions

### 🚀 **Entry Points**
- `training/scripts/launch_real_env_cpu.sh` - Launch script with environment setup
- `training/scripts/train_qwen3_grpo_real_env.py` - Main training orchestrator

### 🧠 **Core Training Components**
- `training/core/qwen_policy_with_value_prompting.py` - Policy with value head and enhanced prompting
- `training/core/grpo_trainer_gradient_fix.py` - GRPO trainer with gradient fixes
- `training/data/trajectory_collector.py` - Parallel environment rollout collection

### 🌍 **Environment & Tools**
- `environments/mcp_tool_environment.py` - RL environment for tool use tasks
- `environments/simple_shared_manager.py` - Manages MCP server connections
- `mcp_tools/limited/*.py` - Real tool servers (financial, search, code execution, etc.)

### ⚙️ **Configuration**
- `training/configs/training_config_qwen3_0.6b.yaml` - Training hyperparameters
- `training/configs/grpo_config_fixed.yaml` - GRPO algorithm settings
- `configs/model_config_temp.yaml` - Model configuration
- `.env` - API keys and environment variables

### 📊 **Data & Monitoring**
- `data/processed/train.json` - Training dataset
- `wandb` - Experiment tracking (optional)
- `weave` - Trajectory logging (optional)

### 📦 **External Dependencies**
- `SkyRL/skyrl-gym/skyrl_gym/envs/base_text_env.py` - Base environment class
- Various Python packages (torch, transformers, peft, accelerate, etc.)

## Execution Flow Summary

1. **Launch Script** activates environment and sets variables
2. **Main Trainer** loads configurations and initializes components
3. **Policy & GRPO Trainer** handle model loading and training algorithms
4. **Trajectory Collector** manages parallel environment rollouts
5. **MCP Environment** coordinates with tool manager for real tool execution
6. **Tool Manager** connects to MCP servers for actual tool calls
7. **MCP Servers** execute real tools (financial APIs, web search, code execution)
8. **Training Loop** uses collected trajectories to update the policy

This creates a complete reinforcement learning pipeline with real environment interaction rather than mock data.