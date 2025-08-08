# SkyRL GRPO Training Module

This module provides complete Group Relative Policy Optimization (GRPO) training infrastructure for fine-tuning Qwen2.5-1.5B-Instruct on multi-turn tool use tasks.

## 📁 Module Structure

```
training/
├── __init__.py                 # Module initialization
├── configs/                    # Configuration files
│   ├── model_config.yaml       # Model and tokenizer settings
│   ├── training_config.yaml    # Training hyperparameters
│   ├── grpo_config.yaml        # GRPO algorithm settings
│   ├── accelerate_config.yaml  # Multi-GPU training config
│   └── deepspeed_config.json   # DeepSpeed ZeRO-3 configuration
├── core/                       # Core training components
│   ├── qwen_policy.py          # Qwen model wrapper with policy methods
│   ├── grpo_trainer.py         # Main GRPO training logic
│   ├── trajectory_buffer.py    # Episode collection and buffering
│   └── distributed_utils.py    # Multi-GPU utilities
├── data/                       # Data loading and processing
│   ├── trajectory_collector.py # Collect rollouts from environment
│   ├── data_loader.py          # Streaming data loader
│   └── preprocessing.py        # Data preprocessing utilities
├── evaluation/                 # Model evaluation
│   ├── online_evaluator.py     # Evaluate during training
│   └── checkpoint_evaluator.py # Evaluate saved checkpoints
├── scripts/                    # Training and evaluation scripts
│   ├── train_grpo.py           # Main training script
│   ├── collect_trajectories.py # Collect episodes for analysis
│   ├── evaluate_checkpoint.py  # Evaluate specific checkpoint
│   └── launch_distributed.py   # Launch distributed training
└── utils/                      # Utility functions
    ├── logging_utils.py        # Logging and monitoring
    ├── checkpoint_utils.py     # Model checkpointing
    └── monitoring.py           # Training metrics and monitoring
```

## 🎯 Training Modes

### LoRA Mode (Development/Testing)
- **Hardware**: Single A100 40GB
- **Features**: 4-bit quantization + LoRA adapters
- **Use Case**: Fast iteration and development

### Full Fine-tuning Mode (Production)
- **Hardware**: 2x A100 40GB minimum
- **Features**: BF16 precision, DeepSpeed ZeRO-3
- **Use Case**: Production-quality model training

## 🚀 Key Features

### GRPO Algorithm Implementation
- **Group Relative Rewards**: Compare multiple rollouts per task
- **KL Divergence Penalty**: Prevent policy drift from reference model
- **GAE Advantage Estimation**: Gamma=0.99, Lambda=0.95
- **Reference Policy Updates**: Every 10k steps for stability

### Integration with Existing Infrastructure
- **MCPToolEnvironment**: Direct integration with SkyRL environment
- **Shared Tool Manager**: Single initialization across all processes
- **Async Tool Execution**: Proper handling of MCP async operations
- **Streaming Data**: Memory-efficient loading of large datasets

### Performance Optimizations
- **DeepSpeed ZeRO-3**: Distributed training with parameter offloading
- **Gradient Checkpointing**: Reduce memory usage during backprop
- **Flash Attention**: Efficient attention computation
- **Connection Pooling**: Reuse MCP server connections

## 📊 Expected Model Behavior

The trained model will produce structured outputs with reasoning and tool calls:

```
<think>
I need to find Apple's current stock price. I'll use the FMP financial tool to get a real-time quote for AAPL ticker symbol.
</think>

<tool_call>{"name": "fmp_get_quote", "arguments": {"symbol": "AAPL"}}</tool_call>
```

## 🔧 Configuration

All configurations are in YAML/JSON format:
- **model_config.yaml**: Model architecture and generation settings
- **training_config.yaml**: Learning rates, batch sizes, optimization
- **grpo_config.yaml**: GRPO algorithm hyperparameters
- **accelerate_config.yaml**: Multi-GPU and distributed settings
- **deepspeed_config.json**: DeepSpeed ZeRO optimization

## 📈 Success Criteria

Training is considered successful when:
1. ✅ LoRA training runs without OOM on single GPU
2. ✅ Full fine-tuning works on 2x A100 with DeepSpeed
3. ✅ Model achieves >50% task completion rate
4. ✅ Training stability (KL divergence controlled, rewards increase)
5. ✅ Can process 100k+ tasks without memory issues

## 🔍 Implementation Status

- [x] Directory structure created
- [x] Configuration templates ready
- [ ] Core GRPO trainer implementation
- [ ] Data loading and trajectory collection
- [ ] Model policy wrapper with log probability computation
- [ ] Evaluation and monitoring systems
- [ ] Training scripts and launchers

## 📝 Next Steps

1. Implement `core/qwen_policy.py` with log probability computation
2. Build `core/grpo_trainer.py` with GRPO algorithm
3. Create `data/trajectory_collector.py` for episode collection
4. Implement evaluation pipeline
5. Create training scripts with proper error handling
6. Add comprehensive testing and smoke tests