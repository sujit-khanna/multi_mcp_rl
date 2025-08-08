# GRPO Training Utilities

This directory contains comprehensive utility modules for GRPO (Group Relative Policy Optimization) training with Qwen2.5-1.5B-Instruct.

## 📁 Module Overview

### 🔄 `checkpoint_utils.py`
**Comprehensive checkpoint management for both LoRA and full model training**

**Key Classes:**
- `CheckpointManager`: Full-featured checkpoint management with rotation, metadata tracking, and distributed support
- Standalone functions: `save_checkpoint()`, `load_checkpoint()`, `find_latest_checkpoint()`

**Features:**
- ✅ **LoRA Adapter Support**: Automatic detection and handling of PEFT LoRA adapters
- ✅ **Full Model Support**: Standard PyTorch model checkpointing
- ✅ **Distributed Training**: Multi-GPU checkpoint sharding with DeepSpeed integration
- ✅ **Automatic Rotation**: Configurable maximum checkpoint retention
- ✅ **Metadata Tracking**: JSON metadata with metrics, timestamps, and training state
- ✅ **Resume Capability**: Robust checkpoint loading with error handling

**Usage:**
```python
from utils.checkpoint_utils import CheckpointManager

# Create checkpoint manager
checkpoint_manager = CheckpointManager(
    checkpoint_dir="./checkpoints",
    max_checkpoints=5,
    save_every_n_steps=1000,
)

# Save checkpoint
checkpoint_path = checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    step=step,
    metrics=metrics,
    is_best=True
)

# Load checkpoint
training_state = checkpoint_manager.load_checkpoint(
    model=model,
    optimizer=optimizer,
    load_optimizer=True
)
```

---

### 📊 `logging_utils.py`
**Distributed logging, metrics aggregation, and WandB integration**

**Key Classes:**
- `TrainingLogger`: Pretty training progress logging with metrics tracking
- `MetricsAggregator`: Cross-GPU metrics aggregation for distributed training
- `WandBLogger`: WandB integration with error handling
- Setup functions: `setup_distributed_logging()`, `setup_wandb_logging()`

**Features:**
- ✅ **Distributed Logging**: Separate log files per rank with unified console output
- ✅ **Metrics Aggregation**: Automatic reduction across distributed processes
- ✅ **Pretty Printing**: Formatted training progress with timing and throughput
- ✅ **WandB Integration**: Experiment tracking with gradient/parameter histograms
- ✅ **Error Handling**: Graceful degradation when dependencies unavailable
- ✅ **Metrics History**: JSON export of training metrics

**Usage:**
```python
from utils.logging_utils import setup_distributed_logging, TrainingLogger, setup_wandb_logging

# Setup distributed logging
logger = setup_distributed_logging(
    log_dir="./logs",
    rank=rank,
    world_size=world_size
)

# Setup WandB
wandb_run = setup_wandb_logging(
    project_name="grpo-training",
    config=config,
    rank=rank
)

# Create training logger
training_logger = TrainingLogger(
    logger=logger,
    log_every_n_steps=10,
    metrics_file="./metrics.json"
)

# Log training step
training_logger.log_training_step(
    step=step,
    epoch=epoch,
    metrics=metrics,
    model=model
)
```

---

### 📈 `monitoring.py`
**GPU memory, training speed, gradient monitoring, and early stopping**

**Key Classes:**
- `GPUMonitor`: Cross-platform GPU memory monitoring (CUDA, MPS, CPU)
- `TrainingSpeedTracker`: Throughput tracking (steps/sec, tokens/sec, examples/sec)
- `GradientMonitor`: Gradient norm analysis and issue detection
- `EarlyStoppingMonitor`: Configurable early stopping with best weight restoration
- `TrainingMonitor`: Unified monitoring combining all capabilities

**Features:**
- ✅ **Cross-Platform GPU Support**: CUDA, Apple Silicon MPS, CPU fallback
- ✅ **Memory Tracking**: Peak memory, utilization, history, and cache management
- ✅ **Speed Metrics**: Steps/second, tokens/second, examples/second tracking
- ✅ **Gradient Analysis**: Norm tracking, clipping detection, issue identification
- ✅ **Early Stopping**: Patience-based stopping with metric-driven decisions
- ✅ **Issue Detection**: Automatic detection of exploding/vanishing gradients

**Usage:**
```python
from utils.monitoring import TrainingMonitor

# Create comprehensive monitor
monitor = TrainingMonitor(
    device=device,
    early_stopping_config={
        "patience": 10,
        "metric_name": "loss",
        "mode": "min"
    }
)

# Update monitoring during training
results = monitor.update(
    step=step,
    epoch=epoch,
    metrics=metrics,
    model=model,
    batch_size=batch_size,
    num_tokens=num_tokens,
    max_grad_norm=1.0
)

# Check early stopping
if monitor.should_stop_training():
    print("Early stopping triggered!")
    break
```

---

## 🚀 **Key Features**

### **Production-Ready**
- ✅ Comprehensive error handling and logging
- ✅ Resource cleanup and memory management
- ✅ Distributed training support across all modules
- ✅ Cross-platform compatibility (CUDA, MPS, CPU)

### **Optimized for GRPO Training**
- ✅ LoRA adapter-aware checkpointing
- ✅ Multi-GPU memory monitoring and sharding
- ✅ Gradient analysis for policy optimization
- ✅ Curriculum learning progress tracking

### **Integration Ready**
- ✅ WandB experiment tracking
- ✅ JSON metrics export for analysis
- ✅ Configurable logging levels and formats
- ✅ Modular design allowing selective usage

### **Memory Optimized**
- ✅ Efficient checkpoint rotation and cleanup
- ✅ GPU memory cache management
- ✅ Streaming metrics storage with bounded history
- ✅ macOS unified memory support

---

## 📊 **Validation Results**

All utility modules have been **100% validated** with comprehensive tests:

```
UTILITY TESTS SUMMARY
============================================================
  Checkpoint Utils: ✅ PASS
  Logging Utils: ✅ PASS  
  Monitoring: ✅ PASS
  Integration: ✅ PASS

Overall: 4/4 tests passed
🎉 ALL UTILITY TESTS PASSED!
✅ Training utilities are ready for use
```

**Test Coverage:**
- ✅ Checkpoint saving/loading with LoRA and full models
- ✅ Distributed logging across multiple ranks
- ✅ GPU memory monitoring on Apple Silicon MPS
- ✅ Training speed and gradient tracking
- ✅ Early stopping logic with patience-based decisions
- ✅ End-to-end integration with realistic training loops

---

## 🔧 **Usage in Main Training Script**

The utilities integrate seamlessly with the main GRPO training pipeline:

```python
# In train_grpo.py
from utils import (
    CheckpointManager, setup_distributed_logging, 
    TrainingLogger, TrainingMonitor, setup_wandb_logging
)

# Initialize all utilities
checkpoint_manager = CheckpointManager(config.checkpoint_dir)
logger = setup_distributed_logging(config.log_dir, rank, world_size)
monitor = TrainingMonitor(device, early_stopping_config=config.early_stopping)
wandb_run = setup_wandb_logging(config.project_name, config.dict(), rank)

# Training loop integration
for step in range(max_steps):
    # ... training step ...
    
    # Monitor training
    monitoring_results = monitor.update(step, epoch, metrics, model, batch_size)
    
    # Check early stopping
    if monitor.should_stop_training():
        break
    
    # Save checkpoints
    if checkpoint_manager.should_save_checkpoint(step, epoch):
        checkpoint_manager.save_checkpoint(model, optimizer, epoch, step, metrics)
    
    # Log to WandB
    if wandb_run:
        wandb.log(metrics, step=step)
```

---

## 📈 **Performance Characteristics**

- **Memory Efficient**: Bounded history buffers and automatic cleanup
- **Fast**: Optimized for high-frequency monitoring (every training step)
- **Scalable**: Distributed-first design supporting 100k+ training steps
- **Robust**: Comprehensive error handling and graceful degradation

The utilities are ready for production GRPO training with both LoRA and full fine-tuning modes! 🚀