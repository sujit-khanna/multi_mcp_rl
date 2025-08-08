# GRPO Training Tests

This directory contains test scripts for validating and profiling the GRPO training pipeline.

## Test Scripts

### 1. smoke_test.py - Quick Validation Test

A comprehensive smoke test that validates all components work together correctly.

**Features:**
- Tests both LoRA and full fine-tuning modes
- Validates model loading, data loading, trajectory collection, and training
- Verifies gradient computation and parameter updates
- Tests checkpoint saving and loading
- Completes in under 2 minutes

**Usage:**
```bash
# Test LoRA mode (default)
python smoke_test.py

# Test LoRA mode explicitly
python smoke_test.py --mode lora

# Test full fine-tuning mode
python smoke_test.py --mode full

# Skip model tests for faster validation
python smoke_test.py --skip_model_tests

# Enable debug logging
python smoke_test.py --debug
```

**What it tests:**
- ✅ Policy initialization (LoRA/full fine-tuning)
- ✅ Data loading with curriculum sampling
- ✅ Mock trajectory collection
- ✅ GRPO training steps with gradient computation
- ✅ Checkpoint saving and loading

### 2. memory_profile.py - Memory Usage Analysis

Comprehensive memory profiling for different training configurations.

**Features:**
- Tests different batch sizes, sequence lengths, and gradient accumulation
- Measures peak memory usage for each configuration
- Provides GPU-specific recommendations
- Optimized for macOS 48GB unified memory
- Generates detailed recommendations for H100/A100 deployment

**Usage:**
```bash
# Profile LoRA mode (default)
python memory_profile.py

# Profile full fine-tuning mode
python memory_profile.py --mode full

# Test specific sequence lengths
python memory_profile.py --sequence_lengths 512,1024,2048

# Limit batch size range
python memory_profile.py --max_batch_size 4

# Quick profile (fewer configurations)
python memory_profile.py --quick

# Save results to specific file
python memory_profile.py --output_file my_profile.json
```

**What it profiles:**
- 📊 Memory usage across batch sizes (1-8)
- 📊 Memory scaling with sequence lengths (512, 1024, 2048)
- 📊 Gradient accumulation impact (1, 2, 4, 8, 16 steps)
- 📊 GPU-specific recommendations (8GB, 16GB, 24GB, 32GB, 40GB, 48GB, 80GB)

## Environment Requirements

### macOS Testing Environment
- **Memory**: 48GB unified memory
- **Device**: MPS (Metal Performance Shaders) or CPU fallback
- **Python**: 3.9+
- **PyTorch**: Latest with MPS support

### Dependencies
```bash
pip install torch transformers peft bitsandbytes
pip install psutil  # For memory monitoring
pip install yaml numpy
```

## Expected Outputs

### Smoke Test Results
```
🔥 GRPO Training Pipeline Smoke Test
Mode: lora
Skip Model Tests: False
Target: Complete in under 2 minutes
--------------------------------------------------

SMOKE TEST RESULTS
==============================================================
Mode: lora
Device: mps
Elapsed Time: 87.3 seconds
Skip Model Tests: False

Test Results:
  policy_initialization: ✅ PASS
  data_loading: ✅ PASS
  trajectory_collection: ✅ PASS
  grpo_training: ✅ PASS
  checkpointing: ✅ PASS

Summary: 5/5 tests passed
🎉 ALL TESTS PASSED!
```

### Memory Profile Results
```
🔍 GRPO Memory Profiler
Mode: lora
Sequence lengths: [512, 1024, 2048]
Max batch size: 8
Max gradient accumulation: 16
--------------------------------------------------

MEMORY PROFILING RESULTS
================================================================================
Mode: lora
Device: mps
Total System Memory: 48.0 GB
Baseline Memory: 0.15 GB

Configurations Tested: 15
Successful: 12
Failed: 3

📊 SUCCESSFUL CONFIGURATIONS:
--------------------------------------------------------------------------------
  SeqLen    Batch  GradAcc   Peak(GB)  Mem/Sample
--------------------------------------------------------------------------------
     512        1        1       2.34       2.340
     512        2        1       3.12       1.560
     512        4        1       4.67       1.168
    1024        1        1       3.45       3.450
    1024        2        1       5.23       2.615
    2048        1        1       6.78       6.780

🎯 GPU-SPECIFIC RECOMMENDATIONS:
--------------------------------------------------------------------------------
8GB: Max Batch = 2, Seq Len = 512, Memory = 3.1GB (39% util)
16GB: Max Batch = 4, Seq Len = 1024, Memory = 8.7GB (54% util)
24GB: Max Batch = 4, Seq Len = 2048, Memory = 12.3GB (51% util)
32GB: Max Batch = 8, Seq Len = 2048, Memory = 18.9GB (59% util)
40GB: Max Batch = 8, Seq Len = 2048, Memory = 18.9GB (47% util)
48GB: Max Batch = 8, Seq Len = 2048, Memory = 18.9GB (39% util)
80GB: Max Batch = 16, Seq Len = 2048, Memory = 35.2GB (44% util)

💡 OPTIMIZATION TIPS:
  • LoRA mode uses 4-bit quantization to reduce memory usage
  • Consider increasing batch size for better GPU utilization
  • Gradient accumulation can simulate larger batches without memory increase
  • macOS Metal Performance Shaders (MPS) provides unified memory access
  • Memory is shared between CPU and GPU on Apple Silicon
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the right directory
   cd skyrl_tool_agent/training/tests
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:../../"
   ```

2. **Model Download Issues**
   ```bash
   # Pre-download model to avoid timeout
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')"
   ```

3. **Memory Issues on macOS**
   ```bash
   # Monitor memory usage
   python memory_profile.py --quick --max_batch_size 2
   ```

4. **MPS Device Issues**
   ```bash
   # Test MPS availability
   python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
   ```

### Performance Tips

- Use `--quick` flag for faster testing during development
- Use `--skip_model_tests` to avoid model loading during smoke tests
- Monitor system memory with Activity Monitor during profiling
- Close other applications to free up memory during testing

## Integration with CI/CD

These tests can be integrated into continuous integration:

```bash
# Quick validation (under 1 minute)
python smoke_test.py --skip_model_tests

# Memory profile for documentation
python memory_profile.py --quick --output_file ci_memory_profile.json
```

The tests are designed to work reliably in the macOS development environment while providing insights for H100 production deployment.