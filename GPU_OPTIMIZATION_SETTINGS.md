# GPU Optimization Settings for A100 40GB

## Updated Batch Size Configuration

### Previous Settings (Conservative)
- **Batch size**: 4
- **Gradient accumulation**: 2
- **Effective batch size**: 8
- **Eval batch size**: 8
- **Rollout batch size**: 4
- **Parallel environments**: 1

### New Settings (Optimized for A100 40GB)
- **Batch size**: 16
- **Gradient accumulation**: 1 (no accumulation needed)
- **Effective batch size**: 16
- **Eval batch size**: 32
- **Rollout batch size**: 16
- **Parallel environments**: 8

## Additional Optimizations

### Data Loading
- **Workers**: 8 (increased from 4)
- **Prefetch factor**: 4 (increased from 2)
- **Cache size**: 5000 (increased from 2000)
- **Pin memory**: Enabled for faster GPU transfer

### Training Configuration
- **Group size**: 4 (increased from 2)
- **Episodes per update**: 8 (increased from 2)
- **Max rollout length**: 20 (increased from 10)

## Expected Benefits

1. **Higher GPU Utilization**: Should see 70-90% GPU usage vs previous 20-40%
2. **Faster Training**: ~2-4x speedup due to larger batches and parallel collection
3. **Better Gradient Estimates**: Larger batches provide more stable gradients
4. **More Efficient Memory Use**: A100's 40GB memory is now properly utilized

## Running the Optimized Training

```bash
# Start training with new settings
./training/scripts/launch_real_env_gpu.sh

# Monitor GPU usage (should see higher utilization)
watch -n 1 nvidia-smi

# Watch training logs
tail -f outputs/real-env-grpo-gpu-*/training.log | grep -E "batch|GPU|memory"
```

## Monitoring Expected Metrics

- **GPU Utilization**: 70-90% (up from ~20%)
- **GPU Memory**: 25-35GB used (up from ~10GB)
- **Training Speed**: 2-4x faster iterations
- **Batch Processing**: 16 samples/batch vs 4

## Fine-tuning Tips

If you encounter OOM errors:
- Reduce batch_size to 12 or 8
- Enable gradient_accumulation_steps: 2
- Reduce rollout_parallel_envs to 4

If GPU utilization is still low:
- Increase batch_size to 20 or 24
- Increase rollout_parallel_envs to 12
- Increase num_workers to 12

## Notes

These settings are specifically optimized for:
- NVIDIA A100 40GB GPU
- Qwen 0.5B model with LoRA
- Mixed precision training (FP16)
- Real environment rollouts with MCP tools