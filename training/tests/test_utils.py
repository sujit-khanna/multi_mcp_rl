#!/usr/bin/env python3
"""
Test Utility Functions for GRPO Training

This script tests all the utility functions to ensure they work correctly
with the training pipeline.
"""

import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

# Import utility modules
from utils.checkpoint_utils import CheckpointManager, save_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.logging_utils import setup_distributed_logging, MetricsAggregator, TrainingLogger, setup_wandb_logging
from utils.monitoring import GPUMonitor, TrainingSpeedTracker, GradientMonitor, EarlyStoppingMonitor, TrainingMonitor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTestModel(nn.Module):
    """Simple model for testing utilities."""
    
    def __init__(self, input_size: int = 100, hidden_size: int = 50, output_size: int = 10):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)


def test_checkpoint_utils():
    """Test checkpoint utilities."""
    logger.info("Testing checkpoint utilities...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test model and optimizer
            model = SimpleTestModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Create checkpoint manager
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=temp_dir,
                max_checkpoints=3,
                save_every_n_steps=5,
            )
            
            # Test saving checkpoints
            for step in range(1, 11):
                if checkpoint_manager.should_save_checkpoint(step, epoch=1):
                    metrics = {
                        "loss": 1.0 / step,  # Decreasing loss
                        "accuracy": step / 10.0,  # Increasing accuracy
                    }
                    
                    checkpoint_path = checkpoint_manager.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=1,
                        step=step,
                        metrics=metrics,
                        is_best=(step == 10),
                    )
                    
                    if checkpoint_path:
                        logger.info(f"Saved checkpoint at step {step}: {checkpoint_path}")
            
            # Test finding latest checkpoint
            latest_checkpoint = checkpoint_manager.find_latest_checkpoint()
            assert latest_checkpoint is not None, "No latest checkpoint found"
            logger.info(f"Latest checkpoint: {latest_checkpoint}")
            
            # Test loading checkpoint
            loaded_state = checkpoint_manager.load_checkpoint(
                checkpoint_path=latest_checkpoint,
                model=model,
                optimizer=optimizer,
            )
            
            assert "step" in loaded_state, "Step not found in loaded state"
            assert "metrics" in loaded_state, "Metrics not found in loaded state"
            logger.info(f"Loaded checkpoint from step {loaded_state['step']}")
            
            # Test best checkpoint
            best_checkpoint = checkpoint_manager.get_best_checkpoint()
            assert best_checkpoint is not None, "No best checkpoint found"
            logger.info(f"Best checkpoint: {best_checkpoint}")
            
            logger.info("✅ Checkpoint utilities test passed")
            return True
            
    except Exception as e:
        logger.error(f"❌ Checkpoint utilities test failed: {e}")
        return False


def test_logging_utils():
    """Test logging utilities."""
    logger.info("Testing logging utilities...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test distributed logging setup
            training_logger = setup_distributed_logging(
                log_dir=temp_dir,
                rank=0,
                world_size=1,
            )
            
            training_logger.info("Test log message")
            
            # Test metrics aggregator
            metrics_aggregator = MetricsAggregator(rank=0, world_size=1)
            
            test_metrics = {"loss": 0.5, "accuracy": 0.8}
            aggregated = metrics_aggregator.aggregate_metrics(test_metrics)
            
            assert "loss" in aggregated, "Loss not in aggregated metrics"
            assert "accuracy" in aggregated, "Accuracy not in aggregated metrics"
            
            # Add metrics to history
            for step in range(10):
                metrics_aggregator.add_metrics({
                    "loss": 1.0 - step * 0.1,
                    "accuracy": step * 0.1,
                }, step)
            
            summary = metrics_aggregator.get_metrics_summary()
            assert "loss" in summary, "Loss not in summary"
            assert "accuracy" in summary, "Accuracy not in summary"
            
            # Test training logger
            metrics_file = Path(temp_dir) / "metrics.json"
            training_logger_obj = TrainingLogger(
                logger=training_logger,
                metrics_aggregator=metrics_aggregator,
                log_every_n_steps=2,
                metrics_file=metrics_file,
            )
            
            # Simulate training steps
            model = SimpleTestModel()
            for step in range(5):
                training_logger_obj.log_training_step(
                    step=step,
                    epoch=1,
                    metrics={"loss": 1.0 - step * 0.1, "accuracy": step * 0.1},
                    model=model,
                )
            
            # Check if metrics file was created
            assert metrics_file.exists(), "Metrics file not created"
            
            logger.info("✅ Logging utilities test passed")
            return True
            
    except Exception as e:
        logger.error(f"❌ Logging utilities test failed: {e}")
        return False


def test_monitoring():
    """Test monitoring utilities."""
    logger.info("Testing monitoring utilities...")
    
    try:
        # Test GPU monitor
        gpu_monitor = GPUMonitor()
        memory_usage = gpu_monitor.get_memory_usage()
        
        assert "allocated_gb" in memory_usage, "allocated_gb not in memory usage"
        assert "total_gb" in memory_usage, "total_gb not in memory usage"
        
        # Update monitoring
        for _ in range(5):
            gpu_monitor.update()
        
        memory_summary = gpu_monitor.get_memory_summary()
        assert "current_allocated_gb" in memory_summary, "current_allocated_gb not in summary"
        
        # Test speed tracker
        speed_tracker = TrainingSpeedTracker(window_size=10)
        
        for step in range(10):
            speed_metrics = speed_tracker.update(
                batch_size=32,
                num_tokens=1024,
                step_time=0.1 + np.random.random() * 0.05,  # Simulate varying step times
            )
            
            if step > 0:  # Skip first step as it might be incomplete
                assert "steps_per_second" in speed_metrics, "steps_per_second not in metrics"
                assert "tokens_per_second" in speed_metrics, "tokens_per_second not in metrics"
        
        # Test gradient monitor
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters())
        gradient_monitor = GradientMonitor(window_size=10)
        
        # Simulate training step with gradients
        for step in range(5):
            # Forward pass
            x = torch.randn(32, 100)
            y = torch.randn(32, 10)
            output = model(x)
            loss = nn.MSELoss()(output, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Monitor gradients
            grad_metrics = gradient_monitor.update(model, max_grad_norm=1.0)
            
            assert "grad_norm" in grad_metrics, "grad_norm not in metrics"
            assert "param_norm" in grad_metrics, "param_norm not in metrics"
            
            optimizer.step()
        
        # Check for gradient issues
        gradient_issues = gradient_monitor.detect_gradient_issues()
        # Should be empty for this simple case
        assert isinstance(gradient_issues, list), "gradient_issues should be a list"
        
        # Test early stopping monitor
        early_stopping = EarlyStoppingMonitor(
            patience=3,
            metric_name="loss",
            mode="min",
        )
        
        # Simulate improving then plateauing loss
        loss_values = [1.0, 0.8, 0.6, 0.5, 0.5, 0.5, 0.5]  # Plateaus after step 3
        
        should_stop = False
        for step, loss_val in enumerate(loss_values):
            result = early_stopping.update(
                metrics={"loss": loss_val},
                step=step,
                model_state=model.state_dict(),
            )
            
            if result["should_stop"]:
                should_stop = True
                break
        
        assert should_stop, "Early stopping should have triggered"
        logger.info(f"Early stopping triggered at step {step}")
        
        # Test comprehensive training monitor
        training_monitor = TrainingMonitor(
            speed_window_size=10,
            gradient_window_size=10,
            early_stopping_config={
                "patience": 5,
                "metric_name": "loss",
                "mode": "min",
            },
        )
        
        # Simulate training steps
        for step in range(10):
            metrics = {
                "loss": 1.0 - step * 0.05,  # Slowly decreasing
                "accuracy": step * 0.1,
            }
            
            # Create mock gradients
            x = torch.randn(32, 100)
            y = torch.randn(32, 10)
            output = model(x)
            loss = nn.MSELoss()(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            
            monitoring_results = training_monitor.update(
                step=step,
                epoch=1,
                metrics=metrics,
                model=model,
                batch_size=32,
                num_tokens=1024,
                max_grad_norm=1.0,
            )
            
            assert "gpu" in monitoring_results, "gpu not in monitoring results"
            assert "speed" in monitoring_results, "speed not in monitoring results"
            assert "gradients" in monitoring_results, "gradients not in monitoring results"
            
            optimizer.step()
        
        # Get monitoring summary
        summary = training_monitor.get_monitoring_summary()
        assert "total_steps" in summary, "total_steps not in summary"
        
        logger.info("✅ Monitoring utilities test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Monitoring utilities test failed: {e}")
        return False


def test_integration():
    """Test integration of all utilities together."""
    logger.info("Testing utility integration...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup components
            training_logger = setup_distributed_logging(temp_dir)
            checkpoint_manager = CheckpointManager(temp_dir, max_checkpoints=5)
            training_monitor = TrainingMonitor()
            
            # Create model and optimizer
            model = SimpleTestModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Simulate training loop
            for step in range(1, 21):
                # Generate training data
                x = torch.randn(32, 100)
                y = torch.randn(32, 10)
                
                # Forward pass
                output = model(x)
                loss = nn.MSELoss()(output, y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Metrics
                metrics = {
                    "loss": loss.item(),
                    "step": step,
                }
                
                # Update monitoring
                monitoring_results = training_monitor.update(
                    step=step,
                    epoch=1,
                    metrics=metrics,
                    model=model,
                    batch_size=32,
                    num_tokens=1024,
                )
                
                # Log training progress (every 5 steps)
                if step % 5 == 0:
                    training_logger.info(
                        f"Step {step}: loss={loss.item():.6f}, "
                        f"GPU: {monitoring_results['gpu'].get('allocated_gb', 0):.2f}GB"
                    )
                
                # Save checkpoints
                if checkpoint_manager.should_save_checkpoint(step, epoch=1):
                    checkpoint_manager.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=1,
                        step=step,
                        metrics=metrics,
                    )
                
                # Check early stopping (if configured)
                if training_monitor.should_stop_training():
                    logger.info(f"Early stopping at step {step}")
                    break
            
            # Final summary
            monitoring_summary = training_monitor.get_monitoring_summary()
            logger.info(f"Training completed. Total steps: {monitoring_summary.get('total_steps', 0)}")
            
            # Cleanup
            training_monitor.cleanup()
            
            logger.info("✅ Integration test passed")
            return True
            
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all utility tests."""
    logger.info("🔧 Running GRPO Training Utilities Tests")
    logger.info("="*60)
    
    tests = [
        ("Checkpoint Utils", test_checkpoint_utils),
        ("Logging Utils", test_logging_utils),
        ("Monitoring", test_monitoring),
        ("Integration", test_integration),
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            success = test_func()
            results[test_name] = success
            if success:
                passed += 1
                logger.info(f"✅ {test_name} test passed")
            else:
                logger.error(f"❌ {test_name} test failed")
        except Exception as e:
            logger.error(f"💥 {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    total = len(tests)
    logger.info("\n" + "="*60)
    logger.info("UTILITY TESTS SUMMARY")
    logger.info("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL UTILITY TESTS PASSED!")
        logger.info("✅ Training utilities are ready for use")
        return True
    else:
        logger.error(f"⚠️  {total - passed} test(s) failed")
        logger.error("❌ Some utilities need attention")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)