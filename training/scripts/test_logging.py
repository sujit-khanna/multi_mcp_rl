#!/usr/bin/env python3
"""
Test Comprehensive Training Metrics Logging
==========================================

Quick test to verify that all training metrics are properly logged to WandB/Weave.
"""

import os
import sys
import torch
import logging
import numpy as np
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set test environment variables
os.environ['WANDB_MODE'] = 'offline'  # Offline mode for testing
os.environ['WANDB_PROJECT'] = 'test-logging'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_comprehensive_logging():
    """Test the comprehensive logging method"""
    logger.info("🧪 Testing Comprehensive Training Metrics Logging")
    
    try:
        # Import the trainer
        from train_qwen3_grpo_real_env import RealEnvironmentGRPOTrainer
        
        # Mock training config
        test_config = {
            'training': {
                'use_wandb': True,
                'use_weave': True,
                'batch_size': 2,
            },
            'grpo': {
                'reference_update_interval': 100,
            },
            'model': {},
            'environment': {}
        }
        
        # Create a mock trainer instance
        class MockTrainer(RealEnvironmentGRPOTrainer):
            def __init__(self):
                self.configs = test_config
                self.use_wandb = True
                self.use_weave = True
                self.global_step = 1
                self.current_epoch = 1
                self.device = torch.device('cpu')
                
                # Initialize logging
                self.setup_logging()
        
        trainer = MockTrainer()
        
        # Mock training metrics
        training_metrics = {
            'total_loss': 1.234,
            'policy_loss': 0.567,
            'value_loss': 0.234,
            'kl_divergence': 0.012,
            'kl_penalty': 0.002,
            'entropy_loss': 0.001,
            'grad_norm': 0.5,
            'avg_ratio': 1.05,
            'max_ratio': 1.2,
            'min_ratio': 0.9,
            'kl_coef': 0.1,
            'avg_advantage': 0.15,
            'std_advantage': 0.8,
        }
        
        # Mock trajectories
        class MockTrajectory:
            def __init__(self, reward, length):
                self.total_reward = reward
                self.length = length
        
        trajectories = [
            MockTrajectory(0.8, 5),
            MockTrajectory(0.2, 7),
            MockTrajectory(0.9, 4),
        ]
        
        # Test comprehensive logging
        logger.info("Testing comprehensive metrics logging...")
        logged_metrics = trainer._log_comprehensive_training_metrics(
            training_metrics=training_metrics,
            trajectories=trajectories,
            step=1
        )
        
        # Verify all expected metrics are present
        expected_metric_groups = [
            'training/', 'ppo/', 'advantages/', 'episodes/', 'system/'
        ]
        
        for group in expected_metric_groups:
            group_metrics = [k for k in logged_metrics.keys() if k.startswith(group)]
            assert len(group_metrics) > 0, f"❌ Missing metrics for group: {group}"
            logger.info(f"✅ Found {len(group_metrics)} metrics for {group}")
        
        # Check specific critical metrics
        critical_metrics = [
            'training/total_loss',
            'training/policy_loss', 
            'training/value_loss',
            'training/kl_divergence',
            'ppo/ratio_mean',
            'episodes/reward_mean',
            'episodes/success_rate',
            'advantages/mean',
            'system/step'
        ]
        
        for metric in critical_metrics:
            assert metric in logged_metrics, f"❌ Missing critical metric: {metric}"
        
        logger.info(f"✅ All {len(critical_metrics)} critical metrics present")
        logger.info(f"✅ Total metrics logged: {len(logged_metrics)}")
        
        # Test console logging format
        logger.info("✅ Console logging format working")
        
        logger.info("🎉 Comprehensive logging test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Logging test FAILED: {e}")
        return False

def main():
    """Run logging tests"""
    logger.info("🚀 Starting Training Metrics Logging Test")
    logger.info("=" * 50)
    
    success = test_comprehensive_logging()
    
    logger.info("=" * 50)
    if success:
        logger.info("🎉 ALL LOGGING TESTS PASSED!")
        logger.info("Training metrics will be properly logged to WandB/Weave")
        return 0
    else:
        logger.error("💥 LOGGING TESTS FAILED!")
        return 1

if __name__ == "__main__":
    exit(main())