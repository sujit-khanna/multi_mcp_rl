#!/usr/bin/env python3
"""
Test script to validate critical training fixes
===============================================

This script tests the 4 critical fixes identified:
1. Forced tool calls disabled during RL
2. Trainer update step execution  
3. Episode termination and reward structure
4. WandB metrics logging

Run with minimal data to verify fixes work.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "training"))

# Set environment variables for testing
os.environ['DEVICE_TYPE'] = 'cpu'  # Use CPU for testing
os.environ['DISABLE_BITSANDBYTES'] = '1'
os.environ['WANDB_MODE'] = 'offline'  # Offline mode for testing
os.environ['RL_MODE'] = 'true'
os.environ['FORCE_RATE'] = '0.0'

from training.scripts.train_qwen3_grpo_real_env import RealEnvironmentGRPOTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_minimal_training():
    """Test minimal training run to validate fixes"""
    
    logger.info("🧪 STARTING CRITICAL FIXES VALIDATION")
    
    # Create minimal config for testing
    config_path = "training/configs/training_config_qwen3_0.6b.yaml"
    
    # Override config for minimal test
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Minimal settings for quick test
    config['num_epochs'] = 1
    config['batch_size'] = 1
    config['data_path'] = 'data/processed/train.json'
    
    # Save test config
    test_config_path = "test_config.yaml"
    with open(test_config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Create trainer
        trainer = RealEnvironmentGRPOTrainer(test_config_path)
        
        # Test 1: Verify policy RL mode
        logger.info("🔧 TEST 1: Policy RL Mode Configuration")
        trainer.setup_logging()
        trainer.load_data()
        trainer.setup_models()
        
        # Check policy configuration
        assert hasattr(trainer.policy, 'rl_mode'), "Policy missing rl_mode attribute"
        assert trainer.policy.rl_mode == True, f"Policy RL mode is {trainer.policy.rl_mode}, expected True"
        assert trainer.policy.force_rate == 0.0, f"Policy force rate is {trainer.policy.force_rate}, expected 0.0"
        logger.info("✅ TEST 1 PASSED: Policy configured for RL mode with no forcing")
        
        # Test 2: Verify trainer setup
        logger.info("🔧 TEST 2: Trainer Setup")
        trainer.setup_trainer()
        assert trainer.trainer is not None, "Trainer not initialized"
        assert hasattr(trainer.trainer, 'train_step'), "Trainer missing train_step method"
        logger.info("✅ TEST 2 PASSED: Trainer properly initialized")
        
        # Test 3: Verify environment setup
        logger.info("🔧 TEST 3: Environment Setup")
        await trainer.setup_environment()
        assert trainer.shared_tool_manager is not None, "Tool manager not initialized"
        assert trainer.trajectory_collector is not None, "Trajectory collector not initialized"
        
        # Check max turns are reduced
        test_task = trainer.train_data[0] if trainer.train_data else {
            "task_metadata": {"complexity": "medium"},
            "prompt": [{"role": "user", "content": "Test task"}]
        }
        env = trainer.trajectory_collector.env_factory(test_task)
        assert env.max_turns <= 8, f"Max turns is {env.max_turns}, should be <= 8"
        logger.info(f"✅ TEST 3 PASSED: Environment setup with max_turns={env.max_turns}")
        
        # Test 4: Single trajectory collection
        logger.info("🔧 TEST 4: Trajectory Collection")
        test_tasks = [test_task]
        trajectories = await trainer.collect_trajectories(test_tasks, num_rollouts=1)
        
        assert len(trajectories) > 0, "No trajectories collected"
        traj = trajectories[0]
        assert hasattr(traj, 'rewards'), "Trajectory missing rewards"
        assert hasattr(traj, 'dones'), "Trajectory missing dones"
        assert hasattr(traj, 'states'), "Trajectory missing states"
        assert hasattr(traj, 'actions'), "Trajectory missing actions"
        
        # Check for episode termination
        assert any(traj.dones), f"Episode never terminated: dones={traj.dones}"
        logger.info(f"✅ TEST 4 PASSED: Collected {len(trajectories)} valid trajectories with termination")
        
        # Test 5: Training step execution
        logger.info("🔧 TEST 5: Training Step Execution")
        metrics = trainer.trainer.train_step(trajectories)
        
        assert metrics is not None, "Trainer returned None metrics"
        assert len(metrics) > 0, f"Trainer returned empty metrics: {metrics}"
        assert 'policy_loss' in metrics or 'total_loss' in metrics, f"Missing loss metrics: {list(metrics.keys())}"
        logger.info(f"✅ TEST 5 PASSED: Training step executed with metrics: {list(metrics.keys())}")
        
        logger.info("🎉 ALL CRITICAL FIXES VALIDATED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        logger.error(f"❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            if hasattr(trainer, 'shared_tool_manager') and trainer.shared_tool_manager:
                await trainer.shared_tool_manager.cleanup()
            if hasattr(trainer, 'trajectory_collector') and trainer.trajectory_collector:
                await trainer.trajectory_collector.cleanup()
        except:
            pass
        
        # Remove test config
        if os.path.exists(test_config_path):
            os.remove(test_config_path)

async def main():
    """Main test function"""
    success = await test_minimal_training()
    if success:
        logger.info("🏆 CRITICAL FIXES VALIDATION: SUCCESS")
        return 0
    else:
        logger.error("💥 CRITICAL FIXES VALIDATION: FAILED") 
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)