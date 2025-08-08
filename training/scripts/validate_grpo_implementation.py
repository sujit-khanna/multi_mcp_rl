#!/usr/bin/env python3
"""
GRPO Implementation Validation Script

This script validates that the GRPO training implementation is working correctly
by testing core components individually and running a minimal training loop.
"""

import sys
import json
import logging
import torch
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_policy_initialization():
    """Test if the policy can be initialized and loaded correctly."""
    
    logger.info("=== Testing Policy Initialization ===")
    
    try:
        from core.qwen_policy import QwenPolicy
        
        # Test policy creation
        policy = QwenPolicy(
            model_config_path="configs/model_config_mps.yaml",
            training_config_path="configs/training_config_mps.yaml",
            use_lora=True,
            device="cpu",
            load_in_4bit=False,
        )
        
        logger.info(f"✅ Policy initialized: {policy.get_trainable_parameters():,} trainable parameters")
        
        # Test conversation formatting
        test_messages = [
            {"role": "user", "content": "Find the stock price of AAPL"}
        ]
        
        formatted = policy.format_conversation(test_messages)
        logger.info(f"✅ Conversation formatting works: {len(formatted)} characters")
        
        # Test action generation
        states = [test_messages]
        actions = policy.generate_action(states, max_new_tokens=10)
        logger.info(f"✅ Action generation works: {actions}")
        
        return policy
        
    except Exception as e:
        logger.error(f"❌ Policy initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_trajectory_collection():
    """Test trajectory collection with a minimal environment."""
    
    logger.info("=== Testing Trajectory Collection ===")
    
    try:
        # Mock environment for testing
        class MockEnvironment:
            def __init__(self, task_data):
                self.task_data = task_data
                self.step_count = 0
                
            def step(self, action):
                self.step_count += 1
                reward = 0.1 if "price" in action.lower() or "stock" in action.lower() else 0.0
                done = self.step_count >= 3
                observation = f"Step {self.step_count} completed. Action was: {action[:50]}..."
                return {
                    "observation": observation,
                    "reward": reward,
                    "done": done,
                    "metadata": {"step": self.step_count}
                }
        
        # Test with mock data
        task_data = {
            "task_metadata": {"task_id": "test_001", "complexity": "easy"},
            "prompt": [{"role": "user", "content": "Find AAPL stock price"}],
            "reward_spec": {"ground_truth": {"expected_tools": ["stock_lookup"]}}
        }
        
        # Create mock environment
        env = MockEnvironment(task_data)
        
        # Test environment step
        result = env.step("I need to find the stock price of AAPL")
        logger.info(f"✅ Environment step works: reward={result['reward']}, done={result['done']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Trajectory collection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_grpo_trainer():
    """Test GRPO trainer initialization and basic functionality."""
    
    logger.info("=== Testing GRPO Trainer ===")
    
    try:
        from core.grpo_trainer import GRPOTrainer, Trajectory
        from core.qwen_policy import QwenPolicy
        
        # Create minimal policy for testing
        policy = test_policy_initialization()
        if not policy:
            return False
        
        # Create reference policy (copy)
        ref_policy = QwenPolicy(
            model_config_path="configs/model_config_mps.yaml",
            training_config_path="configs/training_config_mps.yaml",
            use_lora=True,
            device="cpu",
            load_in_4bit=False,
        )
        
        # Load GRPO config
        with open("configs/grpo_config_mps.yaml", 'r') as f:
            import yaml
            grpo_config = yaml.safe_load(f)
        
        with open("configs/training_config_mps.yaml", 'r') as f:
            training_config = yaml.safe_load(f)
        
        # Initialize trainer
        trainer = GRPOTrainer(
            policy=policy,
            reference_policy=ref_policy,
            grpo_config=grpo_config,
            training_config=training_config,
            device=torch.device("cpu"),
        )
        
        logger.info("✅ GRPO trainer initialized successfully")
        
        # Test trajectory creation
        test_trajectory = Trajectory(
            task_id="test_001",
            states=[[{"role": "user", "content": "Test task"}]],
            actions=["I need to analyze this task."],
            rewards=[0.1],
            dones=[True]
        )
        
        logger.info("✅ Trajectory creation works")
        
        # Test log probability computation
        log_probs = policy.compute_log_probs(test_trajectory.states, test_trajectory.actions)
        logger.info(f"✅ Log probability computation works: shape={log_probs.shape}, values={log_probs}")
        
        # Test advantage computation
        test_trajectory.log_probs = log_probs
        advantages = trainer._compute_advantages(test_trajectory)
        logger.info(f"✅ Advantage computation works: shape={advantages.shape}, mean={advantages.mean():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GRPO trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_computation():
    """Test that gradients are computed correctly during training."""
    
    logger.info("=== Testing Gradient Computation ===")
    
    try:
        from core.qwen_policy import QwenPolicy
        from core.grpo_trainer import GRPOTrainer, Trajectory
        
        # Create policy and trainer
        policy = QwenPolicy(
            model_config_path="configs/model_config_mps.yaml",
            training_config_path="configs/training_config_mps.yaml",
            use_lora=True,
            device="cpu",
            load_in_4bit=False,
        )
        
        ref_policy = QwenPolicy(
            model_config_path="configs/model_config_mps.yaml",
            training_config_path="configs/training_config_mps.yaml",
            use_lora=True,
            device="cpu",
            load_in_4bit=False,
        )
        
        with open("configs/grpo_config_mps.yaml", 'r') as f:
            import yaml
            grpo_config = yaml.safe_load(f)
        
        with open("configs/training_config_mps.yaml", 'r') as f:
            training_config = yaml.safe_load(f)
        
        trainer = GRPOTrainer(
            policy=policy,
            reference_policy=ref_policy,
            grpo_config=grpo_config,
            training_config=training_config,
            device=torch.device("cpu"),
        )
        
        # Enable training mode first
        policy.enable_training_mode()
        
        # Create test trajectories
        trajectories = [
            Trajectory(
                task_id="test_001",
                states=[[{"role": "user", "content": "Find AAPL stock price"}]],
                actions=["I need to look up the stock price for AAPL."],
                rewards=[0.5],
                dones=[False]
            ),
            Trajectory(
                task_id="test_002", 
                states=[[{"role": "user", "content": "What is the weather?"}]],
                actions=["I'll check the weather for you."],
                rewards=[0.3],
                dones=[True]
            )
        ]
        
        # Get parameter values before training
        param_before = {}
        for name, param in policy.model.named_parameters():
            if param.requires_grad:
                param_before[name] = param.data.clone()
        
        # Run training step
        metrics = trainer.train_step(trajectories)
        
        # Check that parameters changed
        param_changed = False
        for name, param in policy.model.named_parameters():
            if param.requires_grad and name in param_before:
                if not torch.allclose(param.data, param_before[name], atol=1e-6):
                    param_changed = True
                    break
        
        logger.info(f"✅ Training step completed: {metrics}")
        logger.info(f"✅ Parameters changed: {param_changed}")
        
        # Validate metrics
        required_metrics = ["policy_loss", "kl_divergence", "avg_total_reward"]
        for metric in required_metrics:
            if metric in metrics:
                logger.info(f"✅ {metric}: {metrics[metric]:.4f}")
            else:
                logger.warning(f"⚠️ Missing metric: {metric}")
        
        return param_changed and all(m in metrics for m in required_metrics)
        
    except Exception as e:
        logger.error(f"❌ Gradient computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_validation():
    """Run all validation tests."""
    
    logger.info("Starting GRPO Implementation Validation")
    logger.info("=" * 60)
    
    tests = [
        ("Policy Initialization", test_policy_initialization),
        ("Trajectory Collection", test_trajectory_collection),
        ("GRPO Trainer", test_grpo_trainer),
        ("Gradient Computation", test_gradient_computation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"{test_name}: ❌ FAILED with exception: {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:25}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Implementation is ready for GPU training.")
        return True
    else:
        logger.error("⚠️ Some tests failed. Fix issues before proceeding to GPU training.")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)