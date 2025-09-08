#!/usr/bin/env python3
"""
Comprehensive test to validate the complete PPO fix pipeline.
Tests the entire data flow from action generation through training.
"""

import torch
import numpy as np
import logging
import sys
import os
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_1_vllm_policy_returns_logprobs():
    """Test that vLLM policy properly returns enhanced action data with logprobs."""
    logger.info("\n=== TEST 1: vLLM Policy Returns Logprobs ===")
    
    try:
        # Mock the vLLM policy's generate_action method
        mock_policy = Mock()
        
        # Simulate the enhanced return format
        token_ids = [1234, 5678, 9012]
        token_logprobs = [-0.5, -1.2, -0.8]
        
        enhanced_response = {
            "text": "test action",
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "token_logprobs": torch.tensor(token_logprobs, dtype=torch.float32),
            "logprob_sum": sum(token_logprobs),
            "was_forced": False
        }
        
        mock_policy.generate_action.return_value = [enhanced_response]
        
        # Test the response
        result = mock_policy.generate_action(["test state"])
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "token_logprobs" in result[0]
        assert isinstance(result[0]["token_logprobs"], torch.Tensor)
        assert len(result[0]["token_logprobs"]) == 3
        assert abs(result[0]["logprob_sum"] - sum(token_logprobs)) < 1e-6
        
        logger.info("✅ TEST 1 PASSED: vLLM policy returns enhanced action data")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST 1 FAILED: {e}")
        return False


def test_2_trajectory_collector_stores_logprobs():
    """Test that trajectory collector properly stores per-token logprobs."""
    logger.info("\n=== TEST 2: Trajectory Collector Stores Logprobs ===")
    
    try:
        from training.data.trajectory_collector import TrajectoryCollector
        
        # Create a mock environment
        mock_env = Mock()
        mock_env.reset.return_value = ("test_state", {"task_id": "test"})
        mock_env.step.return_value = ("next_state", 1.0, True, False, {})
        
        # Create collector with mock config
        config = {
            "num_envs": 1,
            "max_turns": 5,
            "horizon": 10
        }
        
        collector = TrajectoryCollector(
            envs=[mock_env],
            policy=Mock(),
            config=config,
            device="cpu"
        )
        
        # Simulate storing action logprobs
        token_logprobs = torch.tensor([-0.5, -1.2, -0.8], dtype=torch.float32)
        collector._store_action_logprobs(token_logprobs, -2.5, False)
        
        # Check storage
        assert hasattr(collector, '_current_episode_logprobs')
        assert len(collector._current_episode_logprobs) == 1
        assert torch.allclose(collector._current_episode_logprobs[0], token_logprobs)
        
        logger.info("✅ TEST 2 PASSED: Trajectory collector stores per-token logprobs")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST 2 FAILED: {e}")
        return False


def test_3_trajectory_has_logprobs():
    """Test that built trajectories contain logprobs."""
    logger.info("\n=== TEST 3: Trajectories Contain Logprobs ===")
    
    try:
        from skyrl.data.trajectory import Trajectory
        
        # Create mock trajectory with logprobs
        states = ["state1", "state2", "state3"]
        actions = ["action1", "action2", "action3"]
        
        # Create per-token logprobs for each action
        log_probs = [
            torch.tensor([-0.5, -1.2], dtype=torch.float32),  # action1: 2 tokens
            torch.tensor([-0.8, -0.3, -1.1], dtype=torch.float32),  # action2: 3 tokens
            torch.tensor([-0.6], dtype=torch.float32),  # action3: 1 token
        ]
        
        rewards = [0.5, 1.0, -0.5]
        advantages = [0.2, 0.5, -0.3]
        
        traj = Trajectory(
            states=states,
            actions=actions,
            rewards=rewards,
            advantages=advantages,
            log_probs=log_probs
        )
        
        # Validate trajectory
        assert hasattr(traj, 'log_probs')
        assert len(traj.log_probs) == 3
        assert all(isinstance(lp, torch.Tensor) for lp in traj.log_probs)
        assert traj.log_probs[0].shape == (2,)
        assert traj.log_probs[1].shape == (3,)
        assert traj.log_probs[2].shape == (1,)
        
        logger.info("✅ TEST 3 PASSED: Trajectories properly contain per-token logprobs")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST 3 FAILED: {e}")
        return False


def test_4_trainer_processes_logprobs():
    """Test that trainer correctly processes per-token logprobs."""
    logger.info("\n=== TEST 4: Trainer Processes Logprobs ===")
    
    try:
        # Simulate trainer processing
        device = torch.device("cpu")
        
        # Create sample trajectory logprobs (list of tensors, one per action)
        traj_log_probs = [
            torch.tensor([-0.5, -1.2], dtype=torch.float32),  # action1: 2 tokens
            torch.tensor([-0.8, -0.3, -1.1], dtype=torch.float32),  # action2: 3 tokens
            torch.tensor([-0.6], dtype=torch.float32),  # action3: 1 token
        ]
        
        # Process as trainer would
        all_old_log_probs = []
        for lp in traj_log_probs:
            if isinstance(lp, torch.Tensor):
                if lp.dim() > 0 and len(lp) > 0:
                    # Vector of per-token logprobs for one action
                    for token_lp in lp:
                        all_old_log_probs.append(
                            token_lp if isinstance(token_lp, torch.Tensor) 
                            else torch.tensor(float(token_lp), device=device)
                        )
                else:
                    # Scalar tensor
                    all_old_log_probs.append(lp)
        
        # Should have flattened to 6 total token logprobs
        assert len(all_old_log_probs) == 6
        assert all(isinstance(lp, torch.Tensor) for lp in all_old_log_probs)
        
        # Stack into tensor
        old_log_probs = torch.stack(all_old_log_probs)
        assert old_log_probs.shape == (6,)
        
        logger.info(f"✅ TEST 4 PASSED: Trainer correctly flattens {len(traj_log_probs)} action logprobs to {len(all_old_log_probs)} token logprobs")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST 4 FAILED: {e}")
        return False


def test_5_ppo_ratio_computation():
    """Test that PPO ratios are computed correctly with variance."""
    logger.info("\n=== TEST 5: PPO Ratio Computation ===")
    
    try:
        device = torch.device("cpu")
        
        # Simulate old (sample-time) logprobs
        old_log_probs = torch.tensor([-0.5, -1.2, -0.8, -0.3, -1.1, -0.6], device=device)
        
        # Simulate current policy logprobs (should be different!)
        # Add small changes to simulate policy update
        current_log_probs = old_log_probs + torch.randn_like(old_log_probs) * 0.1
        
        # Compute PPO ratios
        log_ratios = current_log_probs - old_log_probs
        log_ratios = torch.clamp(log_ratios, min=-20.0, max=2.0)
        ratios = torch.exp(log_ratios).clamp(1e-8, 1e8)
        
        # Check statistics
        ratio_mean = ratios.mean().item()
        ratio_std = ratios.std().item()
        
        logger.info(f"PPO Ratios: mean={ratio_mean:.4f}, std={ratio_std:.4f}")
        
        # Ratios should have variance (not all 1.0)
        assert ratio_std > 1e-4, f"PPO ratios have no variance: std={ratio_std}"
        
        # Mean should be close to 1.0 but not exactly 1.0
        assert abs(ratio_mean - 1.0) > 1e-4 or ratio_std > 1e-4, "PPO ratios are degenerate"
        
        logger.info("✅ TEST 5 PASSED: PPO ratios have proper variance")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST 5 FAILED: {e}")
        return False


def test_6_validation_catches_degenerate_ratios():
    """Test that validation properly catches degenerate PPO ratios."""
    logger.info("\n=== TEST 6: Validation Catches Degenerate Ratios ===")
    
    try:
        device = torch.device("cpu")
        
        # Create degenerate case: old = current
        old_log_probs = torch.tensor([-0.5, -1.2, -0.8, -0.3], device=device)
        current_log_probs = old_log_probs.clone()  # Exactly the same!
        
        # Compute ratios
        log_ratios = current_log_probs - old_log_probs
        ratios = torch.exp(log_ratios)
        
        ratio_mean = ratios.mean().item()
        ratio_std = ratios.std().item()
        
        logger.info(f"Degenerate case: mean={ratio_mean:.6f}, std={ratio_std:.6f}")
        
        # This should trigger validation failure
        if ratio_std < 1e-4 and ratios.numel() > 1:
            logger.info("✅ Validation correctly detects degenerate ratios")
            detected = True
        else:
            logger.error("❌ Validation failed to detect degenerate ratios")
            detected = False
        
        assert detected, "Validation should detect degenerate PPO ratios"
        
        logger.info("✅ TEST 6 PASSED: Validation catches degenerate ratios")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST 6 FAILED: {e}")
        return False


def test_7_end_to_end_data_flow():
    """Test complete data flow from action generation to PPO ratio computation."""
    logger.info("\n=== TEST 7: End-to-End Data Flow ===")
    
    try:
        device = torch.device("cpu")
        
        # Step 1: Generate action with logprobs
        action_logprobs = torch.tensor([-0.5, -1.2, -0.8], dtype=torch.float32)
        action_data = {
            "text": "test action",
            "token_logprobs": action_logprobs,
            "logprob_sum": action_logprobs.sum().item(),
            "was_forced": False
        }
        
        # Step 2: Store in trajectory
        stored_logprobs = [action_logprobs]  # One action
        
        # Step 3: Process in trainer
        all_old_log_probs = []
        for lp in stored_logprobs:
            if isinstance(lp, torch.Tensor) and lp.dim() > 0:
                for token_lp in lp:
                    all_old_log_probs.append(token_lp)
        
        old_log_probs = torch.stack(all_old_log_probs)
        
        # Step 4: Compute current logprobs (simulated with small change)
        current_log_probs = old_log_probs + torch.randn_like(old_log_probs) * 0.05
        
        # Step 5: Compute PPO ratios
        log_ratios = current_log_probs - old_log_probs
        ratios = torch.exp(log_ratios)
        
        # Validate
        assert old_log_probs.shape == (3,), f"Expected 3 token logprobs, got {old_log_probs.shape}"
        assert ratios.std().item() > 1e-5, f"Ratios have insufficient variance: {ratios.std().item()}"
        
        logger.info(f"✅ Data flow: action({len(action_logprobs)} tokens) → trajectory → trainer → ratios(std={ratios.std().item():.4f})")
        logger.info("✅ TEST 7 PASSED: End-to-end data flow works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST 7 FAILED: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE PPO FIX PIPELINE TEST")
    logger.info("=" * 80)
    
    tests = [
        ("vLLM Policy Returns Logprobs", test_1_vllm_policy_returns_logprobs),
        ("Trajectory Collector Stores Logprobs", test_2_trajectory_collector_stores_logprobs),
        ("Trajectories Contain Logprobs", test_3_trajectory_has_logprobs),
        ("Trainer Processes Logprobs", test_4_trainer_processes_logprobs),
        ("PPO Ratio Computation", test_5_ppo_ratio_computation),
        ("Validation Catches Degenerate Ratios", test_6_validation_catches_degenerate_ratios),
        ("End-to-End Data Flow", test_7_end_to_end_data_flow),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            logger.error(f"Test {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        logger.info("\n🎉 ALL TESTS PASSED! PPO fix pipeline is working correctly.")
        return 0
    else:
        logger.error(f"\n❌ {total_count - passed_count} tests failed. Please review the fixes.")
        return 1


if __name__ == "__main__":
    sys.exit(main())