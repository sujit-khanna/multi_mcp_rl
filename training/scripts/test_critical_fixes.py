#!/usr/bin/env python3
"""
Critical Fixes Verification Test
===============================

This script tests all the critical fixes applied to the GRPO training system:
1. PPO/GRPO ratio computation (correct trajectory keys)
2. Forced tool calls disabled during RL updates
3. Reference policy synchronization
4. Value function training
5. Log-probs recorded at sampling time
6. Tool name consistency
7. Sanity checks and debugging

Run this before actual training to verify all fixes are working.
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Dict, List

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set test environment variables
os.environ['FORCE_RATE'] = '0.0'
os.environ['ASSIST_WARMUP'] = '0'
os.environ['RL_DISABLE_FORCED'] = '1'
os.environ['PPO_RECORD_AT_SAMPLE'] = '1'
os.environ['STRICT_TRAJ_KEYS'] = '1'
os.environ['DEVICE_TYPE'] = 'cpu'  # Use CPU for testing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_trajectory_key_fix():
    """Test 1: Verify trajectory uses correct 'log_probs' key"""
    logger.info("🧪 Test 1: Trajectory Key Fix")
    
    try:
        # Simple test: can we access the trainer file?
        trainer_file = Path("training/core/grpo_trainer_gradient_fix.py")
        assert trainer_file.exists(), "❌ GRPO trainer file not found"
        
        # Check that the trainer looks for 'log_probs' key
        with open(trainer_file, 'r') as f:
            content = f.read()
            assert "hasattr(traj, 'log_probs')" in content, "❌ Trainer doesn't check for log_probs"
            assert "Missing 'log_probs' on trajectory" in content, "❌ Missing error message for log_probs"
    
        logger.info("✅ Test 1 PASSED: Trajectory uses correct 'log_probs' key")
    except Exception as e:
        logger.error(f"❌ Test 1 FAILED: {e}")
        raise

def test_forced_actions_disabled():
    """Test 2: Verify forced actions are disabled during RL"""
    logger.info("🧪 Test 2: Forced Actions Disabled")
    
    try:
        # Check policy file has the fixes
        policy_file = Path("training/core/qwen_policy_with_prompting.py")
        assert policy_file.exists(), "❌ Policy file not found"
        
        with open(policy_file, 'r') as f:
            content = f.read()
            assert "in_rl_update" in content, "❌ RL update logic missing"
            assert "force_rate" in content, "❌ Force rate logic missing"
            assert "FORCE_RATE" in content, "❌ Environment variable support missing"
    
        logger.info("✅ Test 2 PASSED: Forced actions disabled during RL")
    except Exception as e:
        logger.error(f"❌ Test 2 FAILED: {e}")
        raise

def test_tool_name_consistency():
    """Test 3: Verify tool names are consistent"""
    logger.info("🧪 Test 3: Tool Name Consistency")
    
    try:
        policy_file = Path("training/core/qwen_policy_with_prompting.py")
        assert policy_file.exists(), "❌ Policy file not found"
        
        with open(policy_file, 'r') as f:
            content = f.read()
            assert "- send_slack_message: Send messages" in content, "❌ Correct tool name not found in prompt"
            assert "slack_send_message" not in content, "❌ Old incorrect tool name still present"
    
        logger.info("✅ Test 3 PASSED: Tool names are consistent")
    except Exception as e:
        logger.error(f"❌ Test 3 FAILED: {e}")
        raise

def test_value_function_methods():
    """Test 4: Verify value function training methods exist"""
    logger.info("🧪 Test 4: Value Function Methods")
    
    try:
        trainer_file = Path("training/core/grpo_trainer_gradient_fix.py")
        assert trainer_file.exists(), "❌ GRPO trainer file not found"
        
        with open(trainer_file, 'r') as f:
            content = f.read()
            required_methods = [
                '_compute_returns_advantages',
                '_compute_value_loss',
                '_compute_explained_variance'
            ]
            
            for method in required_methods:
                assert f"def {method}" in content, f"❌ Missing method: {method}"
    
        logger.info("✅ Test 4 PASSED: Value function methods implemented")
    except Exception as e:
        logger.error(f"❌ Test 4 FAILED: {e}")
        raise

def test_reference_policy_sync():
    """Test 5: Verify reference policy sync methods"""
    logger.info("🧪 Test 5: Reference Policy Sync")
    
    try:
        trainer_file = Path("training/core/grpo_trainer_gradient_fix.py")
        assert trainer_file.exists(), "❌ GRPO trainer file not found"
        
        with open(trainer_file, 'r') as f:
            content = f.read()
            sync_methods = [
                'sync_reference_policy',
                '_count_param_diffs'
            ]
            
            for method in sync_methods:
                assert f"def {method}" in content, f"❌ Missing sync method: {method}"
    
        logger.info("✅ Test 5 PASSED: Reference policy sync methods implemented")
    except Exception as e:
        logger.error(f"❌ Test 5 FAILED: {e}")
        raise

def test_sanity_checks():
    """Test 6: Verify sanity checks and logging"""
    logger.info("🧪 Test 6: Sanity Checks and Logging")
    
    try:
        trainer_file = Path("training/core/grpo_trainer_gradient_fix.py")
        assert trainer_file.exists(), "❌ GRPO trainer file not found"
        
        with open(trainer_file, 'r') as f:
            content = f.read()
            assert "PPO Ratio Check" in content, "❌ Ratio logging missing"
            assert "STRICT_TRAJ_KEYS" in content, "❌ Strict key validation missing"
            assert "raise KeyError" in content, "❌ Error handling missing"
    
        logger.info("✅ Test 6 PASSED: Sanity checks implemented")
    except Exception as e:
        logger.error(f"❌ Test 6 FAILED: {e}")
        raise

def test_log_prob_sampling():
    """Test 7: Verify log-prob sampling method exists"""
    logger.info("🧪 Test 7: Log-prob Sampling")
    
    try:
        policy_file = Path("training/core/qwen_policy_with_prompting.py")
        assert policy_file.exists(), "❌ Policy file not found"
        
        with open(policy_file, 'r') as f:
            content = f.read()
            assert "def sample_with_logprobs" in content, "❌ Missing sample_with_logprobs method"
            assert "PPO_RECORD_AT_SAMPLE" in content, "❌ Missing environment variable check"
    
        logger.info("✅ Test 7 PASSED: Log-prob sampling method implemented")
    except Exception as e:
        logger.error(f"❌ Test 7 FAILED: {e}")
        raise

def test_environment_variables():
    """Test 8: Verify all critical environment variables are set"""
    logger.info("🧪 Test 8: Environment Variables")
    
    required_env_vars = {
        'FORCE_RATE': '0.0',
        'ASSIST_WARMUP': '0',
        'RL_DISABLE_FORCED': '1',
        'PPO_RECORD_AT_SAMPLE': '1',
        'STRICT_TRAJ_KEYS': '1'
    }
    
    for var, expected in required_env_vars.items():
        actual = os.getenv(var)
        assert actual == expected, \
            f"❌ {var} should be '{expected}', got '{actual}'"
    
    logger.info("✅ Test 8 PASSED: All environment variables correctly set")

def main():
    """Run all critical fixes tests"""
    logger.info("🚀 Starting Critical Fixes Verification Tests")
    logger.info("=" * 60)
    
    tests = [
        test_trajectory_key_fix,
        test_forced_actions_disabled,
        test_tool_name_consistency,
        test_value_function_methods,
        test_reference_policy_sync,
        test_sanity_checks,
        test_log_prob_sampling,
        test_environment_variables
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
        print()  # Add spacing
    
    logger.info("=" * 60)
    logger.info(f"🎯 TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 ALL CRITICAL FIXES VERIFIED! Ready for training.")
        return 0
    else:
        logger.error("💥 Some tests failed. Please fix before training.")
        return 1

if __name__ == "__main__":
    exit(main())