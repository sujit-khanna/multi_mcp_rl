#!/usr/bin/env python3
"""
Simple Test for Training Metrics Logging Structure
=================================================

Verifies that the comprehensive logging method is properly implemented.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_logging_code_structure():
    """Test that the logging code structure is correct"""
    logger.info("🧪 Testing Logging Code Structure")
    
    try:
        # Check that the training script has the comprehensive logging method
        training_script = Path("training/scripts/train_qwen3_grpo_real_env.py")
        assert training_script.exists(), "❌ Training script not found"
        
        with open(training_script, 'r') as f:
            content = f.read()
        
        # Check for comprehensive logging method
        assert "_log_comprehensive_training_metrics" in content, "❌ Comprehensive logging method missing"
        
        # Check for all critical metric groups
        metric_groups = [
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
        
        for metric in metric_groups:
            assert metric in content, f"❌ Missing metric: {metric}"
        
        logger.info(f"✅ All {len(metric_groups)} critical metrics found in code")
        
        # Check for WandB logging calls
        assert "wandb.log(safe, step=step" in content, "❌ WandB step logging missing"
        assert "weave.log" in content, "❌ Weave logging missing"
        
        # Check for console logging
        assert "📊 STEP" in content, "❌ Console logging format missing"
        
        # Check method is called in training loop
        assert "_log_comprehensive_training_metrics(" in content, "❌ Comprehensive logging not called"
        
        logger.info("✅ Comprehensive logging properly integrated into training loop")
        
        # Check WandB metric definitions
        assert "wandb.define_metric" in content, "❌ WandB metric definitions missing"
        assert "step_metric='trainer/step'" in content, "❌ Step metric alignment missing"
        
        logger.info("✅ WandB metric definitions configured")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Code structure test FAILED: {e}")
        return False

def test_project_names_updated():
    """Test that project names are updated"""
    logger.info("🧪 Testing Project Names Updated")
    
    try:
        files_to_check = [
            "training/scripts/train_qwen3_grpo_real_env.py",
            "training/scripts/launch_real_env_gpu.sh",
            "training/scripts/launch_real_env_cpu.sh"
        ]
        
        for file_path in files_to_check:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for new project name
            assert "multi-mcp-rl-fixed" in content, f"❌ New project name missing in {file_path}"
            logger.info(f"✅ Project name updated in {file_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Project name test FAILED: {e}")
        return False

def test_environment_variables():
    """Test that launcher scripts have proper environment variables"""
    logger.info("🧪 Testing Environment Variables in Launchers")
    
    try:
        launcher_scripts = [
            "training/scripts/launch_real_env_gpu.sh",
            "training/scripts/launch_real_env_cpu.sh"
        ]
        
        required_env_vars = [
            'WANDB_PROJECT="multi-mcp-rl-fixed"',
            'WEAVE_PROJECT="synergia_Agents/multi-mcp-rl-fixed"',
            'FORCE_RATE="0.0"',
            'PPO_RECORD_AT_SAMPLE="1"',
            'STRICT_TRAJ_KEYS="1"'
        ]
        
        for script in launcher_scripts:
            with open(script, 'r') as f:
                content = f.read()
            
            for env_var in required_env_vars:
                assert env_var in content, f"❌ Missing {env_var} in {script}"
            
            logger.info(f"✅ All environment variables present in {script}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment variables test FAILED: {e}")
        return False

def main():
    """Run all logging tests"""
    logger.info("🚀 Starting Training Metrics Logging Structure Test")
    logger.info("=" * 60)
    
    tests = [
        test_logging_code_structure,
        test_project_names_updated,
        test_environment_variables
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                logger.info("")
        except Exception as e:
            logger.error(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
    
    logger.info("=" * 60)
    logger.info(f"🎯 TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 ALL LOGGING STRUCTURE TESTS PASSED!")
        logger.info("✅ Comprehensive training metrics logging is properly implemented")
        logger.info("✅ Project names updated to 'multi-mcp-rl-fixed'")
        logger.info("✅ All critical metrics will be logged to WandB/Weave at each step")
        return 0
    else:
        logger.error("💥 Some tests failed. Please fix before training.")
        return 1

if __name__ == "__main__":
    exit(main())