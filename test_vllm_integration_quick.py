#!/usr/bin/env python3
"""
Quick vLLM Integration Test
===========================

This script tests the vLLM integration components without running full training.
It validates that all pieces work together correctly.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required components can be imported."""
    logger.info("🔍 Testing imports...")
    
    try:
        # Test vLLM import
        import vllm
        logger.info(f"✅ vLLM imported successfully (version: {vllm.__version__})")
        
        # Test core training components
        from training.core.qwen_policy_with_vllm_inference import QwenPolicyWithVLLMInference
        from training.core.vllm_inference_wrapper import VLLMInferenceWrapper
        logger.info("✅ vLLM policy components imported successfully")
        
        # Test config loading
        from training.utils.config_loader import load_training_config, load_model_config
        logger.info("✅ Config loader imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_configs():
    """Test that vLLM configs can be loaded."""
    logger.info("🔍 Testing vLLM configurations...")
    
    try:
        from training.utils.config_loader import load_training_config, load_model_config
        
        # Test model config
        model_config_path = "training/configs/model_config_vllm.yaml"
        if not Path(model_config_path).exists():
            logger.error(f"❌ Model config not found: {model_config_path}")
            return False
            
        model_config = load_model_config(model_config_path)
        logger.info(f"✅ Model config loaded: {model_config.get('name', 'Unknown')}")
        
        # Test training config
        training_config_path = "training/configs/training_config_vllm.yaml"
        if not Path(training_config_path).exists():
            logger.error(f"❌ Training config not found: {training_config_path}")
            return False
            
        training_config = load_training_config(training_config_path)
        logger.info(f"✅ Training config loaded")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config loading failed: {e}")
        return False

def test_vllm_wrapper():
    """Test vLLM wrapper initialization."""
    logger.info("🔍 Testing vLLM wrapper...")
    
    try:
        from training.core.vllm_inference_wrapper import VLLMInferenceWrapper
        
        # Test wrapper creation (but don't start server)
        wrapper = VLLMInferenceWrapper(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            max_model_len=2048,
            gpu_memory_utilization=0.1,  # Very conservative
            port=8002,  # Different port for testing
            auto_start=False  # Don't auto-start
        )
        
        logger.info("✅ vLLM wrapper created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ vLLM wrapper test failed: {e}")
        return False

def test_gpu_status():
    """Test GPU availability."""
    logger.info("🔍 Testing GPU status...")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        
        logger.info(f"CUDA available: {cuda_available}")
        logger.info(f"GPU count: {device_count}")
        
        if cuda_available:
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1e9
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1e9
            memory_reserved = torch.cuda.memory_reserved(current_device) / 1e9
            
            logger.info(f"Current device: {current_device} ({device_name})")
            logger.info(f"Memory total: {memory_total:.1f} GB")
            logger.info(f"Memory allocated: {memory_allocated:.1f} GB")
            logger.info(f"Memory reserved: {memory_reserved:.1f} GB")
            logger.info(f"Memory free: {memory_total - memory_reserved:.1f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GPU status check failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting vLLM Integration Quick Test")
    logger.info("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configurations", test_configs),
        ("vLLM Wrapper", test_vllm_wrapper),
        ("GPU Status", test_gpu_status),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.info(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! vLLM integration is ready.")
        return 0
    else:
        logger.info("⚠️  Some tests failed. Check logs above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)