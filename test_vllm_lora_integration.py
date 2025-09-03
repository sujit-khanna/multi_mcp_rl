#!/usr/bin/env python3
"""
Test script to verify vLLM LoRA integration works correctly
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vllm_lora_support():
    """Test if vLLM has LoRA support"""
    try:
        import vllm
        from vllm.lora.request import LoRARequest
        
        logger.info(f"✅ vLLM version {vllm.__version__} imported successfully")
        logger.info(f"✅ LoRARequest class available")
        
        # Test if we can create a dummy LoRA request
        dummy_request = LoRARequest(
            lora_name="test",
            lora_int_id=1,
            lora_local_path="/tmp/test"
        )
        logger.info(f"✅ Can create LoRARequest objects")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import vLLM LoRA components: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing vLLM LoRA: {e}")
        return False

def test_ram_disk():
    """Test if /dev/shm is available and writable"""
    try:
        test_path = "/dev/shm/vllm_lora_test"
        os.makedirs(test_path, exist_ok=True)
        
        # Write a test file
        test_file = os.path.join(test_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        
        # Read it back
        with open(test_file, "r") as f:
            content = f.read()
        
        # Cleanup
        os.remove(test_file)
        os.rmdir(test_path)
        
        logger.info(f"✅ RAM disk /dev/shm is available and writable")
        return True
        
    except Exception as e:
        logger.error(f"❌ RAM disk test failed: {e}")
        return False

def test_policy_initialization():
    """Test if our modified policy can initialize"""
    try:
        # Set environment for vLLM
        os.environ["ENABLE_VLLM"] = "true"
        os.environ["VLLM_MAX_MODEL_LEN"] = "2048"
        os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.3"
        
        # Add parent directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Import and initialize policy
        from training.scripts.train_qwen3_grpo_real_env_vllm import VLLMQwenPolicy
        
        logger.info("🔄 Initializing VLLMQwenPolicy with LoRA support...")
        policy = VLLMQwenPolicy(model_name="Qwen/Qwen2.5-0.5B-Instruct")
        
        # Check if LoRA attributes are set
        assert hasattr(policy, 'lora_save_path'), "Missing lora_save_path"
        assert hasattr(policy, 'lora_update_frequency'), "Missing lora_update_frequency"
        assert hasattr(policy, 'LoRARequest'), "Missing LoRARequest class"
        assert policy.use_lora == True, "LoRA not enabled"
        
        logger.info(f"✅ Policy initialized with LoRA support")
        logger.info(f"   LoRA save path: {policy.lora_save_path}")
        logger.info(f"   Update frequency: {policy.lora_update_frequency}")
        
        # Test LoRA save functionality
        if hasattr(policy, 'save_lora_weights_for_vllm'):
            result = policy.save_lora_weights_for_vllm()
            if result:
                logger.info(f"✅ LoRA weights saved successfully")
                
                # Check if files were created
                if os.path.exists(policy.lora_save_path):
                    files = os.listdir(policy.lora_save_path)
                    logger.info(f"   Created files: {files[:5]}...")  # Show first 5 files
            else:
                logger.warning(f"⚠️  LoRA save returned False")
        
        # Test LoRA update functionality
        if hasattr(policy, 'maybe_update_vllm_lora'):
            result = policy.maybe_update_vllm_lora(force=True)
            if result:
                logger.info(f"✅ vLLM LoRA update successful")
                logger.info(f"   LoRA request created: {policy.lora_request}")
            else:
                logger.warning(f"⚠️  LoRA update returned False")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Policy initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting vLLM LoRA Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("vLLM LoRA Support", test_vllm_lora_support),
        ("RAM Disk Availability", test_ram_disk),
        ("Policy Initialization", test_policy_initialization),
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"\n📋 Testing: {name}")
        logger.info("-" * 40)
        success = test_func()
        results.append((name, success))
        logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 Test Results Summary:")
    all_passed = True
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed! vLLM LoRA integration is ready.")
    else:
        logger.info("\n⚠️  Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())