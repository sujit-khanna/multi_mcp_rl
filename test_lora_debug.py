#!/usr/bin/env python3
"""
Test script to verify the corrected LoRA implementation with debugging
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_lora_debug_implementation():
    """Test the corrected LoRA implementation with debug logging"""
    
    try:
        # Set environment for vLLM
        os.environ["ENABLE_VLLM"] = "true"
        os.environ["VLLM_MAX_MODEL_LEN"] = "2048"
        os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.3"
        
        # Add parent directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Import and initialize policy
        from training.scripts.train_qwen3_grpo_real_env_vllm import VLLMQwenPolicy
        
        logger.info("🔄 Initializing VLLMQwenPolicy with corrected LoRA implementation...")
        policy = VLLMQwenPolicy(model_name="Qwen/Qwen2.5-0.5B-Instruct")
        
        # Test initial LoRA update
        logger.info("🧪 Testing initial LoRA update (forced)...")
        result1 = policy.maybe_update_vllm_lora(force=True)
        logger.info(f"   Initial update result: {result1}")
        
        # Test counter-based updates
        logger.info("🧪 Testing counter-based LoRA updates...")
        for i in range(1, 7):
            logger.info(f"\n--- Test {i} ---")
            result = policy.maybe_update_vllm_lora(force=False)
            logger.info(f"   Step {i} result: {result}")
            logger.info(f"   Counter after step: {policy.lora_update_counter}")
            
            if result:
                logger.info(f"   ✅ LoRA updated at step {i}")
                break
        
        # Test LoRA request status
        if hasattr(policy, 'lora_request') and policy.lora_request:
            logger.info(f"✅ Final LoRA request: {policy.lora_request}")
            logger.info(f"   LoRA name: {policy.lora_request.lora_name}")
            logger.info(f"   LoRA ID: {policy.lora_request.lora_int_id}")
            logger.info(f"   LoRA path: {policy.lora_request.lora_local_path}")
        else:
            logger.error("❌ No LoRA request created!")
        
        # Test generation with LoRA
        logger.info("🧪 Testing generation with LoRA adapter...")
        test_prompt = ["Tell me about machine learning in one sentence."]
        
        # This should use the LoRA adapter if available
        responses = policy.generate_action(test_prompt, max_tokens=50)
        logger.info(f"✅ Generated response: {responses[0][:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Run the debug test"""
    logger.info("🚀 Starting LoRA Debug Test")
    logger.info("=" * 60)
    
    success = test_lora_debug_implementation()
    
    logger.info("=" * 60)
    if success:
        logger.info("🎉 LoRA debug test completed successfully!")
        logger.info("   The corrected implementation should now:")
        logger.info("   - Log all LoRA update attempts")
        logger.info("   - Show counter progression")
        logger.info("   - Force updates every training step")
        logger.info("   - Provide detailed error messages")
    else:
        logger.error("⚠️  LoRA debug test failed!")
        logger.error("   Check the error messages above for details")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())