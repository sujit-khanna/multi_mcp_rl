#!/usr/bin/env python3
"""
Simple vLLM Test
================

Test just the vLLM server functionality without the complex training setup.
"""

import logging
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vllm_server():
    """Test vLLM server startup and basic generation."""
    
    try:
        from vllm import LLM, SamplingParams
        logger.info("✅ vLLM imported successfully")
        
        # Create vLLM model - very small for testing
        logger.info("🚀 Creating vLLM model...")
        llm = LLM(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            max_model_len=1024,  # Small for testing
            gpu_memory_utilization=0.3,  # Conservative
            enforce_eager=True  # Disable CUDA graphs for simplicity
        )
        logger.info("✅ vLLM model created successfully")
        
        # Test generation
        prompts = ["Hello, how are you?"]
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=50,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        logger.info("🎯 Testing generation...")
        start_time = time.time()
        
        outputs = llm.generate(prompts, sampling_params)
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Generation completed in {generation_time:.2f}s")
        
        for output in outputs:
            logger.info(f"Prompt: {output.prompt}")
            logger.info(f"Generated: {output.outputs[0].text}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ vLLM test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting simple vLLM test...")
    
    success = test_vllm_server()
    
    if success:
        logger.info("🎉 vLLM test passed!")
        sys.exit(0)
    else:
        logger.info("❌ vLLM test failed!")
        sys.exit(1)