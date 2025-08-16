#!/usr/bin/env python3
"""
Test vLLM Integration (Fallback Mode)
====================================

This script tests the vLLM integration in fallback mode (using HuggingFace)
to verify the integration works correctly.
"""

import sys
from pathlib import Path

# Add training path
sys.path.append(str(Path(__file__).parent / "training"))

import torch
import logging
logging.basicConfig(level=logging.INFO)

def test_vllm_integration():
    """Test the vLLM integration wrapper"""
    print("🧪 Testing vLLM Integration (Fallback Mode)")
    print("=" * 50)
    
    try:
        from training.core.qwen_policy_with_vllm_inference import QwenPolicyWithVLLMInference
        
        # Create policy with vLLM integration
        print("📝 Creating policy with vLLM integration...")
        policy = QwenPolicyWithVLLMInference(
            model_config_path="training/configs/model_config_qwen3_0.6b.yaml",
            training_config_path="training/configs/training_config_qwen3_0.6b.yaml"
        )
        
        print("✅ Policy created successfully!")
        
        # Test generation
        print("\n🔄 Testing generation...")
        test_states = [[
            {"role": "user", "content": "What is 2 + 2?"}
        ]]
        
        actions = policy.generate_action(test_states, max_new_tokens=50, temperature=0.1)
        
        print(f"📤 Input: What is 2 + 2?")
        print(f"📥 Output: {actions[0]}")
        
        print("\n✅ vLLM integration test completed successfully!")
        print("💡 The integration will use HuggingFace fallback until vLLM is installed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vllm_integration()