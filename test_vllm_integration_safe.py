#!/usr/bin/env python3
"""
Safe vLLM Integration Test
=========================

This script tests the vLLM integration using separate config files
that won't interfere with existing training scripts.
"""

import sys
from pathlib import Path

# Add training path
sys.path.append(str(Path(__file__).parent / "training"))

import torch
import logging
logging.basicConfig(level=logging.INFO)

def test_vllm_integration_safe():
    """Test the vLLM integration with separate configs"""
    print("🧪 Testing vLLM Integration (Safe Mode)")
    print("=" * 50)
    
    try:
        from training.core.qwen_policy_with_vllm_inference import QwenPolicyWithVLLMInference
        
        # Create policy with vLLM-specific configs (won't affect existing training)
        print("📝 Creating policy with vLLM integration (separate configs)...")
        policy = QwenPolicyWithVLLMInference(
            model_config_path="training/configs/model_config_vllm.yaml",
            training_config_path="training/configs/training_config_vllm.yaml"
        )
        
        print("✅ Policy created successfully!")
        
        # Test generation
        print("\n🔄 Testing generation...")
        test_states = [[
            {"role": "user", "content": "What is 2 + 2? Please answer briefly."}
        ]]
        
        actions = policy.generate_action(test_states, max_new_tokens=50, temperature=0.1)
        
        print(f"📤 Input: What is 2 + 2? Please answer briefly.")
        print(f"📥 Output: {actions[0]}")
        
        print("\n✅ vLLM integration test completed successfully!")
        print("💡 This integration uses separate configs and won't affect existing training")
        print("🚀 Ready for vLLM speedup when vLLM is installed!")
        
        # Clean up
        del policy
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vllm_integration_safe()