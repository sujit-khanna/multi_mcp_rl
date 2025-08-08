#!/usr/bin/env python3
"""Simple MPS compatibility test that skips generation"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

def test_mps_basics():
    """Test basic MPS compatibility"""
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        return False
        
    print("✅ MPS is available")
    device = torch.device('mps')
    
    # Test tensor operations
    try:
        x = torch.randn(1, 512, 896, device=device)
        y = torch.randn(1, 512, 896, device=device)
        z = x + y
        print("✅ Basic tensor operations work")
    except Exception as e:
        print(f"❌ Basic tensor operations failed: {e}")
        return False
    
    # Test model loading
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("\nLoading model...")
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            model_max_length=512
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Move to MPS
        model = model.to(device)
        print("✅ Model loaded on MPS")
        
        # Test forward pass only
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        print("✅ Forward pass works")
        
        # Note about generation
        print("\n⚠️  Note: Generation will be performed on CPU due to MPS 4GB limit")
        print("   The training script handles this automatically")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("MPS Compatibility Test (Simplified)")
    print("=" * 50)
    
    if test_mps_basics():
        print("\n✅ MPS compatibility check passed!")
        sys.exit(0)
    else:
        print("\n❌ MPS compatibility check failed!")
        sys.exit(1)