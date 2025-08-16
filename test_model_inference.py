#!/usr/bin/env python3
"""
Standalone test to check if the Qwen model can perform basic inference
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_basic_inference():
    print("🔍 Testing basic Qwen model inference...")
    
    # Set environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    try:
        print("📦 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
        
        print("📦 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to("cuda:0")
        
        print("✅ Model loaded successfully")
        print(f"   Model device: {model.device}")
        print(f"   Model dtype: {model.dtype}")
        print(f"   Training mode: {model.training}")
        
        # Put in eval mode
        model.eval()
        print("✅ Model set to eval mode")
        
        # Simple test
        test_input = "Hello, how are you?"
        print(f"🧪 Testing with input: '{test_input}'")
        
        # Tokenize
        inputs = tokenizer(test_input, return_tensors="pt").to("cuda:0")
        print(f"✅ Tokenization successful: {inputs['input_ids'].shape}")
        
        # Generate
        print("🚀 Starting generation...")
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        print("✅ Generation completed!")
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"📄 Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_inference()
    if success:
        print("🎉 Basic inference test PASSED")
    else:
        print("💥 Basic inference test FAILED")