#!/usr/bin/env python3
"""Test MPS compatibility with reduced settings"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

def test_mps_limits():
    """Test MPS tensor size limits"""
    if not torch.backends.mps.is_available():
        print("MPS not available")
        return
        
    device = torch.device('mps')
    
    print("Testing MPS limits...")
    print(f"MPS is available: {torch.backends.mps.is_available()}")
    print(f"MPS is built: {torch.backends.mps.is_built()}")
    
    # Test different tensor sizes
    sizes = [
        (1, 512, 768),    # OK: ~1.5MB
        (1, 1024, 768),   # OK: ~3MB
        (1, 2048, 768),   # OK: ~6MB
        (1, 4096, 768),   # May fail: ~12MB
        (1, 512, 32000),  # Vocab size test
        (1, 512, 151936), # Qwen vocab size
    ]
    
    for size in sizes:
        try:
            tensor = torch.randn(size, device=device)
            total_bytes = tensor.element_size() * tensor.numel()
            print(f"✅ Size {size}: {total_bytes/1024/1024:.1f}MB - OK")
            del tensor
            torch.mps.empty_cache()
        except Exception as e:
            print(f"❌ Size {size}: FAILED - {str(e)[:100]}")

def test_model_loading():
    """Test loading model with MPS settings"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\nTesting model loading...")
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Load with reduced settings
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=512,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        max_position_embeddings=512  # Override config
    )
    
    print(f"Model loaded, moving to MPS...")
    model = model.to('mps')
    print(f"✅ Model loaded on MPS")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test generation with small input
    print("\nTesting generation...")
    inputs = tokenizer("Hello", return_tensors="pt", max_length=50).to('mps')
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False  # Greedy for testing
        )
    
    print(f"✅ Generation successful: {outputs.shape}")
    
    # Test with longer input
    long_text = "This is a test " * 50  # ~200 tokens
    inputs = tokenizer(long_text, return_tensors="pt", max_length=400, truncation=True).to('mps')
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=tokenizer.pad_token_id
        )
    
    print(f"✅ Long input generation successful: {outputs.shape}")
    
    # Clear cache
    torch.mps.empty_cache()
    print("\n✅ All tests passed!")

def test_batch_processing():
    """Test batch processing limits"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\nTesting batch processing...")
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to('mps')
    
    # Test different batch sizes with fixed sequence length
    batch_sizes = [1, 2, 4, 8]
    seq_length = 256
    
    for bs in batch_sizes:
        try:
            # Create batch
            texts = ["Hello world"] * bs
            inputs = tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True,
                max_length=seq_length,
                truncation=True
            ).to('mps')
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            print(f"✅ Batch size {bs} x seq_len {seq_length}: OK")
            torch.mps.empty_cache()
            
        except Exception as e:
            print(f"❌ Batch size {bs}: FAILED - {str(e)[:100]}")
            break

if __name__ == "__main__":
    print("MPS Compatibility Test")
    print("=" * 50)
    
    test_mps_limits()
    print("\n" + "=" * 50)
    
    test_model_loading()
    print("\n" + "=" * 50)
    
    test_batch_processing()