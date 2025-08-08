#!/usr/bin/env python3
"""Simple test to identify MPS limit issue"""

import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

if torch.backends.mps.is_available():
    device = torch.device('mps')
    
    # Test different model components
    print("\nTesting embedding layer...")
    try:
        # Qwen vocab size is 151936
        embed = torch.nn.Embedding(151936, 896).to(device)
        print(f"✅ Embedding layer OK: {embed.weight.shape}")
        
        # Test with sequence
        input_ids = torch.randint(0, 151936, (1, 512)).to(device)
        output = embed(input_ids)
        print(f"✅ Embedding forward pass OK: {output.shape}")
        
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
    
    print("\nTesting attention...")
    try:
        # Single attention layer
        attn = torch.nn.MultiheadAttention(896, 14, batch_first=True).to(device)
        x = torch.randn(1, 512, 896).to(device)
        output, _ = attn(x, x, x)
        print(f"✅ Attention OK: {output.shape}")
    except Exception as e:
        print(f"❌ Attention failed: {e}")
    
    print("\nTesting transformer...")
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        
        # Load just config first
        config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
        print(f"Model config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
        
        # Try loading with CPU first, then move
        print("\nLoading model on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("Model loaded on CPU, checking size...")
        total_params = sum(p.numel() for p in model.parameters())
        total_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        print(f"Total params: {total_params:,}")
        print(f"Total size: {total_size_mb:.1f} MB")
        
        # Try moving layer by layer
        print("\nTrying to move to MPS layer by layer...")
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        print("✅ Embeddings moved")
        
        # Move one layer at a time
        for i, layer in enumerate(model.model.layers[:5]):  # Just first 5 layers
            try:
                layer.to(device)
                print(f"✅ Layer {i} moved")
            except Exception as e:
                print(f"❌ Layer {i} failed: {e}")
                break
                
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()