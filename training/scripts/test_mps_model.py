#!/usr/bin/env python3
"""
Simple test script to verify Qwen model runs on MPS
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: {device}")
    
    # Load model with minimal settings
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    print(f"Model loaded, moving to {device}...")
    model = model.to(device)
    print("Model on MPS!")
    
    # Test simple generation
    text = "Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    print("Testing generation...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {text}")
    print(f"Output: {response}")
    print("\nSuccess! Model runs on MPS.")
else:
    print("MPS not available!")