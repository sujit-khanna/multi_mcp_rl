#!/usr/bin/env python3
"""Debug MPS generation issue with Qwen model"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

def test_generation_step_by_step():
    """Test generation step by step to find the issue"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import warnings
    warnings.filterwarnings("ignore")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\n1. Loading model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        model_max_length=512
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Check model config
    print(f"Model vocab size: {model.config.vocab_size}")
    print(f"Model hidden size: {model.config.hidden_size}")
    print(f"Model num layers: {model.config.num_hidden_layers}")
    print(f"Model max position embeddings: {model.config.max_position_embeddings}")
    
    # Move to MPS
    print("\n2. Moving to MPS...")
    model = model.to(device)
    model.eval()
    
    # Test simple forward pass
    print("\n3. Testing forward pass...")
    try:
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        print(f"✅ Forward pass OK, logits shape: {outputs.logits.shape}")
        
        # Check logits size
        logits_bytes = outputs.logits.element_size() * outputs.logits.numel()
        print(f"   Logits size: {logits_bytes / 1024 / 1024:.1f}MB")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return
    
    # Test generation with different settings
    print("\n4. Testing generation...")
    
    # Test 1: Minimal generation
    print("\n   Test 1: Minimal generation (1 token)")
    try:
        text = "Hi"
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Disable cache to save memory
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False,
                return_dict_in_generate=True,
                output_scores=False,
                output_attentions=False,
                output_hidden_states=False
            )
        print(f"   ✅ Minimal generation OK: {outputs.sequences.shape}")
    except Exception as e:
        print(f"   ❌ Minimal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Slightly longer
    print("\n   Test 2: Generate 5 tokens")
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False
            )
        print(f"   ✅ 5-token generation OK: {outputs.shape}")
    except Exception as e:
        print(f"   ❌ 5-token generation failed: {e}")
    
    # Test 3: With sampling
    print("\n   Test 3: Generate with sampling disabled")
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   ✅ Sampling generation OK: '{generated_text}'")
    except Exception as e:
        print(f"   ❌ Sampling generation failed: {e}")
    
    # Test 4: Check memory usage during generation
    print("\n5. Testing memory patterns...")
    
    # Create input of different lengths
    for length in [10, 50, 100, 200]:
        try:
            text = "Hello " * (length // 6)  # Roughly 'length' tokens
            inputs = tokenizer(text, return_tensors="pt", max_length=length, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            print(f"\n   Input length: {inputs['input_ids'].shape[1]} tokens")
            
            with torch.no_grad():
                # Forward pass to check intermediate sizes
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_attentions=True,
                    output_hidden_states=True
                )
                
                # Check sizes
                hidden_states_size = sum(h.element_size() * h.numel() for h in outputs.hidden_states) / 1024 / 1024
                attention_size = sum(a.element_size() * a.numel() for a in outputs.attentions) / 1024 / 1024
                logits_size = outputs.logits.element_size() * outputs.logits.numel() / 1024 / 1024
                
                print(f"   Hidden states total: {hidden_states_size:.1f}MB")
                print(f"   Attention weights total: {attention_size:.1f}MB")
                print(f"   Logits size: {logits_size:.1f}MB")
                
                # Try generation
                gen_outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=False
                )
                print(f"   ✅ Generation OK for length {length}")
                
            torch.mps.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Failed at length {length}: {str(e)[:100]}")
            break
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_generation_step_by_step()