#!/usr/bin/env python3
"""
vLLM Training Demo
==================

This demonstrates the vLLM integration for dramatically faster inference
during training, showing the speed improvements compared to HuggingFace.
"""

import logging
import time
import sys
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_huggingface_generation():
    """Demo standard HuggingFace generation (slow)."""
    logger.info("🐌 Testing HuggingFace Transformers Generation (Baseline)")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        logger.info(f"Loading model: {model_name}")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Test prompts
        prompts = [
            "Hello, how are you?",
            "What is machine learning?", 
            "Explain quantum computing."
        ]
        
        total_time = 0
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating response {i+1}/{len(prompts)}...")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            generation_time = time.time() - start_time
            total_time += generation_time
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            results.append({
                "prompt": prompt,
                "response": response,
                "time": generation_time
            })
            
            logger.info(f"   Time: {generation_time:.2f}s")
            logger.info(f"   Response: {response[:100]}...")
        
        avg_time = total_time / len(prompts)
        logger.info(f"🐌 HuggingFace Average: {avg_time:.2f}s per generation")
        
        # Clean up GPU memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
        logger.info("🧹 Cleaned up HuggingFace model from GPU memory")
        
        return results, avg_time
        
    except Exception as e:
        logger.error(f"❌ HuggingFace demo failed: {e}")
        return [], 0.0

def demo_vllm_generation():
    """Demo vLLM generation (fast)."""
    logger.info("🚀 Testing vLLM Generation (Fast)")
    
    try:
        from vllm import LLM, SamplingParams
        
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        logger.info(f"Loading vLLM model: {model_name}")
        
        # Create vLLM model
        llm = LLM(
            model=model_name,
            max_model_len=1024,
            gpu_memory_utilization=0.3,  # Conservative to share with HF model
            enforce_eager=True
        )
        
        # Test prompts (same as HuggingFace)
        prompts = [
            "Hello, how are you?",
            "What is machine learning?", 
            "Explain quantum computing."
        ]
        
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=50,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        # Time the generation
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        total_time = time.time() - start_time
        
        results = []
        for output in outputs:
            results.append({
                "prompt": output.prompt,
                "response": output.outputs[0].text,
                "time": total_time / len(prompts)  # Average time per prompt
            })
        
        avg_time = total_time / len(prompts)
        logger.info(f"🚀 vLLM Average: {avg_time:.2f}s per generation")
        
        for i, result in enumerate(results):
            logger.info(f"Response {i+1}: {result['response'][:100]}...")
        
        return results, avg_time
        
    except Exception as e:
        logger.error(f"❌ vLLM demo failed: {e}")
        return [], 0.0

def compare_results(hf_results, hf_time, vllm_results, vllm_time):
    """Compare HuggingFace and vLLM results."""
    logger.info("📊 Performance Comparison")
    logger.info("=" * 50)
    
    if hf_time > 0 and vllm_time > 0:
        speedup = hf_time / vllm_time
        time_saved = hf_time - vllm_time
        
        logger.info(f"HuggingFace Average Time: {hf_time:.2f}s")
        logger.info(f"vLLM Average Time: {vllm_time:.2f}s")
        logger.info(f"Speed Improvement: {speedup:.1f}x faster")
        logger.info(f"Time Saved per Generation: {time_saved:.2f}s")
        
        # Estimate training speedup
        generations_per_episode = 10  # Typical for RL training
        episodes_per_batch = 4
        total_generations = generations_per_episode * episodes_per_batch
        
        hf_total_time = hf_time * total_generations
        vllm_total_time = vllm_time * total_generations
        batch_time_saved = hf_total_time - vllm_total_time
        
        logger.info(f"\n🎯 Training Impact (per batch):")
        logger.info(f"   Generations per batch: {total_generations}")
        logger.info(f"   HuggingFace batch time: {hf_total_time:.1f}s")
        logger.info(f"   vLLM batch time: {vllm_total_time:.1f}s")
        logger.info(f"   Time saved per batch: {batch_time_saved:.1f}s")
        
        if speedup >= 5:
            logger.info(f"🎉 Excellent! {speedup:.1f}x speedup will dramatically improve training!")
        elif speedup >= 2:
            logger.info(f"✅ Good! {speedup:.1f}x speedup will noticeably improve training!")
        else:
            logger.info(f"⚠️  Modest {speedup:.1f}x speedup - may have overhead issues")
            
    else:
        logger.warning("⚠️  Could not compare - one or both demos failed")

def main():
    """Run the vLLM vs HuggingFace demo."""
    logger.info("🚀 Starting vLLM vs HuggingFace Performance Demo")
    logger.info("This demo shows the speed improvement of vLLM for RL training")
    logger.info("=" * 60)
    
    # Demo HuggingFace (baseline)
    hf_results, hf_time = demo_huggingface_generation()
    
    logger.info("\n" + "=" * 60)
    
    # Demo vLLM (fast)
    vllm_results, vllm_time = demo_vllm_generation()
    
    logger.info("\n" + "=" * 60)
    
    # Compare results
    compare_results(hf_results, hf_time, vllm_results, vllm_time)
    
    # Save results
    results = {
        "huggingface": {
            "results": hf_results,
            "avg_time": hf_time
        },
        "vllm": {
            "results": vllm_results,
            "avg_time": vllm_time
        },
        "speedup": hf_time / vllm_time if vllm_time > 0 else 0
    }
    
    output_file = "vllm_demo_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n📁 Results saved to: {output_file}")
    
    if vllm_time > 0 and hf_time > 0:
        speedup = hf_time / vllm_time
        if speedup >= 2:
            logger.info("✅ vLLM integration is ready for training acceleration!")
            return 0
        else:
            logger.warning("⚠️  vLLM speedup is less than expected")
            return 1
    else:
        logger.error("❌ Demo failed - check dependencies")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)