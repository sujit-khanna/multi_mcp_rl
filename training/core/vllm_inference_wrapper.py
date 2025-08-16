#!/usr/bin/env python3
"""
vLLM Inference Wrapper for SkyRL Integration
============================================

A lightweight wrapper that uses vLLM purely for inference while keeping 
SkyRL's training loop and policy structure intact.

This replaces slow HuggingFace model.generate() calls with fast vLLM inference.
"""

import torch
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import subprocess
import sys

logger = logging.getLogger(__name__)

class VLLMInferenceWrapper:
    """
    Lightweight vLLM wrapper for fast inference.
    
    This class handles:
    1. vLLM server lifecycle management  
    2. Fast generation via vLLM API
    3. Fallback to HuggingFace if vLLM fails
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.3,  # Conservative to coexist with training
        port: int = 8000
    ):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.port = port
        self.server_process = None
        self.fallback_model = None
        self.use_vllm = self._check_vllm_available()
        
        if self.use_vllm:
            logger.info("✅ vLLM available - will use for fast inference")
            self._start_vllm_server()
        else:
            logger.warning("⚠️ vLLM not available - falling back to HuggingFace")
    
    def _check_vllm_available(self) -> bool:
        """Check if vLLM is available in the environment"""
        try:
            import vllm
            return True
        except ImportError:
            return False
    
    def _start_vllm_server(self):
        """Start vLLM server in the background"""
        try:
            # Start vLLM server with conservative memory settings
            cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.model_name,
                "--port", str(self.port),
                "--max-model-len", str(self.max_model_len),
                "--gpu-memory-utilization", str(self.gpu_memory_utilization),
                "--tensor-parallel-size", "1",
                "--disable-log-requests"
            ]
            
            logger.info(f"🚀 Starting vLLM server: {' '.join(cmd)}")
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={"CUDA_VISIBLE_DEVICES": "0"}  # Use same GPU as training
            )
            
            # Wait a moment for server to start
            import time
            time.sleep(10)
            
            # Test if server is responsive
            if self._test_vllm_connection():
                logger.info("✅ vLLM server started successfully")
            else:
                logger.warning("❌ vLLM server failed to start - falling back to HuggingFace")
                self.use_vllm = False
                self._stop_vllm_server()
                
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            self.use_vllm = False
    
    def _test_vllm_connection(self) -> bool:
        """Test if vLLM server is responsive"""
        try:
            import requests
            response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _stop_vllm_server(self):
        """Stop vLLM server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
            logger.info("🛑 vLLM server stopped")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate text using vLLM or fallback to HuggingFace
        """
        if self.use_vllm:
            return self._generate_vllm(prompt, max_tokens, temperature, top_p, stop)
        else:
            return self._generate_fallback(prompt, max_tokens, temperature, top_p, stop)
    
    def _generate_vllm(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]]
    ) -> str:
        """Generate using vLLM API"""
        try:
            import requests
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop or [],
                "stream": False
            }
            
            response = requests.post(
                f"http://localhost:{self.port}/v1/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["text"]
            else:
                logger.warning(f"vLLM API error: {response.status_code} - falling back")
                return self._generate_fallback(prompt, max_tokens, temperature, top_p, stop)
                
        except Exception as e:
            logger.warning(f"vLLM generation failed: {e} - falling back")
            return self._generate_fallback(prompt, max_tokens, temperature, top_p, stop)
    
    def _generate_fallback(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]]
    ) -> str:
        """Fallback to HuggingFace generation"""
        if self.fallback_model is None:
            logger.info("🔄 Loading HuggingFace fallback model...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.fallback_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.fallback_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            if self.fallback_tokenizer.pad_token is None:
                self.fallback_tokenizer.pad_token = self.fallback_tokenizer.eos_token
        
        # Generate with HuggingFace
        inputs = self.fallback_tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.fallback_model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.fallback_tokenizer.pad_token_id,
                eos_token_id=self.fallback_tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Extract generated text
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return self.fallback_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    def __del__(self):
        """Cleanup on destruction"""
        self._stop_vllm_server()


# Convenience function for easy integration
def create_vllm_inference_wrapper(**kwargs) -> VLLMInferenceWrapper:
    """Create a vLLM inference wrapper with default settings"""
    return VLLMInferenceWrapper(**kwargs)


if __name__ == "__main__":
    # Test the wrapper
    wrapper = VLLMInferenceWrapper()
    
    test_prompt = "What is the capital of France?"
    result = wrapper.generate(test_prompt, max_tokens=50)
    
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {result}")