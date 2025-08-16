#!/usr/bin/env python3
"""
vLLM-based Policy Implementation for Fast Inference

This replaces the slow Hugging Face transformers generation with vLLM's
optimized inference engine for significant speed improvements.
"""

import logging
import time
import os
import torch
from typing import List, Dict, Any, Optional
import yaml

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# Chat template imports
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class VLLMQwenPolicy:
    """
    Fast vLLM-based policy for Qwen model inference.
    
    This replaces QwenPolicyWithPrompting with a much faster vLLM implementation
    that should reduce generation time from minutes to seconds.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        config_path: Optional[str] = None,
        gpu_memory_utilization: float = 0.7,
        max_model_len: int = 512,
        enforce_eager: bool = True,  # Disable CUDA graphs for compatibility
        **kwargs
    ):
        """Initialize vLLM policy."""
        
        self.model_name = model_name
        self.config_path = config_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize tokenizer for chat templating
        logger.info(f"🔄 Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set up vLLM engine
        logger.info(f"🚀 Initializing vLLM engine for {model_name}...")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,  # Single GPU
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,  # Disable CUDA graphs
            trust_remote_code=True,
            dtype="float16",
            enable_prefix_caching=True,  # Speed up repeated prefixes
            **kwargs
        )
        
        # Set up sampling parameters from config
        self.sampling_params = self._create_sampling_params()
        
        # Training mode flags
        self.training_mode = False
        self.eval_mode = True
        
        logger.info("✅ vLLM Policy initialized successfully")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Max length: {max_model_len}")
        logger.info(f"   GPU memory utilization: {gpu_memory_utilization}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"📁 Loaded config from {self.config_path}")
            return config
        else:
            # Default config
            logger.info("📝 Using default vLLM config")
            return {
                "generation": {
                    "max_new_tokens": 128,
                    "temperature": 0.05,
                    "top_p": 0.8,
                    "top_k": 10,
                    "repetition_penalty": 1.5,
                    "do_sample": True
                }
            }
    
    def _create_sampling_params(self) -> SamplingParams:
        """Create vLLM sampling parameters from config."""
        
        gen_config = self.config.get("generation", {})
        
        return SamplingParams(
            max_tokens=gen_config.get("max_new_tokens", 128),
            temperature=gen_config.get("temperature", 0.05),
            top_p=gen_config.get("top_p", 0.8),
            top_k=gen_config.get("top_k", 10),
            repetition_penalty=gen_config.get("repetition_penalty", 1.5),
            stop=["</tool_call>", "<|im_end|>", "</think>"],  # Stop sequences
            # Note: vLLM doesn't support length_penalty, use max_tokens for control
        )
    
    def _format_conversation_for_generation(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history for generation."""
        
        # Add aggressive tool calling prompt at the start
        system_prompt = {
            "role": "system",
            "content": """You are a tool-calling assistant. You MUST respond with a tool call in the exact format below. Do NOT write any natural language text.

CRITICAL: Your response MUST start with <tool_call> and follow this exact format:

<tool_call>{"name": "tool_name", "arguments": {"key": "value"}}</tool_call>

Available tools:
- tavily_search: Web search (arguments: {"query": "search terms"})
- execute_python: Run Python code (arguments: {"code": "python code"})
- fmp_get_company_profile: Get company info (arguments: {"symbol": "AAPL"})
- send_slack_message: Send message (arguments: {"message": "text", "channel": "general"})

ALWAYS respond with a tool call starting with <tool_call> and ending with </tool_call>. NO explanations or reasoning."""
        }
        
        # Combine system prompt with conversation
        formatted_conversation = [system_prompt] + conversation_history
        
        # Use tokenizer's chat template
        try:
            formatted_text = self.tokenizer.apply_chat_template(
                formatted_conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted_text
        except Exception as e:
            logger.warning(f"Chat template failed, using fallback: {e}")
            # Fallback formatting
            text = ""
            for msg in formatted_conversation:
                role = msg["role"]
                content = msg["content"]
                text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            text += "<|im_start|>assistant\n"
            return text
    
    def generate_action(self, states: List[List[Dict[str, str]]]) -> List[str]:
        """
        Generate actions using vLLM for much faster inference.
        
        Args:
            states: List of conversation histories
            
        Returns:
            List of generated actions
        """
        
        if not states:
            logger.warning("No states provided for generation")
            return ["I need to analyze this task."]
        
        start_time = time.time()
        logger.info(f"🎯 VLLMQwenPolicy.generate_action called with {len(states)} states")
        logger.info(f"   Sampling params: max_tokens={self.sampling_params.max_tokens}, "
                   f"temp={self.sampling_params.temperature}")
        
        try:
            # Format all conversations for generation
            prompts = []
            for state in states:
                formatted_prompt = self._format_conversation_for_generation(state)
                prompts.append(formatted_prompt)
                
            logger.info(f"📝 Formatted {len(prompts)} prompts for generation")
            logger.info(f"   First prompt length: {len(prompts[0])} chars")
            
            # Generate with vLLM (much faster than transformers)
            logger.info("🚀 Starting vLLM generation...")
            generation_start = time.time()
            
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            generation_time = time.time() - generation_start
            logger.info(f"✅ vLLM generation completed in {generation_time:.2f} seconds")
            logger.info(f"   Generated {len(outputs)} outputs")
            
            # Extract generated text
            actions = []
            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                actions.append(generated_text)
                logger.info(f"   Generated action: {generated_text[:100]}...")
            
            total_time = time.time() - start_time
            logger.info(f"🎉 Total generation time: {total_time:.2f} seconds")
            
            return actions
            
        except Exception as e:
            logger.error(f"❌ Error in vLLM generation: {e}")
            import traceback
            traceback.print_exc()
            # Fallback action
            return [f"Error in generation: {str(e)}"] * len(states)
    
    def compute_log_probs(self, states: List[List[Dict[str, str]]], actions: List[str]) -> List[float]:
        """
        Compute log probabilities for actions (required for GRPO).
        
        Note: This is a simplified implementation. For full GRPO training,
        you might need to use vLLM's logprobs functionality.
        """
        
        logger.info(f"🧮 Computing log probabilities for {len(actions)} actions...")
        
        try:
            # Use vLLM's logprobs functionality
            prompts = []
            for state, action in zip(states, actions):
                formatted_prompt = self._format_conversation_for_generation(state)
                full_prompt = formatted_prompt + action
                prompts.append(full_prompt)
            
            # Generate with logprobs enabled
            sampling_params_with_logprobs = SamplingParams(
                max_tokens=1,  # Just compute logprobs, don't generate
                logprobs=1,  # Return log probabilities
                prompt_logprobs=1,  # Include prompt logprobs
                temperature=0.0  # Deterministic
            )
            
            outputs = self.llm.generate(prompts, sampling_params_with_logprobs)
            
            log_probs = []
            for output in outputs:
                if output.outputs[0].logprobs:
                    # Extract average log probability
                    logprob_sum = sum(token_logprob.logprob for token_logprob in output.outputs[0].logprobs)
                    avg_logprob = logprob_sum / len(output.outputs[0].logprobs)
                    log_probs.append(avg_logprob)
                else:
                    log_probs.append(-1.0)  # Default value
            
            logger.info(f"✅ Computed {len(log_probs)} log probabilities")
            return log_probs
            
        except Exception as e:
            logger.warning(f"Failed to compute log probabilities: {e}")
            # Return default values
            return [-1.0] * len(actions)
    
    def enable_eval_mode(self):
        """Enable evaluation mode."""
        self.eval_mode = True
        self.training_mode = False
        logger.info("📊 vLLM Policy set to evaluation mode")
    
    def enable_training_mode(self):
        """Enable training mode."""
        self.training_mode = True
        self.eval_mode = False
        logger.info("🎓 vLLM Policy set to training mode")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"VLLMQwenPolicy(model={self.model_name}, device={self.device})"


# Factory function to create vLLM policy
def create_vllm_policy(config_path: str = None, **kwargs) -> VLLMQwenPolicy:
    """
    Factory function to create vLLM policy instance.
    
    Args:
        config_path: Path to model configuration file
        **kwargs: Additional arguments for VLLMQwenPolicy
        
    Returns:
        VLLMQwenPolicy instance
    """
    
    logger.info("🏭 Creating vLLM policy instance...")
    
    # Use the temp config file that training actually uses
    if config_path is None:
        config_path = "/home/ubuntu/multi_mcp_rl/training/configs/model_config_temp.yaml"
    
    policy = VLLMQwenPolicy(
        config_path=config_path,
        **kwargs
    )
    
    logger.info("✅ vLLM policy created successfully")
    return policy


if __name__ == "__main__":
    # Test the vLLM policy
    print("🧪 Testing vLLM Policy...")
    
    policy = create_vllm_policy()
    
    # Test generation
    test_states = [[
        {"role": "user", "content": "Search for information about Apple Inc stock performance"}
    ]]
    
    actions = policy.generate_action(test_states)
    print(f"Generated action: {actions[0]}")
    
    print("✅ vLLM Policy test completed!")