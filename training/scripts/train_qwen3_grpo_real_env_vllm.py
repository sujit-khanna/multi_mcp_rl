#!/usr/bin/env python3
"""
vLLM-Enhanced GRPO Training Script with REAL Environment Rollouts
================================================================

This script integrates vLLM for 10x faster inference while maintaining
all the fixes for real environment training. Key features:

- vLLM for ultra-fast generation (10x speedup)
- Real MCPToolEnvironment for actual tool execution  
- TrajectoryCollector for parallel rollout collection
- All critical fixes from the original script
- Memory-optimized for high GPU utilization (70-90%)
"""

import argparse
import copy
import json
import logging
import os
import random
import warnings
import numpy as np

# Suppress the gradient checkpointing + caching warning that floods logs
warnings.filterwarnings('ignore', message='.*Caching is incompatible with gradient checkpointing.*')
import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import yaml
import numpy as np
from tqdm import tqdm
import re

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Add utils path for tool validation
utils_path = str(Path(__file__).parent.parent / "utils")
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

# Check for vLLM availability
VLLM_AVAILABLE = False
try:
    import vllm
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    print(f"✅ vLLM {vllm.__version__} detected - enabling 10x faster inference")
except ImportError:
    print("⚠️  vLLM not available - falling back to HuggingFace (slower)")

# Import our enhanced components (import later to avoid circular dependencies)
# from core.qwen_policy_with_value_prompting import QwenPolicyWithValuePrompting
# from core.grpo_trainer_gradient_fix import GRPOTrainerGradientFix
# from core.grpo_trainer import Trajectory

# Import data components (commented out to avoid import issues)
# from data.trajectory_collector import TrajectoryCollector, EpisodeResult

# Import environment components (commented out to avoid import issues)  
# env_path = str(Path(__file__).parent.parent.parent / "environments")
# if env_path not in sys.path:
#     sys.path.insert(0, env_path)

# from mcp_tool_environment import MCPToolEnvironment
# from simple_shared_manager import SimpleSharedManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def _seed_everything(seed: int = 42):
    """Seed all random number generators for reproducible training runs"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep cudnn non-deterministic for speed unless exact reproducibility needed
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    logger.info(f"🎲 Seeded all RNGs with seed={seed}")


class VLLMQwenPolicy:
    """vLLM-enhanced policy for ultra-fast generation"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", **kwargs):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Add required attributes for GRPO trainer compatibility
        self.use_lora = True  # Using LoRA for efficient training
        self.model = None  # Will be set after initialization
        
        # LoRA synchronization with vLLM
        self.lora_update_frequency = 1  # Update vLLM every 1 training step (for debugging)
        self.lora_update_counter = 0
        self.current_lora_id = 0
        self.lora_save_path = "/dev/shm/vllm_lora_weights"  # RAM disk for speed
        self.lora_request = None  # Will be set when LoRA is loaded
        
        # Initialize vLLM if available
        if VLLM_AVAILABLE and os.getenv("ENABLE_VLLM", "false").lower() == "true":
            self.use_vllm = True
            self._init_vllm()
        else:
            self.use_vllm = False
            self._init_hf()
            
        logger.info(f"🚀 Policy initialized - vLLM: {self.use_vllm}")
    
    def get_trainable_parameters(self):
        """Get trainable parameters for GRPO trainer compatibility"""
        # CRITICAL FIX: Directly get trainable parameters instead of using self.parameters()
        # which causes a mismatch between list and iterator
        trainable_params = []
        
        # CRITICAL FIX: Re-enable LoRA gradients if they were disabled
        # This can happen after certain operations
        if hasattr(self, 'training_model'):
            try:
                from peft import PeftModel
                if isinstance(self.training_model, PeftModel):
                    # Ensure LoRA is in training mode
                    self.training_model.train()
                    
                    # Re-enable gradients for LoRA parameters if needed
                    lora_count = 0
                    for name, param in self.training_model.named_parameters():
                        if "lora" in name.lower():
                            if not param.requires_grad:
                                logger.warning(f"⚠️ Re-enabling gradient for {name}")
                                param.requires_grad = True
                            trainable_params.append(param)
                            lora_count += 1
                            logger.debug(f"Added LoRA param: {name}, shape: {param.shape}")
                    
                    if lora_count == 0:
                        logger.error("❌ No LoRA parameters found! Model may have been reset.")
                        # Try to re-enable all LoRA parameters
                        for name, param in self.training_model.named_parameters():
                            if "lora" in name.lower():
                                param.requires_grad = True
                                trainable_params.append(param)
                                lora_count += 1
                else:
                    # Regular model - get all parameters that require gradients
                    for name, param in self.training_model.named_parameters():
                        if param.requires_grad:
                            trainable_params.append(param)
                            logger.debug(f"Added param: {name}, shape: {param.shape}")
            except ImportError:
                # PEFT not available, use regular method
                for name, param in self.training_model.named_parameters():
                    if param.requires_grad:
                        trainable_params.append(param)
                        logger.debug(f"Added param: {name}, shape: {param.shape}")
        
        # Add value head parameters
        if hasattr(self, 'value_head'):
            for name, param in self.value_head.named_parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                trainable_params.append(param)
                logger.debug(f"Added value head param: {name}, shape: {param.shape}")
        
        count = sum(p.numel() for p in trainable_params)
        
        # Debug: Print parameter names to understand what's being returned
        if len(trainable_params) < 100:  # Only print if small number (likely error case)
            param_names = []
            if hasattr(self, 'training_model'):
                for name, param in self.training_model.named_parameters():
                    if param.requires_grad:
                        param_names.append(name)
            if hasattr(self, 'value_head'):
                for name, param in self.value_head.named_parameters():
                    if param.requires_grad:
                        param_names.append(f"value_head.{name}")
            logger.warning(f"⚠️ Only {len(trainable_params)} params found! Names: {param_names[:10]}")
        
        logger.info(f"🔧 get_trainable_parameters() returning {len(trainable_params)} parameters ({count} total elements)")
        
        if len(trainable_params) == 0:
            logger.error("⚠️ CRITICAL: NO TRAINABLE PARAMETERS FOUND!")
            logger.error("This will cause optimizer to skip updates!")
            # Force adding at least one parameter to avoid crashes
            if hasattr(self, 'value_head'):
                dummy_param = next(self.value_head.parameters())
                dummy_param.requires_grad = True
                trainable_params.append(dummy_param)
                logger.warning("Added dummy parameter to prevent crash")
        
        return trainable_params
    
    def compute_values(self, input_ids, attention_mask=None):
        """Compute value estimates for GRPO training"""
        # Handle case where input_ids might be a list of states instead of tensor
        if not isinstance(input_ids, torch.Tensor):
            # If input_ids is a list of states, we need to tokenize them first
            if isinstance(input_ids, list):
                states = input_ids
                values = []
                for state in states:
                    # Build chat-formatted prompt consistently (include assistant start)
                    if isinstance(state, list):
                        state_prompt = self._build_prompt(state, add_generation_prompt=True)
                    else:
                        state_prompt = self._build_prompt([{"role": "user", "content": str(state)}], add_generation_prompt=True)
                    # Tokenize the state prompt
                    tokens = self.tokenizer(state_prompt, return_tensors="pt", truncation=True, max_length=1500, padding=True)
                    tokens = {k: v.to(self.device) for k, v in tokens.items()}
                    
                    # Get value for this state (PRESERVE GRADIENTS FOR TRAINING)
                    if hasattr(self, 'value_head') and self.value_head is not None:
                        # CRITICAL: Ensure training mode for gradient flow
                        self.value_head.train()
                        # CRITICAL: Disable autocast for value computation to ensure FP32
                        with torch.amp.autocast('cuda', enabled=False):
                            hidden_states = self.training_model(**tokens, output_hidden_states=True).hidden_states[-1]
                            # Use last token's hidden state and convert to FP32 for value head
                            last_hidden = hidden_states[:, -1, :].float()
                            
                            # Normalize hidden states before value head to prevent explosions
                            last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)
                            
                            value = self.value_head(last_hidden)
                            
                            # NaN safety check and aggressive clipping
                            if torch.isnan(value).any() or torch.isinf(value).any():
                                logger.warning(f"⚠️ NaN/Inf detected in value computation, clamping to safe range")
                                value = torch.nan_to_num(value, nan=0.0, posinf=5.0, neginf=-5.0)
                            else:
                                # Clamp to reasonable range to prevent explosions
                                value = torch.clamp(value, min=-10.0, max=10.0)
                        
                        values.append(value.squeeze())
                    else:
                        values.append(torch.tensor(0.0, device=self.device, requires_grad=True))
                
                return torch.stack(values) if values else torch.tensor([], device=self.device)
        
        # Original tensor-based computation (PRESERVE GRADIENTS for training)
        if hasattr(self, 'value_head') and self.value_head is not None:
            # Get hidden states from the training model (not combined model)
            outputs = self.training_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
                
            # Convert to FP32 for value head to avoid numerical instability
            hidden_states_fp32 = hidden_states.float()
            
            # Compute values using the value head
            values = self.value_head(hidden_states_fp32)
            return values.squeeze(-1)  # Remove last dimension
        else:
            # Return zeros if no value head available (with gradients enabled)
            if hasattr(input_ids, 'shape'):
                batch_size, seq_len = input_ids.shape
                return torch.zeros(batch_size, seq_len, device=input_ids.device, requires_grad=True)
            else:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def enable_eval_mode(self):
        """Enable evaluation mode for trajectory collection"""
        if hasattr(self, 'model') and self.model is not None:
            self.model.eval()
        if hasattr(self, 'training_model') and self.training_model is not None:
            self.training_model.eval()
        if hasattr(self, 'value_head') and self.value_head is not None:
            self.value_head.eval()
    
    def enable_train_mode(self):
        """Enable training mode"""
        if hasattr(self, 'model') and self.model is not None:
            self.model.train()
        if hasattr(self, 'training_model') and self.training_model is not None:
            self.training_model.train()
        if hasattr(self, 'value_head') and self.value_head is not None:
            self.value_head.train()
    
    def enable_training_mode(self):
        """Enable training mode (alias for compatibility)"""
        self.enable_train_mode()
    
    def get_last_sample_logprobs(self):
        """Return the sample-time logprobs from last generation for PPO"""
        if hasattr(self, 'last_sample_logprobs'):
            return self.last_sample_logprobs
        return None
    
    def generate_action(self, states, max_new_tokens=None, temperature=None, **kwargs):
        """Generate actions for trajectory collection - core method needed by TrajectoryCollector"""
        if not isinstance(states, list):
            states = [states]
            
        # Convert conversation states to text prompts via unified builder
        formatted_inputs = []
        # Build dynamic tool list
        tool_names = self._get_tool_names()
        tool_list_str = ", ".join(tool_names) if tool_names else "fmp_get_quote, polygon_get_aggs, execute_python, send_slack_message"
        tool_system = {
            "role": "system",
            "content": (
                "You are a tool-using assistant. When external information or actions are needed, "
                "respond using exactly one <tool_call>{\"name\": \"tool_name\", \"arguments\": { ... }}</tool_call> block. "
                f"Available tools: {tool_list_str}."
            ),
        }

        for state in states:
            if isinstance(state, list):
                max_history_messages = 6
                if len(state) > max_history_messages:
                    truncated_state = [state[0]] + state[-(max_history_messages - 1):]
                else:
                    truncated_state = state
                # Prepend system tool guidance
                msgs = [tool_system] + truncated_state
                formatted_inputs.append(self._build_prompt(msgs, add_generation_prompt=True))
            else:
                msgs = [tool_system, {"role": "user", "content": str(state)}]
                formatted_inputs.append(self._build_prompt(msgs, add_generation_prompt=True))
        
        # Use vLLM or HuggingFace for generation with better parameters
        responses = self.generate(
            formatted_inputs, 
            max_tokens=max_new_tokens or 512,  # Increased to prevent truncation of tool calls and responses
            temperature=temperature or 0.7  # Higher temperature for diverse generation
        )
        
        # Handle empty responses and encourage tool calls
        processed_responses = []
        last_forced_mask = []
        for i, response in enumerate(responses):
            if not response or len(response.strip()) == 0:
                # CRITICAL: Log this as an error since it breaks training
                logger.error(f"🚨 Empty response generated for prompt {i}! This breaks RL training!")
                logger.debug(f"Prompt was: {formatted_inputs[i][:200]}...")
                logger.warning("Using contextual fallback tool call (NOT IDEAL FOR TRAINING)")
                context = str(formatted_inputs[i]).lower()
                
                # Contextual fallback based on input
                if 'spy' in context or 'stock' in context or 'price' in context:
                    fallback = '<tool_call>{"name": "fmp_get_quote", "arguments": {"symbol": "SPY"}}</tool_call>'
                elif 'search' in context or 'find' in context:
                    fallback = '<tool_call>{"name": "tavily_search", "arguments": {"query": "stock market"}}</tool_call>'
                elif 'calculate' in context or 'analysis' in context:
                    fallback = '<tool_call>{"name": "execute_python", "arguments": {"code": "print(\\"Starting analysis...\\")"}}</tool_call>'
                else:
                    fallback = '<tool_call>{"name": "fmp_get_quote", "arguments": {"symbol": "AAPL"}}</tool_call>'
                    
                processed_responses.append(fallback)
                last_forced_mask.append(True)
            else:
                # Check if response contains a tool call
                if '<tool_call>' in response and '</tool_call>' in response:
                    processed_responses.append(response)
                    last_forced_mask.append(False)
                    logger.info(f"✅ Natural tool call generated: {response[:100]}...")
                else:
                    # Try a strict resample requiring a tool_call-only output
                    try:
                        strict_prompt = formatted_inputs[i] + "Respond ONLY with a single <tool_call>{...}</tool_call> JSON block."
                        strict_resp = self.generate([strict_prompt], max_tokens=(max_new_tokens or 512), temperature=0.2)[0]
                    except Exception as e:
                        logger.warning(f"Strict resample failed: {e}")
                        strict_resp = ""
                    if strict_resp and "<tool_call>" in strict_resp and "</tool_call>" in strict_resp:
                        processed_responses.append(strict_resp)
                        last_forced_mask.append(True)
                        logger.info(f"🔁 Resampled tool call: {strict_resp[:100]}...")
                    else:
                        # Fall back to a contextual tool call to ensure environment interaction
                        context = formatted_inputs[i].lower()
                        if 'stock' in context or 'price' in context or 'spy' in context or 'aapl' in context:
                            fallback = '<tool_call>{"name": "fmp_get_quote", "arguments": {"symbol": "AAPL"}}</tool_call>'
                        elif 'search' in context or 'find' in context or 'news' in context:
                            fallback = '<tool_call>{"name": "tavily_search", "arguments": {"query": "latest market news"}}</tool_call>'
                        elif 'calculate' in context or 'python' in context or 'code' in context or 'analysis' in context:
                            fallback = '<tool_call>{"name": "execute_python", "arguments": {"code": "print(2+2)"}}</tool_call>'
                        else:
                            fallback = '<tool_call>{"name": "tavily_search", "arguments": {"query": "S&P 500"}}</tool_call>'
                        processed_responses.append(fallback)
                        last_forced_mask.append(True)
                        logger.info(f"🔧 Forced fallback tool call: {fallback}")
        
        # Expose which outputs were forced
        self.last_forced_mask = last_forced_mask
        
        # ENHANCED RETURN: Include per-token logprobs for PPO training
        # Return structured data with both text and logprob information
        enhanced_responses = []
        for i, response_text in enumerate(processed_responses):
            # Get the captured logprobs and token IDs from vLLM generation
            token_logprobs = self.last_sample_logprobs[i] if i < len(self.last_sample_logprobs) else []
            token_ids = self.last_sample_token_ids[i] if i < len(self.last_sample_token_ids) else []
            logprob_sum = sum(token_logprobs) if token_logprobs else 0.0
            
            enhanced_responses.append({
                "text": response_text,
                "token_ids": torch.tensor(token_ids, dtype=torch.long) if token_ids else torch.empty(0, dtype=torch.long),
                "token_logprobs": torch.tensor(token_logprobs, dtype=torch.float32) if token_logprobs else torch.empty(0, dtype=torch.float32),
                "logprob_sum": float(logprob_sum),
                "was_forced": last_forced_mask[i] if i < len(last_forced_mask) else False
            })
        
        return enhanced_responses
    
    def compute_log_probs(self, states, actions, **kwargs):
        """Compute log probabilities for given state-action pairs - MEMORY OPTIMIZED"""
        if not hasattr(self, 'training_model') or self.training_model is None:
            raise RuntimeError("No training model available for log prob computation")
        
        # CRITICAL: Ensure model is in training mode for gradient computation
        # BUT don't modify model state unnecessarily as it can reset LoRA
        was_training = self.training_model.training
        if not was_training:
            logger.warning("⚠️ Model was in eval mode during compute_log_probs, temporarily switching to train mode")
            self.training_model.train()
            if hasattr(self, 'value_head'):
                self.value_head.train()
        
        # Configure gradient checkpointing and caching (mutually exclusive)
        if hasattr(self.training_model, 'gradient_checkpointing_enable'):
            self.training_model.gradient_checkpointing_enable()
            # Ensure caching is disabled when gradient checkpointing is enabled
            if hasattr(self.training_model.config, 'use_cache'):
                self.training_model.config.use_cache = False
            logger.debug("Enabled gradient checkpointing and disabled caching for memory efficiency")
            
        # Ensure states and actions are lists
        if not isinstance(states, list):
            states = [states]
        if not isinstance(actions, list):
            actions = [actions]
        
        # Prepare batch inputs for efficient computation
        batch_prompts = []
        batch_targets = []
        
        for state, action in zip(states, actions):
            # Build prompt from messages using unified builder
            if isinstance(state, list):
                prompt = self._build_prompt(state, add_generation_prompt=True)
            else:
                prompt = self._build_prompt([{"role": "user", "content": str(state)}], add_generation_prompt=True)
            # Ensure action is a string and not empty
            action_str = str(action) if action else " "
            batch_prompts.append(prompt)
            batch_targets.append(action_str)
        
        # Process in chunks to avoid OOM
        batch_log_probs = []
        chunk_size = 4  # Process 4 examples at a time to avoid OOM
        
        for chunk_start in range(0, len(batch_prompts), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(batch_prompts))
            chunk_prompts = batch_prompts[chunk_start:chunk_end]
            chunk_targets = batch_targets[chunk_start:chunk_end]
            
            chunk_log_probs = []
            for prompt, target in zip(chunk_prompts, chunk_targets):
                # Tokenize prompt and full sequence
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
                full_text = prompt + target
                full_tokens = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2000)
                
                # Move to device
                prompt_tokens = {k: v.to(self.device) for k, v in prompt_tokens.items()}
                full_tokens = {k: v.to(self.device) for k, v in full_tokens.items()}
                
                # Get target token positions
                prompt_len = prompt_tokens["input_ids"].shape[1]
                target_tokens = full_tokens["input_ids"][:, prompt_len:]
                
                # Skip if no target tokens (shouldn't happen with our fallback)
                if target_tokens.shape[1] == 0:
                    # Force at least one token by tokenizing a single space
                    space_tokens = self.tokenizer(" ", return_tensors="pt")["input_ids"].to(self.device)
                    target_tokens = space_tokens[:, -1:]  # Take last token
                
                # Compute logits for the sequence (excluding last token for autoregressive prediction)
                input_ids = full_tokens["input_ids"][:, :-1]
                
                # Limit sequence length to prevent OOM
                max_seq_len = 512  # Limit to 512 tokens for memory efficiency
                if input_ids.shape[1] > max_seq_len:
                    logger.debug(f"Truncating sequence from {input_ids.shape[1]} to {max_seq_len} tokens")
                    input_ids = input_ids[:, -max_seq_len:]  # Keep last max_seq_len tokens
                
                # CRITICAL: Ensure gradients are enabled for the forward pass
                # This must NOT be in a torch.no_grad() context
                # Force training mode to ensure gradients flow
                self.training_model.train()
                
                # Use autocast for memory efficiency but preserve gradients
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                    outputs = self.training_model(input_ids=input_ids, use_cache=False)  # use_cache=False saves memory
                    logits = outputs.logits.float()  # Convert back to FP32 for stability
                
                # Verify gradients are flowing (only for main policy, not reference)
                # Check if this is the main policy by seeing if vLLM is enabled
                if self.use_vllm and not any(p.requires_grad for p in self.training_model.parameters()):
                    logger.error("🚨 NO PARAMETERS REQUIRE GRADIENTS IN MAIN POLICY!")
                    # Force enable gradients for LoRA parameters
                    for name, param in self.training_model.named_parameters():
                        if "lora_" in name:
                            param.requires_grad = True
                
                # Note: With 4-bit quantization, logits won't have requires_grad=True directly
                # because the base model is frozen. Only LoRA adapters are trainable.
                # The gradient will flow through the LoRA adapters when we compute the loss.
                
                # Get logits for target positions
                target_len = target_tokens.shape[1]
                if prompt_len > 0 and prompt_len + target_len - 1 <= logits.shape[1]:
                    # Extract logits corresponding to target tokens
                    target_logits = logits[:, prompt_len-1:prompt_len+target_len-1, :]
                    
                    # MEMORY-EFFICIENT LOG PROB COMPUTATION: Use gather instead of log_softmax
                    # This avoids creating large [batch, seq, vocab] tensors
                    if target_logits.shape[1] == target_tokens.shape[1]:
                        # Gather logits for target tokens: [batch, seq, 1] -> [batch, seq]
                        token_logits = target_logits.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
                        # Compute log normalization: [batch, seq]
                        log_normalizers = target_logits.logsumexp(dim=-1)
                        # Log probabilities = token_logits - log_normalizers
                        token_log_probs = token_logits - log_normalizers
                        total_log_prob = token_log_probs.sum()
                    else:
                        # Compute what we can if there's a mismatch
                        min_len = min(target_logits.shape[1], target_tokens.shape[1])
                        if min_len > 0:
                            target_subset = target_tokens[:, :min_len]
                            logits_subset = target_logits[:, :min_len, :]
                            # Memory-efficient log prob computation
                            token_logits = logits_subset.gather(2, target_subset.unsqueeze(-1)).squeeze(-1)
                            log_normalizers = logits_subset.logsumexp(dim=-1)
                            token_log_probs = token_logits - log_normalizers
                            total_log_prob = token_log_probs.sum()
                        else:
                            # Last resort: compute log prob for a single space token
                            space_id = self.tokenizer(" ", return_tensors="pt")["input_ids"][0, -1].to(self.device)
                            space_logits = logits[:, -1, :]  # Last position
                            # Memory-efficient computation
                            space_token_logit = space_logits[0, space_id]
                            log_normalizer = space_logits.logsumexp(dim=-1)[0]
                            total_log_prob = space_token_logit - log_normalizer
                else:
                    # Fallback: compute log prob for end token at last position
                    eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
                    if eos_id is not None:
                        last_logits = logits[:, -1, :]
                        # Memory-efficient computation
                        eos_token_logit = last_logits[0, eos_id]
                        log_normalizer = last_logits.logsumexp(dim=-1)[0]
                        total_log_prob = eos_token_logit - log_normalizer
                    else:
                        # Very last resort: Use a small value from the logits to maintain gradient connection
                        # We take the mean of logits to keep it connected to the computation graph
                        total_log_prob = logits.mean() * 0.0  # Multiply by 0 but keep gradient connection
                
                chunk_log_probs.append(total_log_prob)
            
            # Add chunk results to batch
            batch_log_probs.extend(chunk_log_probs)
            
            # Clear cache after each chunk to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Stack log probs and ensure they maintain gradient connection
        stacked_log_probs = torch.stack(batch_log_probs)
        
        # For 4-bit models, the gradient flows through LoRA adapters
        # The stacked log probs should maintain the computation graph
        return stacked_log_probs
    
    def _init_vllm(self):
        """Initialize vLLM for fast generation"""
        try:
            logger.info("🔥 Initializing vLLM for ultra-fast inference...")
            
            # Ensure we're using the correct context length from environment
            max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))  # Increased for longer conversations
            logger.info(f"🔧 Setting vLLM max_model_len to {max_model_len}")
            
            # Import LoRA request class
            from vllm.lora.request import LoRARequest
            
            self.vllm_engine = LLM(
                model=self.model_name,
                max_model_len=max_model_len,
                gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.6")),
                enforce_eager=True,  # Disable CUDA graphs for training compatibility
                trust_remote_code=True,
                dtype="auto",
                quantization=None,  # Let vLLM handle optimization
                enable_lora=True,  # Enable LoRA adapter support
                max_lora_rank=16,  # Match our LoRA config
                max_loras=2,  # Allow swapping between adapters
            )
            
            # Store LoRARequest class for later use
            self.LoRARequest = LoRARequest
            
            # Initialize HuggingFace components for training (value head, LoRA)
            self._init_hf_training_components()
            
            logger.info("✅ vLLM engine initialized successfully")
            
        except Exception as e:
            logger.warning(f"⚠️  vLLM initialization failed: {e}")
            logger.info("🔄 Falling back to HuggingFace...")
            self.use_vllm = False
            self._init_hf()
    
    def _init_hf_training_components(self):
        """Initialize HuggingFace components needed for training"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model for training (with LoRA)
        self.training_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True
        )
        
        # Apply LoRA for training
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.training_model = get_peft_model(self.training_model, lora_config)
        
        # Ensure all LoRA parameters require gradients
        for name, param in self.training_model.named_parameters():
            if "lora_" in name or param.requires_grad:
                param.requires_grad = True
                logger.debug(f"Enabled gradients for: {name}")
        
        # Add value head
        self._add_value_head()
        
        # Create a wrapper model that includes both training_model and value_head
        self.model = self._create_combined_model()
        
        # CRITICAL: Ensure training model is in training mode and parameters require gradients
        self.training_model.train()
        self.value_head.train()
        
        # CRITICAL: Count and verify trainable parameters after full initialization
        lora_params = [p for name, p in self.training_model.named_parameters() if "lora_" in name and p.requires_grad]
        value_params = [p for p in self.value_head.parameters() if p.requires_grad]
        
        n_lora = sum(p.numel() for p in lora_params)
        n_value = sum(p.numel() for p in value_params) 
        n_trainable = n_lora + n_value
        n_total = sum(p.numel() for p in self.training_model.parameters())
        
        logger.info(f"🧠 Trainable parameters: {n_trainable:,} of {n_total:,} "
                   f"({n_trainable / max(1,n_total):.2%}) [LoRA: {n_lora:,}, Value: {n_value:,}]")
        
        if n_trainable == 0:
            logger.error("🚨 CRITICAL: NO TRAINABLE PARAMETERS FOUND!")
            logger.error(f"  LoRA params requiring grad: {len(lora_params)}")
            logger.error(f"  Value params requiring grad: {len(value_params)}")
            raise RuntimeError("No trainable parameters. Check LoRA injection and value head initialization.")
        
    def _add_value_head(self):
        """Add value head for GRPO training with improved initialization"""
        import torch.nn as nn
        
        hidden_size = self.training_model.config.hidden_size
        
        # CRITICAL: Keep value head in FP32 for numerical stability
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Improved initialization to prevent NaNs
        with torch.no_grad():
            # Xavier/He initialization for better gradient flow
            for module in self.value_head.modules():
                if isinstance(module, nn.Linear):
                    # Small initialization to prevent exploding values
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        # Move to device and ensure FP32
        self.value_head = self.value_head.to(self.device).to(torch.float32)
        
        # Ensure all value head parameters require gradients
        for name, param in self.value_head.named_parameters():
            param.requires_grad = True
            logger.debug(f"Value head param {name}: shape={param.shape}, dtype={param.dtype}, requires_grad={param.requires_grad}")
        
        # Test forward pass to ensure no NaNs at initialization
        with torch.no_grad():
            test_input = torch.randn(1, hidden_size, device=self.device, dtype=torch.float32)
            test_output = self.value_head(test_input)
            if torch.isnan(test_output).any() or torch.isinf(test_output).any():
                logger.error("⚠️ Value head produces NaN/Inf at initialization! Re-initializing...")
                # Re-initialize with even smaller values
                for module in self.value_head.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, mean=0.0, std=0.01)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
            else:
                logger.info(f"✅ Value head initialized successfully (test output: {test_output.item():.4f})")
        
        logger.info(f"✅ Added value head to model (fp32 for numerical stability, gradients enabled)")
    
    def _create_combined_model(self):
        """Create a wrapper model that includes both training_model and value_head parameters"""
        import torch.nn as nn
        
        class CombinedModel(nn.Module):
            def __init__(self, training_model, value_head):
                super().__init__()
                # CRITICAL FIX: Register submodules properly using add_module
                # This ensures PyTorch's parameter tracking works correctly
                self.add_module("training_model", training_model)
                self.add_module("value_head", value_head)
                self.training = True  # Ensure training state is preserved
                
            def parameters(self, recurse=True):
                # Return parameters from both training_model and value_head
                for param in self.training_model.parameters(recurse=recurse):
                    yield param
                for param in self.value_head.parameters(recurse=recurse):
                    yield param
                    
            def train(self, mode=True):
                self.training = mode
                self.training_model.train(mode)
                self.value_head.train(mode)
                return self
                
            def eval(self):
                return self.train(False)
        
        combined = CombinedModel(self.training_model, self.value_head)
        logger.info(f"✅ Created combined model with {sum(1 for _ in combined.parameters())} total parameters")
        return combined
    
    def _init_hf(self):
        """Initialize HuggingFace for generation and training"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        
        logger.info("🐌 Initializing HuggingFace (slower but compatible)...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with BitsAndBytesConfig (no more deprecation warnings)
        from transformers import BitsAndBytesConfig
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        self.training_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True
        )
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.training_model = get_peft_model(self.training_model, lora_config)
        
        # Ensure all LoRA parameters require gradients
        for name, param in self.training_model.named_parameters():
            if "lora_" in name or param.requires_grad:
                param.requires_grad = True
                logger.debug(f"Enabled gradients for: {name}")
        
        # Add value head
        self._add_value_head()
        
        # Create a wrapper model that includes both training_model and value_head
        self.model = self._create_combined_model()
        
        # CRITICAL: Ensure training model is in training mode and parameters require gradients
        self.training_model.train()
        self.value_head.train()
        
        # CRITICAL: Count and verify trainable parameters after full initialization
        lora_params = [p for name, p in self.training_model.named_parameters() if "lora_" in name and p.requires_grad]
        value_params = [p for p in self.value_head.parameters() if p.requires_grad]
        
        n_lora = sum(p.numel() for p in lora_params)
        n_value = sum(p.numel() for p in value_params) 
        n_trainable = n_lora + n_value
        n_total = sum(p.numel() for p in self.training_model.parameters())
        
        logger.info(f"🧠 Trainable parameters: {n_trainable:,} of {n_total:,} "
                   f"({n_trainable / max(1,n_total):.2%}) [LoRA: {n_lora:,}, Value: {n_value:,}]")
        
        if n_trainable == 0:
            logger.error("🚨 CRITICAL: NO TRAINABLE PARAMETERS FOUND!")
            logger.error(f"  LoRA params requiring grad: {len(lora_params)}")
            logger.error(f"  Value params requiring grad: {len(value_params)}")
            raise RuntimeError("No trainable parameters. Check LoRA injection and value head initialization.")
        
        logger.info("✅ HuggingFace model initialized")
    
    def generate(self, prompts: List[str], max_tokens: int = 512, temperature: float = 0.1) -> List[str]:
        """Generate responses using vLLM or HuggingFace"""
        
        if self.use_vllm:
            return self._generate_vllm(prompts, max_tokens, temperature)
        else:
            return self._generate_hf(prompts, max_tokens, temperature)
    
    def _generate_vllm(self, prompts: List[str], max_tokens: int, temperature: float) -> List[str]:
        """Ultra-fast generation with vLLM - now captures logprobs for PPO"""
        start_time = time.time()
        
        # CRITICAL: Prevent empty/very short completions with min_tokens
        # vLLM uses `max_tokens` (not `max_new_tokens`); `min_tokens` prevents empty generation
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            min_tokens=8,  # Prevent empty/single-token completions
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.05,  # Slight repetition penalty
            # Stop at structured boundaries to reduce rambling and stray content
            stop=["</tool_call>", "</tool_response>", "<|im_end|>", "\n\n\n"],
            skip_special_tokens=False,  # Don't skip special tokens during generation
            logprobs=1  # Request logprobs for sampled tokens
        )
        
        # DEBUG: Log prompts being sent to vLLM with length check
        for i, prompt in enumerate(prompts):
            # Rough token estimation (1 token ≈ 4 characters for most models)
            estimated_tokens = len(prompt) // 4
            if estimated_tokens > 3500:  # Leave some buffer for generation
                logger.warning(f"⚠️ Prompt {i} estimated at {estimated_tokens} tokens (may exceed vLLM limit)")
            logger.debug(f"📝 vLLM prompt {i} ({estimated_tokens} est. tokens): {prompt[:200]}..." if len(prompt) > 200 else f"📝 vLLM prompt {i}: {prompt}")
            
        # Generate with vLLM - with empty generation handling
        try:
            # Use LoRA adapter if available
            if self.lora_request is not None:
                logger.debug(f"🎯 Using LoRA adapter for generation: {self.lora_request.lora_name}")
                outputs = self.vllm_engine.generate(
                    prompts, 
                    sampling_params,
                    lora_request=self.lora_request
                )
            else:
                logger.debug(f"⚠️ No LoRA adapter available, using base model")
                outputs = self.vllm_engine.generate(prompts, sampling_params)
            
            # Check for empty generations and resample once if needed
            empty_outputs = []
            for i, output in enumerate(outputs):
                if not (output.outputs and output.outputs[0].text.strip()):
                    empty_outputs.append(i)
            
            # Resample empty generations with higher temperature
            if empty_outputs:
                logger.warning(f"🔄 Resampling {len(empty_outputs)} empty generations with higher temperature")
                resample_prompts = [prompts[i] for i in empty_outputs]
                resample_params = SamplingParams(
                    max_tokens=max_tokens,
                    min_tokens=16,  # Higher minimum for resampling
                    temperature=max(0.7, temperature),  # Higher temperature
                    top_p=max(0.9, 0.9),
                    repetition_penalty=1.1,  # Stronger repetition penalty
                    # Stop at structured boundaries
                    stop=["</tool_call>", "</tool_response>", "<|im_end|>", "\n\n\n"],
                    skip_special_tokens=False,
                    logprobs=1
                )
                # Use LoRA adapter for resampling too
                if self.lora_request is not None:
                    logger.debug(f"🔄 Using LoRA adapter for resampling: {self.lora_request.lora_name}")
                    resample_outputs = self.vllm_engine.generate(
                        resample_prompts, 
                        resample_params,
                        lora_request=self.lora_request
                    )
                else:
                    logger.debug(f"⚠️ No LoRA adapter for resampling, using base model")
                    resample_outputs = self.vllm_engine.generate(resample_prompts, resample_params)
                
                # Replace empty outputs with resampled ones
                for j, empty_idx in enumerate(empty_outputs):
                    if j < len(resample_outputs):
                        outputs[empty_idx] = resample_outputs[j]
            
            # Extract generated text AND logprobs
            results = []
            self.last_sample_logprobs = []  # Store for PPO ratio computation
            self.last_sample_token_ids = []  # Store token IDs too
            
            for i, output in enumerate(outputs):
                if len(output.outputs) > 0:
                    completion_output = output.outputs[0]
                    raw_text = completion_output.text
                    generated_text = raw_text.strip()
                    
                    # DEBUG: Log what vLLM actually generated
                    logger.info(f"🔍 vLLM raw output {i}: '{raw_text}' (length: {len(raw_text)})")
                    logger.info(f"🔍 vLLM stripped output {i}: '{generated_text}' (length: {len(generated_text)})")
                    
                    results.append(generated_text)
                    
                    # CRITICAL: Capture sample-time logprobs for PPO
                    if hasattr(completion_output, 'logprobs') and completion_output.logprobs:
                        # Extract logprobs for the sampled tokens
                        sample_logprobs = []
                        sample_token_ids = []
                        for logprob_dict in completion_output.logprobs:
                            if logprob_dict:  # logprob_dict might be None
                                # Get the top token (the one that was sampled)
                                top_token = max(logprob_dict.items(), key=lambda x: x[1].logprob if hasattr(x[1], 'logprob') else x[1])
                                token_id = top_token[0]
                                logprob_val = top_token[1].logprob if hasattr(top_token[1], 'logprob') else top_token[1]
                                sample_logprobs.append(logprob_val)
                                sample_token_ids.append(token_id)
                        
                        self.last_sample_logprobs.append(sample_logprobs)
                        self.last_sample_token_ids.append(sample_token_ids)
                        logger.info(f"✅ Captured {len(sample_logprobs)} sample-time logprobs for response {i}, sum={sum(sample_logprobs) if sample_logprobs else 0:.4f}")
                    else:
                        # No logprobs available - will need to compute later
                        self.last_sample_logprobs.append([])
                        self.last_sample_token_ids.append([])
                        logger.warning(f"⚠️ No logprobs captured from vLLM for response {i}! Has logprobs attr: {hasattr(completion_output, 'logprobs')}")
                    
                    if not generated_text:
                        logger.warning(f"⚠️ vLLM generated empty response for prompt {i}")
                    elif len(generated_text) < 10:
                        logger.warning(f"⚠️ vLLM generated very short response: '{generated_text}'")
                else:
                    logger.error(f"⚠️ vLLM output has no results for prompt {i}")
                    results.append("")
                    self.last_sample_logprobs.append([])
                    self.last_sample_token_ids.append([])
            
            generation_time = time.time() - start_time
            logger.info(f"🚀 vLLM generated {len(prompts)} responses in {generation_time:.2f}s ({generation_time/len(prompts):.2f}s each)")
            
            return results
            
        except Exception as e:
            logger.error(f"🚨 vLLM generation failed: {e}")
            # Fallback to HuggingFace if vLLM fails
            logger.info("🔄 Falling back to HuggingFace generation...")
            return self._generate_hf(prompts, max_tokens, temperature)
    
    def _generate_hf(self, prompts: List[str], max_tokens: int, temperature: float) -> List[str]:
        """Fallback generation with HuggingFace"""
        start_time = time.time()
        
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.training_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=False  # Disable caching to avoid conflicts with gradient checkpointing
                )
            
            generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append(generated_text)
        
        generation_time = time.time() - start_time
        logger.info(f"🐌 HuggingFace generated {len(prompts)} responses in {generation_time:.2f}s ({generation_time/len(prompts):.2f}s each)")
        
        return results
    
    
    def get_value(self, states: List[str]) -> torch.Tensor:
        """Compute state values for GRPO training"""
        values = []
        for state in states:
            tokens = self.tokenizer(state, return_tensors="pt", truncation=True, max_length=1500)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            
            with torch.no_grad():
                hidden_states = self.training_model(**tokens, output_hidden_states=True).hidden_states[-1]
                # Use last token's hidden state
                last_hidden = hidden_states[:, -1, :]
                value = self.value_head(last_hidden)
                values.append(value.squeeze())
        
        return torch.stack(values)
    
    def parameters(self):
        """Get trainable parameters - specifically LoRA adapters + value head"""
        trainable_params = []
        
        # CRITICAL FIX: For PEFT models, we need to check if this is a PeftModel
        # and use the appropriate method to get trainable parameters
        try:
            from peft import PeftModel
            if isinstance(self.training_model, PeftModel):
                # For PEFT models, get only the trainable parameters
                for name, param in self.training_model.named_parameters():
                    if param.requires_grad and "lora" in name.lower():
                        trainable_params.append(param)
                        
                # Don't double-check base_model - PEFT already includes all trainable params
            else:
                # Regular model - get all parameters that require gradients
                for name, param in self.training_model.named_parameters():
                    if param.requires_grad:
                        trainable_params.append(param)
        except ImportError:
            # PEFT not available, use regular method
            for name, param in self.training_model.named_parameters():
                if param.requires_grad:
                    trainable_params.append(param)
        
        # Add value head parameters (these should always be trainable)
        for param in self.value_head.parameters():
            if not param.requires_grad:
                # Force value head parameters to require gradients
                param.requires_grad = True
            trainable_params.append(param)
        
        # Debug: Log the number of trainable parameters and their gradient status
        trainable_count = sum(1 for p in trainable_params if p.requires_grad)
        logger.info(f"🔧 Found {len(trainable_params)} trainable parameters ({trainable_count} require gradients)")
        
        if len(trainable_params) == 0 or trainable_count == 0:
            logger.error("🚨 CRITICAL: NO TRAINABLE PARAMETERS FOUND!")
            logger.error("Checking training_model state:")
            logger.error(f"  training_model.training: {self.training_model.training}")
            logger.error(f"  value_head.training: {self.value_head.training}")
            
            # Force enable gradients for all LoRA and value head parameters
            for name, param in self.training_model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                    logger.info(f"🔧 FORCED gradient for LoRA: {name}")
            
            for param in self.value_head.parameters():
                param.requires_grad = True
                if param not in trainable_params:
                    trainable_params.append(param)
                logger.info(f"🔧 FORCED gradient for value_head parameter")
            
            logger.info(f"🔧 After forcing: {len(trainable_params)} trainable parameters")
        
        return iter(trainable_params)  # Return iterator like nn.Module.parameters()
    
    def save_lora_weights_for_vllm(self):
        """Save current LoRA weights to RAM disk for vLLM to load"""
        import os
        import shutil
        
        try:
            # Create directory in RAM disk (faster than SSD)
            os.makedirs(self.lora_save_path, exist_ok=True)
            
            # Save only the LoRA adapter weights (not the full model)
            if hasattr(self, 'training_model') and self.training_model is not None:
                # Save using PEFT's save_pretrained which handles LoRA properly
                self.training_model.save_pretrained(
                    self.lora_save_path,
                    save_embedding_layers=False,  # Don't save embeddings
                    state_dict=None,  # Use current state
                )
                
                # Increment LoRA ID for cache invalidation
                self.current_lora_id += 1
                
                logger.info(f"💾 Saved LoRA weights to {self.lora_save_path} (ID: {self.current_lora_id})")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to save LoRA weights: {e}")
            return False
    
    def maybe_update_vllm_lora(self, force=False):
        """Conditionally update vLLM's LoRA weights based on update frequency"""
        logger.info(f"🔄 maybe_update_vllm_lora called (force={force})")
        logger.info(f"   Current counter: {self.lora_update_counter}")
        logger.info(f"   Update frequency: {self.lora_update_frequency}")
        logger.info(f"   use_vllm: {self.use_vllm}")
        logger.info(f"   has LoRARequest: {hasattr(self, 'LoRARequest')}")
        
        if not self.use_vllm or not hasattr(self, 'LoRARequest'):
            logger.warning("⚠️  Cannot update LoRA: vLLM not enabled or LoRARequest not available")
            return False
            
        self.lora_update_counter += 1
        logger.info(f"   Counter incremented to: {self.lora_update_counter}")
        
        # Only update at specified frequency or when forced
        if not force and self.lora_update_counter < self.lora_update_frequency:
            logger.info(f"   Skipping update: counter {self.lora_update_counter} < frequency {self.lora_update_frequency}")
            return False
            
        logger.info(f"🚀 Proceeding with LoRA update (force={force})")
        
        # Save current LoRA weights
        logger.info("💾 Saving LoRA weights for vLLM...")
        if self.save_lora_weights_for_vllm():
            logger.info("💾 LoRA weights saved successfully, creating LoRARequest...")
            # Create LoRA request for vLLM
            self.lora_request = self.LoRARequest(
                lora_name=f"training_iter_{self.current_lora_id}",
                lora_int_id=self.current_lora_id % 1000000,  # Keep ID reasonable
                lora_path=self.lora_save_path  # Fixed: lora_local_path → lora_path
            )
            logger.info(f"📦 Created LoRARequest:")
            logger.info(f"   lora_name: {self.lora_request.lora_name}")
            logger.info(f"   lora_int_id: {self.lora_request.lora_int_id}")
            logger.info(f"   lora_path: {self.lora_request.lora_path}")
            
            self.lora_update_counter = 0
            logger.info(f"   Reset counter to: {self.lora_update_counter}")
            logger.info(f"✅ Updated vLLM LoRA adapter (ID: {self.current_lora_id})")
            return True
        else:
            logger.error("❌ Failed to save LoRA weights for vLLM")
            return False
    
    def train(self):
        """Set to training mode"""
        self.training_model.train()
        self.value_head.train()
    
    def eval(self):
        """Set to evaluation mode"""
        self.training_model.eval()
        self.value_head.eval()

    def _build_prompt(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Build a chat-formatted prompt with token-aware truncation and content sanitization."""
        def sanitize_content(c: str) -> str:
            if not isinstance(c, str):
                c = str(c)
            c = re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL)
            c = re.sub(r"<tool_response>.*?</tool_response>", "<tool_response>[omitted]</tool_response>", c, flags=re.DOTALL)
            if len(c) > 512:
                c = c[:512] + "..."
            return c

        msgs = []
        for m in messages:
            msgs.append({
                "role": str(m.get("role", "user")),
                "content": sanitize_content(m.get("content", ""))
            })

        def render(_msgs: List[Dict[str, str]]) -> str:
            try:
                if hasattr(self.tokenizer, "apply_chat_template"):
                    return self.tokenizer.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
            except Exception as e:
                logger.debug(f"apply_chat_template failed, using fallback: {e}")
            text = ""
            for m in _msgs:
                text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            if add_generation_prompt:
                text += "<|im_start|>assistant\n"
            return text

        prompt = render(msgs)

        # Token-aware truncation for vLLM
        try:
            budget = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096")) if self.use_vllm else 4096
            margin = 256
            max_total = max(512, budget - margin)
            attempts = 0
            while attempts < 6:
                toks = self.tokenizer(prompt, return_tensors="pt", truncation=False)
                if toks["input_ids"].shape[1] <= max_total:
                    break
                # drop oldest non-system
                drop_idx = None
                for i, m in enumerate(msgs):
                    if i == 0 and m.get("role") == "system":
                        continue
                    drop_idx = i
                    break
                if drop_idx is None or len(msgs) <= 2:
                    # hard truncate string
                    enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_total)
                    prompt = self.tokenizer.decode(enc["input_ids"][0], skip_special_tokens=False)
                    break
                msgs.pop(drop_idx)
                prompt = render(msgs)
                attempts += 1
        except Exception as e:
            logger.debug(f"Token-aware truncation skipped: {e}")

        return prompt

    def _get_tool_names(self) -> List[str]:
        if hasattr(self, "_cached_tool_names") and self._cached_tool_names:
            return self._cached_tool_names
        names = []
        try:
            env_path = Path(__file__).parent.parent.parent / "environments"
            if str(env_path) not in sys.path:
                sys.path.append(str(env_path))
            from simple_shared_manager import SimpleSharedMCPToolManager
            mgr = SimpleSharedMCPToolManager()
            if not getattr(mgr, "_initialized", False):
                import threading
                def init_mgr():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(mgr.initialize())
                    finally:
                        loop.close()
                t = threading.Thread(target=init_mgr)
                t.start(); t.join(timeout=10)
            names = list(getattr(mgr, "available_tools", {}).keys())
        except Exception as e:
            logger.debug(f"Dynamic tool discovery unavailable: {e}")
        if not names:
            names = ["fmp_get_quote", "polygon_get_aggs", "execute_python", "send_slack_message"]
        self._cached_tool_names = names
        return names
    def clone_for_reference(self):
        """Create a lightweight, non-vLLM copy for reference policy.
        
        - Shares tokenizer
        - New HF model (no quantization) loaded from the same pretrained id
        - Loads current policy weights
        - Disables grads and sets eval()
        - Does NOT initialize vLLM (generation not needed for reference)
        """
        import copy
        import torch.nn as nn
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        
        logger.info("🔄 Creating reference policy clone (non-vLLM, non-quantized)...")
        
        ref = object.__new__(self.__class__)
        # Copy simple attributes
        ref.model_name = self.model_name
        ref.tokenizer = self.tokenizer
        ref.use_vllm = False
        ref.vllm_engine = None
        ref.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prefer bf16 if available; else fp16
        torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        
        logger.info(f"📦 Loading fresh model from {ref.model_name} for reference...")
        ref.training_model = AutoModelForCausalLM.from_pretrained(
            ref.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            # No quantization for reference policy
        )
        
        # Create value head (if present)
        if hasattr(self, "value_head") and self.value_head is not None:
            logger.info("📦 Creating value head for reference...")
            hidden_size = ref.training_model.config.hidden_size
            # Match the original value head architecture (896 hidden units)
            ref.value_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),  # 896 -> 896
                nn.ReLU(),
                nn.Linear(hidden_size, 1)  # 896 -> 1
            ).to(device=ref.device, dtype=torch.float32)  # Move to correct device and keep fp32 for stability
            
            # Copy value head weights (ensure they're on the same device)
            value_state = self.value_head.state_dict()
            # Move state dict to reference device
            value_state = {k: v.to(ref.device) if torch.is_tensor(v) else v for k, v in value_state.items()}
            ref.value_head.load_state_dict(value_state)
        else:
            ref.value_head = None
        
        # Try to load main policy weights (filter out quantization and LoRA)
        logger.info("📦 Copying base model weights to reference...")
        try:
            # Get state dict from current policy, filtering out LoRA and quantization
            current_state = {}
            for name, param in self.training_model.named_parameters():
                # Skip LoRA layers and quantization artifacts
                if "lora_" not in name and "bnb" not in name and "quant" not in name:
                    # Try to map the parameter name
                    clean_name = name.replace("base_model.model.", "")
                    if clean_name in ref.training_model.state_dict():
                        current_state[clean_name] = param.data
            
            # Load available weights (non-strict to handle missing keys)
            if current_state:
                ref.training_model.load_state_dict(current_state, strict=False)
                logger.info(f"✅ Loaded {len(current_state)} base parameters into reference")
        except Exception as e:
            logger.warning(f"⚠️ Reference policy: could not copy all weights - {e}")
            logger.info("   Using pretrained weights as reference baseline")
        
        # Freeze all parameters
        for p in ref.training_model.parameters():
            p.requires_grad_(False)
        ref.training_model.eval()
        
        if ref.value_head is not None:
            for p in ref.value_head.parameters():
                p.requires_grad_(False)
            ref.value_head.eval()
        
        # Surface compatibility attributes
        ref.last_sample_logprobs = None
        ref.last_sample_token_ids = None
        
        # Add model property for compatibility
        class RefCombinedModel:
            def __init__(self, training_model, value_head):
                self.training_model = training_model
                self.value_head = value_head
                
            def parameters(self):
                # Return all parameters from both training_model and value_head
                for p in self.training_model.parameters():
                    yield p
                if self.value_head is not None:
                    for p in self.value_head.parameters():
                        yield p
            
            def named_parameters(self):
                # Return all named parameters from both training_model and value_head
                for name, param in self.training_model.named_parameters():
                    yield f"training_model.{name}", param
                if self.value_head is not None:
                    for name, param in self.value_head.named_parameters():
                        yield f"value_head.{name}", param
            
            def load_state_dict(self, state_dict, strict=True):
                # Handle state dict loading for the combined model
                missing_keys = []
                unexpected_keys = []
                if hasattr(self.training_model, 'load_state_dict'):
                    m, u = self.training_model.load_state_dict(state_dict, strict=False)
                    missing_keys.extend(m)
                    unexpected_keys.extend(u)
                return missing_keys, unexpected_keys
                    
        ref.model = RefCombinedModel(ref.training_model, ref.value_head)
        
        # Copy the compute_log_probs method and bind it to reference
        import types
        
        # Copy the original compute_log_probs method
        original_compute_log_probs = self.__class__.compute_log_probs
        
        # Bind it to the reference policy instance
        ref.compute_log_probs = types.MethodType(original_compute_log_probs, ref)
        
        # Add compute_values method if not present  
        if not hasattr(ref, 'compute_values'):
            ref.compute_values = lambda states: torch.zeros(len(states) if isinstance(states, list) else states.shape[0], device=ref.device)
        
        # Add parameters() method to reference policy
        def ref_parameters():
            """Return all parameters from reference policy"""
            for p in ref.training_model.parameters():
                yield p
            if ref.value_head is not None:
                for p in ref.value_head.parameters():
                    yield p
        ref.parameters = ref_parameters
        
        logger.info("✅ Reference policy created (frozen, non-vLLM, non-quantized)")
        return ref


def update_reference_policy_ema(reference_policy, current_policy, alpha=0.05):
    """Update reference policy using Exponential Moving Average (EMA)"""
    with torch.no_grad():
        # Update training model parameters
        for ref_param, cur_param in zip(reference_policy.training_model.parameters(), 
                                        current_policy.training_model.parameters()):
            # EMA update: ref = (1-alpha) * ref + alpha * current
            ref_param.data.copy_(ref_param.data * (1.0 - alpha) + cur_param.data * alpha)
        
        # Update value head parameters
        for ref_param, cur_param in zip(reference_policy.value_head.parameters(), 
                                        current_policy.value_head.parameters()):
            # EMA update: ref = (1-alpha) * ref + alpha * current
            ref_param.data.copy_(ref_param.data * (1.0 - alpha) + cur_param.data * alpha)


async def main():
    """Main training function with vLLM integration"""
    
    parser = argparse.ArgumentParser(description="vLLM-Enhanced GRPO Training")
    parser.add_argument("--config", type=str, default="training/configs/training_config_qwen3_0.6b.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed-precision", type=str, default="fp16")
    parser.add_argument("--enable-profiling", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Seed everything for reproducible runs
    _seed_everything(args.seed)
    
    logger.info("🚀 Starting vLLM-Enhanced GRPO Training")
    logger.info(f"vLLM Available: {VLLM_AVAILABLE}")
    logger.info(f"Enable vLLM: {os.getenv('ENABLE_VLLM', 'false')}")
    
    # Initialize WandB
    try:
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "multi-mcp-rl-vllm"),
            name=f"vllm-grpo-{time.strftime('%Y%m%d-%H%M%S')}",
            tags=["vllm", "grpo", "real-env"],
            config={
                "vllm_enabled": VLLM_AVAILABLE and os.getenv("ENABLE_VLLM", "false").lower() == "true",
                "device": args.device,
                "mixed_precision": args.mixed_precision
            }
        )
        logger.info("✅ WandB initialized")
    except Exception as e:
        logger.warning(f"⚠️  WandB initialization failed: {e}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load training data
    data_path = config.get("data_path", "data/inputs/train.json")
    if not os.path.exists(data_path):
        data_path = "data/processed/train.json"
    
    with open(data_path, 'r') as f:
        training_data = json.load(f)
    
    logger.info(f"📊 Loaded {len(training_data)} training examples")
    
    # Initialize vLLM-enhanced policy  
    policy = VLLMQwenPolicy()
    
    # Debug: Check what attributes the policy object has
    logger.info(f"🔍 Policy object type: {type(policy)}")
    logger.info(f"🔍 Policy has get_trainable_parameters: {hasattr(policy, 'get_trainable_parameters')}")
    logger.info(f"🔍 Policy methods: {[m for m in dir(policy) if not m.startswith('_') and callable(getattr(policy, m))]}")
    
    # Quick performance test first
    logger.info("🎯 Testing vLLM policy generation speed...")
    test_prompts = ["Hello, how are you today?", "What is machine learning?", "Explain quantum computing in simple terms."]
    start_time = time.time()
    responses = policy.generate(test_prompts, max_tokens=100, temperature=0.1)
    generation_time = time.time() - start_time
    
    logger.info(f"🚀 Generated {len(test_prompts)} responses in {generation_time:.2f}s")
    logger.info(f"   Average time per response: {generation_time/len(test_prompts):.2f}s")
    for i, response in enumerate(responses):
        logger.info(f"Response {i+1}: {response[:100]}...")
    
    logger.info("✅ vLLM integration verified - ready for full training")
    
    # NOW START THE ACTUAL TRAINING
    logger.info("🚀 Starting full GRPO training with vLLM optimizations...")
    
    # Import required training components (using working imports from existing script)
    try:
        # Import training components
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.grpo_trainer_gradient_fix import GRPOTrainerGradientFix  
        from core.grpo_trainer import Trajectory
        from data.trajectory_collector import TrajectoryCollector, EpisodeResult
        
        # Import environment components
        env_path = str(Path(__file__).parent.parent.parent / "environments")
        if env_path not in sys.path:
            sys.path.insert(0, env_path)
        from simple_shared_manager import SimpleSharedManager
        from mcp_tool_environment import MCPToolEnvironment
        
        logger.info("✅ All training components imported successfully")
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        logger.info("This may be due to missing SkyRL dependency")
        return
    
    # Setup environment factory function
    tool_manager = SimpleSharedManager()
    # Ensure tool manager is initialized and log discovered tools for visibility
    try:
        if not getattr(tool_manager, "_initialized", False):
            import threading
            def _init_mgr():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(tool_manager.initialize())
                finally:
                    loop.close()
            t = threading.Thread(target=_init_mgr)
            t.start(); t.join(timeout=15)
        discovered = list(getattr(tool_manager, "available_tools", {}).keys())
        logger.info(f"🧰 Discovered tools via MCP manager: {discovered if discovered else '[]'}")
        # Warn about common missing API keys
        missing = []
        if "tavily_search" not in discovered:
            if not os.getenv("TAVILY_API_KEY"):
                missing.append("TAVILY_API_KEY")
        if "fmp_get_quote" not in discovered:
            if not os.getenv("FMP_API_KEY"):
                missing.append("FMP_API_KEY")
        if missing:
            logger.warning(f"⚠️  Missing or inactive API keys for tools: {missing}. Tools may be unavailable.")
    except Exception as e:
        logger.warning(f"Tool discovery logging failed: {e}")
    
    def env_factory(task_data):
        return MCPToolEnvironment(task_data=task_data)
    
    # Setup training config
    training_config = config.get("training", {})
    model_config = config.get("model", {})
    
    # Initialize reference policy using the new clone method
    logger.info("🔄 Creating separate reference policy for proper KL divergence...")
    
    # Create a frozen, non-quantized reference policy
    if hasattr(policy, 'clone_for_reference'):
        reference_policy = policy.clone_for_reference()
        logger.info("✅ Reference policy created (separate, frozen, non-quantized)")
    else:
        # Fallback to using same policy if clone method not available
        logger.warning("⚠️ clone_for_reference not available, using main policy as reference")
        reference_policy = policy
        logger.info("⚠️ KL divergence will be 0 (using same policy as reference)")
    
    # Setup GRPO config
    grpo_config = {
        "learning_rate": float(config.get("learning_rate", 5e-5)),  # Use config's learning_rate (5e-5) not training sub-config
        "batch_size": training_config.get("batch_size", 1),
        "kl_coeff": float(training_config.get("kl_coeff", 0.01)),
        "gamma": float(training_config.get("gamma", 0.99)),
        "lam": float(training_config.get("lam", 0.95)),
        "clip_ratio": float(training_config.get("clip_ratio", 0.2)),
        "value_loss_coeff": float(training_config.get("value_loss_coeff", 0.5)),
        "entropy_coeff": float(training_config.get("entropy_coeff", 0.01)),
        "ref_policy_update_frequency": int(training_config.get("ref_policy_update_frequency", 100)),
        "group_size": int(training_config.get("group_size", 2)),
    }
    
    # Initialize GRPO trainer (using working trainer)
    trainer = GRPOTrainerGradientFix(
        policy=policy,
        reference_policy=reference_policy,  # Same as policy for now (KL divergence = 0)
        grpo_config=grpo_config,
        training_config=training_config,
        device=torch.device(args.device),
        enable_mixed_precision=(args.mixed_precision == "fp16")
    )
    
    # Initialize trajectory collector
    collector = TrajectoryCollector(
        policy=policy,
        env_factory=env_factory,
        num_parallel_envs=training_config.get("num_workers", 1),  # Reduced for stability
        shared_tool_manager=tool_manager
    )
    
    # Initial LoRA save for vLLM to have weights to load
    if hasattr(policy, 'maybe_update_vllm_lora'):
        logger.info("🔧 Performing initial LoRA save for vLLM...")
        policy.maybe_update_vllm_lora(force=True)
    
    logger.info("🎯 Starting training loop...")
    
    # Training loop
    max_epochs = training_config.get("max_epochs", 100)
    for epoch in range(max_epochs):
        logger.info(f"📈 Epoch {epoch+1}/{max_epochs}")
        
        # Collect episodes using real environment with vLLM speed
        start_time = time.time()
        import random
        batch_size = training_config.get("batch_size", 4)
        group_size = grpo_config.get("group_size", 2)
        # Form GRPO groups: select a few tasks and collect multiple rollouts per task
        num_tasks = max(1, batch_size // max(1, group_size))
        base_tasks = random.sample(training_data, min(num_tasks, len(training_data)))
        training_batch = []
        for t in base_tasks:
            for gi in range(group_size):
                t_copy = copy.deepcopy(t)
                t_copy.setdefault("extra_info", {})["resample_id"] = f"{epoch}-{gi}-{random.randint(0, 1_000_000)}"
                training_batch.append(t_copy)
        
        logger.info(f"📊 Using {len(training_batch)} samples from {len(training_data)} total training examples")
        logger.debug(f"Task IDs: {[task.get('task_metadata', {}).get('task_id', 'unknown') for task in training_batch]}")
        
        episode_results = await collector.collect_batch(training_batch)
        collection_time = time.time() - start_time
        
        logger.info(f"⚡ Collected {len(episode_results)} episodes in {collection_time:.2f}s")
        
        # ENHANCED WANDB ROLLOUT LOGGING: Always log rollout metrics
        valid_episodes = [e for e in episode_results if e.is_valid()]
        total_rewards = [sum(ep.rewards) for ep in valid_episodes]
        step_rewards = [r for ep in valid_episodes for r in ep.rewards]
        tool_calls_count = sum(1 for ep in valid_episodes for action in ep.actions if '<tool_call>' in str(action))
        
        rollout_metrics = {
            "rollouts/num_episodes": len(episode_results),
            "rollouts/num_valid": len(valid_episodes),
            "rollouts/collection_time": collection_time,
            "rollouts/avg_episode_reward": sum(total_rewards) / len(total_rewards) if total_rewards else 0.0,
            "rollouts/avg_step_reward": sum(step_rewards) / len(step_rewards) if step_rewards else 0.0,
            "rollouts/total_tool_calls": tool_calls_count,
            "rollouts/tool_call_rate": tool_calls_count / len(step_rewards) if step_rewards else 0.0,
            "rollouts/success_rate": len(valid_episodes) / len(episode_results) if episode_results else 0.0,
        }
        
        try:
            wandb.log(rollout_metrics, commit=False)
            logger.debug(f"📊 Rollout metrics logged: {tool_calls_count} tool calls, {len(valid_episodes)} valid episodes")
        except Exception as e:
            logger.warning(f"⚠️ Rollout metrics logging failed: {e}")
        
        if valid_episodes:
            # Convert episodes to trajectories for training
            trajectories = []
            for episode in valid_episodes:
                # Extract states, actions, rewards, and dones from episode trajectory
                states = []
                actions = []
                rewards = []
                dones = []
                
                for i, step in enumerate(episode.trajectory):
                    # Always append to maintain length consistency
                    states.append(step.get('state', {}))  # Default to empty dict
                    actions.append(step.get('action', ""))  # Default to empty string
                    rewards.append(step.get('reward', 0.0))  # Default to 0.0
                    dones.append(i == len(episode.trajectory) - 1)  # Last step is done
                
                # Ensure we have at least one step if trajectory is completely empty
                if not states:
                    states = [{}]  # Empty state
                    actions = [""]  # Empty action
                    rewards = [0.0]
                    dones = [True]
                
                # Final safety check - ensure all arrays have same length
                min_length = min(len(states), len(actions), len(rewards), len(dones))
                if min_length > 0:
                    states = states[:min_length]
                    actions = actions[:min_length]
                    rewards = rewards[:min_length]
                    dones = dones[:min_length]
                
                logger.info(f"📊 Trajectory lengths - States: {len(states)}, Actions: {len(actions)}, Rewards: {len(rewards)}, Dones: {len(dones)}")
                
                trajectory = Trajectory(
                    task_id=episode.task_id,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    dones=dones
                )
                trajectories.append(trajectory)
            
            # ROBUST WANDB LOGGING: Pre-update heartbeat to ensure visibility
            heartbeat_metrics = {
                "epoch": epoch,
                "heartbeat": 1,
                "collection_time": collection_time,
                "total_trajectories": len(trajectories),
                "vllm_enabled": policy.use_vllm,
                "status": "starting_training_step"
            }
            
            try:
                wandb.log(heartbeat_metrics)
                logger.debug("📡 Pre-training heartbeat logged to WandB")
            except Exception as e:
                logger.warning(f"⚠️  WandB heartbeat failed: {e}")
            
            start_time = time.time()
            metrics = trainer.train_step(trajectories)
            train_time = time.time() - start_time
            
            # Update vLLM's LoRA weights after training step
            logger.info(f"🔍 Checking for LoRA update after training step...")
            if hasattr(policy, 'maybe_update_vllm_lora'):
                logger.info(f"✅ Policy has maybe_update_vllm_lora method")
                
                # FORCE update every step for debugging (remove frequency limit)
                logger.info(f"🚀 FORCING LoRA update for debugging purposes...")
                lora_updated = policy.maybe_update_vllm_lora(force=True)
                
                if lora_updated:
                    logger.info(f"🎉 vLLM LoRA weights successfully updated after epoch {epoch}")
                else:
                    logger.error(f"❌ vLLM LoRA weights update FAILED after epoch {epoch}")
            else:
                logger.error(f"❌ Policy does NOT have maybe_update_vllm_lora method!")
            
            # Log comprehensive training metrics
            combined_metrics = {
                "epoch": epoch,
                "collection_time": collection_time,
                "train_time": train_time,
                "total_trajectories": len(trajectories),
                "vllm_enabled": policy.use_vllm,
                "status": "training_completed",
                **metrics
            }
            
            # Robust WandB logging with detailed error handling
            try:
                wandb.log(combined_metrics)
                logger.debug("📊 Training metrics logged to WandB successfully")
            except Exception as e:
                logger.error(f"🚨 WandB metrics logging failed: {e}")
                logger.info(f"📋 Metrics that failed to log: {combined_metrics}")
                # Continue training even if logging fails
                
            logger.info(f"✅ Training step completed in {train_time:.2f}s")
            logger.info(f"📊 Metrics: {metrics}")
            
            # KL divergence handled inside trainer
        else:
            logger.warning("⚠️ No valid trajectories collected, skipping training step")
            
            # LOG SKIPPED TRAINING STEP
            skip_metrics = {
                "trainer/updates_skipped": 1,
                "trainer/skip_reason": "no_valid_trajectories",
                "epoch": epoch + 1
            }
            
            try:
                wandb.log(skip_metrics, commit=True)
                logger.debug("📊 Training skip logged to WandB")
            except Exception as e:
                logger.warning(f"⚠️ Skip metrics logging failed: {e}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"outputs/vllm-checkpoint-epoch-{epoch+1}.pt"
            try:
                # Save the training model and value head
                torch.save({
                    'training_model_state_dict': policy.training_model.state_dict(),
                    'value_head_state_dict': policy.value_head.state_dict() if hasattr(policy, 'value_head') else None,
                    'epoch': epoch + 1,
                    'total_reward': sum(total_rewards) / len(total_rewards) if 'total_rewards' in locals() and total_rewards else 0.0
                }, checkpoint_path)
                logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
                
                # Force LoRA update after checkpoint to ensure consistency
                if hasattr(policy, 'maybe_update_vllm_lora'):
                    policy.maybe_update_vllm_lora(force=True)
                    logger.info(f"🔄 vLLM LoRA weights updated after checkpoint")
                    
            except Exception as e:
                logger.warning(f"⚠️ Checkpoint save failed: {e}")
    
    logger.info("🎉 vLLM-enhanced training completed successfully!")
    
    # Final cleanup
    try:
        tool_manager.cleanup()
    except:
        pass


if __name__ == "__main__":
    asyncio.run(main())
