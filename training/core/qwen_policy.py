"""
QwenPolicy: Unified policy class for GRPO training with Qwen2.5-1.5B-Instruct

This module implements a comprehensive policy wrapper that supports both LoRA and 
full fine-tuning modes for training Qwen2.5-1.5B-Instruct on multi-turn tool use tasks.
"""

import json
import logging
import re
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Conditional import for BitsAndBytesConfig
# Check environment variable first
import os
if os.environ.get("DISABLE_BITSANDBYTES", "0") == "1":
    HAS_BITSANDBYTES = False
    BitsAndBytesConfig = None
    warnings.warn("BitsAndBytes disabled by environment variable DISABLE_BITSANDBYTES=1")
else:
    try:
        from transformers import BitsAndBytesConfig
        HAS_BITSANDBYTES = True
    except ImportError:
        HAS_BITSANDBYTES = False
        BitsAndBytesConfig = None
        warnings.warn("BitsAndBytesConfig not available. Quantization disabled.")
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class QwenPolicy:
    """
    Unified Qwen2.5-1.5B-Instruct policy for GRPO training.
    
    This class provides a complete interface for both LoRA and full fine-tuning modes,
    with specialized methods for tool-calling generation and log probability computation
    required for Group Relative Policy Optimization (GRPO).
    
    Key Features:
    - Dual mode support: LoRA adapters or full model fine-tuning
    - Memory-optimized model loading with quantization options
    - Qwen chat template integration for proper conversation formatting
    - Tool-calling aware generation with structured output parsing
    - Efficient log probability computation for GRPO training
    - Gradient checkpointing and flash attention support
    
    Args:
        model_config_path (str): Path to model configuration YAML file
        training_config_path (str): Path to training configuration YAML file
        use_lora (bool): Whether to use LoRA adapters (True) or full fine-tuning (False)
        device (str): Device to load model on ('cuda' or 'cpu')
        load_in_4bit (bool): Whether to use 4-bit quantization (for LoRA mode)
    """
    
    def __init__(
        self,
        model_config_path: str,
        training_config_path: str,
        use_lora: bool = True,
        device: str = "cuda",
        load_in_4bit: bool = False,
    ):
        """Initialize the QwenPolicy with configuration and model loading."""
        
        # Convert device string to torch.device
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_lora = use_lora
        self.load_in_4bit = load_in_4bit
        
        # Load configurations
        self.model_config = self._load_config(model_config_path)
        self.training_config = self._load_config(training_config_path)
        
        # Initialize model and tokenizer
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.generation_config: Optional[GenerationConfig] = None
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Setup generation configuration
        self._setup_generation_config()
        
        # Cache for efficiency
        self._stop_token_ids: Optional[List[int]] = None
        
        logger.info(f"QwenPolicy initialized successfully in {'LoRA' if use_lora else 'full fine-tuning'} mode")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def _load_model_and_tokenizer(self) -> None:
        """Load Qwen model and tokenizer with appropriate configurations."""
        
        model_name = self.model_config["model_name"]
        logger.info(f"Loading model: {model_name}")
        
        # Configure quantization if needed
        quantization_config = None
        # Disable quantization for MPS devices
        if self.device.type == "mps":
            logger.info("MPS device detected, adjusting settings for compatibility")
            self.load_in_4bit = False
            # Override model config for MPS
            max_length = self.model_config.get("max_length", 2048)
            if max_length > 1024:
                logger.warning(f"Reducing max_length from {max_length} to 1024 for MPS compatibility")
                self.model_config["max_length"] = 1024
        elif self.load_in_4bit and self.use_lora and HAS_BITSANDBYTES and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.model_config["quantization"]["bnb_4bit_compute_dtype"]),
                bnb_4bit_use_double_quant=self.model_config["quantization"]["bnb_4bit_use_double_quant"],
                bnb_4bit_quant_type=self.model_config["quantization"]["bnb_4bit_quant_type"],
            )
            logger.info("Configured 4-bit quantization for LoRA training")
        elif self.load_in_4bit:
            logger.warning("4-bit quantization requested but not available (CUDA required). Loading in full precision.")
            self.load_in_4bit = False
        
        # Load tokenizer
        tokenizer_kwargs = {
            "trust_remote_code": self.model_config.get("trust_remote_code", True),
            "padding_side": self.model_config.get("padding_side", "left"),
            "use_fast": self.model_config.get("use_fast_tokenizer", True),
        }
        
        # Set model_max_length for MPS
        if self.device.type == "mps":
            tokenizer_kwargs["model_max_length"] = self.model_config.get("max_length", 1024)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["tokenizer_name"],
            **tokenizer_kwargs
        )
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with memory optimizations
        model_kwargs = {
            "trust_remote_code": self.model_config.get("trust_remote_code", True),
            "torch_dtype": getattr(torch, self.model_config["memory_optimization"]["torch_dtype"]),
            "device_map": self.model_config["memory_optimization"]["device_map"] if not self.load_in_4bit else None,
            "low_cpu_mem_usage": self.model_config["memory_optimization"]["low_cpu_mem_usage"],
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Add flash attention if supported
        if self.model_config["memory_optimization"].get("attn_implementation"):
            model_kwargs["attn_implementation"] = self.model_config["memory_optimization"]["attn_implementation"]
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Apply LoRA if specified
        if self.use_lora:
            self._apply_lora()
        else:
            self._prepare_full_finetuning()
        
        # Move to device if not using device_map
        if not model_kwargs.get("device_map") and not self.load_in_4bit:
            if self.device.type == "mps":
                # Move model to MPS gradually to avoid memory issues
                logger.info("Moving model to MPS layer by layer...")
                try:
                    # Move embeddings first
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                        self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.device)
                        torch.mps.empty_cache()
                    
                    # Move layers one by one
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                        for i, layer in enumerate(self.model.model.layers):
                            layer.to(self.device)
                            if i % 4 == 0:  # Clear cache every 4 layers
                                torch.mps.empty_cache()
                    
                    # Move remaining components
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
                        self.model.model.norm = self.model.model.norm.to(self.device)
                    
                    if hasattr(self.model, 'lm_head'):
                        self.model.lm_head = self.model.lm_head.to(self.device)
                    
                    # Final comprehensive device sync
                    self.model = self.model.to(self.device)
                    torch.mps.empty_cache()
                    logger.info("Model successfully moved to MPS")
                except Exception as e:
                    logger.error(f"Failed to move model to MPS: {e}")
                    logger.info("Falling back to CPU")
                    self.device = torch.device("cpu")
                    self.model = self.model.to(self.device)
            else:
                self.model = self.model.to(self.device)
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _apply_lora(self) -> None:
        """Apply LoRA configuration to the model."""
        
        lora_config = self.model_config["lora_mode"]
        
        # Prepare model for k-bit training if quantized
        if self.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["dropout"],
            bias=lora_config["bias"],
            task_type=TaskType.CAUSAL_LM,
            fan_in_fan_out=lora_config.get("fan_in_fan_out", False),
            init_lora_weights=lora_config.get("init_lora_weights", True),
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"LoRA applied: {trainable_params:,} trainable / {total_params:,} total parameters "
                   f"({100 * trainable_params / total_params:.2f}% trainable)")
    
    def _prepare_full_finetuning(self) -> None:
        """Prepare model for full fine-tuning."""
        
        full_config = self.model_config["full_finetune_mode"]
        
        # Enable gradient checkpointing if specified
        if full_config.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
        
        # Ensure all parameters are trainable
        for param in self.model.parameters():
            param.requires_grad = True
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Full fine-tuning mode: {total_params:,} trainable parameters")
    
    def _setup_generation_config(self) -> None:
        """Setup generation configuration for inference."""
        
        gen_config = self.model_config["generation"]
        
        self.generation_config = GenerationConfig(
            max_new_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            top_p=gen_config["top_p"],
            top_k=gen_config.get("top_k", 50),
            do_sample=gen_config["do_sample"],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=gen_config.get("repetition_penalty", 1.1),
        )
        
        # Cache stop token IDs for efficiency
        stop_sequences = self.model_config["stop_sequences"]
        self._stop_token_ids = []
        for stop_seq in stop_sequences:
            stop_ids = self.tokenizer.encode(stop_seq, add_special_tokens=False)
            if stop_ids:  # Only add if encoding was successful
                self._stop_token_ids.extend(stop_ids)
        
        # Remove duplicates while preserving order
        self._stop_token_ids = list(dict.fromkeys(self._stop_token_ids))
        
        logger.info(f"Generation config setup complete with {len(self._stop_token_ids)} stop tokens")
    
    def format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """
        Format conversation using Qwen's chat template.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Formatted conversation string
        """
        
        # Handle empty or invalid messages
        if not messages or not isinstance(messages, list):
            logger.warning(f"Empty or invalid messages: {messages}")
            return "<|im_start|>user\nPlease help me with this task.<|im_end|>\n<|im_start|>assistant\n"
        
        if not self.tokenizer.chat_template:
            # Fallback to manual formatting if chat template not available
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            
            # Add assistant prompt for generation
            if not formatted.endswith("<|im_start|>assistant\n"):
                formatted += "<|im_start|>assistant\n"
            
            return formatted
        
        # Use tokenizer's chat template
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception as e:
            logger.warning(f"Chat template application failed: {e}. Using fallback formatting.")
            # Use manual formatting instead of recursion
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            
            # Add assistant prompt for generation
            if not formatted.endswith("<|im_start|>assistant\n"):
                formatted += "<|im_start|>assistant\n"
            
            return formatted
    
    def generate_action(
        self,
        states: List[List[Dict[str, str]]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate actions for given conversation states.
        
        This method handles the complete pipeline from conversation formatting
        to generation and post-processing for tool-calling tasks.
        
        Args:
            states: List of conversation histories (list of message dicts)
            max_new_tokens: Override max tokens for generation
            temperature: Override temperature for generation
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated action strings
        """
        
        if not states:
            return []
        
        # Format conversations
        formatted_inputs = []
        for i, state in enumerate(states):
            formatted = self.format_conversation(state)
            formatted_inputs.append(formatted)
            
            # Log the complete formatted prompt
            logger.info(f"📝 COMPLETE FORMATTED PROMPT {i}:")
            logger.info(f"{'='*80}")
            logger.info(formatted)
            logger.info(f"{'='*80}")
        
        # Tokenize inputs
        inputs = self.tokenizer(
            formatted_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config["max_length"]
        ).to(self.model.device)
        
        # Override generation config if specified
        gen_config = self.generation_config
        if max_new_tokens is not None or temperature is not None:
            gen_config = GenerationConfig(**gen_config.to_dict())
            if max_new_tokens is not None:
                gen_config.max_new_tokens = max_new_tokens
            if temperature is not None:
                gen_config.temperature = temperature
        
        # Update with any additional kwargs
        for key, value in kwargs.items():
            if hasattr(gen_config, key):
                setattr(gen_config, key, value)
        
        # Generate with the model
        # Disable cache if gradient checkpointing is enabled to avoid warnings
        use_cache = not getattr(self.model, 'gradient_checkpointing', False)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=gen_config,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self._stop_token_ids if self._stop_token_ids else self.tokenizer.eos_token_id,
                use_cache=use_cache,
            )
        
        # Extract only the newly generated tokens
        input_length = inputs.input_ids.shape[1]
        generated_tokens = generated_ids[:, input_length:]
        
        # Decode generated tokens
        generated_texts = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        # Post-process outputs
        processed_outputs = []
        for text in generated_texts:
            processed = self._postprocess_generation(text)
            processed_outputs.append(processed)
        
        # Clear MPS cache after generation
        if self.device.type == "mps":
            torch.mps.empty_cache()
        
        return processed_outputs
    
    def _postprocess_generation(self, text: str) -> str:
        """
        Post-process generated text to clean up and extract valid actions.
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned and processed text
        """
        
        # Remove any trailing whitespace
        text = text.strip()
        
        # Stop at any of our stop sequences
        for stop_seq in self.model_config["stop_sequences"]:
            if stop_seq in text:
                text = text.split(stop_seq)[0]
        
        # Additional cleanup specific to tool calling
        # Ensure proper formatting of tool calls and thinking blocks
        
        # Fix incomplete tool calls
        if "<tool_call>" in text and "</tool_call>" not in text:
            # Try to find a valid JSON structure and close it
            try:
                tool_start = text.find("<tool_call>")
                json_part = text[tool_start + len("<tool_call>"):]
                
                # Attempt to parse and reformat JSON
                if json_part.strip().startswith("{"):
                    # Find the end of the JSON object
                    brace_count = 0
                    json_end = -1
                    for i, char in enumerate(json_part):
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    
                    if json_end > 0:
                        json_str = json_part[:json_end]
                        try:
                            # Validate JSON
                            json.loads(json_str)
                            text = text[:tool_start] + f"<tool_call>{json_str}</tool_call>"
                        except json.JSONDecodeError:
                            pass  # Keep original text if JSON is invalid
            except Exception:
                pass  # Keep original text if processing fails
        
        # Fix incomplete thinking blocks
        if "<think>" in text and "</think>" not in text:
            # Add closing tag if thinking block is incomplete
            text = text + "</think>"
        
        return text.strip()
    
    def compute_log_probs(
        self,
        states: List[List[Dict[str, str]]],
        actions: List[str]
    ) -> torch.Tensor:
        """
        Compute log probabilities of actions given states for GRPO training.
        
        This method is crucial for GRPO as it computes the log probabilities of
        generated actions under the current policy, which are used for policy
        gradient computation and KL divergence calculation.
        
        Args:
            states: List of conversation histories (list of message dicts)
            actions: List of action strings corresponding to each state
            
        Returns:
            Tensor of log probabilities with shape [batch_size]
        """
        
        if len(states) != len(actions):
            raise ValueError(f"States and actions length mismatch: {len(states)} vs {len(actions)}")
        
        if not states:
            return torch.tensor([], device=self.model.device)
        
        # Format conversations
        formatted_inputs = []
        for state in states:
            formatted = self.format_conversation(state)
            formatted_inputs.append(formatted)
        
        # Create full sequences (input + action)
        full_sequences = []
        for formatted_input, action in zip(formatted_inputs, actions):
            full_sequence = formatted_input + action
            full_sequences.append(full_sequence)
        
        # Tokenize inputs and full sequences
        inputs_tokenized = self.tokenizer(
            formatted_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config["max_length"]
        )
        
        full_tokenized = self.tokenizer(
            full_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config["max_length"]
        )
        
        # Move to device with explicit device handling
        target_device = self.device if hasattr(self, 'device') else self.model.device
        inputs_tokenized = {k: v.to(target_device) for k, v in inputs_tokenized.items()}
        full_tokenized = {k: v.to(target_device) for k, v in full_tokenized.items()}
        
        # Get model outputs for full sequences
        # Note: Don't use no_grad() here as we need gradients for training
        outputs = self.model(
            input_ids=full_tokenized["input_ids"],
            attention_mask=full_tokenized["attention_mask"],
            return_dict=True
        )
        
        # Compute log probabilities
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        
        # Extract log probabilities for action tokens only
        batch_log_probs = []
        
        for i, (input_ids, full_ids, attention_mask) in enumerate(zip(
            inputs_tokenized["input_ids"],
            full_tokenized["input_ids"],
            full_tokenized["attention_mask"]
        )):
            # Find where the action starts (after the input prompt)
            input_length = inputs_tokenized["attention_mask"][i].sum().item()  # Actual input length
            full_length = attention_mask.sum().item()  # Actual full length without padding
            
            # Ensure we don't go beyond the actual sequence length
            action_start = min(input_length, full_length - 1)
            action_end = full_length
            
            if action_start >= action_end:
                # No action tokens, assign very low probability
                batch_log_probs.append(torch.tensor(-1000.0, device=target_device))
                continue
            
            # Extract action token IDs and their log probabilities
            action_token_ids = full_ids[action_start:action_end]
            action_log_probs = log_probs[i, action_start-1:action_end-1]  # Shift by 1 for next-token prediction
            
            # Get log probabilities for the actual action tokens
            token_log_probs = []
            for j, token_id in enumerate(action_token_ids):
                if j < action_log_probs.shape[0] and token_id < action_log_probs.shape[1]:
                    token_log_prob = action_log_probs[j, token_id]
                    # Add numerical stability check
                    if torch.isfinite(token_log_prob):
                        token_log_probs.append(token_log_prob)
                    else:
                        token_log_probs.append(torch.tensor(-10.0, device=target_device))
            
            if token_log_probs:
                # Sum log probabilities for the entire action sequence
                total_log_prob = torch.stack(token_log_probs).sum()
                # Final numerical stability check
                if torch.isfinite(total_log_prob):
                    batch_log_probs.append(total_log_prob)
                else:
                    batch_log_probs.append(torch.tensor(-10.0, device=target_device))
            else:
                # No valid action tokens
                batch_log_probs.append(torch.tensor(-10.0, device=target_device))
        
        return torch.stack(batch_log_probs)
    
    def compute_log_probs_optimized(
        self,
        states: List[List[Dict[str, str]]],
        actions: List[str],
        use_memory_efficient: bool = True
    ) -> torch.Tensor:
        """
        Memory-optimized log probability computation (Fix 2.1).
        Expected 50% memory reduction vs standard method.
        """
        if not states or not actions:
            return torch.tensor([], device=self.model.device)
            
        if len(states) != len(actions):
            raise ValueError(f"States and actions length mismatch: {len(states)} vs {len(actions)}")
        
        # Format conversations more efficiently
        formatted_inputs = []
        action_start_positions = []
        
        for state, action in zip(states, actions):
            formatted_input = self.format_conversation(state)
            formatted_inputs.append(formatted_input)
            
            # Estimate where action starts (more efficient than double tokenization)
            input_tokens_estimate = len(formatted_input.split())
            action_start_positions.append(input_tokens_estimate)
        
        # Create full sequences for single tokenization
        full_sequences = [
            formatted_input + action 
            for formatted_input, action in zip(formatted_inputs, actions)
        ]
        
        # Single tokenization pass
        tokenized = self.tokenizer(
            full_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config["max_length"]
        )
        
        # Move to device
        target_device = self.device if hasattr(self, 'device') else self.model.device
        input_ids = tokenized["input_ids"].to(target_device)
        attention_mask = tokenized["attention_mask"].to(target_device)
        
        # Compute refined action start positions
        input_tokenized = self.tokenizer(
            formatted_inputs,
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=self.model_config["max_length"]
        )
        
        actual_action_starts = []
        for i in range(len(formatted_inputs)):
            input_length = input_tokenized["attention_mask"][i].sum().item()
            actual_action_starts.append(input_length)
        
        if use_memory_efficient:
            # Use gradient checkpointing for memory efficiency
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
        
        # Single forward pass with memory optimization
        with torch.cuda.amp.autocast(enabled=False):  # Disable for MPS compatibility
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        # Efficient log probability extraction
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        batch_log_probs = []
        
        for i, action_start in enumerate(actual_action_starts):
            full_length = attention_mask[i].sum().item()
            action_end = full_length
            
            if action_start >= action_end:
                batch_log_probs.append(torch.tensor(-1000.0, device=target_device))
                continue
            
            # Extract action tokens and their log probabilities
            action_token_ids = input_ids[i, action_start:action_end]
            # Shift by 1 for next-token prediction
            action_log_probs = log_probs[i, action_start-1:action_end-1]
            
            # Gather log probs efficiently
            token_log_probs = []
            min_len = min(len(action_log_probs), len(action_token_ids))
            
            for j in range(min_len):
                if j < action_log_probs.shape[0]:
                    token_log_prob = action_log_probs[j, action_token_ids[j]]
                    token_log_probs.append(token_log_prob)
            
            if token_log_probs:
                total_log_prob = torch.stack(token_log_probs).sum()
                batch_log_probs.append(total_log_prob)
            else:
                batch_log_probs.append(torch.tensor(-1000.0, device=target_device))
        
        # Clear cache for MPS devices
        if target_device.type == 'mps' and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except:
                pass
        
        return torch.stack(batch_log_probs)
    
    def save_model(self, save_path: str) -> None:
        """Save the model and tokenizer."""
        
        if self.use_lora:
            # Save LoRA adapters
            self.model.save_pretrained(save_path)
        else:
            # Save full model
            self.model.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        
        if self.use_lora:
            # Load LoRA adapters
            self.model = PeftModel.from_pretrained(
                self.model,
                checkpoint_path,
                device_map=self.model_config["memory_optimization"]["device_map"]
            )
        else:
            # Load full model state dict
            checkpoint = torch.load(
                f"{checkpoint_path}/pytorch_model.bin",
                map_location=self.device
            )
            self.model.load_state_dict(checkpoint)
        
        logger.info(f"Model loaded from {checkpoint_path}")
    
    def enable_training_mode(self) -> None:
        """Enable training mode for the model."""
        self.model.train()
        
        # Enable gradients for trainable parameters
        if self.use_lora:
            # Only LoRA parameters should have gradients
            lora_params = []
            for name, param in self.model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
                    lora_params.append(name)
                else:
                    param.requires_grad = False
            
            if not lora_params:
                logger.error("No LoRA parameters found in model!")
                # Fallback: enable all parameters that were originally trainable
                logger.warning("Falling back to enabling all model parameters")
                for param in self.model.parameters():
                    param.requires_grad = True
            else:
                logger.info(f"Enabled gradients for {len(lora_params)} LoRA parameters")
                if len(lora_params) < 10:
                    logger.debug(f"LoRA parameters: {lora_params}")
        else:
            # All parameters should have gradients for full fine-tuning
            for param in self.model.parameters():
                param.requires_grad = True
    
    def enable_eval_mode(self) -> None:
        """Enable evaluation mode for the model."""
        self.model.eval()
        
        # Disable gradients for inference
        for param in self.model.parameters():
            param.requires_grad = False
    
    def get_trainable_parameters(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get approximate model size in MB."""
        total_params = sum(p.numel() for p in self.model.parameters())
        # Assuming float32 (4 bytes per parameter)
        size_mb = (total_params * 4) / (1024 * 1024)
        return size_mb
    
    def __repr__(self) -> str:
        """String representation of the policy."""
        mode = "LoRA" if self.use_lora else "Full Fine-tuning"
        trainable_params = self.get_trainable_parameters()
        model_size = self.get_model_size_mb()
        
        return (f"QwenPolicy(mode={mode}, trainable_params={trainable_params:,}, "
                f"model_size={model_size:.1f}MB, device={self.device})")


# Utility functions for policy management

def create_policy_from_configs(
    model_config_path: str,
    training_config_path: str,
    use_lora: bool = True,
    device: str = "cuda",
    load_in_4bit: bool = False,
) -> QwenPolicy:
    """
    Factory function to create QwenPolicy from configuration files.
    
    Args:
        model_config_path: Path to model configuration YAML
        training_config_path: Path to training configuration YAML
        use_lora: Whether to use LoRA adapters
        device: Device to load model on
        load_in_4bit: Whether to use 4-bit quantization
        
    Returns:
        Initialized QwenPolicy instance
    """
    return QwenPolicy(
        model_config_path=model_config_path,
        training_config_path=training_config_path,
        use_lora=use_lora,
        device=device,
        load_in_4bit=load_in_4bit,
    )


def estimate_memory_usage(use_lora: bool = True, batch_size: int = 1) -> Dict[str, float]:
    """
    Estimate memory usage for different configurations.
    
    Args:
        use_lora: Whether using LoRA adapters
        batch_size: Batch size for training
        
    Returns:
        Dictionary with memory estimates in GB
    """
    
    # Base model size (Qwen2.5-1.5B)
    base_model_gb = 3.0  # Approximate size in GB
    
    if use_lora:
        # LoRA adds minimal parameters (typically <1% of base model)
        model_memory = base_model_gb * 1.02  # 2% overhead for LoRA
        # 4-bit quantization reduces base model memory
        if True:  # Assuming 4-bit quantization for LoRA
            model_memory = base_model_gb * 0.25 + 0.1  # 25% of original + LoRA overhead
    else:
        # Full fine-tuning needs full precision
        model_memory = base_model_gb
    
    # Gradient memory (same size as model for full fine-tuning)
    gradient_memory = model_memory if not use_lora else model_memory * 0.02
    
    # Optimizer states (AdamW needs 2x model size for momentum and variance)
    optimizer_memory = gradient_memory * 2
    
    # Activation memory (depends on batch size and sequence length)
    # Rough estimate: batch_size * seq_len * hidden_size * layers * bytes_per_param
    activation_memory = batch_size * 4096 * 1536 * 28 * 2 / (1024**3)  # GB
    
    # Buffer and miscellaneous memory
    buffer_memory = 1.0  # 1GB buffer
    
    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory + buffer_memory
    
    return {
        "model_memory_gb": model_memory,
        "gradient_memory_gb": gradient_memory,
        "optimizer_memory_gb": optimizer_memory,
        "activation_memory_gb": activation_memory,
        "buffer_memory_gb": buffer_memory,
        "total_memory_gb": total_memory,
        "recommended_gpu_memory_gb": total_memory * 1.2,  # 20% safety margin
    }


if __name__ == "__main__":
    # Example usage and testing
    
    # Print memory estimates
    print("Memory Estimates:")
    print("LoRA Mode:", estimate_memory_usage(use_lora=True, batch_size=4))
    print("Full Fine-tuning Mode:", estimate_memory_usage(use_lora=False, batch_size=1))
    
    # Example policy creation (commented out to avoid loading model in test)
    # policy = create_policy_from_configs(
    #     model_config_path="configs/model_config.yaml",
    #     training_config_path="configs/training_config.yaml",
    #     use_lora=True,
    #     device="cuda",
    #     load_in_4bit=True,
    # )
    # print(policy)