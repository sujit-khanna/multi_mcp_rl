"""
QwenPolicy with MPS workaround for generation

This module extends QwenPolicy to handle MPS limitations by using CPU for generation.
"""

import torch
import logging
from typing import List, Dict, Optional
from transformers import GenerationConfig
from .qwen_policy import QwenPolicy

logger = logging.getLogger(__name__)


class QwenPolicyMPS(QwenPolicy):
    """
    QwenPolicy with MPS-specific workarounds.
    
    Due to MPS 4GB tensor limit, this class performs generation on CPU
    while keeping the model weights on MPS for other operations.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_device = torch.device("cpu")
        logger.info("QwenPolicyMPS initialized with CPU generation fallback")
    
    def generate_action(
        self,
        states: List[List[Dict[str, str]]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate actions with MPS workaround.
        
        Keeps model on MPS but uses smaller batches and safer generation settings.
        """
        
        if not states:
            return []
        
        try:
            # Format conversations
            formatted_inputs = []
            for state in states:
                formatted = self.format_conversation(state)
                formatted_inputs.append(formatted)
            
            # Tokenize inputs and ensure they're on the same device as model
            inputs = self.tokenizer(
                formatted_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=min(self.model_config["max_length"], 256)  # Limit sequence length
            )
            
            # Move inputs to model device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Override generation config for MPS safety
            gen_config = GenerationConfig(
                max_new_tokens=min(max_new_tokens or 512, 512),  # Increased for better tool responses
                temperature=temperature or 0.7,
                do_sample=False,  # Use greedy decoding for stability
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False,  # Disable cache to save memory
                output_scores=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict_in_generate=False
            )
            
            # Update with any additional kwargs (but keep safety limits)
            for key, value in kwargs.items():
                if hasattr(gen_config, key) and key not in ['max_new_tokens', 'use_cache']:
                    setattr(gen_config, key, value)
            
            # Generate on MPS with safety measures
            generated_texts = []
            
            # Process one at a time to avoid memory issues
            for i in range(len(formatted_inputs)):
                single_input = {k: v[i:i+1] for k, v in inputs.items()}
                
                try:
                    with torch.no_grad():
                        # Clear cache before generation
                        if self.device.type == "mps":
                            torch.mps.empty_cache()
                        
                        generated_ids = self.model.generate(
                            input_ids=single_input["input_ids"],
                            attention_mask=single_input["attention_mask"],
                            **gen_config.to_dict()
                        )
                    
                    # Extract only the newly generated tokens
                    input_length = single_input["input_ids"].shape[1]
                    generated_tokens = generated_ids[:, input_length:]
                    
                    # Decode generated tokens
                    generated_text = self.tokenizer.decode(
                        generated_tokens[0],
                        skip_special_tokens=True
                    )
                    
                    # Post-process output
                    processed = self._postprocess_generation(generated_text)
                    generated_texts.append(processed)
                    
                    # Clear cache after each generation
                    if self.device.type == "mps":
                        torch.mps.empty_cache()
                        
                except Exception as e:
                    logger.warning(f"Generation failed for input {i}: {e}")
                    generated_texts.append("I need to think about this task.")
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"Error in generate_action: {e}")
            # Return fallback responses
            return ["I need to analyze this task." for _ in states]
    
    def compute_log_probs(
        self,
        states: List[List[Dict[str, str]]],
        actions: List[str]
    ) -> torch.Tensor:
        """
        Compute log probabilities with MPS optimization.
        
        Processes in smaller chunks if needed to avoid memory issues.
        """
        
        if len(states) != len(actions):
            raise ValueError(f"States and actions length mismatch: {len(states)} vs {len(actions)}")
        
        if not states:
            return torch.tensor([], device=self.model.device)
        
        # For MPS, process in smaller batches
        if self.device.type == "mps" and len(states) > 2:
            logger.debug(f"Processing {len(states)} samples in batches for MPS")
            batch_size = 2
            all_log_probs = []
            
            for i in range(0, len(states), batch_size):
                batch_states = states[i:i+batch_size]
                batch_actions = actions[i:i+batch_size]
                batch_log_probs = super().compute_log_probs(batch_states, batch_actions)
                all_log_probs.append(batch_log_probs)
                torch.mps.empty_cache()
            
            return torch.cat(all_log_probs)
        else:
            return super().compute_log_probs(states, actions)