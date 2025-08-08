"""
QwenPolicy with Value Head: Enhanced policy class for GRPO training with critic

This module extends QwenPolicy to include a value head for computing state values,
which is critical for reducing variance in policy gradient estimation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Dict, Optional, Tuple
from .qwen_policy import QwenPolicy

logger = logging.getLogger(__name__)


class ValueHead(nn.Module):
    """
    Value head network that outputs scalar value estimates for states.
    
    Takes the last hidden state from the language model and projects it
    to a scalar value estimate.
    """
    
    def __init__(self, hidden_size: int, hidden_dim: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_dim
        
        # Two-layer MLP with LayerNorm for stability
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize value head weights with small values"""
        # Use smaller initialization to prevent exploding values
        nn.init.normal_(self.fc1.weight, std=0.001)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.001)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value head.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] tensor from LM
            
        Returns:
            values: [batch_size] tensor of value estimates
        """
        # Apply layer norm
        hidden_states = self.ln(hidden_states)
        
        # Take the last token's hidden state (often best for value estimation)
        # Shape: [batch_size, hidden_size]
        last_hidden = hidden_states[:, -1, :]
        
        # Pass through MLP with numerical stability
        x = self.fc1(last_hidden)
        x = self.activation(x)
        values = self.fc2(x).squeeze(-1)  # [batch_size]
        
        # Clamp values to prevent explosion during training
        values = torch.clamp(values, min=-10.0, max=10.0)
        
        return values


class QwenPolicyWithValue(QwenPolicy):
    """
    QwenPolicy extended with a value head for GRPO training.
    
    This class adds value function capabilities to the base QwenPolicy,
    enabling proper advantage estimation and reduced variance in policy gradients.
    """
    
    def __init__(self, *args, value_head_hidden_dim=1024, **kwargs):
        """Initialize policy with value head"""
        # Remove value_head_hidden_dim from kwargs before passing to parent
        self.value_head_hidden_dim = value_head_hidden_dim
        super().__init__(*args, **kwargs)
        
        # Get hidden size from model config
        if hasattr(self.model, 'config'):
            hidden_size = self.model.config.hidden_size
            logger.info(f"Model hidden size from config: {hidden_size}")
        else:
            # Default for Qwen2.5 models
            hidden_size = 896  # Qwen2.5-0.5B actually has 896 hidden size
            logger.warning(f"Using default hidden size: {hidden_size}")
        
        # Create value head
        self.value_head = ValueHead(hidden_size, self.value_head_hidden_dim).to(self.device)
        
        # Ensure value head parameters require gradients
        for param in self.value_head.parameters():
            param.requires_grad = True
            
        # Initialize value head to output zeros initially to prevent NaN
        with torch.no_grad():
            # Set final layer to output exactly 0.5 (neutral value)
            self.value_head.fc2.weight.data.fill_(0.0)
            self.value_head.fc2.bias.data.fill_(0.5)
            # Also initialize first layer more conservatively
            self.value_head.fc1.weight.data.mul_(0.01)  # Scale down by 100x
        
        logger.info(f"Added value head with {sum(p.numel() for p in self.value_head.parameters())} parameters")
    
    def compute_values(
        self,
        states: List[List[Dict[str, str]]]
    ) -> torch.Tensor:
        """
        Compute value estimates for given states.
        
        Args:
            states: List of conversation histories (list of message dicts)
            
        Returns:
            Tensor of value estimates with shape [batch_size]
        """
        if not states:
            return torch.tensor([], device=self.device)
        
        # Format conversations
        formatted_inputs = []
        for state in states:
            formatted = self.format_conversation(state)
            formatted_inputs.append(formatted)
        
        # Tokenize inputs
        inputs = self.tokenizer(
            formatted_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config["max_length"]
        ).to(self.device)
        
        # Get model outputs with hidden states
        with torch.no_grad() if not self.model.training else torch.enable_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Extract last hidden states
        # outputs.hidden_states is a tuple of tensors, one for each layer
        # We want the last layer's hidden states
        last_hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Debug: Check for NaN in hidden states
        if torch.isnan(last_hidden_states).any() or torch.isinf(last_hidden_states).any():
            logger.error(f"NaN/Inf in last_hidden_states! Shape: {last_hidden_states.shape}")
            logger.error(f"Hidden states stats - min: {last_hidden_states.min()}, max: {last_hidden_states.max()}")
            # Return zeros instead of propagating NaN
            return torch.zeros(len(states), device=self.device, dtype=torch.float32)
        
        # Pass through value head
        values = self.value_head(last_hidden_states)
        
        # Additional safety check
        if torch.isnan(values).any() or torch.isinf(values).any():
            logger.error(f"NaN/Inf in value head output! Returning zeros.")
            return torch.zeros_like(values)
        
        return values
    
    def compute_values_and_log_probs(
        self,
        states: List[List[Dict[str, str]]],
        actions: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both values and log probabilities in a single forward pass.
        More efficient than computing them separately.
        
        Args:
            states: List of conversation histories
            actions: List of action strings
            
        Returns:
            Tuple of (values, log_probs) tensors
        """
        if len(states) != len(actions):
            raise ValueError(f"States and actions length mismatch: {len(states)} vs {len(actions)}")
        
        if not states:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        
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
        
        # Move to device
        target_device = self.device if hasattr(self, 'device') else self.model.device
        inputs_tokenized = {k: v.to(target_device) for k, v in inputs_tokenized.items()}
        full_tokenized = {k: v.to(target_device) for k, v in full_tokenized.items()}
        
        # Single forward pass for both values and log probs
        outputs = self.model(
            input_ids=full_tokenized["input_ids"],
            attention_mask=full_tokenized["attention_mask"],
            output_hidden_states=True,
            return_dict=True
        )
        
        # Compute values from hidden states of input portion
        # We need the hidden states at the position where the input ends
        input_lengths = inputs_tokenized["attention_mask"].sum(dim=1)
        last_hidden_states = outputs.hidden_states[-1]
        
        # Extract hidden states at input end positions
        batch_size = last_hidden_states.shape[0]
        value_hidden_states = []
        for i in range(batch_size):
            pos = min(input_lengths[i] - 1, last_hidden_states.shape[1] - 1)
            value_hidden_states.append(last_hidden_states[i:i+1, pos:pos+1, :])
        
        value_hidden_states = torch.cat(value_hidden_states, dim=0)
        values = self.value_head(value_hidden_states)
        
        # Compute log probabilities for actions
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract log probabilities for action tokens
        batch_log_probs = []
        
        for i in range(batch_size):
            input_length = input_lengths[i].item()
            full_length = full_tokenized["attention_mask"][i].sum().item()
            
            action_start = min(input_length, full_length - 1)
            action_end = full_length
            
            if action_start >= action_end:
                batch_log_probs.append(torch.tensor(-1000.0, device=target_device))
                continue
            
            # Extract action token IDs and their log probabilities
            action_token_ids = full_tokenized["input_ids"][i, action_start:action_end]
            action_log_probs = log_probs[i, action_start-1:action_end-1]
            
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
                total_log_prob = torch.stack(token_log_probs).sum()
                # Final numerical stability check
                if torch.isfinite(total_log_prob):
                    batch_log_probs.append(total_log_prob)
                else:
                    batch_log_probs.append(torch.tensor(-10.0, device=target_device))
            else:
                batch_log_probs.append(torch.tensor(-10.0, device=target_device))
        
        log_probs_tensor = torch.stack(batch_log_probs)
        
        return values, log_probs_tensor
    
    def save_model(self, save_path: str) -> None:
        """Save the model, tokenizer, and value head."""
        # Save base model using parent method
        super().save_model(save_path)
        
        # Save value head separately
        value_head_path = f"{save_path}/value_head.pt"
        torch.save({
            'state_dict': self.value_head.state_dict(),
            'hidden_size': self.value_head.hidden_size,
            'hidden_dim': self.value_head.hidden_dim,
        }, value_head_path)
        
        logger.info(f"Value head saved to {value_head_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model and value head from checkpoint."""
        # Load base model using parent method
        super().load_checkpoint(checkpoint_path)
        
        # Load value head
        value_head_path = f"{checkpoint_path}/value_head.pt"
        if os.path.exists(value_head_path):
            checkpoint = torch.load(value_head_path, map_location=self.device)
            self.value_head.load_state_dict(checkpoint['state_dict'])
            logger.info(f"Value head loaded from {value_head_path}")
        else:
            logger.warning(f"Value head checkpoint not found at {value_head_path}")
    
    def get_trainable_parameters(self) -> int:
        """Get the number of trainable parameters including value head."""
        base_params = super().get_trainable_parameters()
        value_head_params = sum(p.numel() for p in self.value_head.parameters() if p.requires_grad)
        return base_params + value_head_params
    
    def enable_training_mode(self) -> None:
        """Enable training mode for both model and value head."""
        super().enable_training_mode()
        self.value_head.train()
        
        # Ensure value head parameters have gradients
        for param in self.value_head.parameters():
            param.requires_grad = True
    
    def enable_eval_mode(self) -> None:
        """Enable evaluation mode for both model and value head."""
        super().enable_eval_mode()
        self.value_head.eval()
        
        # Disable gradients for value head during eval
        for param in self.value_head.parameters():
            param.requires_grad = False
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save complete checkpoint including model and value head."""
        # Create checkpoint directory
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model and tokenizer using the parent save_model method
        self.save_model(checkpoint_path)
        
        # Save additional training state if needed
        checkpoint_info = {
            'model_config': self.model_config,
            'training_config': self.training_config,
            'use_lora': self.use_lora,
            'value_head_hidden_dim': self.value_head_hidden_dim,
        }
        
        info_path = os.path.join(checkpoint_path, 'checkpoint_info.json')
        import json
        with open(info_path, 'w') as f:
            # Convert any non-serializable items to strings
            serializable_info = {}
            for k, v in checkpoint_info.items():
                try:
                    json.dumps(v)  # Test if serializable
                    serializable_info[k] = v
                except:
                    serializable_info[k] = str(v)
            json.dump(serializable_info, f, indent=2)
        
        logger.info(f"Complete checkpoint saved to {checkpoint_path}")
    
    def __repr__(self) -> str:
        """String representation including value head info."""
        base_repr = super().__repr__()
        value_params = sum(p.numel() for p in self.value_head.parameters())
        return base_repr.replace(")", f", value_head_params={value_params:,})")