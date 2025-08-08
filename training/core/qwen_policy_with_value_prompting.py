#!/usr/bin/env python3
"""
QwenPolicyWithValue enhanced with prompting for untrained models.
"""

from .qwen_policy_with_prompting import QwenPolicyWithPrompting
import torch
import torch.nn as nn
import logging

class ValueHead(nn.Module):
    """Value head for state value estimation."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass returning value estimates."""
        return self.layers(hidden_states).squeeze(-1)

logger = logging.getLogger(__name__)


class QwenPolicyWithValuePrompting(QwenPolicyWithPrompting):
    """QwenPolicy with value head and enhanced prompting."""
    
    def __init__(self, *args, value_head_hidden_dim: int = 1024, **kwargs):
        """Initialize policy with value head."""
        super().__init__(*args, **kwargs)
        
        # Get model hidden size
        if hasattr(self.model.config, 'hidden_size'):
            model_hidden_size = self.model.config.hidden_size
        elif hasattr(self.model.config, 'd_model'):
            model_hidden_size = self.model.config.d_model
        else:
            # Fallback for Qwen models
            model_hidden_size = 896  # Qwen2.5-0.5B hidden size
            logger.warning(f"Could not determine model hidden size, using {model_hidden_size}")
        
        logger.info(f"Model hidden size from config: {model_hidden_size}")
        
        # Add value head
        self.value_head = ValueHead(
            input_dim=model_hidden_size,
            hidden_dim=value_head_hidden_dim
        )
        
        # Move value head to same device as model
        self.value_head = self.value_head.to(self.device)
        
        # Count value head parameters
        value_params = sum(p.numel() for p in self.value_head.parameters())
        logger.info(f"Added value head with {value_params:,} parameters")
        
    def compute_values(self, states):
        """Compute value estimates for states."""
        # Get hidden states from model
        formatted_inputs = []
        for state in states:
            formatted = self.format_conversation(state)
            formatted_inputs.append(formatted)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config["max_length"]
        ).to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Get last hidden states
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Pool hidden states (mean pooling)
        attention_mask = inputs["attention_mask"]
        masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
        summed = masked_hidden.sum(dim=1)
        counts = attention_mask.sum(dim=1, keepdim=True)
        pooled_hidden = summed / counts
        
        # Compute values
        values = self.value_head(pooled_hidden)
        
        return values
    
    def get_trainable_parameters(self) -> int:
        """Get total number of trainable parameters including value head."""
        base_params = super().get_trainable_parameters()
        value_params = sum(p.numel() for p in self.value_head.parameters() if p.requires_grad)
        return base_params + value_params
    
    def enable_training_mode(self):
        """Enable training mode for both model and value head."""
        super().enable_training_mode()
        self.value_head.train()
        for param in self.value_head.parameters():
            param.requires_grad = True
    
    def enable_eval_mode(self):
        """Enable eval mode for both model and value head."""
        super().enable_eval_mode()
        self.value_head.eval()
    
    def save_checkpoint(self, checkpoint_dir: str):
        """Save model and value head checkpoint."""
        import os
        
        # Save base model
        super().save_checkpoint(checkpoint_dir)
        
        # Save value head
        value_head_path = os.path.join(checkpoint_dir, "value_head.pt")
        torch.save(self.value_head.state_dict(), value_head_path)
        logger.info(f"Saved value head to {value_head_path}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model and value head checkpoint."""
        import os
        
        # Load base model
        super().load_checkpoint(checkpoint_dir)
        
        # Load value head
        value_head_path = os.path.join(checkpoint_dir, "value_head.pt")
        if os.path.exists(value_head_path):
            self.value_head.load_state_dict(torch.load(value_head_path, map_location=self.device))
            logger.info(f"Loaded value head from {value_head_path}")
        else:
            logger.warning(f"Value head checkpoint not found at {value_head_path}")