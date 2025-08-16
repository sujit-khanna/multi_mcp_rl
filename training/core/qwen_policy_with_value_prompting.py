#!/usr/bin/env python3
"""
QwenPolicyWithValue enhanced with prompting for untrained models.
"""

from .qwen_policy_with_prompting import QwenPolicyWithPrompting
import torch
import torch.nn as nn
import logging
from typing import List, Dict, Optional

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

        # Store any malformed model outputs for later analysis
        self.malformed_outputs: List[str] = []
        
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

    def generate_action(
        self,
        states: List[List[Dict[str, str]]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """Generate actions and ensure the first token is a tool call.

        If the model fails to begin with a tool call tag, we re-prompt with an
        explicit reminder and log the malformed output for later analysis.
        """

        # First attempt using the standard prompting behaviour
        actions = super().generate_action(
            states,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

        validated_actions: List[str] = []
        forced_mask: List[bool] = []

        reminder = (
            "REMINDER: Respond ONLY with a tool call in the exact format "
            "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"key\": \"value\"}}</tool_call> "
            "and do not include any other text."
        )

        for idx, action in enumerate(actions):
            if action.lstrip().startswith("<tool_call>"):
                validated_actions.append(action)
                forced_mask.append(False)
                continue

            # Log and store malformed output
            logger.warning(
                "Model output did not start with <tool_call>: %s",
                action[:100].replace("\n", " "),
            )
            self.malformed_outputs.append(action)

            # Re-prompt with explicit reminder
            reminder_state = states[idx] + [{"role": "system", "content": reminder}]
            corrected = super().generate_action(
                [reminder_state],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )[0]
            logger.info("Re-prompted model output: %s", corrected)

            validated_actions.append(corrected)
            forced_mask.append(True)

        # Expose mask indicating which actions were corrected
        self.last_forced_mask = forced_mask

        return validated_actions
    
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
        # NOTE: The base class chain (QwenPolicyWithPrompting -> QwenPolicy) does not
        # implement `save_checkpoint`, only `save_model`. The previous call to
        # super().save_checkpoint raised an AttributeError during checkpointing.
        # We instead use the existing `save_model` API which persists the model
        # (LoRA or full) and tokenizer to the given directory.
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.save_model(checkpoint_dir)
        
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
    
    def state_dict(self):
        """Get state dictionary including model and value head."""
        state_dict = {}
        
        # Add model state dict
        if hasattr(self.model, 'state_dict'):
            model_state = self.model.state_dict()
            for key, value in model_state.items():
                state_dict[f'model.{key}'] = value
        
        # Add value head state dict
        value_state = self.value_head.state_dict()
        for key, value in value_state.items():
            state_dict[f'value_head.{key}'] = value
            
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dictionary for model and value head."""
        model_state = {}
        value_state = {}
        
        for key, value in state_dict.items():
            if key.startswith('model.'):
                model_key = key[6:]  # Remove 'model.' prefix
                model_state[model_key] = value
            elif key.startswith('value_head.'):
                value_key = key[11:]  # Remove 'value_head.' prefix
                value_state[value_key] = value
        
        missing_keys = []
        unexpected_keys = []
        
        # Load model state
        if model_state and hasattr(self.model, 'load_state_dict'):
            result = self.model.load_state_dict(model_state, strict=strict)
            if result is not None:
                if isinstance(result, tuple):
                    missing_keys.extend([f'model.{k}' for k in result[0]])
                    unexpected_keys.extend([f'model.{k}' for k in result[1]])
        
        # Load value head state
        if value_state:
            result = self.value_head.load_state_dict(value_state, strict=strict)
            if result is not None:
                if isinstance(result, tuple):
                    missing_keys.extend([f'value_head.{k}' for k in result[0]])
                    unexpected_keys.extend([f'value_head.{k}' for k in result[1]])
        
        return missing_keys, unexpected_keys
    
    def parameters(self):
        """Get all trainable parameters from model and value head."""
        params = []
        
        # Add model parameters
        if hasattr(self.model, 'parameters'):
            params.extend(list(self.model.parameters()))
        
        # Add value head parameters
        params.extend(list(self.value_head.parameters()))
        
        return iter(params)
    
    def named_parameters(self):
        """Get all named parameters from model and value head."""
        named_params = []
        
        # Add model parameters
        if hasattr(self.model, 'named_parameters'):
            for name, param in self.model.named_parameters():
                named_params.append((f'model.{name}', param))
        
        # Add value head parameters
        for name, param in self.value_head.named_parameters():
            named_params.append((f'value_head.{name}', param))
        
        return iter(named_params)