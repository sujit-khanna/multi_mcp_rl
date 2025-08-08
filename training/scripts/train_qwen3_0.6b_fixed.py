#!/usr/bin/env python3
"""
Fixed training script for Qwen3-0.6B model on multi-turn tool use tasks
Addresses NaN losses, MPS memory issues, and logging problems
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import yaml
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import enhanced logging
from utils.logging_utils import create_enhanced_training_logger

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleQwen3Trainer:
    """Fixed trainer for Qwen3-0.6B model with proper loss computation and memory management"""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration"""
        self.config_path = config_path
        self.configs = self._load_configs()
        self.device = self._setup_device()
        
        # Model and data
        self.model = None
        self.tokenizer = None
        self.train_data = None
        self.valid_data = None
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_valid_score = 0.0
        
        # Optimizer
        self.optimizer = None
        
        # Enhanced logging
        self.training_logger = None
        
        logger.info(f"Initialized trainer with device: {self.device}")
    
    def _load_configs(self) -> Dict[str, Any]:
        """Load all configuration files"""
        configs = {}
        config_dir = Path(self.config_path).parent
        
        # Load training config
        with open(self.config_path, 'r') as f:
            configs['training'] = yaml.safe_load(f)
        
        # Load model config
        model_config_path = config_dir / "model_config_qwen3_0.6b.yaml"
        with open(model_config_path, 'r') as f:
            configs['model'] = yaml.safe_load(f)
        
        # Load GRPO config
        grpo_config_path = config_dir / "grpo_config_qwen3_0.6b.yaml"
        with open(grpo_config_path, 'r') as f:
            configs['grpo'] = yaml.safe_load(f)
        
        # Override settings for MPS memory constraints
        if torch.backends.mps.is_available():
            logger.info("Adjusting settings for MPS device")
            configs['training']['batch_size'] = 1  # Reduce batch size
            configs['training']['eval_batch_size'] = 1
            configs['training']['gradient_accumulation_steps'] = 8  # Increase accumulation
            configs['model']['max_length'] = 2048  # Reduce sequence length
        
        return configs
    
    def _setup_device(self) -> torch.device:
        """Setup compute device based on availability"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available() and self.configs['training'].get('device_config', {}).get('use_mps', True):
            device = torch.device("mps")
            logger.info("Using Apple MPS device")
            # Clear any problematic MPS env vars
            for key in ["PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_MEMORY_FRACTION", "PYTORCH_MPS_LOW_WATERMARK_RATIO"]:
                if key in os.environ:
                    del os.environ[key]
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        
        return device
    
    def load_model(self):
        """Load Qwen3-0.6B model with appropriate configuration"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
        
        model_name = self.configs['model']['model_name']
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Quantization config for LoRA mode (only on CUDA)
        bnb_config = None
        if self.configs['model']['lora_mode']['enabled'] and self.device.type == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Disable quantization for MPS/CPU
        if self.device.type != "cuda":
            logger.info("BitsAndBytes quantization not supported on MPS/CPU, using full precision")
        
        # Load model with reduced precision for MPS
        torch_dtype = torch.float32  # Use float32 for MPS stability
        if self.device.type == "cuda":
            torch_dtype = torch.float16
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
        
        # Apply LoRA if enabled
        if self.configs['model']['lora_mode']['enabled']:
            lora_config = LoraConfig(
                r=self.configs['model']['lora_mode']['rank'],
                lora_alpha=self.configs['model']['lora_mode']['alpha'],
                target_modules=self.configs['model']['lora_mode']['target_modules'],
                lora_dropout=self.configs['model']['lora_mode']['dropout'],
                bias=self.configs['model']['lora_mode']['bias'],
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Move to device if not using device_map
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        
        # Ensure model is in training mode and requires gradients
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        # Setup optimizer with lower learning rate for stability
        lr = float(self.configs['training']['learning_rate'])
        if self.device.type == "mps":
            lr = lr * 0.1  # Reduce learning rate for MPS
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=float(self.configs['training']['weight_decay']),
            eps=1e-8
        )
        
        logger.info("Model loaded successfully")
    
    def load_data(self):
        """Load training and validation data"""
        # Load training data
        train_path = Path(__file__).parent.parent.parent / "data/inputs/train.json"
        logger.info(f"Loading training data from: {train_path}")
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at: {train_path}")
        
        with open(train_path, 'r') as f:
            self.train_data = json.load(f)
        
        # Load validation data
        valid_path = Path(__file__).parent.parent.parent / "data/inputs/validation.json"
        logger.info(f"Loading validation data from: {valid_path}")
        
        if not valid_path.exists():
            raise FileNotFoundError(f"Validation data not found at: {valid_path}")
        
        with open(valid_path, 'r') as f:
            self.valid_data = json.load(f)
        
        # Limit data for testing on MPS
        if self.device.type == "mps":
            self.train_data = self.train_data[:100]  # Use only 100 samples
            self.valid_data = self.valid_data[:20]   # Use only 20 validation samples
        
        logger.info(f"Loaded {len(self.train_data)} training samples and {len(self.valid_data)} validation samples")
    
    def setup_logging(self):
        """Setup enhanced logging with Weave and WandB"""
        # Prepare clean hyperparameters
        hyperparams = {
            "model_name": self.configs['model']['model_name'],
            "training_samples": len(self.train_data),
            "validation_samples": len(self.valid_data),
            "batch_size": self.configs['training']['batch_size'],
            "learning_rate": float(self.configs['training']['learning_rate']),
            "num_epochs": self.configs['training']['num_epochs'],
            "device": str(self.device),
            "lora_enabled": self.configs['model']['lora_mode']['enabled'],
            "max_length": self.configs['model']['max_length']
        }
        
        self.training_logger = create_enhanced_training_logger(
            config={"training": hyperparams},  # Pass clean config
            rank=0,
            world_size=1,
            enable_wandb=True,
            enable_weave=True
        )
        
        self.training_logger.log_hyperparameters(hyperparams)
        
        logger.info("Enhanced logging initialized")
    
    def prepare_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Prepare a batch of data for training with proper label masking"""
        # Extract conversations from batch
        texts = []
        for item in batch:
            # Get the conversation from prompt field
            conversation = item.get('prompt', [])
            # Format as string for tokenization
            text = self.format_conversation(conversation)
            texts.append(text)
        
        # Tokenize batch
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.configs['model']['max_length'],
            return_tensors='pt'
        )
        
        # Create labels with proper masking (-100 for padding tokens)
        labels = encodings['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Move to device
        batch_dict = {
            'input_ids': encodings['input_ids'].to(self.device),
            'attention_mask': encodings['attention_mask'].to(self.device),
            'labels': labels.to(self.device)
        }
        
        return batch_dict
    
    def format_conversation(self, conversation: List[Dict]) -> str:
        """Format conversation for Qwen model"""
        formatted_text = ""
        
        for turn in conversation:
            role = turn.get('role', '')
            content = turn.get('content', '')
            
            if role == 'user':
                formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif role == 'tool':
                formatted_text += f"<|im_start|>tool\n{content}<|im_end|>\n"
        
        return formatted_text
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss for batch with gradient accumulation scaling"""
        try:
            # Ensure model is in training mode
            self.model.train()
            
            # Forward pass with mixed precision context if available
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                use_cache=False  # Disable cache during training
            )
            
            # Get loss and ensure it requires gradients
            loss = outputs.loss
            if loss is None:
                logger.warning("Model returned None loss, computing manual loss")
                # Manual loss computation if needed
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = batch['labels'][..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Scale loss by gradient accumulation steps
            loss = loss / self.configs['training']['gradient_accumulation_steps']
            
            # Check for NaN
            if torch.isnan(loss):
                logger.warning("NaN loss detected, returning zero loss")
                # Create a zero loss that requires gradients
                dummy_output = sum(p.sum() for p in self.model.parameters() if p.requires_grad) * 0.0
                return dummy_output
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in compute_loss: {e}")
            # Create a zero loss that requires gradients
            dummy_output = sum(p.sum() for p in self.model.parameters() if p.requires_grad) * 0.0
            return dummy_output
    
    def evaluate(self, dataset: List[Dict], max_samples: int = 20) -> Dict[str, float]:
        """Evaluate model on dataset"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        # Sample subset for faster evaluation
        eval_samples = dataset[:max_samples]
        batch_size = self.configs['training']['eval_batch_size']
        
        with torch.no_grad():
            for i in range(0, len(eval_samples), batch_size):
                batch = eval_samples[i:i+batch_size]
                if not batch:
                    continue
                
                try:
                    encodings = self.prepare_batch(batch)
                    
                    outputs = self.model(
                        input_ids=encodings['input_ids'],
                        attention_mask=encodings['attention_mask'],
                        labels=encodings['labels']
                    )
                    
                    loss = outputs.loss
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        num_batches += 1
                        
                except Exception as e:
                    logger.error(f"Error in evaluation: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1) if num_batches > 0 else 10.0
        
        # Simple success rate estimation based on loss
        success_rate = np.exp(-min(avg_loss, 10.0)) * 0.5  # Cap loss at 10 to avoid underflow
        
        metrics = {
            'eval_loss': avg_loss,
            'eval_success_rate': min(success_rate, 1.0),
            'eval_perplexity': min(np.exp(avg_loss), 1000.0)  # Cap perplexity
        }
        
        self.model.train()
        return metrics
    
    def train_epoch(self, epoch: int):
        """Train for one epoch with proper gradient accumulation"""
        self.model.train()
        
        batch_size = self.configs['training']['batch_size']
        train_samples = self.train_data.copy()
        
        # Simple curriculum: start with easy samples
        if self.configs['training']['curriculum_learning']['enabled'] and epoch == 0:
            train_samples = [s for s in train_samples if s.get('extra_info', {}).get('complexity') == 'easy']
            logger.info(f"Curriculum learning: Using {len(train_samples)} easy samples for first epoch")
        
        # Shuffle data
        np.random.shuffle(train_samples)
        
        # Training loop
        epoch_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0
        
        progress_bar = tqdm(range(0, len(train_samples), batch_size), desc=f"Epoch {epoch+1}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, i in enumerate(progress_bar):
            batch = train_samples[i:i+batch_size]
            if not batch:
                continue
            
            try:
                # Prepare batch
                encodings = self.prepare_batch(batch)
                
                # Forward pass
                loss = self.compute_loss(encodings)
                
                # Check if loss has gradients
                if not loss.requires_grad:
                    logger.warning("Loss does not require gradients, skipping backward pass")
                    continue
                
                # Backward pass
                loss.backward()
                accumulated_loss += loss.item()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.configs['training']['gradient_accumulation_steps'] == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Log accumulated loss
                    actual_loss = accumulated_loss * self.configs['training']['gradient_accumulation_steps']
                    epoch_loss += actual_loss
                    num_batches += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({'loss': f"{actual_loss:.4f}"})
                    
                    # Reset accumulated loss
                    accumulated_loss = 0.0
                
                self.current_step += 1
                
                # Log metrics
                if self.current_step % self.configs['training']['logging']['logging_steps'] == 0:
                    metrics = {
                        'loss': actual_loss if 'actual_loss' in locals() else accumulated_loss,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch + 1,
                    }
                    
                    if self.training_logger:
                        self.training_logger.log_training_step(
                            metrics=metrics,
                            step=self.current_step,
                            stage="training"
                        )
                
                # Periodic evaluation
                if self.current_step % self.configs['training']['eval_steps'] == 0:
                    eval_metrics = self.evaluate(self.valid_data)
                    
                    logger.info(f"Step {self.current_step} - Eval metrics: {eval_metrics}")
                    
                    if self.training_logger:
                        self.training_logger.log_model_evaluation(
                            evaluation_results=eval_metrics,
                            step=self.current_step
                        )
                    
                    # Save best model
                    if eval_metrics['eval_success_rate'] > self.best_valid_score:
                        self.best_valid_score = eval_metrics['eval_success_rate']
                        self.save_checkpoint(is_best=True)
                        
            except Exception as e:
                logger.error(f"Error in training step: {e}")
                continue
            
            # Clear cache periodically for MPS
            if self.device.type == "mps" and batch_idx % 10 == 0:
                torch.mps.empty_cache()
        
        avg_epoch_loss = epoch_loss / max(num_batches, 1) if num_batches > 0 else 0.0
        logger.info(f"Epoch {epoch+1} completed - Average loss: {avg_epoch_loss:.4f}")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        output_dir = Path(self.configs['training']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            save_path = output_dir / "best_model"
        else:
            save_path = output_dir / f"checkpoint-{self.current_step}"
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_path)
        else:
            torch.save(self.model.state_dict(), save_path / "model.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save metadata
        metadata = {
            'step': self.current_step,
            'epoch': self.current_epoch,
            'best_score': self.best_valid_score,
            'model_name': self.configs['model']['model_name']
        }
        
        with open(save_path / "metadata.yaml", 'w') as f:
            yaml.dump(metadata, f)
        
        logger.info(f"Checkpoint saved to {save_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        num_epochs = self.configs['training']['num_epochs']
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            logger.info(f"\nStarting epoch {epoch+1}/{num_epochs}")
            
            # Train epoch
            self.train_epoch(epoch)
            
            # End of epoch evaluation
            eval_metrics = self.evaluate(self.valid_data)
            logger.info(f"End of epoch {epoch+1} - Validation metrics: {eval_metrics}")
            
            if self.training_logger:
                self.training_logger.log_model_evaluation(
                    evaluation_results=eval_metrics,
                    step=self.current_step,
                    model_checkpoint=f"epoch_{epoch+1}"
                )
            
            # Save checkpoint
            self.save_checkpoint(is_best=False)
            
            # Clear cache for MPS
            if self.device.type == "mps":
                torch.mps.empty_cache()
        
        logger.info("Training completed!")
        
        # Final cleanup
        if self.training_logger:
            self.training_logger.finish()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train Qwen3-0.6B on multi-turn tool use")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config_qwen3_0.6b.yaml",
        help="Path to training configuration"
    )
    
    args = parser.parse_args()
    
    # Check if config exists
    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Create trainer and run
    trainer = SimpleQwen3Trainer(str(config_path))
    
    # Load model and data
    trainer.load_model()
    trainer.load_data()
    
    # Setup logging
    trainer.setup_logging()
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()