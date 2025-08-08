#!/usr/bin/env python3
"""
Training script for Qwen3-0.6B model on multi-turn tool use tasks
Optimized for local execution with enhanced logging
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
    """Simplified trainer for Qwen3-0.6B model focused on quick local testing"""
    
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
        
        return configs
    
    def _setup_device(self) -> torch.device:
        """Setup compute device based on availability"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available() and self.configs['training'].get('device_config', {}).get('use_mps', True):
            device = torch.device("mps")
            logger.info("Using Apple MPS device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        
        return device
    
    def load_model(self):
        """Load Qwen3-0.6B model with appropriate configuration"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
        from torch.optim import AdamW
        
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
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
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
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.configs['training']['learning_rate']),
            betas=(0.9, 0.999),
            weight_decay=float(self.configs['training']['weight_decay'])
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
        
        logger.info(f"Loaded {len(self.train_data)} training samples and {len(self.valid_data)} validation samples")
    
    def setup_logging(self):
        """Setup enhanced logging with Weave and WandB"""
        self.training_logger = create_enhanced_training_logger(
            config=self.configs,
            rank=0,
            world_size=1,
            enable_wandb=True,
            enable_weave=True
        )
        
        # Log hyperparameters
        hyperparams = {
            "model": self.configs['model']['model_name'],
            "training_samples": len(self.train_data),
            "validation_samples": len(self.valid_data),
            **self.configs['training'],
            **self.configs['grpo']
        }
        self.training_logger.log_hyperparameters(hyperparams)
        
        logger.info("Enhanced logging initialized")
    
    def prepare_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Prepare a batch of data for training"""
        # Extract conversations from batch
        conversations = []
        for item in batch:
            # Get the conversation from prompt field
            conversation = item.get('prompt', [])
            # Format as string for tokenization
            text = self.format_conversation(conversation)
            conversations.append(text)
        
        # Tokenize batch
        encodings = self.tokenizer(
            conversations,
            padding=True,
            truncation=True,
            max_length=self.configs['model']['max_length'],
            return_tensors='pt'
        )
        
        # Move to device
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        return encodings
    
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
        """Compute training loss for batch"""
        # Simple next-token prediction loss
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['input_ids']  # For causal LM, labels = input_ids
        )
        
        return outputs.loss
    
    def evaluate(self, dataset: List[Dict], max_samples: int = 100) -> Dict[str, float]:
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
                
                encodings = self.prepare_batch(batch)
                loss = self.compute_loss(encodings)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Simple success rate estimation based on loss
        success_rate = np.exp(-avg_loss) * 0.5  # Rough approximation
        
        metrics = {
            'eval_loss': avg_loss,
            'eval_success_rate': min(success_rate, 1.0),
            'eval_perplexity': np.exp(avg_loss)
        }
        
        self.model.train()
        return metrics
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
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
        
        progress_bar = tqdm(range(0, len(train_samples), batch_size), desc=f"Epoch {epoch+1}")
        
        for batch_idx, i in enumerate(progress_bar):
            batch = train_samples[i:i+batch_size]
            if not batch:
                continue
            
            # Prepare batch
            encodings = self.prepare_batch(batch)
            
            # Forward pass
            loss = self.compute_loss(encodings)
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.configs['training']['gradient_accumulation_steps'] == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1
            self.current_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log metrics
            if self.current_step % self.configs['training']['logging']['logging_steps'] == 0:
                metrics = {
                    'loss': loss.item(),
                    'learning_rate': self.configs['training']['learning_rate'],
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
        
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
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
            'config': self.configs
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