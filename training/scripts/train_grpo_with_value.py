#!/usr/bin/env python3
"""
Enhanced GRPO training script with value function support.

This script demonstrates how to use the enhanced GRPO trainer with proper
value function training for reduced variance and faster convergence.
"""

import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any

import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import enhanced components
from training.core.qwen_policy_with_value import QwenPolicyWithValue
from training.core.grpo_trainer_with_value import GRPOTrainerWithValue
from training.core.grpo_trainer import Trajectory
from training.data.trajectory_collector import TrajectoryCollector
from training.utils.logging_utils import create_enhanced_training_logger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedGRPOTraining:
    """Main training class using enhanced GRPO with value function"""
    
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        
        # Load configurations
        self.configs = self._load_configs()
        
        # Initialize enhanced logging
        self.training_logger = None
        if args.enable_logging:
            self._setup_logging()
        
        # Models and trainer
        self.policy = None
        self.reference_policy = None
        self.trainer = None
        
        # Data
        self.train_data = None
        self.valid_data = None
        
        logger.info(f"Enhanced GRPO training initialized on {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        if self.args.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using Apple MPS device")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            device = torch.device(self.args.device)
            logger.info(f"Using specified device: {device}")
        
        return device
    
    def _load_configs(self) -> Dict[str, Any]:
        """Load all configuration files"""
        configs = {}
        
        # Load main training config
        with open(self.args.training_config, 'r') as f:
            configs['training'] = yaml.safe_load(f)
        
        # Load model config
        with open(self.args.model_config, 'r') as f:
            configs['model'] = yaml.safe_load(f)
        
        # Load GRPO config
        with open(self.args.grpo_config, 'r') as f:
            configs['grpo'] = yaml.safe_load(f)
        
        # Ensure value function settings are present
        if 'value_loss_coef' not in configs['grpo']:
            configs['grpo']['value_loss_coef'] = 0.5
            logger.info("Added default value_loss_coef=0.5 to GRPO config")
        
        if 'clip_value_loss' not in configs['grpo']:
            configs['grpo']['clip_value_loss'] = True
            logger.info("Added default clip_value_loss=True to GRPO config")
        
        return configs
    
    def _setup_logging(self):
        """Setup enhanced logging with WandB/Weave"""
        hyperparams = {
            "model": self.configs['model']['model_name'],
            "device": str(self.device),
            "batch_size": self.configs['training'].get('batch_size', 4),
            "learning_rate": self.configs['training'].get('learning_rate', 2e-4),
            "grpo_gamma": self.configs['grpo']['gamma'],
            "grpo_lambda": self.configs['grpo']['gae_lambda'],
            "value_loss_coef": self.configs['grpo']['value_loss_coef'],
            "enhanced_value_training": True  # Flag to indicate we're using the enhanced version
        }
        
        self.training_logger = create_enhanced_training_logger(
            config={"training": hyperparams},
            rank=0,
            world_size=1,
            enable_wandb=self.args.wandb,
            enable_weave=self.args.weave
        )
        
        self.training_logger.log_hyperparameters(hyperparams)
        logger.info("Enhanced logging initialized")
    
    def load_models(self):
        """Load policy and reference policy with value heads"""
        logger.info("Loading enhanced models with value heads...")
        
        # Create main policy with value head
        self.policy = QwenPolicyWithValue(
            model_config_path=self.args.model_config,
            training_config_path=self.args.training_config,
            use_lora=self.configs['model'].get('lora_mode', {}).get('enabled', True),
            device=str(self.device),
            load_in_4bit=self.configs['model'].get('quantization', {}).get('load_in_4bit', False)
        )
        
        # Create reference policy with value head
        self.reference_policy = QwenPolicyWithValue(
            model_config_path=self.args.model_config,
            training_config_path=self.args.training_config,
            use_lora=self.configs['model'].get('lora_mode', {}).get('enabled', True),
            device=str(self.device),
            load_in_4bit=self.configs['model'].get('quantization', {}).get('load_in_4bit', False)
        )
        
        # Initialize reference policy with main policy weights
        with torch.no_grad():
            for ref_param, param in zip(
                self.reference_policy.model.parameters(),
                self.policy.model.parameters()
            ):
                ref_param.data.copy_(param.data)
            
            # Also copy value head weights
            for ref_param, param in zip(
                self.reference_policy.value_head.parameters(),
                self.policy.value_head.parameters()
            ):
                ref_param.data.copy_(param.data)
        
        logger.info(f"Models loaded: {self.policy.get_trainable_parameters():,} trainable parameters")
    
    def load_data(self):
        """Load training and validation data"""
        # Load training data
        with open(self.args.train_data, 'r') as f:
            self.train_data = json.load(f)
        logger.info(f"Loaded {len(self.train_data)} training samples")
        
        # Load validation data if provided
        if self.args.valid_data:
            with open(self.args.valid_data, 'r') as f:
                self.valid_data = json.load(f)
            logger.info(f"Loaded {len(self.valid_data)} validation samples")
    
    def create_trainer(self):
        """Create enhanced GRPO trainer"""
        self.trainer = GRPOTrainerWithValue(
            policy=self.policy,
            reference_policy=self.reference_policy,
            grpo_config=self.configs['grpo'],
            training_config=self.configs['training'],
            device=self.device
        )
        
        logger.info("Enhanced GRPO trainer created")
    
    def create_dummy_trajectories(self, tasks: List[Dict]) -> List[Trajectory]:
        """Create dummy trajectories for testing (replace with real collector)"""
        trajectories = []
        
        for task in tasks[:self.args.batch_size]:
            # Extract conversation from task
            conversation = task.get('prompt', [])
            
            # Create dummy trajectory
            states = []
            actions = []
            rewards = []
            dones = []
            
            # Simulate a 2-turn conversation
            if len(conversation) > 0:
                # Turn 1
                states.append(conversation[:1])  # Just user message
                actions.append("I'll help you with that task.")
                rewards.append(0.5)
                dones.append(False)
                
                # Turn 2
                states.append(conversation)  # Full conversation
                actions.append("Task completed successfully!")
                rewards.append(1.0)
                dones.append(True)
            else:
                # Fallback
                states.append([{"role": "user", "content": "Hello"}])
                actions.append("Hello! How can I help?")
                rewards.append(0.3)
                dones.append(True)
            
            traj = Trajectory(
                task_id=task.get('task_metadata', {}).get('task_id', f"task_{trajectories.__len__()}"),
                states=states,
                actions=actions,
                rewards=rewards,
                dones=dones
            )
            
            trajectories.append(traj)
        
        return trajectories
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        logger.info(f"\nStarting epoch {epoch + 1}/{self.args.num_epochs}")
        
        # Shuffle data
        import random
        train_tasks = self.train_data.copy()
        random.shuffle(train_tasks)
        
        # Training loop
        num_batches = len(train_tasks) // self.args.batch_size
        
        for batch_idx in range(min(num_batches, self.args.max_steps_per_epoch)):
            # Get batch of tasks
            batch_start = batch_idx * self.args.batch_size
            batch_end = batch_start + self.args.batch_size
            batch_tasks = train_tasks[batch_start:batch_end]
            
            # Collect trajectories (using dummy for now)
            trajectories = self.create_dummy_trajectories(batch_tasks)
            
            # Train step
            metrics = self.trainer.train_step(trajectories)
            
            # Log metrics
            if self.training_logger and batch_idx % self.args.log_interval == 0:
                self.training_logger.log_training_step(
                    metrics=metrics,
                    step=self.trainer.step_count,
                    stage="training"
                )
            
            # Print progress
            if batch_idx % self.args.print_interval == 0:
                logger.info(
                    f"Step {self.trainer.step_count}: "
                    f"policy_loss={metrics.get('policy_loss', 0):.4f}, "
                    f"value_loss={metrics.get('value_loss', 0):.4f}, "
                    f"total_loss={metrics.get('total_loss', 0):.4f}, "
                    f"avg_reward={metrics.get('avg_total_reward', 0):.3f}"
                )
            
            # Validation
            if self.valid_data and batch_idx % self.args.eval_interval == 0:
                self.evaluate()
        
        # End of epoch checkpoint
        if self.args.save_checkpoints:
            checkpoint_path = f"{self.args.output_dir}/checkpoint_epoch_{epoch + 1}"
            self.trainer.save_checkpoint(checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def evaluate(self):
        """Run validation"""
        if not self.valid_data:
            return
        
        logger.info("Running validation...")
        
        # Sample validation tasks
        valid_sample = self.valid_data[:self.args.eval_batch_size]
        
        # Collect validation trajectories
        trajectories = self.create_dummy_trajectories(valid_sample)
        
        # Compute validation metrics (without updating)
        total_reward = sum(t.total_reward for t in trajectories)
        avg_reward = total_reward / len(trajectories)
        
        eval_metrics = {
            "eval_avg_reward": avg_reward,
            "eval_num_trajectories": len(trajectories)
        }
        
        if self.training_logger:
            self.training_logger.log_model_evaluation(
                evaluation_results=eval_metrics,
                step=self.trainer.step_count
            )
        
        logger.info(f"Validation: avg_reward={avg_reward:.3f}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting enhanced GRPO training with value function...")
        
        for epoch in range(self.args.num_epochs):
            self.train_epoch(epoch)
        
        # Final checkpoint
        if self.args.save_checkpoints:
            final_path = f"{self.args.output_dir}/final_model"
            self.trainer.save_checkpoint(final_path)
            logger.info(f"Final model saved to {final_path}")
        
        # Cleanup
        if self.training_logger:
            self.training_logger.finish()
        
        logger.info("Training completed!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced GRPO training with value function"
    )
    
    # Config paths
    parser.add_argument(
        "--training-config",
        type=str,
        required=True,
        help="Path to training configuration YAML"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model configuration YAML"
    )
    parser.add_argument(
        "--grpo-config",
        type=str,
        required=True,
        help="Path to GRPO configuration YAML"
    )
    
    # Data paths
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data JSON"
    )
    parser.add_argument(
        "--valid-data",
        type=str,
        default=None,
        help="Path to validation data JSON"
    )
    
    # Training settings
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=100,
        help="Maximum steps per epoch"
    )
    
    # Logging
    parser.add_argument(
        "--enable-logging",
        action="store_true",
        help="Enable enhanced logging"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB logging"
    )
    parser.add_argument(
        "--weave",
        action="store_true",
        help="Enable Weave logging"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Steps between logging"
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=10,
        help="Steps between printing progress"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="Steps between validation"
    )
    
    # Checkpointing
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/enhanced_grpo",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save model checkpoints"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create trainer
    trainer = EnhancedGRPOTraining(args)
    
    # Load everything
    trainer.load_models()
    trainer.load_data()
    trainer.create_trainer()
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()