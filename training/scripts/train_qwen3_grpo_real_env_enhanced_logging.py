#!/usr/bin/env python3
"""
Enhanced GRPO Training Script with Comprehensive WandB Logging
============================================================

This is an enhanced version of the GRPO training script that includes
comprehensive logging of all training metrics including:
- Detailed policy and value losses
- Advantage computation metrics (GAE, normalization, distribution)
- Reward metrics (raw, normalized, per-component breakdown)
- Gradient norms and parameter statistics
- Episode performance and tool usage analytics
- KL divergence, entropy, and clipping metrics

All metrics are logged to WandB with proper organization and dashboard-friendly names.
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import yaml
import numpy as np
from tqdm import tqdm

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Add utils path for enhanced logging
utils_path = str(Path(__file__).parent.parent / "utils")
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

# Import original training components
from core.qwen_policy_with_value_prompting import QwenPolicyWithValuePrompting
from core.grpo_trainer_gradient_fix import GRPOTrainerGradientFix
from core.grpo_trainer import Trajectory
from data.trajectory_collector import TrajectoryCollector, EpisodeResult

# Import environment components
env_path = str(Path(__file__).parent.parent.parent / "environments")
if env_path not in sys.path:
    sys.path.insert(0, env_path)

from simple_shared_manager import SimpleSharedManager
from mcp_tool_environment import MCPToolEnvironment

# Import enhanced logging components
from enhanced_wandb_logger import GRPOTrainingMetricsLogger
from grpo_trainer_logging_patch import enhance_grpo_trainer_logging

# Import WandB and Weave with fallbacks
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    logging.warning("WandB not available - metrics will not be logged to WandB")

try:
    import weave
    HAS_WEAVE = True
except ImportError:
    HAS_WEAVE = False
    logging.warning("Weave not available - detailed episode tracking disabled")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedGRPOTrainer:
    """
    Enhanced GRPO Trainer with comprehensive metrics logging
    """
    
    def __init__(self, config_paths: Dict[str, str], args: argparse.Namespace):
        self.config_paths = config_paths
        self.args = args
        
        # Load configurations
        self.configs = self._load_configs()
        
        # Initialize device
        self.device = self._setup_device()
        
        # Initialize WandB with enhanced configuration
        self.enhanced_logger = None
        self.use_wandb = (
            args.use_wandb and 
            HAS_WANDB and 
            self.configs.get('training', {}).get('use_wandb', True)
        )
        
        if self.use_wandb:
            self._setup_enhanced_wandb_logging()
        
        # Initialize components
        self.policy = None
        self.reference_policy = None
        self.trainer = None
        self.tool_manager = None
        self.trajectory_collector = None
        self.trainer_enhancement = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_score = float('-inf')
        
        logger.info("Enhanced GRPO Trainer initialized with comprehensive logging")
    
    def _setup_enhanced_wandb_logging(self):
        """Setup enhanced WandB logging with detailed metrics tracking"""
        
        # Prepare comprehensive config for logging
        wandb_config = {
            'model': self.configs.get('model', {}),
            'training': self.configs.get('training', {}),
            'grpo': self.configs.get('grpo', {}),
            'environment': {
                'type': 'real_mcp_tools',
                'parallel_envs': self.configs.get('grpo', {}).get('rollout_parallel_envs', 4),
                'max_episode_length': self.configs.get('grpo', {}).get('max_episode_length', 15),
            },
            'hardware': {
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None',
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
            },
            'experiment': {
                'script_version': 'enhanced_logging_v1.0',
                'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
                'args': vars(self.args)
            }
        }
        
        # Initialize enhanced logger
        project_name = os.environ.get('WANDB_PROJECT', 'skyrl-grpo-enhanced-metrics')
        run_name = f"grpo-real-env-enhanced-{time.strftime('%Y%m%d-%H%M%S')}"
        
        self.enhanced_logger = GRPOTrainingMetricsLogger(
            project_name=project_name,
            run_name=run_name,
            config=wandb_config,
            enabled=True
        )
        
        logger.info(f"✅ Enhanced WandB logging initialized: {project_name}/{run_name}")
    
    def _load_configs(self) -> Dict[str, Any]:
        """Load all configuration files"""
        configs = {}
        
        for config_type, config_path in self.config_paths.items():
            try:
                with open(config_path, 'r') as f:
                    configs[config_type] = yaml.safe_load(f)
                logger.info(f"Loaded {config_type} config from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load {config_type} config from {config_path}: {e}")
                raise
        
        return configs
    
    def _setup_device(self) -> torch.device:
        """Setup training device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
        
        return device
    
    def setup_training_components(self):
        """Initialize all training components"""
        
        # Load model configurations
        model_config_path = self.config_paths.get('model')
        training_config_path = self.config_paths.get('training')
        
        # Initialize policy
        logger.info("Initializing policy...")
        value_head_hidden_dim = self.configs['model'].get('value_head_hidden_dim', 1024)
        
        # Enable 4-bit quantization for CUDA if configured
        enable_4bit = (
            self.device.type == 'cuda' and 
            self.configs['model'].get('load_in_4bit', True) and
            self.configs['model'].get('use_lora', True)
        )
        
        self.policy = QwenPolicyWithValuePrompting(
            model_config_path=model_config_path,
            training_config_path=training_config_path,
            use_lora=self.configs['model'].get('use_lora', True),
            device=str(self.device),
            load_in_4bit=enable_4bit,
            value_head_hidden_dim=value_head_hidden_dim
        )
        
        # Create reference policy
        logger.info("Creating reference policy...")
        self.reference_policy = QwenPolicyWithValuePrompting(
            model_config_path=model_config_path,
            training_config_path=training_config_path,
            use_lora=self.configs['model'].get('use_lora', True),
            device=str(self.device),
            load_in_4bit=enable_4bit,
            value_head_hidden_dim=value_head_hidden_dim
        )
        
        # Synchronize reference policy weights
        self.reference_policy.load_state_dict(self.policy.state_dict())
        logger.info("Reference policy synchronized")
        
        # Initialize GRPO trainer
        logger.info("Initializing GRPO trainer...")
        self.trainer = GRPOTrainerGradientFix(
            policy=self.policy,
            reference_policy=self.reference_policy,
            config_path=self.config_paths.get('grpo'),
            device=self.device
        )
        
        # Enhance trainer with comprehensive logging
        if self.use_wandb:
            logger.info("Enhancing trainer with comprehensive metrics logging...")
            self.trainer_enhancement = enhance_grpo_trainer_logging(
                trainer=self.trainer,
                project_name=os.environ.get('WANDB_PROJECT', 'skyrl-grpo-enhanced-metrics'),
                config=self.configs,
                enabled=True
            )
        
        # Initialize tool manager and trajectory collector
        logger.info("Initializing environment components...")
        self.tool_manager = SimpleSharedManager()
        
        grpo_config = self.configs.get('grpo', {})
        self.trajectory_collector = TrajectoryCollector(
            policy=self.policy,
            shared_tool_manager=self.tool_manager,
            num_parallel_envs=grpo_config.get('rollout_parallel_envs', 4),
            timeout_seconds=grpo_config.get('rollout_timeout_seconds', 600),
            max_episode_length=grpo_config.get('max_episode_length', 15),
            retry_failed_episodes=grpo_config.get('retry_failed_episodes', True),
            max_retries=grpo_config.get('max_retries', 3)
        )
        
        logger.info("✅ All training components initialized")
    
    def load_training_data(self) -> List[Dict[str, Any]]:
        """Load and prepare training data"""
        
        data_path = self.configs['training'].get('data_path', 'data/inputs/train.json')
        
        try:
            with open(data_path, 'r') as f:
                training_data = json.load(f)
            
            logger.info(f"Loaded {len(training_data)} training samples from {data_path}")
            
            # Log data statistics
            if self.enhanced_logger:
                data_stats = {
                    'training/dataset_size': len(training_data),
                    'training/data_path': data_path,
                }
                
                # Analyze task complexity distribution
                complexities = {}
                for item in training_data:
                    complexity = item.get('extra_info', {}).get('complexity', 'unknown')
                    complexities[complexity] = complexities.get(complexity, 0) + 1
                
                for complexity, count in complexities.items():
                    data_stats[f'training/complexity_{complexity}'] = count
                
                self.enhanced_logger._log_metrics(data_stats, step=0)
            
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to load training data from {data_path}: {e}")
            raise
    
    async def collect_trajectories(self, tasks: List[Dict[str, Any]], step: int) -> List[Trajectory]:
        """Collect trajectories with enhanced logging"""
        
        logger.info(f"Collecting trajectories for {len(tasks)} tasks...")
        collection_start = time.time()
        
        try:
            # Collect trajectories using the trajectory collector
            episode_results = await self.trajectory_collector.collect_batch(tasks)
            
            collection_time = time.time() - collection_start
            
            # Convert episode results to trajectories
            trajectories = []
            episode_data = []
            
            for i, episode_result in enumerate(episode_results):
                if episode_result and hasattr(episode_result, 'trajectory'):
                    trajectories.append(episode_result.trajectory)
                    
                    # Prepare episode data for logging
                    episode_info = {
                        'episode_id': i,
                        'success': getattr(episode_result, 'success', False),
                        'total_reward': getattr(episode_result.trajectory, 'total_reward', 0.0),
                        'length': getattr(episode_result.trajectory, 'length', 0),
                        'task_complexity': tasks[i].get('extra_info', {}).get('complexity', 'unknown'),
                        'collection_time_seconds': collection_time / len(tasks)
                    }
                    
                    # Add reward breakdown if available
                    if hasattr(episode_result.trajectory, 'reward_breakdown'):
                        episode_info['reward_breakdown'] = episode_result.trajectory.reward_breakdown
                    
                    episode_data.append(episode_info)
            
            # Log collection metrics
            success_rate = sum(1 for ep in episode_data if ep['success']) / len(episode_data) if episode_data else 0.0
            avg_reward = sum(ep['total_reward'] for ep in episode_data) / len(episode_data) if episode_data else 0.0
            avg_length = sum(ep['length'] for ep in episode_data) / len(episode_data) if episode_data else 0.0
            
            collection_metrics = {
                'collection/success_rate': success_rate,
                'collection/average_reward': avg_reward,
                'collection/average_episode_length': avg_length,
                'collection/time_seconds': collection_time,
                'collection/trajectories_collected': len(trajectories),
                'collection/tasks_attempted': len(tasks),
                'training/step': step
            }
            
            if self.enhanced_logger:
                self.enhanced_logger._log_metrics(collection_metrics, step)
                
                # Log detailed episode data
                if self.trainer_enhancement:
                    self.trainer_enhancement.log_episode_batch(
                        step=step,
                        episode_batch=episode_data
                    )
            
            logger.info(f"Collected {len(trajectories)} trajectories in {collection_time:.2f}s")
            logger.info(f"Success rate: {success_rate:.2%}, Avg reward: {avg_reward:.3f}")
            
            return trajectories
            
        except Exception as e:
            logger.error(f"Failed to collect trajectories: {e}")
            
            # Log collection failure
            if self.enhanced_logger:
                failure_metrics = {
                    'collection/failed': 1.0,
                    'collection/success_rate': 0.0,
                    'collection/time_seconds': time.time() - collection_start,
                    'training/step': step
                }
                self.enhanced_logger._log_metrics(failure_metrics, step)
            
            return []
    
    def train_step(self, trajectories: List[Trajectory], step: int, epoch: int) -> Dict[str, float]:
        """Execute one training step with enhanced logging"""
        
        if not trajectories:
            logger.warning("No trajectories available for training step")
            return {}
        
        step_start = time.time()
        
        # Update global step and epoch
        self.global_step = step
        self.current_epoch = epoch
        
        # Perform training step
        logger.debug(f"Training step {step} with {len(trajectories)} trajectories")
        
        training_metrics = self.trainer._update_policy(trajectories)
        
        step_time = time.time() - step_start
        
        # Add step timing and basic info
        training_metrics.update({
            'training/step': step,
            'training/epoch': epoch,
            'training/step_time': step_time,
            'training/trajectories_count': len(trajectories),
        })
        
        # The trainer enhancement will automatically log detailed metrics
        # through the patched _update_policy method
        
        logger.info(
            f"Step {step:6d} | "
            f"Loss: {training_metrics.get('total_loss', 0.0):.4f} | "
            f"Policy: {training_metrics.get('policy_loss', 0.0):.4f} | "
            f"Value: {training_metrics.get('value_loss', 0.0):.4f} | "
            f"KL: {training_metrics.get('kl_divergence', 0.0):.4f} | "
            f"Time: {step_time:.2f}s"
        )
        
        return training_metrics
    
    async def evaluate_model(self, eval_tasks: List[Dict[str, Any]], step: int) -> Dict[str, float]:
        """Evaluate model with enhanced metrics logging"""
        
        logger.info("Starting model evaluation...")
        eval_start = time.time()
        
        try:
            # Collect evaluation trajectories
            eval_trajectories = await self.collect_trajectories(eval_tasks, step)
            
            if not eval_trajectories:
                logger.warning("No evaluation trajectories collected")
                return {'eval_avg_reward': 0.0}
            
            # Calculate evaluation metrics
            eval_rewards = [traj.total_reward for traj in eval_trajectories if hasattr(traj, 'total_reward')]
            eval_successes = [getattr(traj, 'success', False) for traj in eval_trajectories]
            eval_lengths = [getattr(traj, 'length', 0) for traj in eval_trajectories]
            
            eval_metrics = {
                'eval/average_reward': np.mean(eval_rewards) if eval_rewards else 0.0,
                'eval/std_reward': np.std(eval_rewards) if eval_rewards else 0.0,
                'eval/min_reward': np.min(eval_rewards) if eval_rewards else 0.0,
                'eval/max_reward': np.max(eval_rewards) if eval_rewards else 0.0,
                'eval/success_rate': np.mean(eval_successes) if eval_successes else 0.0,
                'eval/average_length': np.mean(eval_lengths) if eval_lengths else 0.0,
                'eval/trajectories_count': len(eval_trajectories),
                'eval/evaluation_time': time.time() - eval_start,
                'training/step': step
            }
            
            # Log evaluation metrics
            if self.enhanced_logger:
                self.enhanced_logger._log_metrics(eval_metrics, step)
            
            logger.info(f"Evaluation completed: Avg reward: {eval_metrics['eval/average_reward']:.3f}, "
                       f"Success rate: {eval_metrics['eval/success_rate']:.2%}")
            
            return eval_metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {'eval_avg_reward': 0.0}
    
    async def train(self):
        """Main training loop with comprehensive logging"""
        
        logger.info("🔥 Starting enhanced GRPO training with comprehensive metrics logging")
        
        # Load training data
        training_data = self.load_training_data()
        
        # Training configuration
        training_config = self.configs['training']
        grpo_config = self.configs['grpo']
        
        num_epochs = training_config.get('num_epochs', 3)
        batch_size = grpo_config.get('rollout_batch_size', 4)
        eval_frequency = training_config.get('eval_steps', 100)
        
        # Prepare training batches
        total_batches = (len(training_data) // batch_size) * num_epochs
        
        logger.info(f"Training for {num_epochs} epochs, {total_batches} total batches")
        
        step = 0
        
        try:
            for epoch in range(num_epochs):
                logger.info(f"\n{'='*60}")
                logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
                logger.info(f"{'='*60}")
                
                # Shuffle training data
                np.random.shuffle(training_data)
                
                # Process batches
                for batch_idx in range(0, len(training_data), batch_size):
                    batch_tasks = training_data[batch_idx:batch_idx + batch_size]
                    
                    step += 1
                    
                    # Collect trajectories
                    trajectories = await self.collect_trajectories(batch_tasks, step)
                    
                    if trajectories:
                        # Training step
                        training_metrics = self.train_step(trajectories, step, epoch + 1)
                        
                        # Evaluation
                        if step % eval_frequency == 0:
                            eval_tasks = training_data[:min(16, len(training_data))]  # Small eval set
                            eval_metrics = await self.evaluate_model(eval_tasks, step)
                            
                            # Check for best model
                            if eval_metrics['eval/average_reward'] > self.best_eval_score:
                                self.best_eval_score = eval_metrics['eval/average_reward']
                                self.save_checkpoint('best_model', step, is_best=True)
                    
                    else:
                        logger.warning(f"No trajectories collected for batch {batch_idx}, skipping training step")
                
                # End of epoch checkpoint
                self.save_checkpoint(f'epoch_{epoch + 1}', step)
                
                logger.info(f"Epoch {epoch + 1} completed")
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Cleanup
            if self.enhanced_logger:
                self.enhanced_logger.finish()
            if self.trainer_enhancement:
                self.trainer_enhancement.finish()
            
            logger.info("✅ Enhanced GRPO training completed")
    
    def save_checkpoint(self, name: str, step: int, is_best: bool = False):
        """Save model checkpoint with enhanced logging"""
        
        checkpoint_dir = Path(self.configs['training'].get('output_dir', './outputs')) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{name}_step_{step}.pt"
        
        # Save checkpoint
        checkpoint_data = {
            'step': step,
            'epoch': self.current_epoch,
            'model_state_dict': self.policy.state_dict(),
            'best_eval_score': self.best_eval_score,
            'config': self.configs
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Log checkpoint
        checkpoint_metrics = {
            'checkpoint/step': step,
            'checkpoint/is_best': 1.0 if is_best else 0.0,
            'checkpoint/best_eval_score': self.best_eval_score
        }
        
        if self.trainer_enhancement:
            self.trainer_enhancement.log_checkpoint(step, str(checkpoint_path), checkpoint_metrics)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}" + (" (BEST)" if is_best else ""))


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description="Enhanced GRPO Training with Comprehensive Logging")
    parser.add_argument("--model-config", type=str, default="training/configs/model_config_qwen3_0.6b.yaml")
    parser.add_argument("--training-config", type=str, default="training/configs/training_config_qwen3_0.6b.yaml")
    parser.add_argument("--grpo-config", type=str, default="training/configs/grpo_config_qwen3_0.6b.yaml")
    parser.add_argument("--use-wandb", action="store_true", default=True, help="Enable WandB logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    # Configuration paths
    config_paths = {
        'model': args.model_config,
        'training': args.training_config,
        'grpo': args.grpo_config
    }
    
    # Initialize and run trainer
    trainer = EnhancedGRPOTrainer(config_paths, args)
    trainer.setup_training_components()
    
    # Run training
    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()