#!/usr/bin/env python3
"""
Main GRPO Training Script

This script implements the complete Group Relative Policy Optimization training
pipeline for multi-turn tool use with Qwen2.5-1.5B-Instruct.

Usage:
    python train_grpo.py --config configs/training_config.yaml --mode lora
    python train_grpo.py --config configs/training_config.yaml --mode full --num_gpus 2
"""

import argparse
import asyncio
import copy
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import yaml
import numpy as np

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Core training imports
from core.qwen_policy import QwenPolicy, create_policy_from_configs
from core.grpo_trainer import GRPOTrainer, Trajectory, create_grpo_trainer
from data.data_loader import TaskDataLoader, create_data_loader
from data.trajectory_collector import TrajectoryCollector, collect_trajectories_for_training

# Environment imports - use path-based imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from environments.mcp_tool_environment import MCPToolEnvironment
from environments.simple_shared_manager import SimpleSharedManager

# Optional dependencies
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: WandB not available. Logging disabled.")

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False
    print("Warning: DeepSpeed not available. Multi-GPU full fine-tuning disabled.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class TrainingSession:
    """
    Main training session manager.
    
    This class manages the complete GRPO training pipeline, including
    component initialization, training loop execution, evaluation,
    checkpointing, and cleanup.
    """
    
    def __init__(self, args: argparse.Namespace):
        """Initialize training session with parsed arguments."""
        
        self.args = args
        self.configs = {}
        # Check for available device based on user preference
        if self.args.device == "auto":
            # Auto-detect best available device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            # Use specified device
            self.device = torch.device(self.args.device)
        
        # Training components
        self.policy: Optional[QwenPolicy] = None
        self.reference_policy: Optional[QwenPolicy] = None
        self.trainer: Optional[GRPOTrainer] = None
        self.data_loader: Optional[TaskDataLoader] = None
        self.trajectory_collector: Optional[TrajectoryCollector] = None
        self.shared_tool_manager: Optional[Any] = None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_eval_score = float('-inf')
        self.early_stopping_counter = 0
        self.training_start_time = None
        
        # Enhanced logging
        self.training_logger = None
        
        # Cleanup handlers
        self.cleanup_functions = []
        self._setup_signal_handlers()
        
        logger.info(f"TrainingSession initialized with device: {self.device}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def load_configs(self) -> None:
        """Load all configuration files."""
        
        logger.info("Loading configuration files...")
        
        # Determine config paths
        config_path = Path(self.args.config)
        config_dir = config_path.parent
        config_stem = config_path.stem
        
        # Check for MPS-specific configs
        if "mps" in config_stem.lower():
            config_files = {
                "model": config_dir / "model_config_mps.yaml",
                "training": config_path,
                "grpo": config_dir / "grpo_config_mps.yaml",
            }
        else:
            config_files = {
                "model": config_dir / "model_config.yaml",
                "training": config_path,
                "grpo": config_dir / "grpo_config.yaml",
            }
        
        # Load each config file
        for config_name, config_path in config_files.items():
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                self.configs[config_name] = yaml.safe_load(f)
            
            logger.info(f"Loaded {config_name} config from {config_path}")
        
        # Override model config based on training mode
        if self.args.mode == "lora":
            self.configs["model"]["lora_mode"]["enabled"] = True
            self.configs["model"]["full_finetune_mode"]["enabled"] = False
            # Disable quantization for MPS
            if self.device.type == "mps":
                self.configs["model"]["quantization"]["load_in_4bit"] = False
                self.configs["model"]["quantization"]["load_in_8bit"] = False
            else:
                self.configs["model"]["quantization"]["load_in_4bit"] = True
        else:
            self.configs["model"]["lora_mode"]["enabled"] = False
            self.configs["model"]["full_finetune_mode"]["enabled"] = True
            self.configs["model"]["quantization"]["load_in_4bit"] = False
        
        logger.info(f"Training mode: {self.args.mode}")
    
    def initialize_logging(self) -> None:
        """Initialize enhanced training logging with WandB and Weave."""
        
        try:
            # Import enhanced logging utilities
            from utils.logging_utils import create_enhanced_training_logger
            
            # Get rank information for distributed training
            rank = int(os.environ.get('RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            
            # Setup enhanced logger
            self.training_logger = create_enhanced_training_logger(
                config=self.configs,
                rank=rank,
                world_size=world_size,
                enable_wandb=not self.args.no_wandb,
                enable_weave=True  # Always enable Weave for detailed tracking
            )
            
            # Log hyperparameters
            hyperparams = {
                "training_mode": self.args.mode,
                "num_gpus": self.args.num_gpus,
                "model": self.configs.get("model", {}),
                "training": self.configs.get("training", {}),
                "grpo": self.configs.get("grpo", {}),
                "environment": self.configs.get("environment", {})
            }
            self.training_logger.log_hyperparameters(hyperparams)
            
            logger.info(f"✅ Enhanced logging initialized (Rank {rank}/{world_size})")
            
            # Add cleanup
            self.cleanup_functions.append(self.training_logger.finish)
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced logging: {e}")
            logger.info("Falling back to basic logging...")
            self.training_logger = None
    
    def initialize_shared_tool_manager(self) -> None:
        """Initialize shared tool manager for all environments."""
        
        logger.info("Initializing shared tool manager...")
        
        if SimpleSharedManager is None:
            logger.error("SimpleSharedManager not available")
            raise RuntimeError("SimpleSharedManager required but not available")
        
        try:
            # Create shared tool manager with appropriate configuration
            self.shared_tool_manager = SimpleSharedManager()
            
            # Add cleanup function
            def cleanup_tool_manager():
                if hasattr(self.shared_tool_manager, 'cleanup'):
                    # Handle both sync and async cleanup
                    import asyncio
                    import inspect
                    cleanup_method = getattr(self.shared_tool_manager, 'cleanup')
                    if inspect.iscoroutinefunction(cleanup_method):
                        # Run async cleanup in new event loop
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(cleanup_method())
                            loop.close()
                        except Exception as e:
                            logger.warning(f"Error in async cleanup: {e}")
                    else:
                        cleanup_method()
            
            self.cleanup_functions.append(cleanup_tool_manager)
            
            logger.info("Shared tool manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize shared tool manager: {e}")
            raise
    
    def initialize_policies(self) -> None:
        """Initialize policy and reference policy."""
        
        logger.info("Initializing policies...")
        
        # Use MPS-optimized policy if on MPS
        if self.device.type == "mps":
            logger.info("Using MPS-optimized policy for Apple Silicon")
            from core.qwen_policy_mps import QwenPolicyMPS
            PolicyClass = QwenPolicyMPS
        else:
            PolicyClass = QwenPolicy
        
        # Determine config files based on device
        if self.device.type == "mps":
            model_config_path = str(Path(self.args.config).parent / "model_config_mps.yaml")
            if not Path(model_config_path).exists():
                model_config_path = str(Path(self.args.config).parent / "model_config.yaml")
        else:
            model_config_path = str(Path(self.args.config).parent / "model_config.yaml")
        
        # Create policy
        self.policy = PolicyClass(
            model_config_path=model_config_path,
            training_config_path=self.args.config,
            use_lora=(self.args.mode == "lora"),
            device=str(self.device),
            load_in_4bit=(self.args.mode == "lora") and self.device.type != "mps",
        )
        
        # Create reference policy (deep copy of initial policy)
        logger.info("Creating reference policy...")
        self.reference_policy = PolicyClass(
            model_config_path=model_config_path,
            training_config_path=self.args.config,
            use_lora=(self.args.mode == "lora"),
            device=str(self.device),
            load_in_4bit=(self.args.mode == "lora") and self.device.type != "mps",
        )
        
        # Set reference policy to evaluation mode
        self.reference_policy.enable_eval_mode()
        
        # Enable training mode for main policy before trainer initialization
        self.policy.enable_training_mode()
        
        logger.info(f"Policies initialized: {self.policy.get_trainable_parameters():,} trainable parameters")
    
    def initialize_data_loader(self) -> None:
        """Initialize data loader with curriculum learning."""
        
        logger.info("Initializing data loader...")
        
        # Get data path from training config or use default
        data_path = self.configs["training"].get("data_path", "../../data/processed/train.json")
        
        # Make path relative to script location
        if not Path(data_path).is_absolute():
            data_path = str(Path(__file__).parent.parent.parent / data_path)
        
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        # Create data loader
        self.data_loader = create_data_loader(
            data_path=data_path,
            cache_size=self.configs["training"].get("cache_size", 1000),
            batch_size=self._get_effective_batch_size(),
            num_shards=self.args.num_gpus if self.args.num_gpus > 1 else 1,
            shard_id=0,  # TODO: Implement proper distributed training
            seed=self.configs["training"].get("seed", 42),
        )
        
        # Log dataset statistics
        stats = self.data_loader.get_dataset_statistics()
        logger.info(f"Dataset loaded: {stats['dataset']['total_tasks']} tasks")
        logger.info(f"Complexity distribution: {dict(stats['dataset']['complexity_counts'])}")
    
    def initialize_trajectory_collector(self) -> None:
        """Initialize trajectory collector for episode collection."""
        
        logger.info("Initializing trajectory collector...")
        
        if MCPToolEnvironment is None:
            raise RuntimeError("MCPToolEnvironment required but not available")
        
        # Create environment factory
        def env_factory(task_data: Dict[str, Any]) -> MCPToolEnvironment:
            """Factory function to create MCPToolEnvironment instances."""
            env = MCPToolEnvironment(task_data)
            # Set shared tool manager
            env.tool_manager = self.shared_tool_manager
            return env
        
        # Create trajectory collector
        grpo_config = self.configs["grpo"]
        
        self.trajectory_collector = TrajectoryCollector(
            policy=self.policy,
            env_factory=env_factory,
            num_parallel_envs=grpo_config.get("rollout_parallel_envs", 4),
            shared_tool_manager=self.shared_tool_manager,
            max_episode_length=grpo_config.get("max_episode_length", 15),
            retry_failed_episodes=grpo_config.get("retry_failed_episodes", True),
            collect_log_probs=True,
        )
        
        # Add cleanup function
        async def cleanup_collector():
            if self.trajectory_collector:
                await self.trajectory_collector.cleanup()
        
        self.cleanup_functions.append(lambda: asyncio.run(cleanup_collector()))
        
        logger.info("Trajectory collector initialized")
    
    def initialize_trainer(self) -> None:
        """Initialize GRPO trainer."""
        
        logger.info("Initializing GRPO trainer...")
        
        self.trainer = GRPOTrainer(
            policy=self.policy,
            reference_policy=self.reference_policy,
            grpo_config=self.configs["grpo"],
            training_config=self.configs["training"],
            device=self.device,
        )
        
        logger.info("GRPO trainer initialized")
    
    def _get_effective_batch_size(self) -> int:
        """Get effective batch size based on training mode."""
        
        if self.args.mode == "lora":
            return self.configs["training"]["lora_mode"]["per_device_train_batch_size"]
        else:
            return self.configs["training"]["full_finetune_mode"]["per_device_train_batch_size"]
    
    def _convert_episodes_to_trajectories(self, episode_results: List[Any], tasks: List[Dict[str, Any]]) -> List[Trajectory]:
        """Convert episode results to GRPO trajectories with proper conversation initialization."""
        
        trajectories = []
        
        for i, episode in enumerate(episode_results):
            if not episode.is_valid():
                continue
            
            # Get the initial prompt from the task
            task = tasks[i] if i < len(tasks) else None
            if not task or "prompt" not in task:
                logger.warning(f"Missing prompt for episode {i}")
                continue
            
            # Extract trajectory data
            states = []
            actions = []
            rewards = []
            dones = []
            
            # Debug: log data structures
            logger.info(f"Task prompt type: {type(task['prompt'])}, length: {len(task['prompt']) if isinstance(task['prompt'], list) else 'N/A'}")
            logger.info(f"Episode trajectory length: {len(episode.trajectory)}")
            if episode.trajectory:
                logger.info(f"First trajectory turn keys: {list(episode.trajectory[0].keys())}")
                logger.info(f"First trajectory turn action sample: {str(episode.trajectory[0].get('action', 'NO_ACTION'))[:100]}...")
            
            # Build conversation states for each turn
            # Start with just the initial user message
            if isinstance(task["prompt"], list) and len(task["prompt"]) > 0:
                # Take only the first user message, not the full conversation
                conversation_history = [task["prompt"][0]]
                logger.debug(f"Starting conversation with: {task['prompt'][0]}")
            else:
                conversation_history = [task["prompt"]] if task["prompt"] else []
                logger.debug(f"Starting conversation with single prompt: {task['prompt'] if task['prompt'] else 'EMPTY'}")
            
            for j, turn_data in enumerate(episode.trajectory):
                # Simple state format like our working isolated test
                state = [
                    {"role": "user", "content": f"Task {episode.task_id} step {j}: {conversation_history[0]['content']}"},
                    {"role": "assistant", "content": f"Working on step {j}"}
                ]
                states.append(state)
                
                # Action and metadata
                action = turn_data["action"]
                reward = turn_data["reward"]
                done = turn_data["done"]
                
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
            
            # Create GRPO trajectory
            if states and actions:
                trajectory = Trajectory(
                    task_id=episode.task_id,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                )
                trajectories.append(trajectory)
        
        return trajectories
    
    async def collect_trajectories(self, tasks: List[Dict[str, Any]]) -> List[Trajectory]:
        """Collect trajectories for a batch of tasks."""
        
        try:
            # Collect episode results
            episode_results = await self.trajectory_collector.collect_batch(
                tasks,
                batch_timeout=self.configs["grpo"].get("rollout_timeout_seconds", 300)
            )
            
            # Convert to GRPO trajectories
            trajectories = self._convert_episodes_to_trajectories(episode_results, tasks)
            
            # Log collection statistics
            valid_episodes = sum(1 for ep in episode_results if ep.is_valid())
            logger.info(f"Collected {len(trajectories)} trajectories from {valid_episodes}/{len(episode_results)} episodes")
            
            # Log episode results to enhanced logger
            if self.training_logger:
                for i, episode in enumerate(episode_results):
                    if episode.is_valid():
                        # Extract episode metadata
                        task_metadata = tasks[i] if i < len(tasks) else {}
                        
                        # Create episode results for logging
                        episode_data = {
                            "task_id": task_metadata.get("task_metadata", {}).get("task_id", f"episode_{i}"),
                            "complexity": task_metadata.get("task_metadata", {}).get("complexity", "unknown"),
                            "success": episode.success,
                            "total_reward": episode.total_reward,
                            "turn_count": episode.turns,
                            "tool_calls": len(episode.tool_calls) if hasattr(episode, 'tool_calls') else 0,
                            "reward_breakdown": getattr(episode, 'reward_breakdown', {})
                        }
                        
                        self.training_logger.log_episode_results(
                            episode_results=episode_data,
                            step=self.current_step,
                            task_metadata=task_metadata.get("task_metadata", {})
                        )
            
            return trajectories
            
        except Exception as e:
            logger.error(f"Error collecting trajectories: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def evaluate_model(self, epoch: int) -> Dict[str, float]:
        """Evaluate model on validation tasks."""
        
        logger.info(f"Evaluating model at epoch {epoch}...")
        
        try:
            # Get evaluation batch
            eval_batch = self.data_loader.get_batch(
                batch_size=self.configs["training"].get("eval_batch_size", 4)
            )
            
            if not eval_batch:
                logger.warning("No evaluation tasks available")
                return {}
            
            # Collect evaluation trajectories
            eval_trajectories = asyncio.run(self.collect_trajectories(eval_batch))
            
            if not eval_trajectories:
                logger.warning("No valid evaluation trajectories")
                return {}
            
            # Compute evaluation metrics
            total_rewards = [sum(traj.rewards) for traj in eval_trajectories]
            episode_lengths = [len(traj.rewards) for traj in eval_trajectories]
            success_rate = sum(1 for reward in total_rewards if reward > 0) / len(total_rewards)
            
            eval_metrics = {
                "eval_avg_reward": np.mean(total_rewards),
                "eval_std_reward": np.std(total_rewards),
                "eval_success_rate": success_rate,
                "eval_avg_episode_length": np.mean(episode_lengths),
                "eval_num_episodes": len(eval_trajectories),
            }
            
            logger.info(f"Evaluation results: avg_reward={eval_metrics['eval_avg_reward']:.3f}, "
                       f"success_rate={eval_metrics['eval_success_rate']:.3f}")
            
            return eval_metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save training checkpoint."""
        
        try:
            checkpoint_dir = Path(self.configs["training"]["output_dir"]) / f"checkpoint-epoch-{epoch}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save trainer checkpoint (includes policies)
            self.trainer.save_checkpoint(str(checkpoint_dir), include_optimizer=True)
            
            # Save additional metadata
            metadata = {
                "epoch": epoch,
                "step": self.current_step,
                "metrics": metrics,
                "config": self.configs,
                "args": vars(self.args),
            }
            
            with open(checkpoint_dir / "metadata.yaml", 'w') as f:
                yaml.dump(metadata, f)
            
            logger.info(f"Checkpoint saved to {checkpoint_dir}")
            
            # Clean up old checkpoints if configured
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints to save disk space."""
        
        try:
            save_limit = self.configs["training"].get("save_total_limit", 3)
            if save_limit <= 0:
                return
            
            output_dir = Path(self.configs["training"]["output_dir"])
            checkpoints = sorted([
                d for d in output_dir.glob("checkpoint-epoch-*") 
                if d.is_dir()
            ])
            
            # Remove oldest checkpoints
            while len(checkpoints) > save_limit:
                oldest = checkpoints.pop(0)
                logger.info(f"Removing old checkpoint: {oldest}")
                import shutil
                shutil.rmtree(oldest)
                
        except Exception as e:
            logger.warning(f"Error cleaning up checkpoints: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            # Load trainer checkpoint
            self.trainer.load_checkpoint(checkpoint_path, load_optimizer=True)
            
            # Load metadata
            metadata_path = Path(checkpoint_path) / "metadata.yaml"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                
                self.current_epoch = metadata.get("epoch", 0)
                self.current_step = metadata.get("step", 0)
                self.best_eval_score = metadata.get("metrics", {}).get("eval_success_rate", float('-inf'))
            
            logger.info(f"Checkpoint loaded: epoch={self.current_epoch}, step={self.current_step}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise
    
    async def train(self) -> None:
        """Main training loop with comprehensive error handling and logging."""
        
        logger.info("🚀 Starting GRPO training...")
        self.training_start_time = time.time()
        
        try:
            # Training configuration
            logger.info("📋 Loading training configuration...")
            num_epochs = self.configs["training"]["num_epochs"]
            eval_steps = self.configs["training"]["eval_steps"]
            save_steps = self.configs["training"]["save_steps"]
            early_stopping_patience = self.configs["training"].get("early_stopping", {}).get("patience", 5)
            
            logger.info(f"Training config: epochs={num_epochs}, eval_steps={eval_steps}, save_steps={save_steps}")
            
            # Training loop
            for epoch in range(self.current_epoch, num_epochs):
                logger.info(f"📊 Starting epoch {epoch + 1}/{num_epochs}")
                epoch_start_time = time.time()
                
                try:
                    # Get epoch iterator
                    logger.debug("Getting epoch iterator from data loader...")
                    batch_iterator = self.data_loader.get_epoch_iterator(
                        batch_size=self._get_effective_batch_size(),
                        epoch=epoch,
                        total_epochs=num_epochs,
                        use_curriculum=True,
                    )
                    logger.info("Epoch iterator created successfully")
                    
                    epoch_metrics = []
                    
                    # Process batches
                    logger.info("🔄 Starting batch processing...")
                    for batch_idx, tasks in enumerate(batch_iterator):
                        logger.info(f"📦 Processing batch {batch_idx + 1}")
                        step_start_time = time.time()
                        
                        try:
                            logger.info(f"Batch contains {len(tasks)} tasks")
                            
                            # Collect trajectories
                            logger.info("🎯 Starting trajectory collection...")
                            trajectories = await self.collect_trajectories(tasks)
                            logger.info(f"Trajectory collection completed: {len(trajectories)} trajectories")
                            
                            if not trajectories:
                                logger.warning(f"No valid trajectories in batch {batch_idx}")
                                continue
                            
                            # Update policy
                            logger.info("🧠 About to call trainer.train_step...")
                            try:
                                train_metrics = self.trainer.train_step(trajectories)
                                logger.info(f"✅ Training step completed successfully")
                                logger.debug(f"Training metrics keys: {list(train_metrics.keys()) if train_metrics else 'None'}")
                                
                                # Log training metrics to both WandB and Weave
                                if train_metrics and self.training_logger:
                                    logger.info(f"📊 Logging training metrics: step={train_metrics.get('step', 0)}")
                                    logger.info(f"📊 Metrics keys: {list(train_metrics.keys())}")
                                    logger.info(f"📊 Key metrics: policy_loss={train_metrics.get('policy_loss', 'MISSING')}, "
                                               f"kl_div={train_metrics.get('kl_divergence', 'MISSING')}, "
                                               f"total_loss={train_metrics.get('total_loss', 'MISSING')}")
                                    self.training_logger.log_training_step(
                                        metrics=train_metrics,
                                        step=train_metrics.get('step', 0),
                                        stage='training'
                                    )
                            except Exception as e:
                                logger.error(f"❌ Training step failed: {type(e).__name__}: {e}")
                                logger.error(f"Training step traceback:\n{traceback.format_exc()}")
                                raise
                            
                        except Exception as e:
                            logger.error(f"❌ Error in batch {batch_idx}: {type(e).__name__}: {e}")
                            logger.error(f"Batch error traceback:\n{traceback.format_exc()}")
                            raise
                        
                except Exception as e:
                    logger.error(f"❌ Error in epoch {epoch}: {type(e).__name__}: {e}")
                    logger.error(f"Epoch error traceback:\n{traceback.format_exc()}")
                    raise
                    
                    # Update step counter
                    self.current_step += 1
                    
                    # Add timing metrics
                    step_time = time.time() - step_start_time
                    train_metrics.update({
                        "step_time": step_time,
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                    })
                    
                    epoch_metrics.append(train_metrics)
                    
                    # Log metrics with enhanced logger
                    if self.training_logger:
                        self.training_logger.log_training_step(
                            metrics=train_metrics,
                            step=self.current_step,
                            stage="training"
                        )
                        
                        # Log model parameters and gradients (periodically to avoid overhead)
                        if self.current_step % 100 == 0:
                            self.training_logger.log_model_parameters(
                                model=self.policy.model,
                                step=self.current_step
                            )
                    
                    # Periodic evaluation
                    if self.current_step % eval_steps == 0:
                        eval_metrics = self.evaluate_model(epoch)
                        
                        if eval_metrics:
                            train_metrics.update(eval_metrics)
                            
                            # Log evaluation metrics
                            if self.training_logger:
                                self.training_logger.log_model_evaluation(
                                    evaluation_results=eval_metrics,
                                    step=self.current_step
                                )
                            
                            # Check for best model
                            current_score = eval_metrics.get("eval_success_rate", 0.0)
                            if current_score > self.best_eval_score:
                                self.best_eval_score = current_score
                                self.early_stopping_counter = 0
                                
                                # Save best model
                                best_model_dir = Path(self.configs["training"]["output_dir"]) / "best_model"
                                best_model_dir.mkdir(parents=True, exist_ok=True)
                                self.trainer.save_checkpoint(str(best_model_dir))
                                logger.info(f"New best model saved: {current_score:.3f}")
                            else:
                                self.early_stopping_counter += 1
                    
                    # Periodic checkpointing
                    if self.current_step % save_steps == 0:
                        self.save_checkpoint(epoch, train_metrics)
                    
                    # Early stopping check
                    if (early_stopping_patience > 0 and 
                        self.early_stopping_counter >= early_stopping_patience):
                        logger.info(f"Early stopping triggered after {self.early_stopping_counter} evaluations")
                        return
                    
                    # Log progress
                    if batch_idx % 10 == 0:
                        logger.info(f"Epoch {epoch + 1}, Batch {batch_idx}: "
                                   f"policy_loss={train_metrics.get('policy_loss', 0):.4f}, "
                                   f"kl_div={train_metrics.get('kl_divergence', 0):.4f}")
                
                # End of epoch
                self.current_epoch = epoch + 1
                epoch_time = time.time() - epoch_start_time
                
                if epoch_metrics:
                    avg_metrics = {
                        f"epoch_avg_{k}": np.mean([m.get(k, 0) for m in epoch_metrics])
                        for k in ["policy_loss", "kl_divergence", "avg_total_reward"]
                    }
                    avg_metrics["epoch_time"] = epoch_time
                    
                    logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s: {avg_metrics}")
                    
                    # Log epoch metrics
                    if self.training_logger:
                        self.training_logger.log_training_step(
                            metrics=avg_metrics,
                            step=self.current_step,
                            stage="epoch_summary"
                        )
                
                # Save checkpoint at end of epoch
                self.save_checkpoint(epoch + 1, avg_metrics if epoch_metrics else {})
            
            # Training completed
            total_time = time.time() - self.training_start_time
            logger.info(f"Training completed in {total_time:.1f}s after {num_epochs} epochs")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            # Final checkpoint
            try:
                final_metrics = {"final_step": self.current_step, "final_epoch": self.current_epoch}
                self.save_checkpoint(self.current_epoch, final_metrics)
            except Exception as e:
                logger.error(f"Error saving final checkpoint: {e}")
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        
        logger.info("Cleaning up training session...")
        
        for cleanup_fn in reversed(self.cleanup_functions):
            try:
                cleanup_fn()
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
        
        logger.info("Cleanup completed")
    
    async def run(self) -> None:
        """Run the complete training pipeline with detailed error tracking."""
        
        try:
            logger.info("🚀 Starting GRPO training session...")
            
            # Load configurations
            logger.info("📋 Loading configurations...")
            self.load_configs()
            logger.info("✅ Configurations loaded")
            
            # Initialize components in order with detailed logging
            logger.info("🔧 Initializing training components...")
            
            logger.info("1/6 Initializing logging...")
            self.initialize_logging()
            logger.info("✅ Logging initialized")
            
            logger.info("2/6 Initializing shared tool manager...")
            self.initialize_shared_tool_manager()
            logger.info("✅ Tool manager initialized")
            
            logger.info("3/6 Initializing policies...")
            self.initialize_policies()
            logger.info("✅ Policies initialized")
            
            logger.info("4/6 Initializing data loader...")
            self.initialize_data_loader()
            logger.info("✅ Data loader initialized")
            
            logger.info("5/6 Initializing trajectory collector...")
            self.initialize_trajectory_collector()
            logger.info("✅ Trajectory collector initialized")
            
            logger.info("6/6 Initializing GRPO trainer...")
            self.initialize_trainer()
            logger.info("✅ GRPO trainer initialized")
            
            # Resume from checkpoint if specified
            if self.args.resume_from:
                logger.info(f"📂 Resuming from checkpoint: {self.args.resume_from}")
                self.load_checkpoint(self.args.resume_from)
                logger.info("✅ Checkpoint loaded")
            
            logger.info("🎯 All components initialized successfully. Starting training...")
            
            # Run training
            await self.train()
            
        except Exception as e:
            logger.error(f"Training session failed: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            self.cleanup()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="GRPO Training for Multi-Turn Tool Use",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["lora", "full"],
        help="Training mode: 'lora' for LoRA adapters, 'full' for full fine-tuning"
    )
    
    # Optional arguments
    parser.add_argument(
        "--resume_from",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory from config"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        help="Override data path from config"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu", "auto"],
        default="auto",
        help="Device to use for training (auto will detect best available)"
    )
    
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable WandB logging"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    if args.mode == "full" and args.num_gpus < 2:
        print("Warning: Full fine-tuning recommended with 2+ GPUs")
    
    if args.num_gpus > 1 and not HAS_DEEPSPEED:
        print("Error: DeepSpeed required for multi-GPU training")
        sys.exit(1)
    
    # Disable wandb if requested
    if args.no_wandb:
        globals()['HAS_WANDB'] = False
    
    # Create and run training session
    logger.info(f"Starting GRPO training with config: {args.config}")
    logger.info(f"Training mode: {args.mode}, GPUs: {args.num_gpus}")
    
    try:
        session = TrainingSession(args)
        asyncio.run(session.run())
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()