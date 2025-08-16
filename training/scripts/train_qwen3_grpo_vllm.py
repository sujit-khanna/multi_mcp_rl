#!/usr/bin/env python3
"""
FAST vLLM-based GRPO training script with REAL environment rollouts
=================================================================

This script replaces the slow Hugging Face transformers with vLLM for
dramatically faster inference while maintaining the real environment
rollouts and GRPO training.

Key improvements:
- vLLM for 10x+ faster inference
- Real MCPToolEnvironment for actual tool execution
- TrajectoryCollector for parallel rollout collection
- SimpleSharedManager for efficient MCP tool management
- Proper reward computation based on actual task completion
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

# vLLM should be available in Python 3.12 virtual environment

import torch
import yaml
import numpy as np
from tqdm import tqdm

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Add utils path for tool validation
utils_path = str(Path(__file__).parent.parent / "utils")
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

# Import our vLLM policy instead of slow HF policy
from core.vllm_policy import create_vllm_policy
from core.grpo_trainer_gradient_fix import GRPOTrainerGradientFix
from core.grpo_trainer import Trajectory

# Import data components
from data.trajectory_collector import TrajectoryCollector, EpisodeResult

# Import environment components
env_path = str(Path(__file__).parent.parent.parent / "environments")
if env_path not in sys.path:
    sys.path.insert(0, env_path)

# Import with proper module handling
import environments.mcp_tool_environment as mcp_tool_environment
from environments.mcp_tool_environment_with_logging import MCPToolEnvironmentWithLogging
from environments.simple_shared_manager import SimpleSharedManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    logger.warning("WandB not available. Install with: pip install wandb")

try:
    import weave
    HAS_WEAVE = True
except ImportError:
    HAS_WEAVE = False
    logger.warning("Weave not available. Install with: pip install weave")


class VLLMRealEnvironmentGRPOTrainer:
    """Fast vLLM-based GRPO trainer using REAL environment rollouts"""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration"""
        self.config_path = config_path
        self.configs = self._load_configs()
        
        # Set seeds for determinism
        self._set_seeds()
        
        self.device = self._setup_device()
        
        # Components
        self.policy = None
        self.reference_policy = None
        self.trainer = None
        self.shared_tool_manager = None
        self.trajectory_collector = None
        
        # Data
        self.train_data = None
        self.valid_data = None
        
        # Training state
        self.global_step = 0
        self.best_eval_score = -float('inf')
        
        # Logging
        self.use_wandb = HAS_WANDB and self.configs['training'].get('use_wandb', True)
        self.use_weave = HAS_WEAVE and self.configs['training'].get('use_weave', True)
        
    def _load_configs(self) -> Dict[str, Any]:
        """Load all configuration files"""
        with open(self.config_path, 'r') as f:
            main_config = yaml.safe_load(f)
        
        configs = {'main': main_config}
        
        # Load GRPO config (use fixed version with our improvements)
        grpo_config_path = Path(self.config_path).parent / "grpo_config_fixed.yaml"
        if grpo_config_path.exists():
            with open(grpo_config_path, 'r') as f:
                configs['grpo'] = yaml.safe_load(f)
            logger.info(f"Loaded enhanced GRPO config from {grpo_config_path}")
        else:
            # Fallback to default
            grpo_config_path = Path(self.config_path).parent / "grpo_config.yaml"
            with open(grpo_config_path, 'r') as f:
                configs['grpo'] = yaml.safe_load(f)
            logger.warning("Using default GRPO config - consider using grpo_config_fixed.yaml")
        
        # Extract training config - support both nested and flat formats
        if 'training' in main_config:
            configs['training'] = main_config['training']
        else:
            # Extract training params from main config
            training_keys = [
                'num_epochs', 'batch_size', 'learning_rate', 'weight_decay', 
                'warmup_steps', 'use_mixed_precision', 'use_wandb', 'use_weave',
                'output_dir', 'save_every'
            ]
            configs['training'] = {k: v for k, v in main_config.items() if k in training_keys}
            
            # Set defaults
            configs['training'].setdefault('num_epochs', main_config.get('num_epochs', 3))
            configs['training'].setdefault('batch_size', main_config.get('batch_size', 4))
            configs['training'].setdefault('learning_rate', main_config.get('learning_rate', 5e-5))
            configs['training'].setdefault('use_mixed_precision', True)
            configs['training'].setdefault('use_wandb', True)
            configs['training'].setdefault('use_weave', True)
            configs['training'].setdefault('output_dir', main_config.get('output_dir', 'outputs/real-env-grpo'))
            configs['training'].setdefault('save_every', 1)
        
        configs['model'] = main_config.get('model', {
            'name': 'Qwen/Qwen2.5-0.5B-Instruct',
            'use_lora': False,  # vLLM doesn't support LoRA currently
            'lora_config': {},
            'value_head_hidden_dim': 512
        })
        configs['data'] = main_config.get('data', {})
        
        # Environment configuration - optimized for vLLM speed
        configs['environment'] = main_config.get('environment', {
            'num_parallel_envs': 1,  # Single environment due to concurrency fixes
            'max_episode_length': 5,  # Short episodes for fast iteration
            'retry_failed_episodes': False,  # No retries for speed
            'collect_log_probs': True,
            'executor_max_workers': 1  # Single worker due to concurrency fixes
        })
        
        return configs
    
    def _set_seeds(self) -> None:
        """Set all random seeds for deterministic training"""
        import random
        
        seed = self.configs['training'].get('seed', 42)
        logger.info(f"Setting seeds for determinism: {seed}")
        
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Make PyTorch operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for additional determinism
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def _setup_device(self) -> torch.device:
        """Setup compute device with proper configuration"""
        device_type = os.environ.get('DEVICE_TYPE', 'auto')
        
        if device_type == 'cpu':
            logger.info("Using CPU device (forced by DEVICE_TYPE env var)")
            return torch.device('cpu')
        elif device_type == 'mps':
            if torch.backends.mps.is_available():
                logger.info("Using MPS device (forced by DEVICE_TYPE env var)")
                return torch.device('mps')
            else:
                logger.warning("MPS not available, falling back to CPU")
                return torch.device('cpu')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return device
        elif torch.backends.mps.is_available():
            logger.info("CUDA not available, using MPS")
            return torch.device('mps')
        else:
            logger.info("No GPU available, using CPU")
            return torch.device('cpu')

    def _create_output_dir(self) -> str:
        """Create unique output directory"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if self.device.type == 'cuda':
            output_dir = f"outputs/vllm-grpo-gpu-{timestamp}"
        else:
            output_dir = f"outputs/vllm-grpo-{self.device.type}-{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        # Copy config files to output directory for reproducibility
        import shutil
        shutil.copy2(self.config_path, f"{output_dir}/")
        
        grpo_config_path = Path(self.config_path).parent / "grpo_config_fixed.yaml"
        if grpo_config_path.exists():
            shutil.copy2(grpo_config_path, f"{output_dir}/")
        
        return output_dir

    def _init_logging(self, output_dir: str):
        """Initialize logging systems"""
        
        # Setup file logging
        log_file = os.path.join(output_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Get root logger and add file handler
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
        
        # Initialize WandB
        if self.use_wandb:
            try:
                wandb.init(
                    project="vllm-grpo-real-env",
                    config=self.configs,
                    name=f"vllm-grpo-{time.strftime('%Y%m%d-%H%M%S')}",
                    tags=["vllm", "grpo", "real-env", "fast-inference"]
                )
                logger.info("✅ WandB initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                self.use_wandb = False
        
        # Initialize Weave
        if self.use_weave:
            try:
                weave.init("vllm-grpo-training")
                logger.info("✅ Weave initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Weave: {e}")
                self.use_weave = False

    def setup_components(self):
        """Setup all training components with vLLM"""
        
        logger.info("🚀 Setting up vLLM-based training components...")
        
        # Create output directory
        output_dir = self._create_output_dir()
        self._init_logging(output_dir)
        
        # Load data
        self._load_data()
        
        # Setup environment manager
        self._setup_environment_manager()
        
        # Setup vLLM policies (much faster than HF!)
        self._setup_vllm_policies()
        
        # Setup trajectory collector with single environment
        self._setup_trajectory_collector()
        
        # Setup GRPO trainer
        self._setup_grpo_trainer()
        
        logger.info("✅ All components setup successfully with vLLM!")

    def _load_data(self):
        """Load training and validation data"""
        
        # Data paths
        data_dir = Path(__file__).parent.parent.parent / "data"
        
        # Try multiple data paths
        train_paths = [
            data_dir / "processed" / "train.json",
            data_dir / "inputs" / "train.json",
            data_dir / "train.json"
        ]
        
        train_path = None
        for path in train_paths:
            if path.exists():
                train_path = path
                break
        
        if train_path is None:
            raise FileNotFoundError(f"Training data not found in any of: {train_paths}")
        
        logger.info(f"Loading training data from: {train_path}")
        with open(train_path, 'r') as f:
            self.train_data = json.load(f)
        
        logger.info(f"Loaded {len(self.train_data)} training examples")
        
        # Validation data (optional)
        valid_paths = [
            data_dir / "processed" / "valid.json",
            data_dir / "inputs" / "valid.json",
            data_dir / "valid.json"
        ]
        
        for path in valid_paths:
            if path.exists():
                logger.info(f"Loading validation data from: {path}")
                with open(path, 'r') as f:
                    self.valid_data = json.load(f)
                break
        
        if self.valid_data:
            logger.info(f"Loaded {len(self.valid_data)} validation examples")
        else:
            logger.info("No validation data found")

    def _setup_environment_manager(self):
        """Setup shared tool manager for environments"""
        
        logger.info("Setting up shared environment manager...")
        
        # Setup shared tool manager with API keys
        from environments.simple_shared_manager import SimpleSharedManager
        
        self.shared_tool_manager = SimpleSharedManager()
        
        # Load API keys
        parent_dir = Path(__file__).parent.parent.parent.parent
        env_file = parent_dir / ".env"
        
        if env_file.exists():
            from dotenv import load_dotenv
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from: {env_file}")
        else:
            logger.warning(f"No .env file found at: {env_file}")
        
        logger.info("✅ Environment manager setup completed")

    def _setup_vllm_policies(self):
        """Setup vLLM policies for fast inference"""
        
        logger.info("🔥 Setting up vLLM policies for ULTRA-FAST inference...")
        
        # Model config path - use the correct temp config
        model_config_path = Path(__file__).parent.parent / "configs" / "model_config_temp.yaml"
        
        if not model_config_path.exists():
            model_config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
        
        logger.info(f"Using model config: {model_config_path}")
        
        # Create vLLM policy (much faster than HF transformers)
        logger.info("Creating vLLM policy...")
        self.policy = create_vllm_policy(
            config_path=str(model_config_path),
            gpu_memory_utilization=0.6,  # Conservative memory usage
            max_model_len=512,  # Short context for fast generation
            enforce_eager=True  # Disable CUDA graphs for compatibility
        )
        
        # For GRPO we need a reference policy (for now, just use the same instance)
        # TODO: Implement proper reference policy with frozen weights
        logger.info("Creating reference vLLM policy...")
        self.reference_policy = create_vllm_policy(
            config_path=str(model_config_path),
            gpu_memory_utilization=0.6,
            max_model_len=512,
            enforce_eager=True
        )
        
        # Enable training mode for policy, eval mode for reference
        self.policy.enable_training_mode()
        self.reference_policy.enable_eval_mode()
        
        logger.info("✅ vLLM policies created - expect 10x+ faster inference!")

    def _setup_trajectory_collector(self):
        """Setup trajectory collector with single environment"""
        
        logger.info("Setting up trajectory collector...")
        
        # Environment factory for creating MCPToolEnvironment instances
        def env_factory(task_data: Dict[str, Any]) -> MCPToolEnvironmentWithLogging:
            """Create MCPToolEnvironment instance for trajectory collection"""
            
            # Extract environment parameters
            max_turns = self.configs['environment'].get('max_episode_length', 5)
            
            return MCPToolEnvironmentWithLogging(
                max_turns=max_turns,
                tool_manager=self.shared_tool_manager,
                task_metadata=task_data.get('task_metadata', {}),
                reward_spec=task_data.get('reward_spec', {}),
                prompt=task_data.get('prompt', [])
            )
        
        # Create trajectory collector with single environment (due to concurrency fixes)
        self.trajectory_collector = TrajectoryCollector(
            policy=self.policy,
            env_factory=env_factory,
            num_parallel_envs=1,  # CRITICAL FIX: Force single environment
            shared_tool_manager=self.shared_tool_manager,
            max_episode_length=self.configs['environment'].get('max_episode_length', 5),
            retry_failed_episodes=self.configs['environment'].get('retry_failed_episodes', False),
            collect_log_probs=self.configs['environment'].get('collect_log_probs', True),
            executor_max_workers=1  # CRITICAL FIX: Force single worker
        )
        
        logger.info("✅ Trajectory collector setup completed")

    def _setup_grpo_trainer(self):
        """Setup GRPO trainer"""
        
        logger.info("Setting up GRPO trainer...")
        
        # Create GRPO trainer
        self.trainer = GRPOTrainerGradientFix(
            policy=self.policy,
            reference_policy=self.reference_policy,
            config=self.configs['grpo']
        )
        
        logger.info("✅ GRPO trainer setup completed")

    async def collect_trajectories(self, tasks: List[Dict[str, Any]]) -> List[EpisodeResult]:
        """Collect trajectories using real environment rollouts"""
        
        logger.info(f"🎮 Collecting trajectories for {len(tasks)} tasks...")
        start_time = time.time()
        
        # Use trajectory collector for parallel episode collection
        episode_results = await self.trajectory_collector.collect_batch(tasks)
        
        collection_time = time.time() - start_time
        logger.info(f"✅ Trajectory collection completed in {collection_time:.2f}s")
        
        # Log statistics
        valid_episodes = [ep for ep in episode_results if ep.is_valid()]
        logger.info(f"   Valid episodes: {len(valid_episodes)}/{len(episode_results)}")
        
        if valid_episodes:
            avg_reward = np.mean([ep.total_reward for ep in valid_episodes])
            avg_turns = np.mean([ep.turns for ep in valid_episodes])
            logger.info(f"   Average reward: {avg_reward:.3f}")
            logger.info(f"   Average turns: {avg_turns:.1f}")
        
        return episode_results

    def convert_episodes_to_trajectories(self, episode_results: List[EpisodeResult]) -> List[Trajectory]:
        """Convert episode results to GRPO trajectory format"""
        
        trajectories = []
        
        for episode in episode_results:
            if not episode.is_valid():
                continue
            
            # Extract data from episode
            states = []
            actions = []
            rewards = []
            values = []
            log_probs = []
            
            # Reconstruct conversation states
            conversation_so_far = episode.initial_prompt.copy()
            
            for turn_data in episode.trajectory:
                # State is conversation before this action
                states.append(conversation_so_far.copy())
                
                # Action and metadata
                action = turn_data['action']
                reward = turn_data['reward']
                metadata = turn_data.get('metadata', {})
                
                actions.append(action)
                rewards.append(reward)
                
                # Extract log probability
                log_prob = metadata.get('log_prob', -1.0)
                log_probs.append(log_prob)
                
                # For now, use dummy values (TODO: compute actual values)
                values.append(0.0)
                
                # Update conversation history
                conversation_so_far.append({"role": "assistant", "content": action})
                
                # Add observation if present
                observation = turn_data.get('observation', '')
                if observation:
                    conversation_so_far.append({"role": "user", "content": observation})
            
            if states and actions:
                trajectory = Trajectory(
                    task_id=episode.task_id,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    values=values,
                    log_probs=log_probs,
                    dones=[False] * (len(rewards) - 1) + [True]  # Last turn is done
                )
                trajectories.append(trajectory)
        
        logger.info(f"Converted {len(trajectories)} valid episodes to trajectories")
        return trajectories

    async def train_step(self, batch_tasks: List[Dict[str, Any]]):
        """Perform one training step with real environment rollouts"""
        
        logger.info(f"🎓 Starting training step with {len(batch_tasks)} tasks")
        step_start_time = time.time()
        
        # 1. Collect trajectories using real environments
        logger.info("📊 Phase 1: Collecting real environment trajectories...")
        episode_results = await self.collect_trajectories(batch_tasks)
        
        # 2. Convert to GRPO format
        logger.info("🔄 Phase 2: Converting episodes to GRPO trajectories...")
        trajectories = self.convert_episodes_to_trajectories(episode_results)
        
        if not trajectories:
            logger.warning("No valid trajectories collected, skipping training step")
            return
        
        # 3. Perform GRPO update
        logger.info("⚡ Phase 3: Performing GRPO policy update...")
        try:
            metrics = self.trainer.train_step(trajectories)
            
            if metrics:
                logger.info(f"✅ Training step completed successfully")
                logger.info(f"   Policy loss: {metrics.get('policy_loss', 'N/A')}")
                logger.info(f"   Value loss: {metrics.get('value_loss', 'N/A')}")
                logger.info(f"   KL divergence: {metrics.get('kl_div', 'N/A')}")
                
                # Log to WandB
                if self.use_wandb:
                    wandb.log({
                        **metrics,
                        "step": self.global_step,
                        "trajectories_collected": len(trajectories),
                        "valid_episodes": len([ep for ep in episode_results if ep.is_valid()]),
                        "step_time": time.time() - step_start_time
                    })
                    
                    # Debug ping for WandB
                    logger.info(f"📊 WandB metrics logged for step {self.global_step}")
            else:
                logger.warning("Training step returned no metrics")
        
        except Exception as e:
            logger.error(f"❌ Error in GRPO training step: {e}")
            import traceback
            traceback.print_exc()

    async def train(self):
        """Main training loop with real environment rollouts"""
        
        logger.info("🚀 Starting vLLM-based GRPO training with real environments!")
        
        # Training parameters
        num_epochs = self.configs['training'].get('num_epochs', 3)
        batch_size = self.configs['training'].get('batch_size', 4)
        
        # Limit training data for faster iteration during development
        max_train_examples = 50  # Reduce for faster experimentation
        train_data_subset = self.train_data[:max_train_examples]
        logger.info(f"Using {len(train_data_subset)} training examples (limited for fast iteration)")
        
        for epoch in range(num_epochs):
            logger.info(f"🌟 Starting epoch {epoch + 1}/{num_epochs}")
            
            # Create batches
            batches = [
                train_data_subset[i:i + batch_size] 
                for i in range(0, len(train_data_subset), batch_size)
            ]
            
            for batch_idx, batch in enumerate(batches):
                logger.info(f"📦 Processing batch {batch_idx + 1}/{len(batches)} (size: {len(batch)})")
                
                try:
                    await self.train_step(batch)
                    self.global_step += 1
                    
                    # Log progress
                    if self.global_step % 5 == 0:
                        logger.info(f"✨ Completed {self.global_step} training steps")
                
                except Exception as e:
                    logger.error(f"❌ Error in training step: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        logger.info("🎉 Training completed!")

    async def cleanup(self):
        """Cleanup resources"""
        
        logger.info("🧹 Cleaning up resources...")
        
        # Cleanup trajectory collector
        if self.trajectory_collector:
            await self.trajectory_collector.cleanup()
        
        # Cleanup shared tool manager
        if self.shared_tool_manager:
            if hasattr(self.shared_tool_manager, 'cleanup'):
                await self.shared_tool_manager.cleanup()
        
        # Cleanup logging
        if self.use_wandb:
            wandb.finish()
        
        logger.info("✅ Cleanup completed")


async def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description="vLLM-based GRPO Training with Real Environments")
    parser.add_argument(
        "--config", 
        type=str, 
        default="training/configs/training_config_qwen3_0.6b.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🐛 Debug mode enabled")
    
    # Initialize trainer
    trainer = VLLMRealEnvironmentGRPOTrainer(args.config)
    
    try:
        # Setup all components
        trainer.setup_components()
        
        # Run training
        await trainer.train()
        
    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Cleanup
        await trainer.cleanup()


if __name__ == "__main__":
    # Run with asyncio
    asyncio.run(main())