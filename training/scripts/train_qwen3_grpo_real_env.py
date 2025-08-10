#!/usr/bin/env python3
"""
Enhanced GRPO training script with REAL environment rollouts
============================================================

This script fixes the fundamental issue where training was using mock trajectories
instead of actual RL with environment interaction. Now uses:
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

# Import our enhanced components
from core.qwen_policy_with_value_prompting import QwenPolicyWithValuePrompting
from core.grpo_trainer_gradient_fix import GRPOTrainerGradientFix
from core.grpo_trainer import Trajectory

# Import data components
from data.trajectory_collector import TrajectoryCollector, EpisodeResult

# Import environment components
env_path = str(Path(__file__).parent.parent.parent / "environments")
if env_path not in sys.path:
    sys.path.insert(0, env_path)

# Import with proper module handling
import mcp_tool_environment
from mcp_tool_environment_with_logging import MCPToolEnvironmentWithLogging
from simple_shared_manager import SimpleSharedManager

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


class RealEnvironmentGRPOTrainer:
    """GRPO trainer using REAL environment rollouts instead of mock data"""
    
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
            'name': 'Qwen/Qwen2.5-0.5B-Instruct',  # Use smaller model for MPS
            'use_lora': True,
            'lora_config': {},
            'value_head_hidden_dim': 512  # Smaller hidden dim for MPS
        })
        configs['data'] = main_config.get('data', {})
        
        # Environment configuration
        configs['environment'] = main_config.get('environment', {
            'num_parallel_envs': 4,
            'max_episode_length': 8,  # Reduced to prevent timeouts with untrained model
            'retry_failed_episodes': False,  # Disable retries for faster iteration
            'collect_log_probs': True
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
        
        if device_type == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("Using Apple Silicon MPS device")
            else:
                device = torch.device('cpu')
                logger.warning("No GPU available, using CPU (training will be slow)")
        else:
            device = torch.device(device_type)
            logger.info(f"Using specified device: {device_type}")
        
        return device
    
    def setup_logging(self):
        """Setup WandB and Weave logging"""
        if self.use_wandb:
            wandb.init(
                project=os.environ.get('WANDB_PROJECT', 'skyrl-grpo-real-env'),
                name=f"grpo-real-env-{time.strftime('%Y%m%d-%H%M%S')}",
                config={
                    **self.configs.get('grpo', {}),
                    **self.configs.get('training', {}),
                    **self.configs.get('environment', {}),
                    'device': str(self.device),
                    'real_environment': True,
                    'phase1_fixes': ['value_function', 'ref_policy_updates', 'gradient_clipping']
                }
            )
            # Define a consistent step metric so charts align by training step
            try:
                wandb.define_metric('trainer/step')
                wandb.define_metric('trainer/*', step_metric='trainer/step')
                wandb.define_metric('rollouts/*', step_metric='trainer/step')
                wandb.define_metric('eval/*', step_metric='trainer/step')
            except Exception:
                pass
            logger.info("WandB logging initialized")
        
        if self.use_weave:
            weave.init(project_name=os.environ.get('WEAVE_PROJECT', 'skyrl-grpo-real-env'))
            logger.info("Weave logging initialized")

    def _log_wandb(self, payload: dict, step: int, commit: bool = True) -> None:
        """Safely log numeric scalars to WandB with a unified step key.

        Converts tensors and numpy scalars to Python numbers to avoid silent drops.
        """
        if not self.use_wandb:
            return
        safe = {}
        for k, v in payload.items():
            try:
                if hasattr(v, 'item'):
                    safe[k] = v.item()
                else:
                    # numpy and python scalars
                    import numpy as _np
                    if isinstance(v, (_np.floating, _np.integer)):
                        safe[k] = float(v)
                    elif isinstance(v, (int, float)):
                        safe[k] = v
            except Exception:
                continue
        safe['trainer/step'] = int(step)
        try:
            wandb.log(safe, commit=commit)
        except Exception as e:
            logger.warning(f"Failed to log to WandB: {e}")
    
    def load_data(self):
        """Load training and validation data"""
        # Support both old and new config formats
        if 'data' in self.configs and 'train_path' in self.configs['data']:
            data_path = Path(self.configs['data']['train_path'])
        elif 'data_path' in self.configs['main']:
            data_path = Path(self.configs['main']['data_path'])
        else:
            # Default fallback
            data_path = Path("data/processed/train.json")
            logger.warning(f"No data path specified in config, using default: {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at {data_path}")
        
        with open(data_path, 'r') as f:
            all_data = json.load(f)
        
        # Split data
        split_ratio = self.configs.get('data', {}).get('train_split', 0.9)
        split_idx = int(len(all_data) * split_ratio)
        
        self.train_data = all_data[:split_idx]
        self.valid_data = all_data[split_idx:]
        
        logger.info(f"Loaded {len(self.train_data)} training and {len(self.valid_data)} validation examples")
        
        # Debug: show first example
        if self.train_data:
            first_example = self.train_data[0]
            logger.info(f"First training example keys: {list(first_example.keys())}")
            if 'prompt' in first_example and first_example['prompt']:
                logger.info(f"First prompt has {len(first_example['prompt'])} messages")
                logger.info(f"First message role: {first_example['prompt'][0].get('role', 'unknown')}")
        else:
            logger.warning("No training data loaded!")
    
    def setup_models(self):
        """Setup policy and reference policy with value heads"""
        # Create temporary model config files for QwenPolicy
        model_config_path = "configs/model_config_temp.yaml"
        training_config_path = "configs/training_config_temp.yaml"
        
        # Create model config compatible with QwenPolicy
        model_name = self.configs['model'].get('name', 'Qwen/Qwen2.5-0.5B-Instruct')
        
        # Adjust max_length based on device
        max_length = 2048 if self.device.type == 'cuda' else 1024  # Larger for CUDA
        
        model_config = {
            'model_name': model_name,
            'tokenizer_name': model_name,  # Use same name for tokenizer
            'torch_dtype': 'float16' if self.device.type == 'cuda' else 'float32',  # FP16 for CUDA
            'device_map': 'auto' if self.device.type == 'cuda' else None,  # Auto device map for CUDA
            'trust_remote_code': True,
            'max_length': max_length,  # Reduced for MPS
            'stop_sequences': ["</tool_call>", "</think>", "<|im_end|>"],
            'memory_optimization': {
                'torch_dtype': 'float16' if self.device.type == 'cuda' else 'float32',
                'load_in_4bit': False,
                'load_in_8bit': False,
                'device_map': 'auto' if self.device.type == 'cuda' else None,
                'low_cpu_mem_usage': True,
                'trust_remote_code': True
            },
            'quantization': {
                'bnb_4bit_compute_dtype': 'float16',
                'bnb_4bit_use_double_quant': True,
                'bnb_4bit_quant_type': 'nf4'
            },
            'lora_mode': {
                'enabled': self.configs['model'].get('use_lora', True),
                'r': 16,
                'alpha': 32,
                'dropout': 0.1,
                'bias': "none",
                'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                'fan_in_fan_out': False,
                'init_lora_weights': True
            },
            'generation': {
                'max_new_tokens': 512,
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 50,
                'do_sample': True,
                'pad_token_id': 151643,  # Standard Qwen pad token
                'eos_token_id': 151645,
                'repetition_penalty': 1.1
            }
        }
        
        # Save temporary configs
        import os
        os.makedirs('configs', exist_ok=True)
        
        with open(model_config_path, 'w') as f:
            yaml.dump(model_config, f)
        
        with open(training_config_path, 'w') as f:
            yaml.dump(self.configs['training'], f)
        
        # Create policy with value head
        logger.info("Creating policy with value head...")
        value_head_hidden_dim = self.configs['model'].get('value_head_hidden_dim', 512)  # Smaller for MPS
        
        # Enable 4-bit quantization for CUDA, disable for MPS/CPU
        enable_4bit = (self.device.type == 'cuda' and 
                      self.configs['model'].get('load_in_4bit', True) and
                      self.configs['model'].get('use_lora', True))
        
        self.policy = QwenPolicyWithValuePrompting(
            model_config_path=model_config_path,
            training_config_path=training_config_path,
            use_lora=self.configs['model'].get('use_lora', True),
            device=str(self.device),
            load_in_4bit=enable_4bit,
            value_head_hidden_dim=value_head_hidden_dim
        )
        
        # Create reference policy (copy of initial policy)
        logger.info("Creating reference policy...")
        self.reference_policy = QwenPolicyWithValuePrompting(
            model_config_path=model_config_path,
            training_config_path=training_config_path,
            use_lora=self.configs['model'].get('use_lora', True),
            device=str(self.device),
            load_in_4bit=enable_4bit,
            value_head_hidden_dim=value_head_hidden_dim
        )
        
        # Copy initial weights to reference policy
        logger.info("Synchronizing reference policy weights...")
        with torch.no_grad():
            for ref_param, param in zip(
                self.reference_policy.model.parameters(),
                self.policy.model.parameters()
            ):
                ref_param.data.copy_(param.data)
            
            # Also sync value heads
            for ref_param, param in zip(
                self.reference_policy.value_head.parameters(),
                self.policy.value_head.parameters()
            ):
                ref_param.data.copy_(param.data)
        
        # Enable training mode for policy, eval mode for reference
        self.policy.enable_training_mode()
        self.reference_policy.enable_eval_mode()
        
        # Update device to match actual policy device (in case of fallback)
        self.device = self.policy.device
        
        logger.info(f"Models initialized with {self.policy.get_trainable_parameters():,} trainable parameters")
        logger.info(f"Actual device after model setup: {self.device}")
    
    def setup_trainer(self):
        """Setup enhanced GRPO trainer with all fixes"""
        # Determine if we should use mixed precision
        enable_mixed_precision = (
            self.configs['training'].get('use_mixed_precision', True) and
            self.device.type == 'cuda'
        )
        
        logger.info(f"Creating enhanced GRPO trainer (mixed precision: {enable_mixed_precision})")
        
        self.trainer = GRPOTrainerGradientFix(
            policy=self.policy,
            reference_policy=self.reference_policy,
            grpo_config=self.configs['grpo'],
            training_config=self.configs['training'],
            device=self.device,
            enable_mixed_precision=enable_mixed_precision
        )
        
        logger.info("Enhanced GRPO trainer initialized with:")
        logger.info(f"  - Value function training ✓")
        logger.info(f"  - Reference policy updates every {self.trainer.ref_update_freq} steps ✓")
        logger.info(f"  - Gradient clipping fix for mixed precision ✓")
        logger.info(f"  - Max gradient norm: {self.trainer.max_grad_norm}")
        
        # Enable WandB gradient tracking if wandb is initialized
        if self.use_wandb and wandb.run is not None:
            # Watch the model for gradient tracking
            wandb.watch(
                self.policy.model,  # Watch the main policy model
                log="all",  # Log gradients and parameters
                log_freq=10,  # Log every 10 batches
                log_graph=False  # Don't log computation graph (too large)
            )
            # Also watch the value head separately
            wandb.watch(
                self.policy.value_head,
                log="all",
                log_freq=10,
                log_graph=False
            )
            logger.info("✅ WandB gradient tracking enabled for policy and value head")
    
    async def setup_environment(self):
        """Setup shared tool manager and trajectory collector"""
        logger.info("Setting up real environment components...")
        
        # Initialize shared tool manager
        self.shared_tool_manager = SimpleSharedManager()
        await self.shared_tool_manager.initialize()
        
        logger.info(f"Shared tool manager initialized with {len(self.shared_tool_manager.get_available_tools())} tools")
        
        # Perform tool validation preflight check
        from utils.tool_validator import validate_tools_before_training
        
        data_paths = [
            "data/processed/train.json",
            "data/inputs/train.json"
        ]
        config_paths = [self.config_path]
        
        validation_report = await validate_tools_before_training(
            self.shared_tool_manager, 
            data_paths, 
            config_paths
        )
        
        # Store validation results for later use
        self.tool_validation_report = validation_report
        
        # Create environment factory with enhanced logging
        def env_factory(task_data: Dict[str, Any]) -> MCPToolEnvironmentWithLogging:
            env = MCPToolEnvironmentWithLogging(task_data)
            # Set the shared tool manager
            env.tool_manager = self.shared_tool_manager
            env.available_tools = [tool['name'] for tool in self.shared_tool_manager.get_available_tools()]
            logger.info(f"Created environment with {len(env.available_tools)} available tools")
            return env
        
        # Initialize trajectory collector
        env_config = self.configs['environment']
        self.trajectory_collector = TrajectoryCollector(
            policy=self.policy,
            env_factory=env_factory,
            num_parallel_envs=env_config.get('num_parallel_envs', 4),
            shared_tool_manager=self.shared_tool_manager,
            max_episode_length=env_config.get('max_episode_length', 15),
            retry_failed_episodes=env_config.get('retry_failed_episodes', True),
            collect_log_probs=env_config.get('collect_log_probs', True)
        )
        
        logger.info("✅ Real environment setup complete!")
    
    async def collect_trajectories(self, tasks: List[Dict], num_rollouts: int = 1) -> List[Trajectory]:
        """Collect REAL trajectories using MCPToolEnvironment"""
        logger.info(f"\n{'#'*80}")
        logger.info(f"🎮 COLLECTING REAL TRAJECTORIES")
        logger.info(f"   Tasks: {len(tasks)}")
        logger.info(f"   Rollouts per task: {num_rollouts}")
        logger.info(f"   Available tools: {len(self.shared_tool_manager.get_available_tools())}")
        logger.info(f"{'#'*80}\n")
        
        # Expand tasks for multiple rollouts
        expanded_tasks = []
        for task in tasks:
            for _ in range(num_rollouts):
                expanded_tasks.append(task)
        
        # Collect episodes using real environment
        episode_results = await self.trajectory_collector.collect_batch(expanded_tasks)
        
        # Log episode results summary
        logger.info(f"\n📊 EPISODE COLLECTION SUMMARY:")
        logger.info(f"   Total episodes: {len(episode_results)}")
        valid_episodes = [e for e in episode_results if e.is_valid()]
        logger.info(f"   Valid episodes: {len(valid_episodes)}")
        logger.info(f"   Failed episodes: {len(episode_results) - len(valid_episodes)}")
        
        # Log tool usage statistics
        if valid_episodes:
            if hasattr(valid_episodes[0], 'tools_used') and isinstance(valid_episodes[0].tools_used, int):
                total_tool_calls = sum(e.tools_used for e in valid_episodes)
            else:
                # tools_used might be a list
                total_tool_calls = sum(len(e.tools_used) if isinstance(e.tools_used, list) else e.tools_used for e in valid_episodes)
            logger.info(f"   Total tool calls made: {total_tool_calls}")
            logger.info(f"   Average tools per episode: {total_tool_calls / len(valid_episodes):.2f}")
        else:
            logger.info(f"   No valid episodes to report tool usage")
        
        # Convert EpisodeResult to GRPO Trajectory format
        trajectories = []
        for i, episode in enumerate(episode_results):
            if not episode.is_valid():
                logger.warning(f"Skipping invalid episode {i}: {episode.task_id} - {episode.error}")
                continue
            
            # Log trajectory details
            logger.debug(f"\nProcessing trajectory {i}:")
            logger.debug(f"   Task ID: {episode.task_id}")
            logger.debug(f"   Turns: {episode.turns}")
            logger.debug(f"   Total reward: {episode.total_reward:.3f}")
            logger.debug(f"   Tools used: {episode.tools_used}")
            
            # Extract trajectory data
            states = []
            actions = []
            rewards = []
            dones = []
            old_log_probs = []
            
            # Build conversation states from trajectory
            # CRITICAL FIX: Initialize with the initial prompt to match rollout states
            conversation_history = copy.deepcopy(episode.initial_prompt) if hasattr(episode, 'initial_prompt') else []
            for turn_data in episode.trajectory:
                # Add the state before this action
                states.append(conversation_history.copy())
                
                # Extract action, reward, done
                actions.append(turn_data['action'])
                rewards.append(turn_data['reward'])
                dones.append(turn_data['done'])
                
                # Extract log prob if available
                log_prob = turn_data.get('metadata', {}).get('log_prob')
                if log_prob is not None:
                    old_log_probs.append(torch.tensor(log_prob, device=self.device))
                else:
                    # Compute if missing
                    with torch.no_grad():
                        lp = self.policy.compute_log_probs([conversation_history], [turn_data['action']])
                        old_log_probs.append(lp[0] if len(lp) > 0 else torch.tensor(-10.0, device=self.device))
                
                # Update conversation history
                conversation_history.append({"role": "assistant", "content": turn_data['action']})
                if turn_data.get('observation'):
                    conversation_history.append({"role": "user", "content": turn_data['observation']})
            
            if states and actions:
                traj = Trajectory(
                    task_id=episode.task_id,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    dones=dones
                )
                # Store old log probs for GRPO
                traj.old_log_probs = old_log_probs
                
                trajectories.append(traj)
        
        logger.info(f"\n✅ TRAJECTORY CONVERSION COMPLETE")
        logger.info(f"   Valid trajectories: {len(trajectories)}/{len(episode_results)}")
        
        # Log detailed statistics
        if trajectories:
            rewards = [traj.total_reward for traj in trajectories]
            lengths = [traj.length for traj in trajectories]
            
            logger.info(f"\n📈 TRAJECTORY STATISTICS:")
            logger.info(f"   Average reward: {np.mean(rewards):.3f} (std: {np.std(rewards):.3f})")
            logger.info(f"   Min/Max reward: {np.min(rewards):.3f} / {np.max(rewards):.3f}")
            logger.info(f"   Average length: {np.mean(lengths):.1f} (std: {np.std(lengths):.1f})")
            logger.info(f"   Min/Max length: {np.min(lengths)} / {np.max(lengths)}")
            
            # Check if rewards are all the same (indicating mock data)
            unique_rewards = set()
            for traj in trajectories:
                for r in traj.rewards:
                    unique_rewards.add(r)
            
            if len(unique_rewards) <= 2:
                logger.warning(f"⚠️ WARNING: Only {len(unique_rewards)} unique reward values found: {unique_rewards}")
                logger.warning(f"This suggests mock rewards instead of real environment feedback!")
        
        logger.info(f"{'#'*80}\n")
        
        return trajectories
    
    async def train_epoch(self, epoch: int):
        """Train for one epoch with REAL environment rollouts"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting Epoch {epoch} with REAL ENVIRONMENT")
        logger.info(f"{'='*50}")
        
        # Shuffle training data
        train_tasks = self.train_data.copy()
        np.random.shuffle(train_tasks)
        
        # Training loop
        batch_size = self.configs['training'].get('batch_size', 4)
        num_batches = len(train_tasks) // batch_size
        
        logger.info(f"Using batch size: {batch_size}")
        
        epoch_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'kl_divergence': [],
            'grad_norm': [],
            'rewards': [],
            'trajectory_lengths': []
        }
        
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch}")
        
        for batch_idx in progress_bar:
            # Get batch of tasks
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(train_tasks))
            batch_tasks = train_tasks[batch_start:batch_end]
            
            # Collect REAL trajectories
            trajectories = await self.collect_trajectories(batch_tasks, num_rollouts=1)
            
            if not trajectories:
                logger.warning(f"No valid trajectories collected for batch {batch_idx}")
                continue
            
            # Training step
            try:
                metrics = self.trainer.train_step(trajectories)
                
                # Update metrics
                for key in ['policy_loss', 'value_loss', 'kl_divergence', 'grad_norm']:
                    if key in metrics:
                        epoch_metrics[key].append(metrics[key])
                
                # Track trajectory statistics
                trajectory_rewards = [traj.total_reward for traj in trajectories]
                trajectory_lengths = [traj.length for traj in trajectories]
                epoch_metrics['rewards'].extend(trajectory_rewards)
                epoch_metrics['trajectory_lengths'].extend(trajectory_lengths)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics.get('total_loss', 0):.4f}",
                    'kl': f"{metrics.get('kl_divergence', 0):.4f}",
                    'reward': f"{np.mean(trajectory_rewards):.3f}"
                })
                
                # Log to WandB with clear namespaces
                rollout_log = {
                    'rollouts/avg_reward': np.mean(trajectory_rewards) if trajectory_rewards else 0,
                    'rollouts/avg_length': np.mean(trajectory_lengths) if trajectory_lengths else 0,
                    'rollouts/min_reward': np.min(trajectory_rewards) if trajectory_rewards else 0,
                    'rollouts/max_reward': np.max(trajectory_rewards) if trajectory_rewards else 0,
                    'rollouts/num_trajectories': len(trajectories),
                }
                trainer_log = {
                    'trainer/total_loss': metrics.get('total_loss', 0),
                    'trainer/policy_loss': metrics.get('policy_loss', 0),
                    'trainer/value_loss': metrics.get('value_loss', 0),
                    'trainer/kl_divergence': metrics.get('kl_divergence', 0),
                    'trainer/kl_coef': metrics.get('kl_coef', 0),
                    'trainer/avg_advantage': metrics.get('avg_advantage', 0),
                    'trainer/std_advantage': metrics.get('std_advantage', 0),
                    'trainer/learning_rate': self.trainer.optimizer.param_groups[0]['lr'],
                    'trainer/epoch': int(epoch),
                }
                # Useful system traces
                system_log = {
                    'system/gpu_memory_allocated': (torch.cuda.memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0,
                    'system/gpu_memory_reserved': (torch.cuda.memory_reserved() / (1024**3)) if torch.cuda.is_available() else 0,
                }
                self._log_wandb({**rollout_log, **trainer_log, **system_log}, step=self.global_step, commit=True)
                
                self.global_step += 1
                
                # Clear MPS cache every N steps
                if self.device.type == 'mps' and torch.backends.mps.is_available():
                    if self.global_step % 10 == 0:
                        try:
                            torch.mps.empty_cache()
                        except:
                            pass
                
            except Exception as e:
                logger.error(f"Error in training step: {e}")
                continue
        
        # Compute epoch statistics
        epoch_stats = {}
        for key, values in epoch_metrics.items():
            if values and key not in ['rewards', 'trajectory_lengths']:
                epoch_stats[f'epoch_{key}_mean'] = np.mean(values)
                epoch_stats[f'epoch_{key}_std'] = np.std(values)
        
        # Add trajectory statistics
        if epoch_metrics['rewards']:
            epoch_stats['epoch_avg_reward'] = np.mean(epoch_metrics['rewards'])
            epoch_stats['epoch_avg_length'] = np.mean(epoch_metrics['trajectory_lengths'])
        
        logger.info(f"Epoch {epoch} completed:")
        for key, value in epoch_stats.items():
            logger.info(f"  {key}: {value:.4f}")
        # Epoch summary logging
        if epoch_stats:
            epoch_log = {f'trainer/{k}': v for k, v in epoch_stats.items()}
            epoch_log['trainer/epoch'] = int(epoch)
            self._log_wandb(epoch_log, step=self.global_step, commit=True)
        
        return epoch_stats
    
    async def evaluate(self, epoch: int):
        """Evaluate on validation set with real environment"""
        logger.info(f"\nEvaluating epoch {epoch} with real environment...")
        
        # Use smaller subset for evaluation
        eval_subset_size = min(20, len(self.valid_data))
        
        # Set policy to eval mode
        self.policy.enable_eval_mode()
        
        # Collect validation trajectories
        val_trajectories = await self.collect_trajectories(
            self.valid_data[:eval_subset_size],
            num_rollouts=1
        )
        
        # Set policy back to training mode
        self.policy.enable_training_mode()
        
        if not val_trajectories:
            logger.warning("No validation trajectories collected")
            return {}
        
        # Compute evaluation metrics
        total_rewards = [traj.total_reward for traj in val_trajectories]
        trajectory_lengths = [traj.length for traj in val_trajectories]
        
        eval_metrics = {
            'eval_avg_reward': np.mean(total_rewards),
            'eval_std_reward': np.std(total_rewards),
            'eval_avg_length': np.mean(trajectory_lengths),
            'eval_num_trajectories': len(val_trajectories)
        }
        
        logger.info(f"Validation results:")
        for key, value in eval_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Check if this is the best model
        if eval_metrics['eval_avg_reward'] > self.best_eval_score:
            self.best_eval_score = eval_metrics['eval_avg_reward']
            self.save_checkpoint('best_model', is_best=True)
        
        eval_log = {f'eval/{k}': v for k, v in eval_metrics.items()}
        eval_log['trainer/epoch'] = int(epoch)
        self._log_wandb(eval_log, step=self.global_step, commit=True)
        
        return eval_metrics
    
    def save_checkpoint(self, name: str, is_best: bool = False):
        """Save training checkpoint"""
        output_dir = Path(self.configs['training'].get('output_dir', 'outputs'))
        checkpoint_dir = output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        self.policy.save_checkpoint(str(checkpoint_dir / 'policy'))
        self.reference_policy.save_checkpoint(str(checkpoint_dir / 'reference_policy'))
        
        # Save trainer state
        self.trainer.save_checkpoint(str(checkpoint_dir))
        
        # Save metadata
        metadata = {
            'global_step': self.global_step,
            'best_eval_score': self.best_eval_score,
            'is_best': is_best,
            'configs': self.configs,
            'real_environment': True
        }
        
        with open(checkpoint_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def _log_wandb(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """Log metrics to WandB"""
        if self.use_wandb:
            try:
                wandb.log(metrics, step=step, commit=commit)
                logger.debug(f"WandB logged metrics: {list(metrics.keys())}")
            except Exception as e:
                logger.warning(f"Failed to log to WandB: {e}")
        
        if HAS_WEAVE and hasattr(weave, 'log'):
            try:
                weave.log(metrics)
                logger.debug(f"Weave logged metrics: {list(metrics.keys())}")
            except Exception as e:
                logger.warning(f"Failed to log to Weave: {e}")
    
    async def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with real environment rollouts"""
        logger.info(f"\n{'='*30} EPOCH {epoch} {'='*30}")
        
        epoch_stats = {
            'epoch': epoch,
            'trajectories_collected': 0,
            'total_reward': 0.0,
            'avg_reward': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'kl_divergence': 0.0,
            'failed_episodes': 0,
            'successful_episodes': 0
        }
        
        batch_size = self.configs['training'].get('batch_size', 4)
        
        # Collect trajectories for this epoch
        logger.info(f"Collecting {batch_size} trajectories...")
        
        try:
            trajectories = await self.trajectory_collector.collect_batch(
                tasks=self.train_data[:batch_size],  # Use first batch_size tasks
                batch_timeout=180.0  # 3 minute timeout per batch
            )
            
            epoch_stats['trajectories_collected'] = len(trajectories)
            
            # Convert EpisodeResult to Trajectory objects for GRPO
            grpo_trajectories = []
            for episode_result in trajectories:
                if episode_result.is_valid():
                    # Convert EpisodeResult to Trajectory format
                    trajectory = Trajectory(
                        task_id=episode_result.task_id,
                        states=[],  # Will be populated from trajectory data
                        actions=[], 
                        rewards=[],
                        dones=[],
                        log_probs=None,
                        values=None
                    )
                    
                    # Extract data from episode trajectory
                    for step in episode_result.trajectory:
                        trajectory.actions.append(step.get('action', ''))
                        trajectory.rewards.append(step.get('reward', 0.0))
                        trajectory.dones.append(step.get('done', False))
                        trajectory.states.append(step.get('state', []))
                    
                    # Set total reward and length
                    trajectory.total_reward = sum(trajectory.rewards) 
                    trajectory.length = len(trajectory.actions)
                    
                    grpo_trajectories.append(trajectory)
            
            epoch_stats['trajectories_collected'] = len(grpo_trajectories)
            
            # Calculate trajectory statistics  
            rewards = [traj.total_reward for traj in grpo_trajectories]
            epoch_stats['total_reward'] = sum(rewards)
            epoch_stats['avg_reward'] = np.mean(rewards) if rewards else 0.0
            
            # Count successes/failures
            for traj in grpo_trajectories:
                if traj.total_reward > 0.5:  # Success threshold
                    epoch_stats['successful_episodes'] += 1
                else:
                    epoch_stats['failed_episodes'] += 1
            
            logger.info(f"Collected {len(grpo_trajectories)} trajectories")
            logger.info(f"Average reward: {epoch_stats['avg_reward']:.3f}")
            logger.info(f"Success rate: {epoch_stats['successful_episodes']}/{len(grpo_trajectories)}")
            
            # Train the policy with GRPO
            if len(grpo_trajectories) > 0:
                logger.info("Training policy with GRPO...")
                
                # Train with GRPO trajectories
                training_metrics = self.trainer.train_step(grpo_trajectories)
                
                # Update epoch stats with training metrics
                epoch_stats.update({
                    'policy_loss': training_metrics.get('policy_loss', 0.0),
                    'value_loss': training_metrics.get('value_loss', 0.0), 
                    'kl_divergence': training_metrics.get('kl_divergence', 0.0)
                })
                
                self.global_step += 1
                
                # Log to WandB
                wandb_metrics = {
                    'train/epoch': epoch,
                    'train/avg_reward': epoch_stats['avg_reward'],
                    'train/total_reward': epoch_stats['total_reward'],
                    'train/success_rate': epoch_stats['successful_episodes'] / len(grpo_trajectories),
                    'train/policy_loss': epoch_stats['policy_loss'],
                    'train/value_loss': epoch_stats['value_loss'],
                    'train/kl_divergence': epoch_stats['kl_divergence'],
                    'train/trajectories_collected': epoch_stats['trajectories_collected'],
                    'trainer/global_step': self.global_step
                }
                
                self._log_wandb(wandb_metrics, step=self.global_step, commit=True)
                
                logger.info(f"Training metrics - Policy Loss: {epoch_stats['policy_loss']:.4f}, "
                           f"Value Loss: {epoch_stats['value_loss']:.4f}, "
                           f"KL Div: {epoch_stats['kl_divergence']:.4f}")
            
        except Exception as e:
            logger.error(f"Error during training epoch {epoch}: {e}")
            epoch_stats['failed_episodes'] = batch_size
            
            # Still log the failure
            self._log_wandb({
                'train/epoch': epoch,
                'train/error': 1,
                'train/avg_reward': 0.0,
                'trainer/global_step': self.global_step
            }, step=self.global_step, commit=True)
        
        return epoch_stats
    
    async def train(self):
        """Main training loop with REAL environment rollouts"""
        logger.info("\n" + "="*50)
        logger.info("Starting REAL ENVIRONMENT GRPO Training")
        logger.info("="*50)
        
        # Setup
        self.setup_logging()
        self.load_data()
        self.setup_models()
        self.setup_trainer()
        await self.setup_environment()
        
        # Training loop
        num_epochs = self.configs['training'].get('num_epochs', 3)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            epoch_stats = await self.train_epoch(epoch)
            
            # Evaluate (optional - skip if no validation data)
            try:
                if hasattr(self, 'valid_data') and self.valid_data:
                    eval_stats = await self.evaluate(epoch)
                else:
                    logger.info("Skipping evaluation - no validation data available")
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
            
            # Save checkpoint
            if epoch % self.configs['training'].get('save_every', 1) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}')
        
        # Final save
        self.save_checkpoint('final_model')
        
        # Cleanup
        await self.shared_tool_manager.cleanup()
        await self.trajectory_collector.cleanup()
        
        logger.info("\n" + "="*50)
        logger.info("Training completed!")
        logger.info(f"Best evaluation score: {self.best_eval_score:.4f}")
        logger.info("="*50)
        
        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Real Environment GRPO Training')
    parser.add_argument(
        '--config',
        type=str,
        default='training/configs/training_config_qwen3_0.6b.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'mps', 'cpu', 'auto'],
        default='auto',
        help='Device to use for training'
    )
    parser.add_argument(
        '--mixed-precision',
        type=str,
        choices=['no', 'fp16', 'bf16'],
        default='no',
        help='Mixed precision training mode'
    )
    parser.add_argument(
        '--enable-profiling',
        action='store_true',
        help='Enable GPU profiling for performance analysis'
    )
    args = parser.parse_args()
    
    # Override device if specified
    if args.device != 'auto':
        os.environ['DEVICE_TYPE'] = args.device
    
    # Set mixed precision if specified
    if args.mixed_precision != 'no':
        os.environ['MIXED_PRECISION'] = args.mixed_precision
    
    # Create trainer
    trainer = RealEnvironmentGRPOTrainer(args.config)
    
    # Run training with asyncio
    asyncio.run(trainer.train())


if __name__ == '__main__':
    main()