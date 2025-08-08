#!/usr/bin/env python3
"""
Enhanced GRPO training script for Qwen2.5-1.5B-Instruct with all Phase 1 fixes:
- Value function training (Fix 1.1)
- Reference policy updates (Fix 1.2)  
- Gradient clipping for mixed precision (Fix 3.1)
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

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our enhanced components
from core.qwen_policy_with_value import QwenPolicyWithValue
from core.grpo_trainer_gradient_fix import GRPOTrainerGradientFix
from core.grpo_trainer import Trajectory
from environments.mcp_tool_environment import MCPToolEnvironment

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


class EnhancedGRPOTrainer:
    """Enhanced GRPO trainer using all Phase 1 improvements"""
    
    def __init__(self, config_path: str):
        """Initialize enhanced trainer with configuration"""
        self.config_path = config_path
        self.configs = self._load_configs()
        self.device = self._setup_device()
        
        # Components
        self.policy = None
        self.reference_policy = None
        self.trainer = None
        self.environment = None
        
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
            configs['training'].setdefault('batch_size', main_config.get('batch_size', 1))
            configs['training'].setdefault('learning_rate', main_config.get('learning_rate', 5e-5))
            configs['training'].setdefault('use_mixed_precision', True)
            configs['training'].setdefault('use_wandb', True)
            configs['training'].setdefault('use_weave', True)
            configs['training'].setdefault('output_dir', main_config.get('output_dir', 'outputs/enhanced-grpo'))
            configs['training'].setdefault('save_every', 1)
        
        configs['model'] = main_config.get('model', {
            'name': 'Qwen/Qwen2.5-1.5B-Instruct',
            'use_lora': True,
            'lora_config': {},
            'value_head_hidden_dim': 1024
        })
        configs['data'] = main_config.get('data', {})
        
        return configs
    
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
                project=os.environ.get('WANDB_PROJECT', 'skyrl-grpo-enhanced'),
                name=f"grpo-enhanced-{time.strftime('%Y%m%d-%H%M%S')}",
                config={
                    **self.configs['grpo'],
                    **self.configs['training'],
                    'device': str(self.device),
                    'phase1_fixes': ['value_function', 'ref_policy_updates', 'gradient_clipping']
                }
            )
            logger.info("WandB logging initialized")
        
        if self.use_weave:
            weave.init(project_name=os.environ.get('WEAVE_PROJECT', 'skyrl-grpo-enhanced'))
            logger.info("Weave logging initialized")
    
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
        
        # Create model config compatible with QwenPolicy - use same model as working script
        model_name = self.configs['model'].get('name', 'Qwen/Qwen2.5-0.5B-Instruct')  # Same as working Qwen3-0.6B script
        
        # Use same max_length as working script
        max_length = 2048  # Same as the working script
        
        model_config = {
            'model_name': model_name,
            'tokenizer_name': model_name,  # Use same name for tokenizer
            'torch_dtype': 'float32',  # Use float32 for MPS
            'device_map': None,  # Handle device mapping ourselves
            'trust_remote_code': True,
            'max_length': max_length,  # Reduced for MPS
            'stop_sequences': ["</tool_call>", "</think>", "<|im_end|>"],
            'memory_optimization': {
                'torch_dtype': 'float32',  # MPS compatible
                'load_in_4bit': False,
                'load_in_8bit': False,
                'device_map': None,
                'low_cpu_mem_usage': True,
                'trust_remote_code': True
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
        value_head_hidden_dim = self.configs['model'].get('value_head_hidden_dim', 1024)
        
        self.policy = QwenPolicyWithValue(
            model_config_path=model_config_path,
            training_config_path=training_config_path,
            use_lora=self.configs['model'].get('use_lora', True),
            device=str(self.device),
            load_in_4bit=False,  # Disable for MPS compatibility
            value_head_hidden_dim=value_head_hidden_dim
        )
        
        # Create reference policy (copy of initial policy)
        logger.info("Creating reference policy...")
        self.reference_policy = QwenPolicyWithValue(
            model_config_path=model_config_path,
            training_config_path=training_config_path,
            use_lora=self.configs['model'].get('use_lora', True),
            device=str(self.device),
            load_in_4bit=False,
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
    
    def setup_environment(self):
        """Setup MCP tool environment"""
        env_config = self.configs.get('environment', {})
        
        # For training, we might use a mock environment or real MCP servers
        # This depends on your setup
        logger.info("Setting up training environment...")
        
        # For now, we'll assume trajectories are pre-collected
        # In a full implementation, you'd set up MCPToolEnvironment here
        pass
    
    def collect_trajectories(self, tasks: List[Dict], num_rollouts: int = 1) -> List[Trajectory]:
        """Collect trajectories for training"""
        trajectories = []
        
        for task in tasks:
            for _ in range(num_rollouts):
                # In a real implementation, you'd use the environment to collect trajectories
                # For now, we'll create mock trajectories from the data
                traj = self._create_trajectory_from_task(task)
                if traj:
                    trajectories.append(traj)
        
        return trajectories
    
    def _create_trajectory_from_task(self, task: Dict) -> Optional[Trajectory]:
        """Create a trajectory from task data"""
        # Extract task components
        prompt = task.get('prompt', [])
        if not prompt:
            logger.warning("Task missing prompt")
            return None
        
        # Extract user prompt and assistant responses
        user_messages = [msg for msg in prompt if msg.get('role') == 'user']
        assistant_messages = [msg for msg in prompt if msg.get('role') == 'assistant']
        
        if not user_messages:
            logger.warning("No user messages found in task")
            return None
        
        # Create states and actions from the conversation
        states = []
        actions = []
        rewards = []
        
        # Apply sliding window to prevent memory blowup (Fix 2.2)
        truncated_prompt = self._truncate_conversation_history(prompt, max_tokens=1024)
        
        # Extract user and assistant messages from truncated prompt
        user_messages = [msg for msg in truncated_prompt if msg.get('role') == 'user']
        assistant_messages = [msg for msg in truncated_prompt if msg.get('role') == 'assistant']
        
        # Start with user message
        current_state = user_messages[:1] if user_messages else prompt[:1]
        states.append(current_state)
        
        # Add assistant responses as actions
        for i, assistant_msg in enumerate(assistant_messages):
            content = assistant_msg.get('content', '')
            if content.strip():
                actions.append(content)
                
                # Simple reward based on message length and tool usage
                reward = 0.5
                if '<tool_call>' in content:
                    reward += 0.3  # Bonus for tool use
                if '<think>' in content:
                    reward += 0.2  # Bonus for reasoning
                
                rewards.append(reward)
                
                # Add next state (conversation so far)
                if i < len(assistant_messages) - 1:
                    next_state = user_messages[:1] + assistant_messages[:i+1]
                    states.append(next_state)
        
        if not actions:
            logger.warning("No assistant actions found in task")
            return None
        
        # All episodes end
        dones = [False] * (len(actions) - 1) + [True]
        
        # Create trajectory
        task_id = task.get('data_source', 'unknown') + '_' + str(hash(str(task)))[:8]
        
        traj = Trajectory(
            task_id=task_id,
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones
        )
        
        # Add log probs (required for GRPO) - use standard computation for now (Fix 2.1 disabled temporarily)
        try:
            with torch.no_grad():
                # Use standard log prob computation (optimized version has NaN issues)
                log_probs = self.policy.compute_log_probs(states, actions)
                # Store as tensors, not floats, with NaN checking
                processed_log_probs = []
                for i, lp in enumerate(log_probs):
                    if hasattr(lp, 'clone'):
                        tensor_lp = lp.clone().detach()
                    else:
                        tensor_lp = torch.tensor(float(lp), device=self.device)
                    
                    # Check for NaN and replace with dummy value (on correct device)
                    if torch.isnan(tensor_lp) or torch.isinf(tensor_lp):
                        logger.warning(f"NaN/Inf in log_prob {i}, replacing with -10.0")
                        tensor_lp = torch.tensor(-10.0, device=self.device)
                    
                    processed_log_probs.append(tensor_lp)
                
                traj.old_log_probs = processed_log_probs
                
                # Clear MPS cache to prevent memory buildup
                if self.device.type == 'mps' and torch.backends.mps.is_available():
                    try:
                        torch.mps.empty_cache()
                    except:
                        pass  # Ignore MPS cache errors
        except Exception as e:
            logger.warning(f"Failed to compute log probs for trajectory: {e}")
            # Use dummy log probs as tensors
            traj.old_log_probs = [torch.tensor(-2.0) for _ in actions]
            
            # Clear MPS cache on error too
            if self.device.type == 'mps' and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except:
                    pass
        
        return traj
    
    def _truncate_conversation_history(self, messages: List[Dict], max_tokens: int = 1024) -> List[Dict]:
        """
        Maintain sliding window of conversation history (Fix 2.2).
        Prevents memory blowup from unbounded prompt growth.
        """
        if not messages:
            return messages
            
        # Always keep system message and current user message
        system_msgs = [m for m in messages if m.get('role') == 'system']
        user_msgs = [m for m in messages if m.get('role') == 'user']
        current_user = user_msgs[-1:] if user_msgs else []
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        def estimate_tokens(msg_list):
            return sum(len(msg.get('content', '') if isinstance(msg, dict) else str(msg)) for msg in msg_list) // 4
        
        total_tokens = estimate_tokens(system_msgs + current_user)
        
        if total_tokens >= max_tokens:
            # If even system + current user exceeds limit, truncate current user
            return system_msgs + current_user
        
        # Add previous turns until we hit limit
        other_messages = [m for m in messages if m not in system_msgs + current_user]
        truncated_messages = system_msgs.copy()
        
        # Add messages in reverse order (most recent first)
        for msg in reversed(other_messages):
            msg_tokens = estimate_tokens([msg])
            if total_tokens + msg_tokens < max_tokens:
                # Insert after system messages but before current user
                insert_pos = len(system_msgs)
                truncated_messages.insert(insert_pos, msg)
                total_tokens += msg_tokens
            else:
                break
        
        # Add current user message at the end
        truncated_messages.extend(current_user)
        
        # Log truncation if it happened
        if len(truncated_messages) < len(messages):
            logger.debug(f"Truncated conversation from {len(messages)} to {len(truncated_messages)} messages")
        
        return truncated_messages
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting Epoch {epoch}")
        logger.info(f"{'='*50}")
        
        # Shuffle training data
        train_tasks = self.train_data.copy()
        np.random.shuffle(train_tasks)
        
        # Training loop - use smaller batch size for MPS
        default_batch_size = 1 if self.device.type == 'mps' else 2
        batch_size = self.configs['training'].get('batch_size', default_batch_size)
        num_batches = len(train_tasks) // batch_size
        
        logger.info(f"Using batch size: {batch_size} for device: {self.device}")
        
        epoch_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'kl_divergence': [],
            'grad_norm': [],
            'rewards': []
        }
        
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch}")
        
        for batch_idx in progress_bar:
            # Get batch of tasks
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(train_tasks))
            batch_tasks = train_tasks[batch_start:batch_end]
            
            # Collect trajectories
            trajectories = self.collect_trajectories(batch_tasks, num_rollouts=1)
            
            if not trajectories:
                continue
            
            # Training step
            try:
                metrics = self.trainer.train_step(trajectories)
                
                # Update metrics
                for key in ['policy_loss', 'value_loss', 'kl_divergence', 'grad_norm']:
                    if key in metrics:
                        epoch_metrics[key].append(metrics[key])
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics.get('total_loss', 0):.4f}",
                    'kl': f"{metrics.get('kl_divergence', 0):.4f}",
                    'grad': f"{metrics.get('grad_norm', 0):.2f}"
                })
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'step': self.global_step,
                        'epoch': epoch,
                        **metrics
                    })
                
                self.global_step += 1
                
                # Clear MPS cache every N steps (Fix 6.2 - MPS optimizations)
                if self.device.type == 'mps' and torch.backends.mps.is_available():
                    # Clear cache every 10 steps as recommended
                    if self.global_step % 10 == 0:
                        try:
                            torch.mps.empty_cache()
                            logger.debug(f"Cleared MPS cache at step {self.global_step}")
                        except:
                            pass
                
            except Exception as e:
                logger.error(f"Error in training step: {e}")
                # Clear cache on error too
                if self.device.type == 'mps' and torch.backends.mps.is_available():
                    try:
                        torch.mps.empty_cache()
                    except:
                        pass
                continue
        
        # Compute epoch statistics
        epoch_stats = {}
        for key, values in epoch_metrics.items():
            if values:
                epoch_stats[f'epoch_{key}_mean'] = np.mean(values)
                epoch_stats[f'epoch_{key}_std'] = np.std(values)
        
        logger.info(f"Epoch {epoch} completed:")
        for key, value in epoch_stats.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return epoch_stats
    
    def evaluate(self, epoch: int):
        """Evaluate on validation set"""
        logger.info(f"\nEvaluating epoch {epoch}...")
        
        # Simple evaluation - compute average reward on validation tasks
        # Use smaller subset for MPS to avoid memory issues
        eval_subset_size = 5 if self.device.type == 'mps' else 20
        val_trajectories = self.collect_trajectories(
            self.valid_data[:eval_subset_size],  # Evaluate on subset
            num_rollouts=1
        )
        
        # Clear MPS cache after trajectory collection
        if self.device.type == 'mps' and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except:
                pass
        
        if not val_trajectories:
            logger.warning("No validation trajectories collected")
            return {}
        
        total_rewards = []
        for traj in val_trajectories:
            total_rewards.append(sum(traj.rewards))
        
        eval_metrics = {
            'eval_avg_reward': np.mean(total_rewards),
            'eval_std_reward': np.std(total_rewards),
            'eval_num_trajectories': len(val_trajectories)
        }
        
        logger.info(f"Validation results:")
        for key, value in eval_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Check if this is the best model
        if eval_metrics['eval_avg_reward'] > self.best_eval_score:
            self.best_eval_score = eval_metrics['eval_avg_reward']
            self.save_checkpoint('best_model', is_best=True)
        
        if self.use_wandb:
            wandb.log({
                'epoch': epoch,
                **eval_metrics
            })
        
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
            'configs': self.configs
        }
        
        with open(checkpoint_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def train(self):
        """Main training loop"""
        logger.info("\n" + "="*50)
        logger.info("Starting Enhanced GRPO Training")
        logger.info("="*50)
        
        # Setup
        self.setup_logging()
        self.load_data()
        self.setup_models()
        self.setup_trainer()
        self.setup_environment()
        
        # Training loop
        num_epochs = self.configs['training'].get('num_epochs', 3)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            epoch_stats = self.train_epoch(epoch)
            
            # Evaluate
            eval_stats = self.evaluate(epoch)
            
            # Save checkpoint
            if epoch % self.configs['training'].get('save_every', 1) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}')
        
        # Final save
        self.save_checkpoint('final_model')
        
        logger.info("\n" + "="*50)
        logger.info("Training completed!")
        logger.info(f"Best evaluation score: {self.best_eval_score:.4f}")
        logger.info("="*50)
        
        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Enhanced GRPO Training')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config_qwen3_0.6b.yaml',
        help='Path to training configuration file'
    )
    args = parser.parse_args()
    
    # Create trainer
    trainer = EnhancedGRPOTrainer(args.config)
    
    # Run training
    trainer.train()


if __name__ == '__main__':
    main()