"""
GRPO Trainer: Group Relative Policy Optimization for Multi-Turn Tool Use

This module implements a complete GRPO trainer for fine-tuning Qwen2.5-1.5B-Instruct
on multi-turn tool use tasks. GRPO optimizes policies by comparing multiple rollouts
per task to compute relative rewards and advantages.
"""

import copy
import logging
import time
import traceback
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

# Disable BitsAndBytes entirely due to MPS compatibility issues
HAS_BITSANDBYTES = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trajectory:
    """
    Container for a single trajectory (episode) with all necessary data for GRPO.
    
    Attributes:
        task_id (str): Unique identifier for the task
        states (List): List of conversation states (list of message dicts)
        actions (List[str]): List of generated actions
        rewards (List[float]): List of rewards for each step
        dones (List[bool]): List of done flags for each step
        log_probs (Optional[torch.Tensor]): Log probabilities under current policy
        values (Optional[torch.Tensor]): Value function estimates (if available)
        advantages (Optional[torch.Tensor]): Computed advantages (filled by trainer)
        total_reward (float): Sum of all rewards in trajectory
        length (int): Number of steps in trajectory
    """
    
    def __init__(
        self,
        task_id: str,
        states: List[List[Dict[str, str]]],
        actions: List[str],
        rewards: List[float],
        dones: List[bool],
        log_probs: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
    ):
        self.task_id = task_id
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.log_probs = log_probs
        self.values = values
        self.advantages: Optional[torch.Tensor] = None
        
        # Computed properties
        self.total_reward = sum(rewards)
        self.length = len(states)
        
        # Validation
        if not (len(states) == len(actions) == len(rewards) == len(dones)):
            raise ValueError("All trajectory components must have the same length")
    
    def to_device(self, device: torch.device) -> 'Trajectory':
        """Move tensor attributes to specified device."""
        new_traj = copy.deepcopy(self)
        
        # Handle log_probs - might be a list or tensor
        if self.log_probs is not None:
            if isinstance(self.log_probs, list):
                # Convert list to tensor if needed
                if self.log_probs and isinstance(self.log_probs[0], torch.Tensor):
                    new_traj.log_probs = torch.stack(self.log_probs).to(device)
                else:
                    # Keep as-is if not tensors
                    new_traj.log_probs = self.log_probs
            elif isinstance(self.log_probs, torch.Tensor):
                new_traj.log_probs = self.log_probs.to(device)
        
        # Handle values
        if self.values is not None:
            if isinstance(self.values, torch.Tensor):
                new_traj.values = self.values.to(device)
        
        # Handle advantages
        if self.advantages is not None:
            if isinstance(self.advantages, torch.Tensor):
                new_traj.advantages = self.advantages.to(device)
        
        # Handle any additional tensor attributes that might exist
        if hasattr(self, 'forced_mask') and self.forced_mask is not None:
            if isinstance(self.forced_mask, torch.Tensor):
                new_traj.forced_mask = self.forced_mask.to(device)
        
        return new_traj


class GRPOTrainer:
    """
    Group Relative Policy Optimization Trainer for Multi-Turn Tool Use.
    
    This trainer implements GRPO, which optimizes policies by comparing multiple
    rollouts per task to compute relative rewards. It uses Generalized Advantage
    Estimation (GAE) and policy gradient methods with KL divergence penalties.
    
    Key Features:
    - Group-based relative reward computation
    - GAE advantage estimation with configurable gamma and lambda
    - Policy gradient optimization with clipping
    - KL divergence penalty for stability
    - Reference policy updates
    - Support for both LoRA and full fine-tuning
    - Comprehensive metrics tracking
    
    Args:
        policy: The policy to be trained (QwenPolicy instance)
        reference_policy: Reference policy for KL penalty computation
        grpo_config: GRPO algorithm configuration dictionary
        training_config: Training configuration dictionary
        device: Device to run training on
        environment_factory: Factory function to create environments (optional)
    """
    
    def __init__(
        self,
        policy,  # QwenPolicy instance
        reference_policy,  # QwenPolicy instance for reference
        grpo_config: Dict[str, Any],
        training_config: Dict[str, Any],
        device: torch.device = torch.device("cuda"),
        environment_factory: Optional[Callable] = None,
    ):
        """Initialize the GRPO trainer with policies and configurations."""
        
        self.policy = policy
        self.reference_policy = reference_policy
        self.grpo_config = grpo_config
        self.training_config = training_config
        self.device = device
        self.environment_factory = environment_factory
        
        # Training state
        self.step_count = 0
        self.epoch_count = 0
        self.total_trajectories_seen = 0
        self.best_eval_score = float('-inf')
        
        # GRPO hyperparameters
        self.gamma = grpo_config.get("gamma", 0.99)
        self.gae_lambda = grpo_config.get("gae_lambda", 0.95)
        self.clip_ratio = grpo_config.get("clip_ratio", 0.2)
        self.kl_penalty_coef = grpo_config.get("kl_penalty_coef", 0.1)
        self.target_kl = grpo_config.get("target_kl_divergence", 0.01)
        self.entropy_coef = grpo_config.get("entropy_coef", 0.01)
        self.value_loss_coef = grpo_config.get("value_loss_coef", 0.5)
        self.normalize_advantages = grpo_config.get("normalize_advantages", True)
        self.advantage_epsilon = float(grpo_config.get("advantage_epsilon", 1e-8))
        
        # Reference policy update settings
        self.ref_update_freq = grpo_config.get("ref_policy_update_frequency", 10000)
        self.ref_ema_alpha = grpo_config.get("ref_policy_ema_alpha", 0.99)
        
        # Group settings
        self.group_size = grpo_config.get("group_size", 4)
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Initialize metrics tracking
        self.metrics_history = []
        self.current_metrics = {}
        
        # Adaptive KL penalty
        self.adaptive_kl = grpo_config.get("adaptive_kl_penalty", True)
        self.kl_warmup_steps = grpo_config.get("kl_penalty_warmup_steps", 500)
        
        logger.info(f"GRPOTrainer initialized with {self.policy.get_trainable_parameters():,} trainable parameters")
        logger.info(f"GRPO config: gamma={self.gamma}, lambda={self.gae_lambda}, "
                   f"clip={self.clip_ratio}, kl_coef={self.kl_penalty_coef}")
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer based on training configuration."""
        
        # Get learning rate based on training mode
        if self.policy.use_lora:
            lr = self.training_config.get(
                "lora_learning_rate",
                self.training_config.get("learning_rate", 5e-6)
            )
        else:
            lr = self.training_config.get(
                "full_finetune_learning_rate",
                self.training_config.get("learning_rate", 5e-6)
            )
        # Coerce to float if YAML or env provided as string
        try:
            lr = float(lr)
        except Exception:
            logger.warning(f"Learning rate '{lr}' is not a float; falling back to 5e-6")
            lr = 5e-6
        
        # Get optimizer parameters
        def _as_float(v, default):
            try:
                return float(v)
            except Exception:
                return default
        weight_decay = _as_float(self.training_config.get("weight_decay", 0.01), 0.01)
        adam_beta1 = _as_float(self.training_config.get("adam_beta1", 0.9), 0.9)
        adam_beta2 = _as_float(self.training_config.get("adam_beta2", 0.95), 0.95)
        adam_epsilon = _as_float(self.training_config.get("adam_epsilon", 1e-5), 1e-5)
        
        # Debug: Check all parameters and their gradient status
        # Include both model and value head parameters if available
        all_params = list(self.policy.model.named_parameters())
        trainable_params = [p for p in self.policy.model.parameters() if p.requires_grad]
        
        # Add value head parameters if this is a policy with value head
        if hasattr(self.policy, 'value_head'):
            value_params = list(self.policy.value_head.named_parameters())
            all_params.extend([('value_head.' + name, p) for name, p in value_params])
            trainable_params.extend([p for p in self.policy.value_head.parameters() if p.requires_grad])
        
        trainable_names = [name for name, p in all_params if p.requires_grad]
        
        logger.info(f"Model has {len(all_params)} total parameters")
        logger.info(f"Found {len(trainable_params)} trainable parameters")
        
        if not trainable_params:
            logger.error("No trainable parameters found!")
            logger.error("Parameter gradient status:")
            for name, param in all_params[:20]:  # Show first 20 for debugging
                logger.error(f"  {name}: requires_grad={param.requires_grad}")
            
            # Try to re-enable training mode
            logger.warning("Attempting to re-enable training mode...")
            self.policy.enable_training_mode()
            
            # Check again  
            trainable_params = [p for p in self.policy.model.parameters() if p.requires_grad]
            if hasattr(self.policy, 'value_head'):
                trainable_params.extend([p for p in self.policy.value_head.parameters() if p.requires_grad])
            trainable_names = [name for name, p in all_params if p.requires_grad]
            
            if not trainable_params:
                logger.error("Still no trainable parameters after re-enabling training mode!")
                logger.error("This suggests gradients are being disabled somewhere else")
                raise ValueError("No trainable parameters found even after re-enabling training mode!")
            else:
                logger.info(f"Successfully found {len(trainable_params)} trainable parameters after re-enabling")
                logger.info(f"Trainable parameter examples: {trainable_names[:5]}")
        
        # Always use standard AdamW optimizer (BitsAndBytes disabled for MPS compatibility)
        self.optimizer = AdamW(
            trainable_params,
            lr=lr,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
            weight_decay=weight_decay,
        )
        logger.info(f"Using AdamW optimizer with lr={lr} for {len(trainable_params)} parameters")
    
    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        
        scheduler_type = self.training_config.get("lr_scheduler_type", "cosine")
        try:
            warmup_steps = int(self.training_config.get("warmup_steps", 100))
        except Exception:
            warmup_steps = 100
        try:
            max_steps = int(self.training_config.get("max_steps", 10000))
        except Exception:
            max_steps = 10000
        
        if scheduler_type == "cosine":
            # Warmup + Cosine annealing
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max_steps - warmup_steps,
                eta_min=self.optimizer.param_groups[0]['lr'] * 0.1
            )
            
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        else:
            # Linear warmup only
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
        
        logger.info(f"Setup {scheduler_type} scheduler with {warmup_steps} warmup steps")
    
    def train_step(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """
        Perform a single GRPO training step.
        
        This method implements the core GRPO algorithm:
        1. Group trajectories by task for relative reward computation
        2. Compute advantages using GAEon
        3. Update policy with GRPO objective (policy loss + KL penalty)
        4. Update reference policy if needed
        5. Return comprehensive metrics
        
        Args:
            trajectories: List of Trajectory objects for this training step
            
        Returns:
            Dictionary containing training metrics
        """
        
        logger.info(f"Starting train_step with {len(trajectories)} trajectories")
        
        if not trajectories:
            logger.warning("No trajectories provided for training step")
            return {}
        
        # Debug trajectory data
        for i, traj in enumerate(trajectories[:2]):  # Check first 2 trajectories
            logger.info(f"Trajectory {i}: task_id={traj.task_id}, length={traj.length}, "
                       f"rewards={traj.rewards}, states_count={len(traj.states)}, "
                       f"actions_count={len(traj.actions)}")
        
        start_time = time.time()
        
        try:
            # Move trajectories to the policy's device
            policy_device = getattr(self.policy.model, "device", self.device)
            logger.info(f"Moving {len(trajectories)} trajectories to device {policy_device}")
            trajectories = [traj.to_device(policy_device) for traj in trajectories]
            logger.info("✅ Trajectories moved to device successfully")
            
            # Group trajectories by task for relative reward computation
            logger.info("Grouping trajectories by task...")
            grouped_trajectories = self._group_trajectories_by_task(trajectories)
            logger.info(f"✅ Grouped into {len(grouped_trajectories)} task groups")
            
            # Compute relative rewards within each group
            logger.info("Computing relative rewards...")
            self._compute_relative_rewards(grouped_trajectories)
            logger.info("✅ Relative rewards computed")
            
            # Compute advantages using GAE
            logger.info("Computing advantages using GAE...")
            for group_idx, group in enumerate(grouped_trajectories.values()):
                logger.debug(f"Processing group {group_idx + 1}/{len(grouped_trajectories)}")
                for traj in group:
                    traj.advantages = self._compute_advantages(traj)
            logger.info("✅ Advantages computed")
            
            # Flatten trajectories back to list
            logger.info("Flattening trajectories...")
            all_trajectories = []
            for group in grouped_trajectories.values():
                all_trajectories.extend(group)
            logger.info(f"✅ Flattened to {len(all_trajectories)} trajectories")
            
            # Compute current policy log probabilities
            logger.info("Computing current policy log probabilities...")
            self._compute_current_log_probs(all_trajectories)
            logger.info("✅ Log probabilities computed")
            
            # Update policy
            logger.info("Updating policy with GRPO...")
            
            # Debug: Check model training state before policy update
            is_training = self.policy.model.training
            trainable_count = sum(1 for p in self.policy.model.parameters() if p.requires_grad)
            logger.info(f"Model training state: {is_training}, trainable params: {trainable_count}")

            # Ensure model and value head are in training mode
            if not is_training:
                logger.warning("Model not in training mode! Re-enabling...")
            self.policy.enable_training_mode()
            if hasattr(self.policy, 'value_head'):
                self.policy.value_head.train()

            policy_metrics = self._update_policy(all_trajectories)
            logger.info("✅ Policy updated successfully")
            
        except Exception as e:
            logger.error(f"❌ Error in train_step: {type(e).__name__}: {e}")
            logger.error(f"Train step error traceback:\n{traceback.format_exc()}")
            raise
        
        # Update reference policy if needed
        if self.step_count % self.ref_update_freq == 0:
            self._update_reference_policy()
        
        # Update step count
        self.step_count += 1
        self.total_trajectories_seen += len(trajectories)
        
        # Learning rate step
        self.scheduler.step()
        
        # Compile metrics
        step_time = time.time() - start_time
        metrics = {
            "step": self.step_count,
            "epoch": self.epoch_count,
            "trajectories_processed": len(trajectories),
            "total_trajectories": self.total_trajectories_seen,
            "step_time": step_time,
            "trajectories_per_second": len(trajectories) / step_time,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            **policy_metrics,
        }
        
        # Add trajectory statistics
        traj_stats = self._compute_trajectory_statistics(trajectories)
        metrics.update(traj_stats)
        
        # Store metrics
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
        
        # Log important metrics
        if self.step_count % 10 == 0:
            logger.info(f"Step {self.step_count}: "
                       f"policy_loss={metrics.get('policy_loss', 0):.4f}, "
                       f"kl_div={metrics.get('kl_divergence', 0):.4f}, "
                       f"avg_reward={metrics.get('avg_total_reward', 0):.3f}")
        
        return metrics
    
    def _group_trajectories_by_task(self, trajectories: List[Trajectory]) -> Dict[str, List[Trajectory]]:
        """Group trajectories by task_id for relative reward computation."""
        
        grouped = defaultdict(list)
        for traj in trajectories:
            grouped[traj.task_id].append(traj)
        
        # Ensure each group has at least one trajectory
        grouped_dict = dict(grouped)
        
        # Log group statistics
        group_sizes = [len(group) for group in grouped_dict.values()]
        logger.debug(f"Grouped {len(trajectories)} trajectories into {len(grouped_dict)} tasks. "
                    f"Group sizes: min={min(group_sizes)}, max={max(group_sizes)}, "
                    f"avg={np.mean(group_sizes):.1f}")
        
        return grouped_dict
    
    def _compute_relative_rewards(self, grouped_trajectories: Dict[str, List[Trajectory]]) -> None:
        """
        Compute relative rewards within each group.
        
        For each group of trajectories on the same task, we normalize rewards
        relative to the group performance to reduce variance and improve training signal.
        """
        
        for task_id, trajectories in grouped_trajectories.items():
            if len(trajectories) < 2:
                # Single trajectory - no relative computation needed
                continue
            
            # Get total rewards for ranking
            total_rewards = [traj.total_reward for traj in trajectories]
            
            # Compute relative rewards based on ranking within group
            if self.grpo_config.get("reward_normalization", "group") == "group":
                # Normalize within group (zero mean, unit variance)
                mean_reward = np.mean(total_rewards)
                std_reward = np.std(total_rewards) + 1e-8  # Avoid division by zero
                
                for traj, total_reward in zip(trajectories, total_rewards):
                    # Apply normalization to all step rewards proportionally
                    normalization_factor = (total_reward - mean_reward) / std_reward
                    if traj.total_reward != 0:
                        reward_scale = normalization_factor / traj.total_reward
                        traj.rewards = [r * reward_scale for r in traj.rewards]
                    traj.total_reward = sum(traj.rewards)
            
            elif self.grpo_config.get("reward_normalization", "group") == "rank":
                # Rank-based rewards
                sorted_indices = np.argsort(total_rewards)
                ranks = np.empty_like(sorted_indices)
                ranks[sorted_indices] = np.arange(len(total_rewards))
                
                # Convert ranks to rewards (higher rank = higher reward)
                rank_rewards = (ranks - np.mean(ranks)) / (np.std(ranks) + 1e-8)
                
                for traj, rank_reward in zip(trajectories, rank_rewards):
                    # Scale step rewards by rank reward
                    if traj.total_reward != 0:
                        reward_scale = rank_reward / traj.total_reward
                        traj.rewards = [r * reward_scale for r in traj.rewards]
                    else:
                        # Assign small rewards for zero-reward trajectories
                        traj.rewards = [rank_reward / len(traj.rewards)] * len(traj.rewards)
                    traj.total_reward = sum(traj.rewards)
    
    def _compute_advantages(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        GAE computes advantages that balance bias and variance in policy gradient
        estimation. It uses exponentially weighted averages of n-step advantages.
        
        Args:
            trajectory: Trajectory to compute advantages for
            
        Returns:
            Tensor of advantages with shape [trajectory_length]
        """
        
        rewards = torch.tensor(trajectory.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(trajectory.dones, dtype=torch.bool, device=self.device)
        
        # If we have value estimates, use them; otherwise assume zero values
        if trajectory.values is not None:
            values = trajectory.values
        else:
            values = torch.zeros_like(rewards)
        
        # Compute GAE advantages
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        
        # Work backwards through the trajectory
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # Last step
                next_value = 0.0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]
            
            # Compute TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t].float()) - values[t]
            
            # Compute GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t].float()) * gae
            advantages[t] = gae
        
        # Normalize advantages if specified
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.advantage_epsilon)
        
        return advantages
    
    def _compute_current_log_probs(self, trajectories: List[Trajectory]) -> None:
        """Compute log probabilities under current policy for all trajectories."""
        
        for traj in trajectories:
            # Compute log probabilities for this trajectory
            log_probs = self.policy.compute_log_probs(traj.states, traj.actions)
            traj.log_probs = log_probs
    
    def _update_policy(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """
        Update policy using GRPO objective.
        
        The GRPO objective combines:
        1. Clipped policy gradient loss
        2. KL divergence penalty from reference policy
        3. Optional entropy regularization
        
        Args:
            trajectories: List of trajectories with computed advantages
            
        Returns:
            Dictionary of policy update metrics
        """
        
        logger.info(f"_update_policy called with {len(trajectories)} trajectories")
        
        if not trajectories:
            logger.warning("No trajectories provided to _update_policy")
            return {}
        
        try:
            # Collect all data
            logger.info("Collecting trajectory data...")
            all_states = []
            all_actions = []
            all_advantages = []
            all_old_log_probs = []
            
            for i, traj in enumerate(trajectories):
                logger.debug(f"Processing trajectory {i}: {len(traj.states)} states, {len(traj.actions)} actions")
                all_states.extend(traj.states)
                all_actions.extend(traj.actions)
                if traj.advantages is not None:
                    all_advantages.extend(traj.advantages.tolist())
                else:
                    all_advantages.extend([0.0] * len(traj.actions))
                if traj.log_probs is not None:
                    all_old_log_probs.extend(traj.log_probs.tolist())
                else:
                    all_old_log_probs.extend([0.0] * len(traj.actions))
            
            logger.info(f"Collected data: {len(all_states)} states, {len(all_actions)} actions, {len(all_advantages)} advantages")
            
            if not all_states:
                logger.warning("No states collected from trajectories")
                return {}
            
            # Convert to tensors
            logger.info("Converting data to tensors...")
            advantages = torch.tensor(all_advantages, dtype=torch.float32, device=self.device)
            old_log_probs = torch.tensor(all_old_log_probs, dtype=torch.float32, device=self.device)
            logger.info(f"Tensors created: advantages shape={advantages.shape}, old_log_probs shape={old_log_probs.shape}")
            
            # Compute current policy log probabilities
            logger.info("Computing current policy log probabilities...")
            logger.info(f"Sample states (first 2): {all_states[:2]}")
            logger.info(f"Sample actions (first 2): {all_actions[:2]}")
            try:
                current_log_probs = self.policy.compute_log_probs(all_states, all_actions)
                logger.info(f"Current log probs computed: shape={current_log_probs.shape}")
            except Exception as e:
                logger.error(f"Error computing current log probs: {type(e).__name__}: {e}")
                logger.error(f"States length: {len(all_states)}, Actions length: {len(all_actions)}")
                logger.error(f"First state structure: {type(all_states[0]) if all_states else 'empty'}")
                logger.error(f"First action: {repr(all_actions[0]) if all_actions else 'empty'}")
                raise
            
            # Compute reference policy log probabilities for KL penalty
            logger.info("Computing reference policy log probabilities...")
            with torch.no_grad():
                ref_log_probs = self.reference_policy.compute_log_probs(all_states, all_actions)
            logger.info(f"Reference log probs computed: shape={ref_log_probs.shape}")
            
        except Exception as e:
            logger.error(f"Error in data collection phase: {type(e).__name__}: {e}")
            logger.error(f"Data collection traceback:\n{traceback.format_exc()}")
            raise
        
        # Compute policy ratios
        log_ratios = current_log_probs - old_log_probs
        ratios = torch.exp(log_ratios)
        
        # Clamp ratios for numerical stability
        ratios = torch.clamp(ratios, 0.1, 10.0)
        
        # Compute clipped policy loss (PPO-style)
        policy_loss_unclipped = -advantages * ratios
        policy_loss_clipped = -advantages * torch.clamp(
            ratios, 1 - self.clip_ratio, 1 + self.clip_ratio
        )
        policy_loss = torch.mean(torch.max(policy_loss_unclipped, policy_loss_clipped))
        
        # Compute KL divergence penalty
        # Correct KL divergence computation: KL(p||q) = sum(p * log(p/q))
        # For log probabilities: KL = exp(current_log_probs) * (current_log_probs - ref_log_probs)
        # However, for stability, we use the approximation: KL ≈ mean((current - ref)^2)
        log_ratio = current_log_probs - ref_log_probs
        kl_divergence = torch.mean(log_ratio ** 2) * 0.5  # Quadratic approximation of KL
        
        # Adaptive KL penalty
        if self.adaptive_kl and self.step_count < self.kl_warmup_steps:
            kl_coef = self.kl_penalty_coef * (self.step_count / self.kl_warmup_steps)
        else:
            kl_coef = self.kl_penalty_coef
        
        kl_penalty = kl_coef * kl_divergence
        
        # Total loss
        total_loss = policy_loss + kl_penalty
        
        # Entropy regularization (if specified)
        entropy_loss = 0.0
        if self.entropy_coef > 0:
            # Approximate entropy from log probabilities
            entropy = -current_log_probs.mean()
            entropy_loss = -self.entropy_coef * entropy
            total_loss += entropy_loss
        
        # Ensure total_loss requires gradients for backprop
        if not total_loss.requires_grad:
            logger.error("total_loss does not require gradients! This suggests model is not in training mode.")
            raise RuntimeError("Loss tensor does not require gradients - check model training mode")
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping and tracking
        if self.grpo_config.get("gradient_clipping_enabled", True):
            clip_value = self.grpo_config.get("gradient_clipping_value", 1.0)
            self.grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.model.parameters(),
                max_norm=clip_value
            ).item()
        else:
            # Calculate gradient norm even if not clipping
            total_norm = 0
            for p in self.policy.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            self.grad_norm = total_norm ** 0.5
        
        # Optimizer step
        self.optimizer.step()
        
        # Compile metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_divergence": kl_divergence.item(),
            "kl_penalty": kl_penalty.item(),
            "total_loss": total_loss.item(),
            "avg_ratio": ratios.mean().item(),
            "max_ratio": ratios.max().item(),
            "min_ratio": ratios.min().item(),
            "avg_advantage": advantages.mean().item(),
            "std_advantage": advantages.std().item(),
            "kl_coef": kl_coef,
            "grad_norm": self.grad_norm if hasattr(self, 'grad_norm') else 0,
        }
        
        if entropy_loss != 0:
            metrics["entropy_loss"] = entropy_loss.item()
        
        # Check for training instabilities
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning("NaN or Inf detected in loss! Skipping this update.")
            return {"error": "nan_loss"}
        
        if kl_divergence.item() > self.target_kl * 10:
            logger.warning(f"High KL divergence detected: {kl_divergence.item():.4f}")
        
        return metrics
    
    def _update_reference_policy(self) -> None:
        """Update reference policy using exponential moving average."""
        
        logger.info(f"Updating reference policy at step {self.step_count}")
        
        # EMA update: ref_param = alpha * ref_param + (1 - alpha) * current_param
        with torch.no_grad():
            for ref_param, current_param in zip(
                self.reference_policy.model.parameters(),
                self.policy.model.parameters()
            ):
                ref_param.data = (
                    self.ref_ema_alpha * ref_param.data +
                    (1 - self.ref_ema_alpha) * current_param.data
                )
    
    def _compute_trajectory_statistics(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """Compute statistics about the trajectories."""
        
        if not trajectories:
            return {}
        
        total_rewards = [traj.total_reward for traj in trajectories]
        lengths = [traj.length for traj in trajectories]
        
        stats = {
            "num_trajectories": len(trajectories),
            "avg_total_reward": np.mean(total_rewards),
            "std_total_reward": np.std(total_rewards),
            "min_total_reward": np.min(total_rewards),
            "max_total_reward": np.max(total_rewards),
            "avg_trajectory_length": np.mean(lengths),
            "std_trajectory_length": np.std(lengths),
            "min_trajectory_length": np.min(lengths),
            "max_trajectory_length": np.max(lengths),
        }
        
        return stats
    
    def save_checkpoint(self, checkpoint_path: str, include_optimizer: bool = True) -> None:
        """Save training checkpoint."""
        
        checkpoint = {
            "step_count": self.step_count,
            "epoch_count": self.epoch_count,
            "total_trajectories_seen": self.total_trajectories_seen,
            "best_eval_score": self.best_eval_score,
            "grpo_config": self.grpo_config,
            "training_config": self.training_config,
            "metrics_history": self.metrics_history,
        }
        
        if include_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save policy
        self.policy.save_model(f"{checkpoint_path}/policy")
        self.reference_policy.save_model(f"{checkpoint_path}/reference_policy")
        
        # Save checkpoint metadata
        torch.save(checkpoint, f"{checkpoint_path}/trainer_checkpoint.pt")
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> None:
        """Load training checkpoint."""
        
        # Load checkpoint metadata
        checkpoint = torch.load(f"{checkpoint_path}/trainer_checkpoint.pt", map_location=self.device)
        
        self.step_count = checkpoint["step_count"]
        self.epoch_count = checkpoint["epoch_count"]
        self.total_trajectories_seen = checkpoint["total_trajectories_seen"]
        self.best_eval_score = checkpoint["best_eval_score"]
        self.metrics_history = checkpoint.get("metrics_history", [])
        
        if load_optimizer:
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load policies
        self.policy.load_checkpoint(f"{checkpoint_path}/policy")
        self.reference_policy.load_checkpoint(f"{checkpoint_path}/reference_policy")
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 steps
        
        summary = {
            "total_steps": self.step_count,
            "total_trajectories": self.total_trajectories_seen,
            "current_lr": self.optimizer.param_groups[0]['lr'],
            "recent_avg_reward": np.mean([m.get("avg_total_reward", 0) for m in recent_metrics]),
            "recent_policy_loss": np.mean([m.get("policy_loss", 0) for m in recent_metrics]),
            "recent_kl_div": np.mean([m.get("kl_divergence", 0) for m in recent_metrics]),
            "best_eval_score": self.best_eval_score,
        }
        
        return summary
    
    def __repr__(self) -> str:
        """String representation of the trainer."""
        return (f"GRPOTrainer(step={self.step_count}, "
                f"trainable_params={self.policy.get_trainable_parameters():,}, "
                f"device={self.device})")


# Utility functions for trainer setup and management

def create_grpo_trainer(
    policy,
    reference_policy,
    grpo_config_path: str,
    training_config_path: str,
    device: torch.device = torch.device("cuda"),
) -> GRPOTrainer:
    """
    Factory function to create GRPOTrainer from configuration files.
    
    Args:
        policy: Policy to be trained
        reference_policy: Reference policy for KL penalty
        grpo_config_path: Path to GRPO configuration YAML
        training_config_path: Path to training configuration YAML
        device: Device for training
        
    Returns:
        Initialized GRPOTrainer instance
    """
    
    import yaml
    
    # Load configurations
    with open(grpo_config_path, 'r') as f:
        grpo_config = yaml.safe_load(f)
    
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    
    return GRPOTrainer(
        policy=policy,
        reference_policy=reference_policy,
        grpo_config=grpo_config,
        training_config=training_config,
        device=device,
    )


if __name__ == "__main__":
    # Example usage and testing
    print("GRPO Trainer module loaded successfully!")
    
    # Test trajectory creation
    test_traj = Trajectory(
        task_id="test_001",
        states=[[{"role": "user", "content": "Hello"}]],
        actions=["Hi there!"],
        rewards=[1.0],
        dones=[True],
    )
    print(f"Test trajectory: {test_traj.length} steps, reward={test_traj.total_reward}")