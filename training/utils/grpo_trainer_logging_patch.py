#!/usr/bin/env python3
"""
GRPO Trainer Logging Enhancement Patch
=====================================

This module patches the existing GRPO trainer to include comprehensive
training metrics logging including policy losses, value function metrics,
advantage computations, reward breakdowns, and more.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch

from .enhanced_wandb_logger import GRPOTrainingMetricsLogger

logger = logging.getLogger(__name__)


class GRPOTrainerLoggingEnhancement:
    """
    Enhancement class that adds comprehensive logging to GRPO trainer
    """
    
    def __init__(
        self,
        trainer,
        project_name: str = "skyrl-grpo-detailed",
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ):
        self.trainer = trainer
        self.metrics_logger = GRPOTrainingMetricsLogger(
            project_name=project_name,
            config=config,
            enabled=enabled
        ) if enabled else None
        
        self.step_start_time = time.time()
        self.episode_metrics_buffer = []
        
        # Patch the trainer methods
        self._patch_trainer_methods()
        
        logger.info("✅ GRPO trainer enhanced with comprehensive metrics logging")
    
    def _patch_trainer_methods(self):
        """Patch trainer methods to include enhanced logging"""
        
        # Store original methods
        self.trainer._original_update_policy = self.trainer._update_policy
        self.trainer._original_compute_advantages = getattr(self.trainer, '_compute_advantages', None)
        
        # Replace with enhanced versions
        self.trainer._update_policy = self._enhanced_update_policy
        
        if hasattr(self.trainer, '_compute_advantages'):
            self.trainer._compute_advantages = self._enhanced_compute_advantages
    
    def _enhanced_update_policy(self, trajectories):
        """Enhanced policy update with comprehensive logging"""
        if not self.metrics_logger:
            return self.trainer._original_update_policy(trajectories)
        
        step_start = time.time()
        
        # Call original method
        result = self.trainer._original_update_policy(trajectories)
        
        # Extract and log detailed metrics
        try:
            self._log_detailed_training_metrics(trajectories, result, step_start)
        except Exception as e:
            logger.warning(f"Failed to log detailed training metrics: {e}")
        
        return result
    
    def _enhanced_compute_advantages(self, trajectories):
        """Enhanced advantage computation with logging"""
        if not self.trainer._original_compute_advantages:
            return trajectories
        
        # Call original method
        result = self.trainer._original_compute_advantages(trajectories)
        
        # Log advantage metrics
        try:
            self._log_advantage_metrics(result)
        except Exception as e:
            logger.warning(f"Failed to log advantage metrics: {e}")
        
        return result
    
    def _log_detailed_training_metrics(
        self,
        trajectories,
        training_result: Dict[str, float],
        step_start_time: float
    ):
        """Log comprehensive training metrics"""
        
        step = getattr(self.trainer, 'global_step', 0)
        epoch = getattr(self.trainer, 'current_epoch', 0)
        
        # Basic training step metrics
        step_time = time.time() - step_start_time
        
        # Extract policy metrics from training result
        policy_loss = training_result.get('policy_loss', 0.0)
        value_loss = training_result.get('value_loss', 0.0)
        entropy = training_result.get('entropy', 0.0)
        kl_divergence = training_result.get('kl_divergence', 0.0)
        total_loss = training_result.get('total_loss', policy_loss + value_loss)
        
        # Log training step summary
        self.metrics_logger.log_training_step_summary(
            step=step,
            epoch=epoch,
            total_loss=total_loss,
            learning_rate=getattr(self.trainer, 'learning_rate', 5e-6),
            batch_size=len(trajectories),
            time_per_step=step_time
        )
        
        # Extract detailed metrics from trajectories and model
        self._extract_and_log_policy_metrics(trajectories, training_result, step)
        self._extract_and_log_value_metrics(trajectories, training_result, step)
        self._extract_and_log_reward_metrics(trajectories, step)
        
        # Log gradient metrics if available
        if hasattr(self.trainer, 'policy') and hasattr(self.trainer.policy, 'model'):
            total_grad_norm = training_result.get('grad_norm', 0.0)
            max_grad_norm = getattr(self.trainer, 'max_grad_norm', 1.0)
            
            self.metrics_logger.log_gradient_metrics(
                step=step,
                model=self.trainer.policy.model,
                total_grad_norm=total_grad_norm,
                max_grad_norm=max_grad_norm,
                grad_clip_fraction=training_result.get('clip_fraction', None)
            )
    
    def _extract_and_log_policy_metrics(self, trajectories, training_result, step):
        """Extract and log detailed policy metrics"""
        
        # Policy-specific metrics
        policy_loss = training_result.get('policy_loss', 0.0)
        entropy = training_result.get('entropy', 0.0)
        kl_divergence = training_result.get('kl_divergence', 0.0)
        clip_fraction = training_result.get('clip_fraction', 0.0)
        
        # Calculate ratio statistics from trajectories
        ratios = []
        for traj in trajectories:
            if hasattr(traj, 'ratios') and traj.ratios is not None:
                if torch.is_tensor(traj.ratios):
                    ratios.extend(traj.ratios.detach().cpu().numpy().tolist())
                else:
                    ratios.extend(traj.ratios)
        
        ratio_mean = np.mean(ratios) if ratios else 1.0
        ratio_std = np.std(ratios) if ratios else 0.0
        
        # Old approximate KL (from importance sampling)
        old_approx_kl = None
        if ratios:
            # Approximate KL from ratios: E[r - 1 - log(r)]
            ratios_np = np.array(ratios)
            old_approx_kl = np.mean(ratios_np - 1.0 - np.log(ratios_np + 1e-8))
        
        self.metrics_logger.log_policy_metrics(
            step=step,
            policy_loss=policy_loss,
            entropy=entropy,
            kl_divergence=kl_divergence,
            clip_fraction=clip_fraction,
            ratio_mean=ratio_mean,
            ratio_std=ratio_std,
            old_approx_kl=old_approx_kl
        )
    
    def _extract_and_log_value_metrics(self, trajectories, training_result, step):
        """Extract and log value function metrics"""
        
        value_loss = training_result.get('value_loss', 0.0)
        
        # Collect value predictions and returns from trajectories
        all_values = []
        all_returns = []
        all_old_values = []
        
        for traj in trajectories:
            if hasattr(traj, 'values') and traj.values is not None:
                values = traj.values
                if torch.is_tensor(values):
                    all_values.extend(values.detach().cpu().numpy())
                else:
                    all_values.extend(values)
            
            if hasattr(traj, 'returns') and traj.returns is not None:
                returns = traj.returns
                if torch.is_tensor(returns):
                    all_returns.extend(returns.detach().cpu().numpy())
                else:
                    all_returns.extend(returns)
            
            # Old values for clipping analysis
            if hasattr(traj, 'old_values') and traj.old_values is not None:
                old_values = traj.old_values
                if torch.is_tensor(old_values):
                    all_old_values.extend(old_values.detach().cpu().numpy())
                else:
                    all_old_values.extend(old_values)
        
        if all_values and all_returns:
            values_tensor = torch.tensor(all_values)
            returns_tensor = torch.tensor(all_returns)
            old_values_tensor = torch.tensor(all_old_values) if all_old_values else None
            
            self.metrics_logger.log_value_function_metrics(
                step=step,
                value_loss=value_loss,
                value_predictions=values_tensor,
                returns=returns_tensor,
                old_values=old_values_tensor
            )
    
    def _log_advantage_metrics(self, trajectories):
        """Log advantage-related metrics"""
        if not self.metrics_logger:
            return
        
        step = getattr(self.trainer, 'global_step', 0)
        
        # Collect advantages from trajectories
        all_advantages = []
        all_returns = []
        all_values = []
        
        for traj in trajectories:
            if hasattr(traj, 'advantages') and traj.advantages is not None:
                advantages = traj.advantages
                if torch.is_tensor(advantages):
                    all_advantages.extend(advantages.detach().cpu().numpy())
                else:
                    all_advantages.extend(advantages)
            
            # Collect returns and values for GAE quality analysis
            if hasattr(traj, 'returns') and traj.returns is not None:
                returns = traj.returns
                if torch.is_tensor(returns):
                    all_returns.extend(returns.detach().cpu().numpy())
                else:
                    all_returns.extend(returns)
            
            if hasattr(traj, 'values') and traj.values is not None:
                values = traj.values
                if torch.is_tensor(values):
                    all_values.extend(values.detach().cpu().numpy())
                else:
                    all_values.extend(values)
        
        if all_advantages:
            advantages_tensor = torch.tensor(all_advantages)
            returns_tensor = torch.tensor(all_returns) if all_returns else None
            values_tensor = torch.tensor(all_values) if all_values else None
            
            # Get GAE parameters from trainer config
            gae_lambda = getattr(self.trainer, 'gae_lambda', 0.95)
            gamma = getattr(self.trainer, 'gamma', 0.99)
            normalize_advantages = getattr(self.trainer, 'normalize_advantages', True)
            
            self.metrics_logger.log_advantage_metrics(
                step=step,
                advantages=advantages_tensor,
                gae_lambda=gae_lambda,
                gamma=gamma,
                normalize_advantages=normalize_advantages,
                returns=returns_tensor,
                values=values_tensor
            )
    
    def _extract_and_log_reward_metrics(self, trajectories, step):
        """Extract and log detailed reward metrics"""
        
        episode_rewards = []
        raw_rewards = []
        success_flags = []
        episode_lengths = []
        reward_breakdown = {}
        
        for traj in trajectories:
            # Total episode reward
            if hasattr(traj, 'total_reward'):
                episode_rewards.append(traj.total_reward)
            elif hasattr(traj, 'rewards') and traj.rewards:
                episode_rewards.append(sum(traj.rewards))
            
            # Raw rewards (before normalization)
            if hasattr(traj, 'raw_rewards') and traj.raw_rewards:
                raw_rewards.extend(traj.raw_rewards)
            
            # Success flag
            if hasattr(traj, 'success'):
                success_flags.append(traj.success)
            elif hasattr(traj, 'task_completed'):
                success_flags.append(traj.task_completed)
            
            # Episode length
            if hasattr(traj, 'length'):
                episode_lengths.append(traj.length)
            elif hasattr(traj, 'states'):
                episode_lengths.append(len(traj.states))
            
            # Reward breakdown by component
            if hasattr(traj, 'reward_breakdown') and traj.reward_breakdown:
                for component, value in traj.reward_breakdown.items():
                    if component not in reward_breakdown:
                        reward_breakdown[component] = []
                    reward_breakdown[component].append(value)
        
        if episode_rewards:
            self.metrics_logger.log_reward_metrics(
                step=step,
                episode_rewards=episode_rewards,
                raw_rewards=raw_rewards if raw_rewards else None,
                reward_breakdown=reward_breakdown if reward_breakdown else None,
                success_flags=success_flags if success_flags else None,
                episode_lengths=episode_lengths if episode_lengths else None
            )
    
    def log_episode_batch(
        self,
        step: int,
        episode_batch: List[Dict[str, Any]],
        tool_usage_stats: Optional[Dict[str, Any]] = None
    ):
        """Log a batch of episodes with detailed analysis"""
        if self.metrics_logger:
            self.metrics_logger.log_episode_details(
                step=step,
                episode_data=episode_batch,
                tool_usage_stats=tool_usage_stats
            )
    
    def log_checkpoint(self, step: int, checkpoint_path: str, metrics: Dict[str, float]):
        """Log model checkpoint"""
        if self.metrics_logger:
            self.metrics_logger.log_model_checkpoints(step, checkpoint_path, metrics)
    
    def finish(self):
        """Finish logging session"""
        if self.metrics_logger:
            self.metrics_logger.finish()


def enhance_grpo_trainer_logging(
    trainer,
    project_name: str = "skyrl-grpo-detailed-metrics",
    config: Optional[Dict[str, Any]] = None,
    enabled: bool = True
) -> GRPOTrainerLoggingEnhancement:
    """
    Enhance an existing GRPO trainer with comprehensive logging
    
    Args:
        trainer: The GRPO trainer instance to enhance
        project_name: WandB project name for logging
        config: Configuration dictionary to log
        enabled: Whether to enable logging
    
    Returns:
        Enhancement instance that can be used for additional logging
    """
    
    return GRPOTrainerLoggingEnhancement(
        trainer=trainer,
        project_name=project_name,
        config=config,
        enabled=enabled
    )