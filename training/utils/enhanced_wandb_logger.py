#!/usr/bin/env python3
"""
Enhanced WandB Logger for GRPO Training Metrics
============================================

This module provides comprehensive logging of all critical training metrics including:
- Policy and value losses with detailed breakdowns
- Advantage metrics (GAE, normalization, variance)
- Reward metrics (raw, normalized, clipped, per-turn)
- KL divergence and entropy metrics
- Gradient norms and parameter statistics
- Episode performance and success rates
- Tool usage statistics and efficiency metrics
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False

logger = logging.getLogger(__name__)


class GRPOTrainingMetricsLogger:
    """
    Enhanced logger specifically for GRPO training metrics with comprehensive tracking
    of all key training components including policy/value losses, advantages, rewards, etc.
    """
    
    def __init__(
        self,
        project_name: str = "skyrl-grpo-training",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ):
        self.enabled = enabled and WANDB_AVAILABLE
        self.wandb_run = None
        self.training_start_time = time.time()
        
        # Initialize WandB
        if self.enabled:
            try:
                self.wandb_run = wandb.init(
                    project=project_name,
                    name=run_name,
                    config=config or {},
                    resume="allow",
                    settings=wandb.Settings(start_method="fork")
                )
                
                # Define metric hierarchies for better organization
                self._setup_metric_definitions()
                
                logger.info(f"✅ Enhanced GRPO metrics logging initialized: {project_name}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                self.enabled = False
    
    def _setup_metric_definitions(self):
        """Setup WandB metric definitions for better dashboard organization"""
        if not self.enabled:
            return
        
        try:
            # Define step as the primary metric
            wandb.define_metric("training/step")
            
            # Policy metrics
            wandb.define_metric("policy/*", step_metric="training/step")
            wandb.define_metric("policy/loss", summary="min")
            wandb.define_metric("policy/entropy", summary="last")
            wandb.define_metric("policy/kl_divergence", summary="last")
            wandb.define_metric("policy/clip_fraction", summary="last")
            
            # Value function metrics
            wandb.define_metric("value/*", step_metric="training/step")
            wandb.define_metric("value/loss", summary="min")
            wandb.define_metric("value/explained_variance", summary="max")
            wandb.define_metric("value/prediction_error", summary="min")
            
            # Advantage metrics
            wandb.define_metric("advantages/*", step_metric="training/step")
            wandb.define_metric("advantages/mean", summary="last")
            wandb.define_metric("advantages/std", summary="last")
            wandb.define_metric("advantages/gae_lambda", summary="last")
            
            # Reward metrics
            wandb.define_metric("rewards/*", step_metric="training/step")
            wandb.define_metric("rewards/mean_episode", summary="max")
            wandb.define_metric("rewards/std_episode", summary="last")
            wandb.define_metric("rewards/success_rate", summary="max")
            
            # Gradient metrics
            wandb.define_metric("gradients/*", step_metric="training/step")
            wandb.define_metric("gradients/norm", summary="last")
            wandb.define_metric("gradients/clip_fraction", summary="last")
            
            # Episode metrics
            wandb.define_metric("episodes/*", step_metric="training/step")
            wandb.define_metric("episodes/length_mean", summary="last")
            wandb.define_metric("episodes/success_rate", summary="max")
            
            # Tool usage metrics
            wandb.define_metric("tools/*", step_metric="training/step")
            wandb.define_metric("tools/usage_efficiency", summary="max")
            wandb.define_metric("tools/error_rate", summary="min")
            
        except Exception as e:
            logger.warning(f"Failed to setup WandB metric definitions: {e}")
    
    def log_policy_metrics(
        self,
        step: int,
        policy_loss: float,
        entropy: float,
        kl_divergence: float,
        clip_fraction: float,
        ratio_mean: float,
        ratio_std: float,
        old_approx_kl: Optional[float] = None,
        clipfrac_threshold: float = 0.1
    ):
        """Log detailed policy-related metrics"""
        if not self.enabled:
            return
        
        metrics = {
            "policy/loss": policy_loss,
            "policy/entropy": entropy,
            "policy/kl_divergence": kl_divergence,
            "policy/kl_divergence_old_approx": old_approx_kl or 0.0,
            "policy/clip_fraction": clip_fraction,
            "policy/ratio_mean": ratio_mean,
            "policy/ratio_std": ratio_std,
            "policy/high_clip_fraction": 1.0 if clip_fraction > clipfrac_threshold else 0.0,
            "training/step": step
        }
        
        # Add policy health indicators
        metrics["policy/entropy_healthy"] = 1.0 if entropy > 0.01 else 0.0
        metrics["policy/kl_healthy"] = 1.0 if kl_divergence < 0.1 else 0.0
        metrics["policy/ratio_healthy"] = 1.0 if 0.5 < ratio_mean < 2.0 else 0.0
        
        self._log_metrics(metrics, step)
    
    def log_value_function_metrics(
        self,
        step: int,
        value_loss: float,
        value_predictions: torch.Tensor,
        returns: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
        clip_range: float = 0.2
    ):
        """Log comprehensive value function metrics"""
        if not self.enabled:
            return
        
        # Convert tensors to numpy for calculations
        values_np = value_predictions.detach().cpu().numpy()
        returns_np = returns.detach().cpu().numpy()
        
        # Calculate explained variance
        var_returns = np.var(returns_np)
        explained_var = 0.0
        if var_returns > 0:
            explained_var = 1.0 - np.var(returns_np - values_np) / var_returns
        
        # Calculate prediction errors
        prediction_errors = returns_np - values_np
        mae = np.mean(np.abs(prediction_errors))
        mse = np.mean(prediction_errors ** 2)
        
        metrics = {
            "value/loss": value_loss,
            "value/explained_variance": explained_var,
            "value/prediction_mae": mae,
            "value/prediction_mse": mse,
            "value/prediction_error_mean": np.mean(prediction_errors),
            "value/prediction_error_std": np.std(prediction_errors),
            "value/predictions_mean": np.mean(values_np),
            "value/predictions_std": np.std(values_np),
            "value/returns_mean": np.mean(returns_np),
            "value/returns_std": np.std(returns_np),
            "training/step": step
        }
        
        # Value function health indicators
        metrics["value/explained_var_healthy"] = 1.0 if explained_var > 0.1 else 0.0
        metrics["value/prediction_error_healthy"] = 1.0 if mae < 2.0 else 0.0
        
        # Value clipping metrics if old values are provided
        if old_values is not None:
            old_values_np = old_values.detach().cpu().numpy()
            clipped_values = np.clip(old_values_np, 
                                   old_values_np - clip_range, 
                                   old_values_np + clip_range)
            clip_fraction = np.mean(np.abs(values_np - clipped_values) > 1e-6)
            metrics["value/clip_fraction"] = clip_fraction
        
        self._log_metrics(metrics, step)
    
    def log_advantage_metrics(
        self,
        step: int,
        advantages: torch.Tensor,
        gae_lambda: float,
        gamma: float,
        normalize_advantages: bool = True,
        returns: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None
    ):
        """Log comprehensive advantage estimation metrics"""
        if not self.enabled:
            return
        
        # Convert to numpy
        advantages_np = advantages.detach().cpu().numpy()
        
        # Basic advantage statistics
        metrics = {
            "advantages/mean": np.mean(advantages_np),
            "advantages/std": np.std(advantages_np),
            "advantages/min": np.min(advantages_np),
            "advantages/max": np.max(advantages_np),
            "advantages/abs_mean": np.mean(np.abs(advantages_np)),
            "advantages/positive_fraction": np.mean(advantages_np > 0),
            "advantages/gae_lambda": gae_lambda,
            "advantages/gamma": gamma,
            "advantages/normalized": 1.0 if normalize_advantages else 0.0,
            "training/step": step
        }
        
        # Advantage distribution metrics
        percentiles = [5, 25, 50, 75, 95]
        for p in percentiles:
            metrics[f"advantages/percentile_{p}"] = np.percentile(advantages_np, p)
        
        # GAE quality metrics
        if returns is not None and values is not None:
            returns_np = returns.detach().cpu().numpy()
            values_np = values.detach().cpu().numpy()
            
            # TD residuals (should correlate with advantages)
            td_residuals = returns_np - values_np
            correlation = np.corrcoef(advantages_np, td_residuals)[0, 1]
            metrics["advantages/td_correlation"] = correlation if not np.isnan(correlation) else 0.0
            
            # Advantage signal-to-noise ratio
            signal_power = np.var(advantages_np)
            noise_power = np.var(advantages_np - td_residuals) if len(advantages_np) > 1 else 0.0
            snr = signal_power / (noise_power + 1e-8)
            metrics["advantages/signal_to_noise_ratio"] = snr
        
        # Advantage health indicators
        metrics["advantages/healthy_distribution"] = 1.0 if 0.3 < np.std(advantages_np) < 3.0 else 0.0
        metrics["advantages/reasonable_range"] = 1.0 if np.max(np.abs(advantages_np)) < 10.0 else 0.0
        
        self._log_metrics(metrics, step)
    
    def log_reward_metrics(
        self,
        step: int,
        episode_rewards: List[float],
        raw_rewards: Optional[List[float]] = None,
        normalized_rewards: Optional[List[float]] = None,
        reward_breakdown: Optional[Dict[str, List[float]]] = None,
        success_flags: Optional[List[bool]] = None,
        episode_lengths: Optional[List[int]] = None
    ):
        """Log comprehensive reward metrics and distributions"""
        if not self.enabled:
            return
        
        if not episode_rewards:
            return
        
        episode_rewards_np = np.array(episode_rewards)
        
        # Basic reward statistics
        metrics = {
            "rewards/mean_episode": np.mean(episode_rewards_np),
            "rewards/std_episode": np.std(episode_rewards_np),
            "rewards/min_episode": np.min(episode_rewards_np),
            "rewards/max_episode": np.max(episode_rewards_np),
            "rewards/total_episodes": len(episode_rewards),
            "training/step": step
        }
        
        # Success rate if available
        if success_flags:
            success_rate = np.mean(success_flags)
            metrics["rewards/success_rate"] = success_rate
            metrics["rewards/failure_rate"] = 1.0 - success_rate
            
            # Success vs failure reward analysis
            if len(success_flags) > 0:
                success_rewards = [r for r, s in zip(episode_rewards, success_flags) if s]
                failure_rewards = [r for r, s in zip(episode_rewards, success_flags) if not s]
                
                if success_rewards:
                    metrics["rewards/success_mean"] = np.mean(success_rewards)
                if failure_rewards:
                    metrics["rewards/failure_mean"] = np.mean(failure_rewards)
        
        # Episode length analysis
        if episode_lengths:
            lengths_np = np.array(episode_lengths)
            metrics["rewards/episode_length_mean"] = np.mean(lengths_np)
            metrics["rewards/episode_length_std"] = np.std(lengths_np)
            
            # Reward efficiency (reward per turn)
            efficiency = episode_rewards_np / (lengths_np + 1e-8)
            metrics["rewards/efficiency_mean"] = np.mean(efficiency)
            metrics["rewards/efficiency_std"] = np.std(efficiency)
        
        # Raw vs normalized rewards comparison
        if raw_rewards:
            raw_np = np.array(raw_rewards)
            metrics["rewards/raw_mean"] = np.mean(raw_np)
            metrics["rewards/raw_std"] = np.std(raw_np)
            metrics["rewards/normalization_factor"] = np.mean(episode_rewards_np) / (np.mean(raw_np) + 1e-8)
        
        if normalized_rewards:
            norm_np = np.array(normalized_rewards)
            metrics["rewards/normalized_mean"] = np.mean(norm_np)
            metrics["rewards/normalized_std"] = np.std(norm_np)
        
        # Detailed reward breakdown
        if reward_breakdown:
            for component, values in reward_breakdown.items():
                if values:
                    values_np = np.array(values)
                    metrics[f"rewards/breakdown_{component}_mean"] = np.mean(values_np)
                    metrics[f"rewards/breakdown_{component}_std"] = np.std(values_np)
                    metrics[f"rewards/breakdown_{component}_contribution"] = np.mean(values_np) / (np.mean(episode_rewards_np) + 1e-8)
        
        # Reward distribution percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            metrics[f"rewards/percentile_{p}"] = np.percentile(episode_rewards_np, p)
        
        # Reward health indicators
        metrics["rewards/positive_fraction"] = np.mean(episode_rewards_np > 0)
        metrics["rewards/improving_trend"] = self._calculate_trend(episode_rewards_np)
        
        self._log_metrics(metrics, step)
    
    def log_gradient_metrics(
        self,
        step: int,
        model: nn.Module,
        total_grad_norm: float,
        max_grad_norm: float,
        grad_clip_fraction: Optional[float] = None
    ):
        """Log detailed gradient and parameter metrics"""
        if not self.enabled:
            return
        
        metrics = {
            "gradients/total_norm": total_grad_norm,
            "gradients/max_norm_limit": max_grad_norm,
            "gradients/clip_fraction": grad_clip_fraction or 0.0,
            "gradients/norm_ratio": total_grad_norm / (max_grad_norm + 1e-8),
            "training/step": step
        }
        
        # Gradient health indicators
        metrics["gradients/healthy_norm"] = 1.0 if total_grad_norm < max_grad_norm * 0.8 else 0.0
        metrics["gradients/exploding"] = 1.0 if total_grad_norm > max_grad_norm * 5.0 else 0.0
        metrics["gradients/vanishing"] = 1.0 if total_grad_norm < 1e-6 else 0.0
        
        # Layer-wise gradient analysis
        try:
            layer_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    layer_norms[name] = grad_norm
            
            if layer_norms:
                # Aggregate by layer type
                policy_norms = [v for k, v in layer_norms.items() if 'policy' in k.lower() or 'lm_head' in k.lower()]
                value_norms = [v for k, v in layer_norms.items() if 'value' in k.lower() or 'v_head' in k.lower()]
                
                if policy_norms:
                    metrics["gradients/policy_layers_mean"] = np.mean(policy_norms)
                    metrics["gradients/policy_layers_max"] = np.max(policy_norms)
                
                if value_norms:
                    metrics["gradients/value_layers_mean"] = np.mean(value_norms)
                    metrics["gradients/value_layers_max"] = np.max(value_norms)
                    
        except Exception as e:
            logger.debug(f"Could not compute layer-wise gradient norms: {e}")
        
        self._log_metrics(metrics, step)
    
    def log_training_step_summary(
        self,
        step: int,
        epoch: int,
        total_loss: float,
        learning_rate: float,
        batch_size: int,
        time_per_step: float,
        memory_usage: Optional[Dict[str, float]] = None
    ):
        """Log high-level training step summary metrics"""
        if not self.enabled:
            return
        
        metrics = {
            "training/step": step,
            "training/epoch": epoch,
            "training/total_loss": total_loss,
            "training/learning_rate": learning_rate,
            "training/batch_size": batch_size,
            "training/time_per_step": time_per_step,
            "training/steps_per_second": 1.0 / (time_per_step + 1e-8),
            "training/training_time_hours": (time.time() - self.training_start_time) / 3600.0
        }
        
        # Memory usage metrics
        if memory_usage:
            for key, value in memory_usage.items():
                metrics[f"system/{key}"] = value
        
        # GPU memory if available
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / 1e9  # GB
                gpu_memory_cached = torch.cuda.memory_reserved() / 1e9  # GB
                metrics["system/gpu_memory_allocated_gb"] = gpu_memory
                metrics["system/gpu_memory_cached_gb"] = gpu_memory_cached
                metrics["system/gpu_memory_utilization"] = gpu_memory / (torch.cuda.get_device_properties(0).total_memory / 1e9)
            except Exception:
                pass
        
        self._log_metrics(metrics, step)
    
    def log_episode_details(
        self,
        step: int,
        episode_data: List[Dict[str, Any]],
        tool_usage_stats: Optional[Dict[str, Any]] = None
    ):
        """Log detailed episode information including tool usage"""
        if not self.enabled or not episode_data:
            return
        
        # Aggregate episode statistics
        episode_lengths = [len(ep.get('turns', [])) for ep in episode_data]
        episode_rewards = [ep.get('total_reward', 0) for ep in episode_data]
        success_flags = [ep.get('success', False) for ep in episode_data]
        
        metrics = {
            "episodes/count": len(episode_data),
            "episodes/length_mean": np.mean(episode_lengths),
            "episodes/length_std": np.std(episode_lengths),
            "episodes/reward_mean": np.mean(episode_rewards),
            "episodes/success_rate": np.mean(success_flags),
            "training/step": step
        }
        
        # Tool usage analysis
        if tool_usage_stats:
            for key, value in tool_usage_stats.items():
                if isinstance(value, (int, float)):
                    metrics[f"tools/{key}"] = value
        
        # Reasoning quality analysis
        reasoning_scores = []
        for ep in episode_data:
            for turn in ep.get('turns', []):
                if 'reasoning_score' in turn:
                    reasoning_scores.append(turn['reasoning_score'])
        
        if reasoning_scores:
            metrics["episodes/reasoning_score_mean"] = np.mean(reasoning_scores)
            metrics["episodes/reasoning_score_std"] = np.std(reasoning_scores)
        
        self._log_metrics(metrics, step)
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend direction of values (-1 to 1)"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.corrcoef(x, values)[0, 1] if len(values) > 1 else 0.0
        return np.clip(slope, -1.0, 1.0)
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Internal method to log metrics to WandB"""
        if not self.enabled:
            return
        
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to WandB: {e}")
    
    def log_model_checkpoints(self, step: int, checkpoint_path: str, metrics: Dict[str, float]):
        """Log model checkpoint information"""
        if not self.enabled:
            return
        
        try:
            # Log checkpoint metrics
            checkpoint_metrics = {
                "checkpoints/step": step,
                **{f"checkpoints/{k}": v for k, v in metrics.items()}
            }
            wandb.log(checkpoint_metrics, step=step)
            
            # Save checkpoint as artifact
            artifact = wandb.Artifact(
                name=f"model_checkpoint_step_{step}",
                type="model",
                description=f"Model checkpoint at step {step}"
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
            
        except Exception as e:
            logger.warning(f"Failed to log checkpoint to WandB: {e}")
    
    def finish(self):
        """Finish WandB logging session"""
        if self.enabled and self.wandb_run:
            try:
                wandb.finish()
                logger.info("✅ WandB training metrics logging session completed")
            except Exception as e:
                logger.warning(f"Failed to finish WandB session: {e}")


# Convenience function for integration
def create_grpo_metrics_logger(
    project_name: str = "skyrl-grpo-training",
    config: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None
) -> GRPOTrainingMetricsLogger:
    """Create and initialize GRPO training metrics logger"""
    
    if run_name is None:
        run_name = f"grpo-training-{time.strftime('%Y%m%d-%H%M%S')}"
    
    return GRPOTrainingMetricsLogger(
        project_name=project_name,
        run_name=run_name,
        config=config,
        enabled=True
    )