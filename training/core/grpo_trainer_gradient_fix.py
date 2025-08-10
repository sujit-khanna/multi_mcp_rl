"""
GRPO Trainer with Fixed Gradient Clipping for Mixed Precision

This module extends the base GRPO trainer to properly handle gradient clipping
when using mixed precision (AMP) training. The key fix is to unscale gradients
BEFORE clipping them, otherwise the clipping has no effect.
"""

import logging
import torch
from torch.cuda.amp import GradScaler
from typing import Dict, List, Optional, Any

from .grpo_trainer_fixed_ref_policy import GRPOTrainerFixedRefPolicy
from .grpo_trainer import Trajectory

logger = logging.getLogger(__name__)


class GRPOTrainerGradientFix(GRPOTrainerFixedRefPolicy):
    """
    GRPO trainer with proper gradient clipping for mixed precision training.
    
    Key improvements:
    1. Unscales gradients before clipping when using mixed precision
    2. Properly tracks gradient norms for monitoring
    3. Removes aggressive ratio pre-clamping that constrains trust region
    4. Adds numerical stability without over-constraining
    """
    
    def __init__(
        self,
        policy,
        reference_policy,
        grpo_config: Dict[str, Any],
        training_config: Dict[str, Any],
        device: Optional[torch.device] = None,
        enable_mixed_precision: bool = False,
    ):
        """
        Initialize GRPO trainer with gradient clipping fixes.
        
        Args:
            policy: The policy to train
            reference_policy: Reference policy for KL divergence
            grpo_config: GRPO algorithm configuration
            training_config: Training hyperparameters
            device: Device to use for training
            enable_mixed_precision: Whether to use automatic mixed precision
        """
        super().__init__(
            policy=policy,
            reference_policy=reference_policy,
            grpo_config=grpo_config,
            training_config=training_config,
            device=device
        )
        
        # Mixed precision setup
        self.use_mixed_precision = enable_mixed_precision and (
            device.type == "cuda" and torch.cuda.is_available()
        )
        
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled with GradScaler")
        else:
            self.scaler = None
            logger.info("Mixed precision training disabled")
        
        # Gradient norm tracking
        self.grad_norm_history = []
        self.max_grad_norm = grpo_config.get('max_grad_norm', 1.0)
        
        # Get value loss coefficient from parent classes
        self.value_coef = grpo_config.get('value_loss_coef', 0.5)
        
    def _update_policy(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """
        Update policy with proper gradient clipping for mixed precision.
        
        This implementation fixes the gradient clipping order to ensure it works
        correctly with automatic mixed precision (AMP).
        """
        logger.info(f"Updating policy with {len(trajectories)} trajectories")
        
        try:
            # Collect all data for batch processing
            all_states = []
            all_actions = []
            all_advantages = []
            all_old_log_probs = []
            
            for traj_idx, traj in enumerate(trajectories):
                all_states.extend(traj.states)
                all_actions.extend(traj.actions)
                all_advantages.extend([traj.advantages[i] for i in range(traj.length)])

                # Robust handling of missing or incomplete old_log_probs.
                # If absent, compute them on-the-fly using the policy for the
                # original (pre-update) states/actions.
                if getattr(traj, 'old_log_probs', None) is None or len(traj.old_log_probs) != len(traj.actions):
                    try:
                        with torch.no_grad():
                            computed = self.policy.compute_log_probs(traj.states, traj.actions)
                        # Persist back to the trajectory for transparency and future steps
                        traj.old_log_probs = [p for p in computed]
                    except Exception as compute_err:
                        raise ValueError(
                            f"Trajectory missing old_log_probs and recomputation failed at index {traj_idx}: {type(compute_err).__name__}: {compute_err}"
                        )

                # At this point traj.old_log_probs must exist and align with actions
                for lp in traj.old_log_probs:
                    if isinstance(lp, torch.Tensor):
                        all_old_log_probs.append(lp)
                    else:
                        all_old_log_probs.append(torch.tensor(float(lp), device=self.device))
            
            # Convert to tensors
            old_log_probs = torch.stack(all_old_log_probs)
            advantages = torch.tensor(all_advantages, device=self.device, dtype=torch.float32)
            
            # Normalize advantages
            if self.grpo_config.get("normalize_advantages", True):
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute current log probabilities
            current_log_probs = self.policy.compute_log_probs(all_states, all_actions)
            
            # Compute reference policy log probabilities for KL penalty
            with torch.no_grad():
                ref_log_probs = self.reference_policy.compute_log_probs(all_states, all_actions)
            
        except Exception as e:
            logger.error(f"Error in data collection phase: {type(e).__name__}: {e}")
            raise
        
        # Compute policy ratios with numerical stability
        log_ratios = current_log_probs - old_log_probs
        # Clamp log ratios to prevent exp() overflow
        log_ratios = torch.clamp(log_ratios, min=-20.0, max=2.0)
        ratios = torch.exp(log_ratios)
        
        # FIX 3.2: Remove aggressive ratio pre-clamping
        # Only clamp for numerical stability, not policy constraints
        ratios = torch.clamp(ratios, 1e-8, 1e8)  # Prevent numerical issues only
        
        # Compute clipped policy loss (PPO-style)
        policy_loss_unclipped = -advantages * ratios
        policy_loss_clipped = -advantages * torch.clamp(
            ratios, 1 - self.clip_ratio, 1 + self.clip_ratio
        )
        policy_loss = torch.mean(torch.max(policy_loss_unclipped, policy_loss_clipped))
        
        # Compute KL divergence penalty
        # Correct KL divergence computation with clamping for stability
        log_ratio = current_log_probs - ref_log_probs
        # Clamp log ratios to prevent extreme values
        log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
        kl_divergence = torch.mean(log_ratio ** 2) * 0.5  # Quadratic approximation of KL
        # Additional clamp on KL divergence itself
        kl_divergence = torch.clamp(kl_divergence, min=0.0, max=100.0)
        
        # Adaptive KL penalty
        if self.adaptive_kl and self.step_count < self.kl_warmup_steps:
            kl_coef = self.kl_penalty_coef * (self.step_count / self.kl_warmup_steps)
        else:
            kl_coef = self.kl_penalty_coef
        
        kl_penalty = kl_coef * kl_divergence
        
        # Total loss
        total_loss = policy_loss + kl_penalty
        
        # Add value loss if we have value function
        value_loss = 0.0
        if hasattr(self, '_compute_value_loss'):
            # Collect returns for value loss
            all_returns = []
            for traj in trajectories:
                if hasattr(traj, 'returns') and traj.returns is not None:
                    all_returns.extend(traj.returns)
            
            if all_returns:
                returns = torch.tensor(all_returns, device=self.device, dtype=torch.float32)
                value_loss = self._compute_value_loss(all_states, returns)
                total_loss += self.value_coef * value_loss
        
        # Entropy regularization
        entropy_loss = 0.0
        if self.entropy_coef > 0:
            entropy = -current_log_probs.mean()
            entropy_loss = -self.entropy_coef * entropy
            total_loss += entropy_loss
        
        # FIX 3.1: Proper gradient handling for mixed precision
        grad_norm = self._optimization_step(total_loss)
        
        # Store grad_norm as instance variable for external access
        self.grad_norm = grad_norm
        
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
            "grad_norm": grad_norm,
        }
        
        if value_loss != 0:
            metrics["value_loss"] = value_loss.item()
        
        if entropy_loss != 0:
            metrics["entropy_loss"] = entropy_loss.item()
        
        # Check for training instabilities with detailed debugging
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error("🚨 NaN/Inf detected in loss components:")
            logger.error(f"  - policy_loss: {policy_loss.item() if torch.isfinite(policy_loss) else 'NaN/Inf'}")
            logger.error(f"  - value_loss: {value_loss.item() if torch.isfinite(value_loss) else 'NaN/Inf'}")
            logger.error(f"  - kl_penalty: {kl_penalty.item() if torch.isfinite(kl_penalty) else 'NaN/Inf'}")
            logger.error(f"  - kl_divergence: {kl_divergence.item() if torch.isfinite(kl_divergence) else 'NaN/Inf'}")
            logger.error(f"  - advantages mean: {advantages.mean().item() if torch.isfinite(advantages.mean()) else 'NaN/Inf'}")
            logger.error(f"  - log_ratios mean: {log_ratios.mean().item() if torch.isfinite(log_ratios.mean()) else 'NaN/Inf'}")
            logger.error(f"  - current_log_probs mean: {current_log_probs.mean().item() if torch.isfinite(current_log_probs.mean()) else 'NaN/Inf'}")
            logger.error(f"  - old_log_probs mean: {old_log_probs.mean().item() if torch.isfinite(old_log_probs.mean()) else 'NaN/Inf'}")
            # Return partial metrics instead of error to track what's happening
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "kl_divergence": 0.0,
                "total_loss": 0.0,
                "grad_norm": float('nan'),
                "error": "nan_loss"
            }
        
        if kl_divergence.item() > self.target_kl * 10:
            logger.warning(f"High KL divergence detected: {kl_divergence.item():.4f}")
        
        return metrics
    
    def _optimization_step(self, loss: torch.Tensor) -> float:
        """
        Perform optimization step with proper gradient handling for mixed precision.
        
        This is the key fix: unscale gradients BEFORE clipping when using AMP.
        
        Args:
            loss: The loss tensor to optimize
            
        Returns:
            The gradient norm after clipping
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        if self.use_mixed_precision:
            # Scale loss and backward
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
            
            # CRITICAL: Unscale gradients BEFORE clipping
            self.scaler.unscale_(self.optimizer)
            
            # Now clip gradients (they are unscaled)
            # Get all parameters including value head
            all_params = list(self.policy.model.parameters())
            if hasattr(self.policy, 'value_head'):
                all_params.extend(list(self.policy.value_head.parameters()))
                
            grad_norm = torch.nn.utils.clip_grad_norm_(
                all_params,
                self.max_grad_norm
            )
            
            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
        else:
            # Standard gradient flow
            loss.backward()
            
            # Clip gradients
            # Get all parameters including value head
            all_params = list(self.policy.model.parameters())
            if hasattr(self.policy, 'value_head'):
                all_params.extend(list(self.policy.value_head.parameters()))
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                all_params,
                self.max_grad_norm
            )
            
            # Check if gradients are valid before stepping
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                logger.error(f"NaN/Inf gradient norm detected: {grad_norm}. Skipping optimizer step.")
                # Clear gradients and return
                self.optimizer.zero_grad()
                return float('nan')
            
            # Standard optimizer step
            self.optimizer.step()
            
            # Debug: Check if parameters actually updated
            if self.step_count % 10 == 0:
                # Sample a parameter to check if it's changing
                sample_param = next(iter(self.policy.model.parameters()))
                logger.debug(f"Sample param mean: {sample_param.data.mean().item():.6f}")
                if hasattr(self.policy, 'value_head'):
                    value_param = next(iter(self.policy.value_head.parameters()))
                    logger.debug(f"Value param mean: {value_param.data.mean().item():.6f}")
        
        # Track gradient norm
        self.grad_norm_history.append(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm)
        
        # Log if gradient norm is high
        if grad_norm > self.max_grad_norm * 2:
            logger.warning(f"High gradient norm detected: {grad_norm:.4f}")
        
        return grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
    
    def get_gradient_stats(self) -> Dict[str, float]:
        """Get statistics about gradient norms."""
        if not self.grad_norm_history:
            return {}
        
        recent_norms = self.grad_norm_history[-100:]  # Last 100 steps
        
        return {
            "avg_grad_norm": sum(recent_norms) / len(recent_norms),
            "max_grad_norm": max(recent_norms),
            "min_grad_norm": min(recent_norms),
            "grad_norm_std": torch.std(torch.tensor(recent_norms)).item(),
        }