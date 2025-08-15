"""
GRPO Trainer with Fixed Gradient Clipping for Mixed Precision

This module extends the base GRPO trainer to properly handle gradient clipping
when using mixed precision (AMP) training. The key fix is to unscale gradients
BEFORE clipping them, otherwise the clipping has no effect.
"""

import logging
import torch
import numpy as np
from torch import amp
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
            self.scaler = amp.GradScaler(device_type="cuda")
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

                # CRITICAL FIX: Use correct key name and fail loudly if missing
                import os
                strict_keys = os.getenv("STRICT_TRAJ_KEYS", "1") == "1"
                
                # Check for log_probs (the expected key)
                if not hasattr(traj, 'log_probs') or traj.log_probs is None:
                    if strict_keys:
                        raise KeyError(
                            f"Missing 'log_probs' on trajectory {traj_idx}. "
                            f"Expected 'log_probs', not 'old_log_probs'. "
                            f"Fix trajectory collection to store with correct key."
                        )
                    # Fallback: try old_log_probs
                    if hasattr(traj, 'old_log_probs') and traj.old_log_probs is not None:
                        logger.warning(f"Using deprecated 'old_log_probs' key for trajectory {traj_idx}")
                        traj.log_probs = traj.old_log_probs
                    else:
                        # Compute on the fly as last resort
                        logger.warning(f"Computing log_probs on-the-fly for trajectory {traj_idx} (suboptimal for PPO)")
                        with torch.no_grad():
                            computed = self.policy.compute_log_probs(traj.states, traj.actions)
                        traj.log_probs = [p for p in computed]
                
                # Validate alignment
                if len(traj.log_probs) != len(traj.actions):
                    raise ValueError(
                        f"Trajectory {traj_idx}: log_probs length {len(traj.log_probs)} != actions length {len(traj.actions)}"
                    )
                
                # Convert to tensors
                for lp in traj.log_probs:
                    if isinstance(lp, torch.Tensor):
                        all_old_log_probs.append(lp)
                    else:
                        all_old_log_probs.append(torch.tensor(float(lp), device=self.device))
                
                # Collect forced mask if available
                if hasattr(traj, 'forced_mask') and traj.forced_mask is not None:
                    if not hasattr(self, '_forced_masks'):
                        self._forced_masks = []
                    self._forced_masks.extend(traj.forced_mask.tolist())
            
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
        
        # Apply forced mask to keep PPO on-policy
        forced_list = getattr(self, '_forced_masks', None)
        if forced_list is not None and len(forced_list) == len(old_log_probs):
            forced_mask = torch.tensor(forced_list, dtype=torch.bool, device=self.device)
        else:
            forced_mask = torch.zeros_like(old_log_probs, dtype=torch.bool)

        unforced_mask = ~forced_mask

        # Compute policy ratios with numerical stability on unforced steps
        log_ratios_full = current_log_probs - old_log_probs
        log_ratios_full = torch.clamp(log_ratios_full, min=-20.0, max=2.0)
        ratios_full = torch.exp(log_ratios_full).clamp(1e-8, 1e8)

        log_ratios = log_ratios_full[unforced_mask]
        ratios = ratios_full[unforced_mask]
        adv_unforced = advantages[unforced_mask]

        # SANITY CHECK: Log ratio statistics
        ratio_mean = ratios.mean().item() if ratios.numel() > 0 else 1.0
        ratio_std = ratios.std().item() if ratios.numel() > 0 else 0.0
        logger.info(f"PPO Ratio Check - mean: {ratio_mean:.3f}, std: {ratio_std:.3f}")

        # Compute clipped policy loss (PPO-style) on unforced steps; if none, fall back to REINFORCE on all
        if adv_unforced.numel() > 0:
            policy_loss_unclipped = -adv_unforced * ratios
            policy_loss_clipped = -adv_unforced * torch.clamp(
                ratios, 1 - self.clip_ratio, 1 + self.clip_ratio
            )
            policy_loss = torch.mean(torch.max(policy_loss_unclipped, policy_loss_clipped))
        else:
            logger.warning("No unforced steps available. Falling back to REINFORCE over all steps.")
            policy_loss = -(advantages * current_log_probs).mean()
        
        # Compute KL divergence penalty
        log_ratio = torch.clamp(current_log_probs - ref_log_probs, min=-10.0, max=10.0)
        if unforced_mask.any():
            kl_divergence = torch.mean((log_ratio[unforced_mask]) ** 2) * 0.5
        else:
            kl_divergence = torch.mean(log_ratio ** 2) * 0.5
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

        # Optional small BC term on forced steps to stabilize (does not affect PPO ratios)
        bc_loss = 0.0
        if forced_mask.any():
            bc_loss = (-current_log_probs[forced_mask]).mean()
            total_loss = total_loss + 0.01 * bc_loss
        
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
            "ppo/forced_fraction": forced_mask.float().mean().item() if forced_mask.numel() > 0 else 0.0,
            "ppo/unforced_count": int(unforced_mask.sum().item()),
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
    
    def sync_reference_policy(self):
        """
        Robust reference policy synchronization that works with LoRA/PEFT.
        """
        logger.info("Synchronizing reference policy...")
        
        with torch.no_grad():
            # Get source state dict (includes LoRA weights if applicable)
            src_state = self.policy.state_dict()

            # Try to include LoRA/PEFT adapter weights explicitly
            try:
                from peft import get_peft_model_state_dict
                if hasattr(self.policy, 'model') and hasattr(self.policy.model, 'peft_config'):
                    lora_state = get_peft_model_state_dict(self.policy.model)
                    src_state.update({k: v for k, v in lora_state.items()})
                    logger.info(f"Included {len(lora_state)} LoRA parameters in sync")
            except Exception as e:
                logger.debug(f"No LoRA state to sync: {e}")

            # Filter out non-trainable buffers (quantization artifacts)
            exclude = (".quant_state", ".absmax", ".quant_map", ".zeros", ".scales")
            filtered_state = {k: v for k, v in src_state.items() if not any(e in k for e in exclude)}

            skipped = [k for k in src_state.keys() if any(e in k for e in exclude)]
            if skipped:
                logger.debug(f"Skipped {len(skipped)} non-trainable buffers during sync")

            # Load into reference policy
            missing, unexpected = self.reference_policy.load_state_dict(filtered_state, strict=False)

            if missing:
                logger.warning(f"Missing keys in reference sync: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys in reference sync: {unexpected}")

            # Copy value head explicitly if present
            if hasattr(self.policy, 'value_head') and hasattr(self.reference_policy, 'value_head'):
                self.reference_policy.value_head.load_state_dict(self.policy.value_head.state_dict())
                logger.info("Updated reference policy value head")
        
        # Freeze reference policy parameters
        for param in self.reference_policy.parameters():
            param.requires_grad_(False)
        
        # Verification check
        diffs = self._count_param_diffs(self.policy, self.reference_policy)
        logger.info(f"Reference sync check: {diffs} tensors differ (should be 0 right after sync)")
        
        return diffs
    
    def _count_param_diffs(self, model_a, model_b) -> int:
        """Count number of parameter tensors that differ between models."""
        diffs = 0
        with torch.no_grad():
            params_a = dict(model_a.named_parameters())
            params_b = dict(model_b.named_parameters())
            
            for name in params_a:
                if name in params_b:
                    if not torch.allclose(params_a[name], params_b[name], atol=1e-6):
                        diffs += 1
                        logger.debug(f"Parameter diff in {name}: max_diff={torch.max(torch.abs(params_a[name] - params_b[name])).item():.6f}")
        
        return diffs
    
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
    
    def _compute_returns_advantages(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Tensor [T, B] of rewards
            values: Tensor [T+1, B] of value predictions (includes next state value)
            dones: Tensor [T, B] of done flags (1.0 after terminal states)
            gamma: Discount factor
            lam: GAE lambda parameter
            
        Returns:
            returns: Tensor [T, B] of returns
            advantages: Tensor [T, B] of advantages
        """
        T, B = rewards.shape
        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(B, device=rewards.device)
        
        # Work backwards through time
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            advantages[t] = lastgaelam
        
        # Returns are advantages + baseline
        returns = advantages + values[:-1]  # Remove last value (next state)
        
        return returns, advantages
    
    def _compute_value_loss(self, states, returns, clip_vloss=None):
        """
        Compute value function loss.
        
        Args:
            states: List of states
            returns: Tensor of target returns
            clip_vloss: Optional value clipping range
            
        Returns:
            Value loss tensor
        """
        if not hasattr(self.policy, 'compute_values'):
            logger.warning("Policy does not have compute_values method, skipping value loss")
            return torch.tensor(0.0, device=self.device)
        
        # Compute current value predictions
        values_pred = self.policy.compute_values(states)
        
        if clip_vloss is None:
            # Simple MSE loss
            value_loss = 0.5 * (values_pred - returns).pow(2).mean()
        else:
            # PPO-style clipped value loss
            # Note: This would require old values for proper clipping
            # For now, use simple MSE
            value_loss = 0.5 * (values_pred - returns).pow(2).mean()
        
        # Log value function metrics
        with torch.no_grad():
            explained_var = self._compute_explained_variance(returns, values_pred)
            logger.debug(f"Value loss: {value_loss.item():.4f}, Explained variance: {explained_var:.3f}")
        
        return value_loss
    
    def _compute_explained_variance(self, targets, predictions):
        """Compute explained variance of value function."""
        targets_np = targets.detach().cpu().numpy()
        predictions_np = predictions.detach().cpu().numpy()
        
        var_targets = np.var(targets_np)
        if var_targets == 0:
            return 0.0
        
        return 1 - np.var(targets_np - predictions_np) / var_targets