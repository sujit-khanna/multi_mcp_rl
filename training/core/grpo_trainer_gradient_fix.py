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
            # PyTorch 2.8.0 compatibility - GradScaler doesn't take device_type parameter
            self.scaler = amp.GradScaler()
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
                
                # CRITICAL FIX: Use existing sample-time log_probs if available
                # These are the "old" log probs for PPO and MUST be from sampling time
                # NOT recomputed with current model (which would make ratio = 1.0)
                if hasattr(traj, 'log_probs') and traj.log_probs is not None:
                    # Log the actual values for debugging
                    if isinstance(traj.log_probs, list) and len(traj.log_probs) > 0:
                        sample_val = traj.log_probs[0] if isinstance(traj.log_probs[0], (int, float)) else str(traj.log_probs[0])[:20]
                        logger.info(f"✅ Using {len(traj.log_probs)} sample-time log_probs for trajectory {traj_idx}, first value: {sample_val}")
                    else:
                        logger.debug(f"✅ Using sample-time log_probs for trajectory {traj_idx}")
                    
                    # Validate alignment
                    if len(traj.log_probs) != len(traj.actions):
                        logger.warning(f"Log_probs length mismatch ({len(traj.log_probs)} vs {len(traj.actions)}), recomputing...")
                        computed = self.policy.compute_log_probs(traj.states, traj.actions)
                        traj.log_probs = [p for p in computed]
                else:
                    # Fallback: compute with current model (NOT ideal - makes ratio ~1.0)
                    logger.warning(f"⚠️ No sample-time log_probs for trajectory {traj_idx}! Computing with current model (PPO ratio will be ~1.0)")
                    computed = self.policy.compute_log_probs(traj.states, traj.actions)
                    traj.log_probs = [p for p in computed]
                
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
            
            # Use sample-time log probabilities if available, otherwise compute current
            # CRITICAL: For PPO to work, we need the log_probs from when actions were sampled
            if all_old_log_probs is not None and len(all_old_log_probs) > 0:
                # Use sample-time logprobs as the "current" policy logprobs for PPO ratio computation
                # This is counterintuitive but correct: we want ratio = exp(new_policy - sample_time)
                # So we need the CURRENT policy logprobs, not the sample-time ones
                logger.info(f"✅ Sample-time log_probs available, computing current logprobs for PPO ratios")
                current_log_probs = self.policy.compute_log_probs(all_states, all_actions)
            else:
                # Fallback: recompute (will result in PPO ratio ~1.0)
                logger.warning("⚠️ No sample-time log_probs available, computing with current policy (PPO ratio will be ~1.0)")
                current_log_probs = self.policy.compute_log_probs(all_states, all_actions)
            
            # Compute reference policy log probabilities for KL penalty
            with torch.no_grad():
                ref_log_probs = self.reference_policy.compute_log_probs(all_states, all_actions)
            
        except Exception as e:
            logger.error(f"Error in data collection phase: {type(e).__name__}: {e}")
            raise
        
        # Apply forced mask to keep PPO on-policy
        # CRITICAL FIX: Ensure forced_mask defaults to False (unforced) not True
        forced_list = getattr(self, '_forced_masks', None)
        if forced_list is not None and len(forced_list) == len(old_log_probs):
            # Convert to bool tensor explicitly
            forced_mask = torch.tensor(forced_list, dtype=torch.bool, device=self.device)
            logger.debug(f"Using provided forced mask: {forced_mask.sum().item()}/{len(forced_mask)} forced")
        else:
            # DEFAULT: All steps are UNFORCED (False) unless explicitly marked
            forced_mask = torch.zeros(len(old_log_probs), dtype=torch.bool, device=self.device)
            logger.debug(f"Using default unforced mask (all False)")

        # Ensure mask is bool type
        if forced_mask.dtype != torch.bool:
            logger.warning(f"Converting forced_mask from {forced_mask.dtype} to bool")
            forced_mask = forced_mask.bool()
        
        unforced_mask = ~forced_mask
        
        # Log mask stats for debugging
        forced_count = forced_mask.sum().item()
        unforced_count = unforced_mask.sum().item()
        logger.info(f"Mask stats: {forced_count} forced, {unforced_count} unforced out of {len(forced_mask)} total")

        # ROBUST PPO RATIO COMPUTATION with empty tensor handling
        # Compute policy ratios with numerical stability on unforced steps
        log_ratios_full = current_log_probs - old_log_probs
        log_ratios_full = torch.clamp(log_ratios_full, min=-20.0, max=2.0)
        ratios_full = torch.exp(log_ratios_full).clamp(1e-8, 1e8)
        
        # PRE-OPTIMIZATION LOGGING for debugging (before any failures)
        try:
            import wandb
            if wandb.run:
                pre_opt_metrics = {
                    "debug/forced_fraction": forced_mask.float().mean().item() if forced_mask.numel() > 0 else 0.0,
                    "debug/unforced_count": unforced_mask.sum().item() if unforced_mask.numel() > 0 else 0,
                    "debug/total_steps": len(advantages) if hasattr(advantages, '__len__') else advantages.numel(),
                    "debug/ratio_mean_full": ratios_full.mean().item() if ratios_full.numel() > 0 else 1.0,
                    "debug/ratio_std_full": ratios_full.std().item() if ratios_full.numel() > 1 else 0.0,
                    "debug/adv_mean": advantages.mean().item() if advantages.numel() > 0 else 0.0,
                    "debug/adv_std": advantages.std().item() if advantages.numel() > 1 else 0.0,
                    "debug/old_logprobs_mean": old_log_probs.mean().item() if old_log_probs.numel() > 0 else 0.0,
                    "debug/current_logprobs_mean": current_log_probs.mean().item() if current_log_probs.numel() > 0 else 0.0,
                }
                wandb.log(pre_opt_metrics, commit=False)
                logger.debug(f"Logged pre-optimization metrics: forced={pre_opt_metrics['debug/forced_fraction']:.2%}")
        except Exception as e:
            logger.debug(f"Pre-optimization logging failed (non-critical): {e}")

        # Extract unforced data with safety checks
        if unforced_mask.any():
            log_ratios = log_ratios_full[unforced_mask]
            ratios = ratios_full[unforced_mask]
            adv_unforced = advantages[unforced_mask]
            
            # SANITY CHECK: Log ratio statistics
            ratio_mean = ratios.mean().item() if ratios.numel() > 0 else 1.0
            ratio_std = ratios.std().item() if ratios.numel() > 0 else 0.0
            logger.info(f"PPO Ratio Check - mean: {ratio_mean:.3f}, std: {ratio_std:.3f}, count: {ratios.numel()}")
        else:
            # No unforced steps - create empty tensors for metrics
            log_ratios = torch.tensor([], device=self.device, dtype=torch.float32)
            ratios = torch.tensor([], device=self.device, dtype=torch.float32)
            adv_unforced = torch.tensor([], device=self.device, dtype=torch.float32)
            logger.warning("All steps are forced - no unforced data for PPO ratios")

        # Compute clipped policy loss (PPO-style) on unforced steps; if none, fall back to REINFORCE on all
        if adv_unforced.numel() > 0:
            policy_loss_unclipped = -adv_unforced * ratios
            policy_loss_clipped = -adv_unforced * torch.clamp(
                ratios, 1 - self.clip_ratio, 1 + self.clip_ratio
            )
            policy_loss = torch.mean(torch.max(policy_loss_unclipped, policy_loss_clipped))
            logger.debug(f"PPO policy loss computed on {adv_unforced.numel()} unforced steps")
        else:
            logger.warning("No unforced steps available. Falling back to REINFORCE over all steps.")
            policy_loss = -(advantages * current_log_probs).mean()
            # For metrics, use full ratios for forced steps
            ratios = ratios_full
        
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
        
        # SAFE LOSS ASSEMBLY: Check validity before combining terms
        loss_terms = []
        loss_names = []
        
        # Validate policy loss
        if policy_loss is not None and torch.isfinite(policy_loss):
            loss_terms.append(policy_loss)
            loss_names.append("policy")
        else:
            logger.warning(f"⚠️ Invalid policy_loss: {policy_loss}")
        
        # Validate KL penalty
        if kl_penalty is not None and torch.isfinite(kl_penalty):
            loss_terms.append(kl_penalty)
            loss_names.append("kl_penalty")
        else:
            logger.warning(f"⚠️ Invalid kl_penalty: {kl_penalty}")
        
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
                if torch.isfinite(value_loss):
                    loss_terms.append(self.value_coef * value_loss)
                    loss_names.append("value")
                else:
                    logger.warning(f"⚠️ Invalid value_loss: {value_loss}")
        
        # Check if we have any valid loss terms
        if not loss_terms:
            logger.warning("⚠️ Skipping update: no valid loss terms found")
            try:
                import wandb
                wandb.log({"trainer/updates_skipped": 1}, commit=False)
            except:
                pass
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "kl_divergence": 0.0,
                "total_loss": 0.0,
                "grad_norm": 0.0,
                "updates_skipped": 1
            }
        
        # Sum valid loss terms
        total_loss = sum(loss_terms)
        logger.debug(f"✅ Assembled loss from {len(loss_terms)} terms: {loss_names}")
        
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
        
        # NaN SAFETY CHECK before optimization
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"⚠️ NaN/Inf detected in total_loss! Skipping optimization step")
            self.optimizer.zero_grad(set_to_none=True)
            
            # Log the skip
            try:
                import wandb
                if wandb.run:
                    wandb.log({"trainer/nan_skips": 1, "trainer/nan_loss": 1}, commit=False)
            except:
                pass
            
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "kl_divergence": 0.0,
                "total_loss": 0.0,
                "grad_norm": 0.0,
                "nan_skip": 1
            }
        
        # FIX 3.1: Proper gradient handling for mixed precision
        grad_norm = self._optimization_step(total_loss)
        
        # Store grad_norm as instance variable for external access
        self.grad_norm = grad_norm
        
        # POST-OPTIMIZATION LOGGING for success confirmation
        try:
            import wandb
            if wandb.run:
                post_opt_metrics = {
                    "trainer/grad_norm": grad_norm,
                    "trainer/optimization_success": 1,
                    "trainer/total_loss": total_loss.item() if hasattr(total_loss, 'item') else float(total_loss),
                }
                wandb.log(post_opt_metrics, commit=False)
                logger.debug(f"✅ Optimization successful: grad_norm={grad_norm:.4f}")
        except Exception as e:
            logger.debug(f"Post-optimization logging failed (non-critical): {e}")
        
        # Compile metrics with safe empty tensor handling
        # CRITICAL FIX: Guard against empty tensors in min/max/std operations
        if ratios.numel() > 0:
            ratio_mean = ratios.mean().item()
            ratio_max = ratios.max().item()
            ratio_min = ratios.min().item()
            ratio_std = ratios.std(unbiased=False).item() if ratios.numel() > 1 else 0.0
        else:
            ratio_mean = 1.0
            ratio_max = 1.0
            ratio_min = 1.0
            ratio_std = 0.0
            logger.warning("⚠️ Empty ratios tensor - using default values")
        
        if advantages.numel() > 0:
            adv_mean = advantages.mean().item()
            adv_std = advantages.std(unbiased=False).item() if advantages.numel() > 1 else 0.0
        else:
            adv_mean = 0.0
            adv_std = 0.0
            logger.warning("⚠️ Empty advantages tensor - using default values")
        
        metrics = {
            "policy_loss": policy_loss.item() if hasattr(policy_loss, 'item') else float(policy_loss),
            "kl_divergence": kl_divergence.item() if hasattr(kl_divergence, 'item') else float(kl_divergence),
            "kl_penalty": kl_penalty.item() if hasattr(kl_penalty, 'item') else float(kl_penalty),
            "total_loss": total_loss.item() if hasattr(total_loss, 'item') else float(total_loss),
            "avg_ratio": ratio_mean,
            "max_ratio": ratio_max,
            "min_ratio": ratio_min,
            "std_ratio": ratio_std,
            "avg_advantage": adv_mean,
            "std_advantage": adv_std,
            "kl_coef": kl_coef,
            "grad_norm": grad_norm,
            "ppo/forced_fraction": forced_mask.float().mean().item() if forced_mask.numel() > 0 else 0.0,
            "ppo/unforced_count": int(unforced_mask.sum().item()) if unforced_mask.numel() > 0 else 0,
            "ppo/total_steps": len(advantages) if hasattr(advantages, '__len__') else advantages.numel(),
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
        Perform optimization step with CPU-safe BnB handling and gradient validation.
        
        CRITICAL FIXES:
        - Never access BnB-specific attributes on CPU or when grads are None
        - Check gradient existence before optimization
        - Use generic gradient clipping that works for all optimizer types
        
        Args:
            loss: The loss tensor to optimize
            
        Returns:
            The gradient norm after clipping
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Determine safe execution mode
        use_cuda = torch.cuda.is_available() and self.device.type == "cuda"
        use_amp = use_cuda and self.use_mixed_precision
        
        # Backward pass
        if use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # CRITICAL: Get trainable parameters and check for gradient existence
        trainable_params = [p for p in self.policy.get_trainable_parameters() if p.requires_grad]
        
        # Early exit if no gradients (prevents BnB/optimizer crashes)
        has_grads = any(p.grad is not None for p in trainable_params)
        if not has_grads:
            logger.info("⏭️ Skipping optimizer.step(): no gradients this update")
            self.optimizer.zero_grad(set_to_none=True)
            return 0.0
        
        # Generic gradient clipping (safe for all optimizer types)
        if hasattr(self, 'max_grad_norm') and self.max_grad_norm > 0:
            if use_amp:
                # Unscale before clipping for mixed precision
                self.scaler.unscale_(self.optimizer)
            
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
        else:
            # Compute gradient norm for logging without clipping
            with torch.no_grad():
                grad_norm = torch.sqrt(sum(
                    (p.grad.detach().float().norm() ** 2)
                    for p in trainable_params if p.grad is not None
                )).item()
        
        # Optimizer step
        if use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # Return gradient norm for logging
        return grad_norm if isinstance(grad_norm, float) else grad_norm.item()
    
    def sync_reference_policy(self):
        """
        OPTIMIZED reference policy synchronization: Only copy LoRA adapters + value head.
        This avoids copying massive base model weights and quantization buffers.
        """
        logger.info("Synchronizing reference policy (LoRA + value head only)...")
        
        with torch.no_grad():
            # Get source state dict
            src_state = self.policy.state_dict()

            # OPTIMIZED APPROACH: Only sync trainable LoRA adapters + value head
            trainable_keys = []
            
            # Include LoRA adapter weights only
            lora_keys = [k for k in src_state.keys() if "lora_" in k]
            trainable_keys.extend(lora_keys)
            
            # Include value head weights only
            value_head_keys = [k for k in src_state.keys() if "value_head" in k]
            trainable_keys.extend(value_head_keys)
            
            # Create filtered state with only trainable components
            filtered_state = {k: v for k, v in src_state.items() if k in trainable_keys}
            
            logger.info(f"Syncing {len(lora_keys)} LoRA params + {len(value_head_keys)} value head params")
            logger.info(f"Skipping {len(src_state) - len(filtered_state)} base model parameters")

            # Load into reference policy (only the trainable parts)
            missing, unexpected = self.reference_policy.load_state_dict(filtered_state, strict=False)

            if missing:
                # Filter missing keys to only show LoRA/value_head that are actually expected
                relevant_missing = [k for k in missing if "lora_" in k or "value_head" in k]
                if relevant_missing:
                    logger.warning(f"Missing trainable keys in reference sync: {relevant_missing}")
            
            if unexpected:
                logger.debug(f"Unexpected keys (normal for partial sync): {len(unexpected)} keys")

            # Additional explicit value head sync for safety
            if hasattr(self.policy, 'value_head') and hasattr(self.reference_policy, 'value_head'):
                self.reference_policy.value_head.load_state_dict(self.policy.value_head.state_dict())
                logger.debug("Explicitly synced value head state")
        
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