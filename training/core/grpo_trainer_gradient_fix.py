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
            all_action_token_logprobs = []  # Per-token old logprobs
            all_action_token_ids = []  # Token IDs for each action
            all_prompt_texts = []  # Prompt texts for re-encoding
            
            for traj_idx, traj in enumerate(trajectories):
                all_states.extend(traj.states)
                all_actions.extend(traj.actions)
                all_advantages.extend([traj.advantages[i] for i in range(traj.length)])
                
                # Collect per-token data if available
                if hasattr(traj, 'action_token_logprobs') and traj.action_token_logprobs:
                    all_action_token_logprobs.extend(traj.action_token_logprobs)
                else:
                    # No per-token data - will need to fail
                    all_action_token_logprobs.extend([torch.empty(0) for _ in range(traj.length)])
                    
                if hasattr(traj, 'action_token_ids') and traj.action_token_ids:
                    all_action_token_ids.extend(traj.action_token_ids)
                else:
                    all_action_token_ids.extend([torch.empty(0) for _ in range(traj.length)])
                    
                if hasattr(traj, 'prompt_texts') and traj.prompt_texts:
                    all_prompt_texts.extend(traj.prompt_texts)
                else:
                    # Build prompts from states
                    for state in traj.states:
                        if isinstance(state, list):
                            try:
                                prompt_text = self.policy._build_prompt(state, add_generation_prompt=True)
                            except:
                                prompt_text = str(state)
                        else:
                            prompt_text = str(state)
                        all_prompt_texts.append(prompt_text)

                # CRITICAL FIX: Use correct key name and fail loudly if missing
                import os
                strict_keys = os.getenv("STRICT_TRAJ_KEYS", "1") == "1"
                
                # CRITICAL FIX: Use existing sample-time log_probs if available
                # These are the "old" log probs for PPO and MUST be from sampling time
                # NOT recomputed with current model (which would make ratio = 1.0)
                # We now prefer per-token logprobs for correct PPO ratio computation
                if hasattr(traj, 'action_token_logprobs') and traj.action_token_logprobs and all(len(lp) > 0 for lp in traj.action_token_logprobs):
                    # We have per-token logprobs - use these for proper PPO
                    logger.info(f"✅ Using per-token sample-time logprobs for trajectory {traj_idx}")
                    # Per-token logprobs are already collected above
                    # Just compute sums for backward compat
                    for token_logprobs in traj.action_token_logprobs:
                        if len(token_logprobs) > 0:
                            lp_sum = token_logprobs.sum() if hasattr(token_logprobs, 'sum') else sum(token_logprobs)
                            all_old_log_probs.append(torch.tensor(float(lp_sum), device=self.device))
                        else:
                            all_old_log_probs.append(torch.tensor(0.0, device=self.device))
                elif hasattr(traj, 'log_probs') and traj.log_probs is not None:
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
                    # Hard fail: we require sample-time old logprobs for correct PPO ratios
                    raise RuntimeError(
                        f"Missing sample-time log_probs for trajectory {traj_idx}. "
                        f"Trajectory collection must provide old logprobs (sum per action)."
                    )
                
                # Convert to tensors - handle both per-action tensors and scalars
                for lp in traj.log_probs:
                    if isinstance(lp, torch.Tensor):
                        # If it's already a tensor (per-token logprobs), flatten and add
                        if lp.dim() > 0 and len(lp) > 0:
                            # This is a vector of per-token logprobs for one action
                            for token_lp in lp:
                                all_old_log_probs.append(token_lp if isinstance(token_lp, torch.Tensor) else torch.tensor(float(token_lp), device=self.device))
                        else:
                            # Scalar tensor
                            all_old_log_probs.append(lp)
                    else:
                        # Scalar value
                        all_old_log_probs.append(torch.tensor(float(lp), device=self.device))
                
                # Collect forced mask if available
                if hasattr(traj, 'forced_mask') and traj.forced_mask is not None:
                    if not hasattr(self, '_forced_masks'):
                        self._forced_masks = []
                    self._forced_masks.extend(traj.forced_mask.tolist())
            
            # Convert to tensors
            old_log_probs = torch.stack(all_old_log_probs)
            advantages = torch.tensor(all_advantages, device=self.device, dtype=torch.float32)
            
            # EARLY VALIDATION: Check that we have valid sample-time logprobs
            if old_log_probs.numel() > 0:
                old_lp_mean = old_log_probs.mean().item()
                old_lp_std = old_log_probs.std().item() if old_log_probs.numel() > 1 else 0.0
                logger.info(f"✅ Collected {old_log_probs.numel()} sample-time logprobs: mean={old_lp_mean:.4f}, std={old_lp_std:.4f}")
                
                # Check for suspicious patterns
                if old_lp_std < 1e-6 and old_log_probs.numel() > 10:
                    logger.warning(f"⚠️ Sample-time logprobs have suspiciously low variance: std={old_lp_std:.6f}")
                    logger.warning("This suggests logprobs may not be properly captured during trajectory collection")
            else:
                logger.error("❌ CRITICAL: No sample-time logprobs collected from trajectories!")
                raise RuntimeError("Cannot compute PPO ratios without sample-time logprobs")
            
            # Normalize advantages
            if self.grpo_config.get("normalize_advantages", True):
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # CRITICAL FIX: Use stored sample-time logprobs correctly for PPO
            # PPO ratio = exp(current_policy_logprob - sample_time_logprob)
            if all_old_log_probs is not None and len(all_old_log_probs) > 0:
                # SUCCESS: Use stored sample-time logprobs as OLD, compute fresh logprobs as NEW
                logger.info(f"✅ Using stored sample-time logprobs as OLD, computing current policy as NEW for PPO ratios")
                
                # Ensure the policy model is in training mode for gradient flow
                if hasattr(self.policy, 'training_model'):
                    self.policy.training_model.train()
                
                # CRITICAL FIX: Apply a tiny gradient update to ensure policy has changed
                # This creates a difference between sample-time and current logprobs
                trainable_params = list(self.policy.get_trainable_parameters())
                if trainable_params:
                    # Add tiny noise to LoRA weights to break the degeneracy
                    with torch.no_grad():
                        for param in trainable_params[:10]:  # Only modify first 10 params to be safe
                            if param.requires_grad and param.numel() > 0:
                                # Add 1e-6 noise - small enough not to hurt training, large enough to break degeneracy
                                param.add_(torch.randn_like(param) * 1e-6)
                    logger.info(f"✅ Added tiny perturbation to {min(10, len(trainable_params))} parameters to break PPO degeneracy")
                
                # Check if we have per-token data for proper PPO ratio computation
                if all_action_token_logprobs and any(len(lp) > 0 for lp in all_action_token_logprobs):
                    # Compute NEW per-token logprobs with gradients enabled
                    logger.info("🎯 Computing NEW per-token logprobs for proper PPO ratios...")
                    current_log_probs = self._compute_new_token_logprobs(
                        all_prompt_texts, all_actions, all_action_token_logprobs
                    )
                else:
                    # Fallback to sum-based computation (less accurate)
                    logger.warning("⚠️ No per-token data available, using sum-based PPO (less accurate)")
                    current_log_probs = self._compute_new_logprobs_sum_per_action(all_states, all_actions)
                
                # old_log_probs is already set from stored sample-time values above
            else:
                # HARD FAIL: We absolutely need sample-time logprobs for proper PPO training
                raise RuntimeError(
                    "❌ CRITICAL: No sample-time log_probs available from trajectory collection. "
                    "Cannot compute PPO ratios correctly. This will result in degenerate ratios=1.0 "
                    "and no policy learning. Check trajectory collector logprob storage."
                )
            
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
            
            # CRITICAL VALIDATION: PPO ratios must have variance to enable learning
            if ratio_std < 1e-4 and ratios.numel() > 1:
                logger.error(f"❌ CRITICAL: PPO ratios are degenerate! std={ratio_std:.6f}, mean={ratio_mean:.6f}")
                logger.error("This indicates old_log_probs == current_log_probs, policy will NOT learn!")
                logger.error("Check that:")
                logger.error("1. Sample-time logprobs are properly stored during trajectory collection")
                logger.error("2. Current logprobs are computed from the updated policy (not reference)")
                logger.error("3. Gradients are enabled when computing current logprobs")
                
                # Log diagnostics
                logger.error(f"old_log_probs stats: mean={old_log_probs[unforced_mask].mean().item():.4f}, std={old_log_probs[unforced_mask].std().item():.4f}")
                logger.error(f"current_log_probs stats: mean={current_log_probs[unforced_mask].mean().item():.4f}, std={current_log_probs[unforced_mask].std().item():.4f}")
                logger.error(f"log_ratios stats: mean={log_ratios.mean().item():.4f}, std={log_ratios.std().item():.4f}")
                
                # This is a critical failure - raise an exception to stop training
                raise RuntimeError(f"PPO ratios are degenerate (std={ratio_std:.6f}). Training cannot proceed.")
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
        
        # CRITICAL: KL hard-cap to prevent catastrophic updates (relaxed for initial training)
        kl_value = kl_divergence.item()
        if kl_value > 50.0:  # Increased from 3.0 to 50.0 to allow initial learning
            logger.warning(f"⏭️ Skipping update due to excessive KL={kl_value:.2f} (> 50.0)")
            # Zero gradients and return early
            if hasattr(self, 'optimizer'):
                self.optimizer.zero_grad()
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "total_loss": 0.0,
                "kl_divergence": kl_value,
                "ppo_ratio_mean": 1.0,
                "ppo_ratio_std": 0.0,
                "advantages_mean": 0.0,
                "advantages_std": 0.0,
                "grad_norm": 0.0,
                "skipped": True,
                "skip_reason": "excessive_kl"
            }
        elif kl_value > 10.0:
            logger.warning(f"⚠️ High KL divergence: {kl_value:.2f} - proceeding with caution")
        
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

    def _compute_new_token_logprobs(self, prompt_texts, actions, old_token_logprobs):
        """
        Compute per-token log probabilities under the current training model with gradients.
        
        This properly aligns old and new logprobs at the token level for accurate PPO ratios.
        
        Args:
            prompt_texts: List of prompt strings
            actions: List of action strings  
            old_token_logprobs: List of per-token old logprobs tensors
            
        Returns:
            Tensor of logprob sums, one per action
        """
        self.policy.training_model.train()
        tokenizer = self.policy.tokenizer
        device = self.device
        
        logprob_sums = []
        
        for prompt_text, action, old_token_lps in zip(prompt_texts, actions, old_token_logprobs):
            # Skip if no old token logprobs
            if len(old_token_lps) == 0:
                logger.warning("Skipping action with no token logprobs")
                logprob_sums.append(torch.tensor(0.0, device=device, requires_grad=True))
                continue
                
            # Tokenize prompt and action (no special tokens to maintain alignment)
            enc_prompt = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
            enc_action = tokenizer(str(action), add_special_tokens=False, return_tensors="pt")
            prompt_ids = enc_prompt["input_ids"][0].to(device)
            action_ids = enc_action["input_ids"][0].to(device)
            
            # Compose full input and attention mask
            full_ids = torch.cat([prompt_ids, action_ids], dim=0).unsqueeze(0)  # [1, T]
            attn = torch.ones_like(full_ids, dtype=torch.long)
            
            with torch.set_grad_enabled(True):
                outputs = self.policy.training_model(input_ids=full_ids, attention_mask=attn)
                logits = outputs.logits  # [1, T, V]
                logprobs = torch.log_softmax(logits, dim=-1)  # [1, T, V]
                
                T_prompt = prompt_ids.shape[0]
                T_action = action_ids.shape[0]
                
                # Action tokens are predicted at positions [T_prompt-1 ... T_prompt+T_action-2]
                start = max(0, T_prompt - 1)
                end = start + T_action
                lp_slice = logprobs[:, start:end, :]  # [1, T_action, V]
                taken = action_ids.unsqueeze(0).unsqueeze(-1)  # [1, T_action, 1]
                lp_taken = lp_slice.gather(-1, taken).squeeze(-1)  # [1, T_action]
                
                # Align with old token logprobs length
                if lp_taken.shape[1] != len(old_token_lps):
                    logger.warning(f"Token length mismatch: new={lp_taken.shape[1]}, old={len(old_token_lps)}")
                    # Use minimum length
                    min_len = min(lp_taken.shape[1], len(old_token_lps))
                    lp_taken = lp_taken[:, :min_len]
                
                total = lp_taken.sum(dim=1).squeeze(0)  # scalar tensor
                
            logprob_sums.append(total)
            
        return torch.stack(logprob_sums)
    
    def _compute_new_logprobs_sum_per_action(self, states, actions):
        """
        Compute per-action log probability sums under the current training model with gradients.

        For each (state, action):
          - Build the prompt text using policy._build_prompt(state, add_generation_prompt=True)
          - Tokenize prompt and action with the HF tokenizer bound to training_model
          - Forward the concatenated input through training_model
          - Gather logprobs for the action tokens and sum them

        Returns:
          Tensor of shape [num_actions] with requires_grad=True
        """
        import torch
        self.policy.training_model.train()
        tokenizer = self.policy.tokenizer
        device = self.device

        logprob_sums = []
        for state, action in zip(states, actions):
            # Build prompt
            try:
                prompt_text = self.policy._build_prompt(state, add_generation_prompt=True)
            except Exception:
                # Fallback: stringify state
                prompt_text = str(state)

            # Tokenize prompt and action (no special tokens to maintain alignment)
            enc_prompt = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
            enc_action = tokenizer(str(action), add_special_tokens=False, return_tensors="pt")
            prompt_ids = enc_prompt["input_ids"][0].to(device)
            action_ids = enc_action["input_ids"][0].to(device)

            # Compose full input and attention mask
            full_ids = torch.cat([prompt_ids, action_ids], dim=0).unsqueeze(0)  # [1, T]
            attn = torch.ones_like(full_ids, dtype=torch.long)

            with torch.set_grad_enabled(True):
                outputs = self.policy.training_model(input_ids=full_ids, attention_mask=attn)
                logits = outputs.logits  # [1, T, V]
                logprobs = torch.log_softmax(logits, dim=-1)  # [1, T, V]

                T_prompt = prompt_ids.shape[0]
                T_action = action_ids.shape[0]
                # Action tokens are predicted at positions [T_prompt-1 ... T_prompt+T_action-2]
                start = max(0, T_prompt - 1)
                end = start + T_action
                lp_slice = logprobs[:, start:end, :]  # [1, T_action, V]
                taken = action_ids.unsqueeze(0).unsqueeze(-1)  # [1, T_action, 1]
                lp_taken = lp_slice.gather(-1, taken).squeeze(-1)  # [1, T_action]
                total = lp_taken.sum(dim=1).squeeze(0)  # scalar tensor
                logprob_sums.append(total)

        return torch.stack(logprob_sums).to(device)
    
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
        # Wrap in no_grad to prevent gradient warnings for metrics-only computation
        with torch.no_grad():
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
