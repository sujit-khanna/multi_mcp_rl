"""
Enhanced GRPO Trainer with Value Function Training

This module extends the base GRPO trainer to include proper value function training,
addressing the critical missing component identified in the code review.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any
from .grpo_trainer import GRPOTrainer, Trajectory

logger = logging.getLogger(__name__)


class GRPOTrainerWithValue(GRPOTrainer):
    """
    Enhanced GRPO trainer that includes value function training.
    
    This addresses the critical gap where advantages were computed using values
    but the value function was never trained, leading to high-variance gradients.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize enhanced trainer with value loss coefficient"""
        super().__init__(*args, **kwargs)
        
        # Value loss configuration
        self.value_loss_coef = self.grpo_config.get("value_loss_coef", 0.5)
        self.clip_value_loss = self.grpo_config.get("clip_value_loss", True)
        self.value_clip_range = self.grpo_config.get("value_clip_range", 0.2)
        
        # Ensure we're using a policy with value head
        if not hasattr(self.policy, 'compute_values'):
            raise ValueError(
                "Policy must have compute_values method. "
                "Use QwenPolicyWithValue instead of QwenPolicy."
            )
        
        # Add value head parameters to optimizer if not already included
        self._ensure_value_head_in_optimizer()
        
        logger.info(f"Enhanced GRPO trainer initialized with value_loss_coef={self.value_loss_coef}")
    
    def _ensure_value_head_in_optimizer(self):
        """Ensure value head parameters are included in optimizer"""
        if hasattr(self.policy, 'value_head'):
            # Get current optimizer param groups
            current_params = []
            for group in self.optimizer.param_groups:
                current_params.extend(group['params'])
            
            # Check if value head params are already included
            value_head_params = list(self.policy.value_head.parameters())
            missing_params = []
            
            for param in value_head_params:
                if param.requires_grad and not any(p is param for p in current_params):
                    missing_params.append(param)
            
            if missing_params:
                # Add missing value head parameters to optimizer
                self.optimizer.add_param_group({
                    'params': missing_params,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'betas': self.optimizer.param_groups[0].get('betas', (0.9, 0.999)),
                    'eps': self.optimizer.param_groups[0].get('eps', 1e-8),
                    'weight_decay': self.optimizer.param_groups[0].get('weight_decay', 0.01),
                })
                logger.info(f"Added {len(missing_params)} value head parameters to optimizer")
    
    def _compute_advantages(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Compute advantages using trained value function.
        
        Override parent method to ensure we use actual value estimates
        instead of zeros.
        """
        rewards = torch.tensor(trajectory.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(trajectory.dones, dtype=torch.bool, device=self.device)
        
        # Compute values using the trained value function
        with torch.no_grad():
            values = self.policy.compute_values(trajectory.states)
            
            # Add numerical stability check for values
            if torch.isnan(values).any() or torch.isinf(values).any():
                print(f"WARNING: NaN/Inf detected in values: {values}")
                # Return simple advantages based on rewards only
                logger.warning("Using reward-based advantages due to NaN values")
                rewards = torch.tensor(trajectory.rewards, dtype=torch.float32, device=self.device)
                # Simple advantage: reward - baseline (mean reward)
                baseline = rewards.mean() if len(rewards) > 1 else 0.5
                advantages = rewards - baseline
                trajectory.values = torch.full_like(rewards, baseline)
                trajectory.returns = rewards
                return advantages
        
        # Store values in trajectory for later use in value loss
        trajectory.values = values
        
        # Compute returns for value loss
        returns = torch.zeros_like(rewards)
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t].float())
        
        # Store returns for value loss computation
        trajectory.returns = returns
        
        # Compute GAE advantages
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t].float()) - values[t]
            
            # Check for numerical issues in delta
            if torch.isnan(delta) or torch.isinf(delta):
                delta = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t].float()) * gae
            
            # Check for numerical issues in gae
            if torch.isnan(gae) or torch.isinf(gae):
                gae = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                
            advantages[t] = gae
        
        # Normalize advantages with additional numerical stability
        if self.normalize_advantages and len(advantages) > 1:
            mean_adv = advantages.mean()
            std_adv = advantages.std()
            
            # Check for numerical issues in normalization
            if torch.isnan(mean_adv) or torch.isinf(mean_adv):
                mean_adv = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            if torch.isnan(std_adv) or torch.isinf(std_adv) or std_adv < self.advantage_epsilon:
                std_adv = torch.tensor(1.0, device=self.device, dtype=torch.float32)
                
            advantages = (advantages - mean_adv) / (std_adv + self.advantage_epsilon)
            
            # Final check for normalized advantages
            advantages = torch.nan_to_num(advantages, nan=0.0, posinf=5.0, neginf=-5.0)
        
        return advantages
    
    def _compute_value_loss(
        self,
        states: List[List[Dict[str, str]]],
        returns: torch.Tensor,
        old_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute value function loss with optional PPO-style clipping.
        
        Args:
            states: List of conversation states
            returns: Target returns computed from rewards
            old_values: Previous value estimates for clipping (optional)
            
        Returns:
            Value loss tensor
        """
        # Get current value predictions
        current_values = self.policy.compute_values(states)
        
        # Add numerical stability checks
        if torch.isnan(current_values).any() or torch.isinf(current_values).any():
            logger.warning(f"NaN/Inf in current_values: {current_values}")
            current_values = torch.nan_to_num(current_values, nan=0.0, posinf=10.0, neginf=-10.0)
            
        if torch.isnan(returns).any() or torch.isinf(returns).any():
            logger.warning(f"NaN/Inf in returns: {returns}")
            returns = torch.nan_to_num(returns, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Basic MSE loss with numerical stability
        value_loss = F.mse_loss(current_values, returns)
        
        # Additional stability check on the loss itself
        if torch.isnan(value_loss) or torch.isinf(value_loss):
            logger.warning(f"NaN/Inf in value_loss, setting to 0.0")
            value_loss = torch.tensor(0.0, device=current_values.device, dtype=torch.float32)
        
        # Optional: PPO-style value clipping
        if self.clip_value_loss and old_values is not None:
            # Clipped value predictions
            value_pred_clipped = old_values + torch.clamp(
                current_values - old_values,
                -self.value_clip_range,
                self.value_clip_range
            )
            value_loss_clipped = F.mse_loss(value_pred_clipped, returns)
            
            # Take maximum of clipped and unclipped loss
            value_loss = torch.max(value_loss, value_loss_clipped)
        
        return value_loss
    
    def _update_policy(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """
        Update policy and value function using GRPO objective.
        
        Enhanced to include value function training alongside policy updates.
        """
        logger.info(f"Enhanced _update_policy called with {len(trajectories)} trajectories")
        
        if not trajectories:
            logger.warning("No trajectories provided to _update_policy")
            return {}
        
        try:
            # Collect all data including returns
            all_states = []
            all_actions = []
            all_advantages = []
            all_old_log_probs = []
            all_returns = []
            all_old_values = []
            
            for traj in trajectories:
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
                
                # Collect returns and old values for value loss
                if hasattr(traj, 'returns') and traj.returns is not None:
                    all_returns.extend(traj.returns.tolist())
                else:
                    # Fallback: compute returns if not available
                    rewards = torch.tensor(traj.rewards, dtype=torch.float32, device=self.device)
                    dones = torch.tensor(traj.dones, dtype=torch.bool, device=self.device)
                    returns = torch.zeros_like(rewards)
                    returns[-1] = rewards[-1]
                    for t in reversed(range(len(rewards) - 1)):
                        returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t].float())
                    all_returns.extend(returns.tolist())
                
                if traj.values is not None:
                    all_old_values.extend(traj.values.tolist())
                else:
                    all_old_values.extend([0.0] * len(traj.actions))
            
            # Convert to tensors
            advantages = torch.tensor(all_advantages, dtype=torch.float32, device=self.device)
            old_log_probs = torch.tensor(all_old_log_probs, dtype=torch.float32, device=self.device)
            returns = torch.tensor(all_returns, dtype=torch.float32, device=self.device)
            old_values = torch.tensor(all_old_values, dtype=torch.float32, device=self.device) if all_old_values else None
            
            # Compute current policy log probabilities and values efficiently
            logger.info("Computing current policy outputs...")
            if hasattr(self.policy, 'compute_values_and_log_probs'):
                # Use combined method if available for efficiency
                current_values, current_log_probs = self.policy.compute_values_and_log_probs(
                    all_states, all_actions
                )
            else:
                # Compute separately
                current_log_probs = self.policy.compute_log_probs(all_states, all_actions)
                current_values = self.policy.compute_values(all_states)
            
            # Compute reference policy log probabilities for KL penalty
            with torch.no_grad():
                ref_log_probs = self.reference_policy.compute_log_probs(all_states, all_actions)
            
            # === POLICY LOSS ===
            # Compute policy ratios
            log_ratios = current_log_probs - old_log_probs
            ratios = torch.exp(log_ratios)
            
            # Clamp ratios for numerical stability (less aggressive than original)
            ratios = torch.clamp(ratios, 1e-8, 1e8)
            
            # Compute clipped policy loss (PPO-style)
            policy_loss_unclipped = -advantages * ratios
            policy_loss_clipped = -advantages * torch.clamp(
                ratios, 1 - self.clip_ratio, 1 + self.clip_ratio
            )
            policy_loss = torch.mean(torch.max(policy_loss_unclipped, policy_loss_clipped))
            
            # === VALUE LOSS ===
            value_loss = self._compute_value_loss(all_states, returns, old_values)
            
            # === KL DIVERGENCE ===
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
            
            # === ENTROPY REGULARIZATION ===
            entropy_loss = 0.0
            if self.entropy_coef > 0:
                entropy = -current_log_probs.mean()
                entropy_loss = -self.entropy_coef * entropy
            
            # === TOTAL LOSS ===
            total_loss = policy_loss + self.value_loss_coef * value_loss + kl_penalty + entropy_loss
            
            # Ensure total_loss requires gradients
            if not total_loss.requires_grad:
                logger.error("total_loss does not require gradients!")
                raise RuntimeError("Loss tensor does not require gradients")
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.grpo_config.get("gradient_clipping_enabled", True):
                clip_value = self.grpo_config.get("gradient_clipping_value", 1.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.policy.model.parameters()) + list(self.policy.value_head.parameters()),
                    max_norm=clip_value
                )
            else:
                grad_norm = 0.0
            
            # Optimizer step
            self.optimizer.step()
            
            # Compile metrics
            metrics = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "kl_divergence": kl_divergence.item(),
                "kl_penalty": kl_penalty.item(),
                "total_loss": total_loss.item(),
                "avg_ratio": ratios.mean().item(),
                "max_ratio": ratios.max().item(),
                "min_ratio": ratios.min().item(),
                "avg_advantage": advantages.mean().item(),
                "std_advantage": advantages.std().item(),
                "avg_value": current_values.mean().item(),
                "avg_return": returns.mean().item(),
                "value_error": (current_values - returns).abs().mean().item(),
                "kl_coef": kl_coef,
                "grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
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
            
        except Exception as e:
            logger.error(f"Error in enhanced _update_policy: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def train_step(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """
        Enhanced train step that ensures reference policy updates.
        
        Overrides parent to ensure reference policy is updated at correct intervals.
        """
        # Call parent train_step
        metrics = super().train_step(trajectories)
        
        # Ensure reference policy update happens (parent might not call it)
        if self.step_count % self.ref_update_freq == 0 and self.step_count > 0:
            self._update_reference_policy()
            logger.info(f"Reference policy updated at step {self.step_count}")
            metrics["reference_policy_updated"] = True
        
        return metrics