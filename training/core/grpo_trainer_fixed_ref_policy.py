"""
Enhanced GRPO Trainer with Fixed Reference Policy Updates

This module ensures reference policy updates happen at appropriate intervals
and includes proper value head synchronization.
"""

import torch
import logging
from typing import Dict, Any
from .grpo_trainer_with_value import GRPOTrainerWithValue

logger = logging.getLogger(__name__)


class GRPOTrainerFixedRefPolicy(GRPOTrainerWithValue):
    """
    GRPO trainer with properly wired reference policy updates.
    
    Fixes:
    - Reference policy update frequency (default 100 steps instead of 10000)
    - Ensures value head is also updated in reference policy
    - Better logging and verification
    - Tracks KL divergence to monitor drift
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with better reference policy defaults"""
        super().__init__(*args, **kwargs)
        
        # Override default ref update frequency to something more reasonable
        if self.ref_update_freq == 10000:
            logger.warning(
                f"Default ref_update_freq of 10000 is too high. "
                f"Overriding to 100 steps."
            )
            self.ref_update_freq = 100
        
        # Track reference policy updates
        self.ref_updates_count = 0
        self.last_kl_before_update = 0.0
        
        # Ensure reference policy also has value head
        if hasattr(self.policy, 'value_head') and hasattr(self.reference_policy, 'value_head'):
            logger.info("Both policy and reference policy have value heads - will sync both")
        else:
            logger.warning("Value heads missing - reference policy updates may be incomplete")
        
        logger.info(f"Reference policy will be updated every {self.ref_update_freq} steps")
    
    def _update_reference_policy(self) -> None:
        """
        Update reference policy using exponential moving average.
        
        Enhanced to:
        - Update both model and value head parameters
        - Log KL divergence before/after update
        - Verify update actually happened
        """
        logger.info(f"Updating reference policy at step {self.step_count}")
        
        # Track parameters before update for verification
        ref_param_before = None
        if hasattr(self.reference_policy, 'model'):
            ref_param_before = next(self.reference_policy.model.parameters()).clone()
        
        # Update model parameters
        with torch.no_grad():
            # Update main model parameters
            for ref_param, current_param in zip(
                self.reference_policy.model.parameters(),
                self.policy.model.parameters()
            ):
                ref_param.data = (
                    self.ref_ema_alpha * ref_param.data +
                    (1 - self.ref_ema_alpha) * current_param.data
                )
            
            # Update value head parameters if present
            if hasattr(self.policy, 'value_head') and hasattr(self.reference_policy, 'value_head'):
                for ref_param, current_param in zip(
                    self.reference_policy.value_head.parameters(),
                    self.policy.value_head.parameters()
                ):
                    ref_param.data = (
                        self.ref_ema_alpha * ref_param.data +
                        (1 - self.ref_ema_alpha) * current_param.data
                    )
                logger.info("Updated reference policy value head")
        
        # Verify update happened
        if ref_param_before is not None:
            ref_param_after = next(self.reference_policy.model.parameters())
            param_change = (ref_param_after - ref_param_before).abs().mean().item()
            logger.info(f"Reference policy parameter change: {param_change:.6f}")
            
            if param_change < 1e-8:
                logger.warning("Reference policy parameters did not change!")
        
        self.ref_updates_count += 1
        logger.info(f"Reference policy update #{self.ref_updates_count} completed")
    
    def train_step(self, trajectories) -> Dict[str, float]:
        """
        Enhanced train step with proper reference policy updates and monitoring.
        """
        # Get metrics from parent
        metrics = super().train_step(trajectories)
        
        # Add reference policy tracking
        metrics["ref_updates_count"] = self.ref_updates_count
        metrics["steps_since_ref_update"] = self.step_count % self.ref_update_freq
        
        # Monitor KL divergence trend
        current_kl = metrics.get("kl_divergence", 0.0)
        
        # Check if KL is growing too large (emergency update)
        if current_kl > self.target_kl * 5 and self.step_count > 50:
            logger.warning(
                f"KL divergence ({current_kl:.4f}) exceeds 5x target "
                f"({self.target_kl * 5:.4f}). Forcing reference policy update."
            )
            self._update_reference_policy()
            metrics["emergency_ref_update"] = True
        
        return metrics
    
    def compute_kl_divergence_stats(self, trajectories) -> Dict[str, float]:
        """
        Compute detailed KL divergence statistics for monitoring.
        
        Returns dict with mean, std, max KL divergence across trajectories.
        """
        all_kl_divs = []
        
        with torch.no_grad():
            for traj in trajectories:
                if traj.states and traj.actions:
                    # Compute log probs under both policies
                    current_log_probs = self.policy.compute_log_probs(
                        traj.states, traj.actions
                    )
                    ref_log_probs = self.reference_policy.compute_log_probs(
                        traj.states, traj.actions
                    )
                    
                    # KL divergence for this trajectory
                    kl_div = (current_log_probs - ref_log_probs).mean()
                    all_kl_divs.append(kl_div.item())
        
        if all_kl_divs:
            return {
                "kl_mean": np.mean(all_kl_divs),
                "kl_std": np.std(all_kl_divs),
                "kl_max": np.max(all_kl_divs),
                "kl_min": np.min(all_kl_divs)
            }
        else:
            return {
                "kl_mean": 0.0,
                "kl_std": 0.0,
                "kl_max": 0.0,
                "kl_min": 0.0
            }
    
    def _should_update_reference_policy(self) -> bool:
        """
        Determine if reference policy should be updated.
        
        Enhanced logic that considers:
        - Regular update schedule
        - KL divergence threshold
        - Minimum steps before first update
        """
        # Don't update in first few steps
        if self.step_count < 10:
            return False
        
        # Regular scheduled update
        if self.step_count % self.ref_update_freq == 0:
            return True
        
        # Emergency update if KL too high (handled in train_step)
        return False
    
    def get_reference_policy_stats(self) -> Dict[str, Any]:
        """Get statistics about reference policy updates"""
        return {
            "ref_updates_count": self.ref_updates_count,
            "ref_update_freq": self.ref_update_freq,
            "ref_ema_alpha": self.ref_ema_alpha,
            "steps_since_update": self.step_count % self.ref_update_freq,
            "next_update_step": (
                (self.step_count // self.ref_update_freq + 1) * self.ref_update_freq
            )
        }


# Import numpy for stats
import numpy as np