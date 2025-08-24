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
    
    def _filter_state_dict_for_sync(self, state_dict):
        """
        CRITICAL FIX: Filter state dict to only include trainable LoRA and value head parameters.
        Excludes bitsandbytes quantization buffers that shouldn't be synced.
        """
        filtered = {}
        excluded_count = 0
        
        for key, value in state_dict.items():
            # Include LoRA adapter parameters
            if "lora_" in key:
                filtered[key] = value
            # Include value head parameters  
            elif "value_head" in key:
                filtered[key] = value
            # Exclude bitsandbytes quantization buffers
            elif any(pattern in key for pattern in [
                "nested_absmax", "nested_quant_map", "quant_state", 
                "bitsandbytes", "absmax", "quant_map", "SCB"
            ]):
                excluded_count += 1
                continue
            # Include other trainable parameters
            else:
                filtered[key] = value
        
        if excluded_count > 0:
            logger.info(f"Filtered out {excluded_count} quantization buffers from reference sync")
            
        return filtered
    
    def _update_reference_policy(self) -> None:
        """
        Update reference policy by copying the full policy state.
        Includes LoRA adapter weights if present and EMA on value head.
        """
        logger.info(f"Updating reference policy at step {self.step_count}")

        with torch.no_grad():
            # 1) Copy model state but FILTER OUT BnB quantization buffers
            src = self.policy.model
            ref = self.reference_policy.model
            from collections import OrderedDict
            src_sd = OrderedDict(src.state_dict())
            try:
                from peft import get_peft_model_state_dict  # type: ignore
                src_sd.update(get_peft_model_state_dict(src))
            except Exception:
                pass
            
            # CRITICAL FIX: Filter out bitsandbytes quantization buffers 
            filtered_sd = self._filter_state_dict_for_sync(src_sd)
            
            missing, unexpected = ref.load_state_dict(filtered_sd, strict=False)
            if missing or unexpected:
                logger.debug(f"Ref load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
                if unexpected:
                    logger.debug(f"Filtered out {len(src_sd) - len(filtered_sd)} quantization buffers")

            # 2) Copy value head with EMA if present
            if hasattr(self.policy, "value_head") and hasattr(self.reference_policy, "value_head"):
                ref_v = self.reference_policy.value_head.state_dict()
                cur_v = self.policy.value_head.state_dict()
                for k in ref_v:
                    if k in cur_v:
                        # Ensure both tensors are on the same device before EMA update
                        cur_tensor = cur_v[k].to(ref_v[k].device)
                        ref_v[k].copy_(self.ref_ema_alpha * ref_v[k] + (1 - self.ref_ema_alpha) * cur_tensor)
                self.reference_policy.value_head.load_state_dict(ref_v, strict=False)
                logger.info("Updated reference policy value head (EMA)")

            # 3) Freeze reference policy parameters
            for p in self.reference_policy.parameters():
                p.requires_grad_(False)

        # 4) Verify copy by counting differing tensors (should be 0 right after copy)
        diffs = 0
        with torch.no_grad():
            cur_params = dict(self.policy.model.named_parameters())
            for name, p_ref in self.reference_policy.model.named_parameters():
                p_cur = cur_params.get(name)
                if p_cur is not None:
                    # Ensure both tensors are on the same device for comparison
                    try:
                        if not torch.allclose(p_ref.to(p_cur.device), p_cur, atol=1e-6):
                            diffs += 1
                    except Exception:
                        # Shape mismatch or other issues - count as different
                        diffs += 1
        logger.info(f"Reference sync done — tensors_differ={diffs} (may be > 0 due to LoRA/quantization differences)")

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