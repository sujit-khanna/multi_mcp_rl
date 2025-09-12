# training/utils/ppo_diagnostics.py
import torch
import math
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PPODiagnostics:
    """Comprehensive PPO diagnostics from mathematical analysis section 8."""
    
    def __init__(self, clip_range: float = 0.2):
        self.clip_range = clip_range
        self.metrics: Dict[str, float] = {}
        
    def compute_diagnostics(
        self, 
        ratios: torch.Tensor,
        advantages: torch.Tensor, 
        unforced_mask: torch.Tensor,
        old_token_ids: Optional[torch.Tensor] = None,
        new_token_ids: Optional[torch.Tensor] = None,
        kl_per_token: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute all diagnostic metrics from mathematical analysis section 8.
        
        Args:
            ratios: PPO ratios (exp(new_logprobs - old_logprobs))
            advantages: Per-token advantages
            unforced_mask: Boolean mask for learnable tokens
            old_token_ids: Token IDs from sampling time (for alignment check)
            new_token_ids: Token IDs from current policy (for alignment check)
            kl_per_token: Per-token KL contributions (optional)
        """
        self.metrics = {}
        
        if not unforced_mask.any():
            logger.warning("No unforced tokens available for diagnostics")
            return self._empty_metrics()
        
        # Only consider unforced tokens
        r = ratios[unforced_mask]
        a = advantages[unforced_mask]
        
        if r.numel() == 0:
            return self._empty_metrics()
        
        # 1. Active fraction (non-clipped tokens)
        active_mask = (r >= 1 - self.clip_range) & (r <= 1 + self.clip_range)
        self.metrics['active_fraction'] = active_mask.float().mean().item()
        
        # 2. Clipped fraction
        self.metrics['clipped_fraction'] = 1.0 - self.metrics['active_fraction']
        
        # 3. Mean ratio by advantage sign
        pos_mask = a > 0
        neg_mask = a < 0
        self.metrics['ratio_mean_positive_adv'] = r[pos_mask].mean().item() if pos_mask.any() else 1.0
        self.metrics['ratio_mean_negative_adv'] = r[neg_mask].mean().item() if neg_mask.any() else 1.0
        
        # 4. Token alignment check
        if old_token_ids is not None and new_token_ids is not None:
            min_len = min(len(old_token_ids), len(new_token_ids))
            if min_len > 0:
                matches = sum(1 for o, n in zip(old_token_ids[:min_len], new_token_ids[:min_len]) if o == n)
                self.metrics['token_alignment'] = matches / min_len
            else:
                self.metrics['token_alignment'] = 0.0
        else:
            self.metrics['token_alignment'] = 1.0  # Assume aligned if not provided
        
        # 5. Log-ratio statistics (more stable than raw ratios)
        log_r = torch.log(r.clamp_min(1e-8))
        self.metrics['log_ratio_mean'] = log_r.mean().item()
        self.metrics['log_ratio_std'] = log_r.std().item()
        
        # 6. Per-token ratio distribution
        self.metrics['ratio_p10'] = torch.quantile(r, 0.1).item()
        self.metrics['ratio_p50'] = torch.quantile(r, 0.5).item()  
        self.metrics['ratio_p90'] = torch.quantile(r, 0.9).item()
        self.metrics['ratio_mean'] = r.mean().item()
        self.metrics['ratio_std'] = r.std().item()
        
        # 7. Advantage distribution
        self.metrics['advantage_mean'] = a.mean().item()
        self.metrics['advantage_std'] = a.std().item()
        self.metrics['advantage_p10'] = torch.quantile(a, 0.1).item()
        self.metrics['advantage_p90'] = torch.quantile(a, 0.9).item()
        
        # 8. Masked vs unmasked KL (if provided)
        if kl_per_token is not None:
            self.metrics['kl_masked_mean'] = kl_per_token[unforced_mask].mean().item()
            self.metrics['kl_unmasked_mean'] = kl_per_token.mean().item()
            self.metrics['kl_schema_overhead'] = self.metrics['kl_unmasked_mean'] - self.metrics['kl_masked_mean']
        
        # 9. Masking statistics
        total_tokens = len(unforced_mask)
        unforced_tokens = unforced_mask.sum().item()
        self.metrics['unforced_fraction'] = unforced_tokens / total_tokens if total_tokens > 0 else 0.0
        self.metrics['unforced_count'] = unforced_tokens
        self.metrics['forced_count'] = total_tokens - unforced_tokens
        
        # Log warnings for problematic conditions
        self._log_warnings()
        
        return self.metrics
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics when no valid data is available."""
        return {
            'active_fraction': 0.0,
            'clipped_fraction': 1.0,
            'ratio_mean_positive_adv': 1.0,
            'ratio_mean_negative_adv': 1.0,
            'token_alignment': 0.0,
            'log_ratio_mean': 0.0,
            'log_ratio_std': 0.0,
            'ratio_p10': 1.0,
            'ratio_p50': 1.0,
            'ratio_p90': 1.0,
            'ratio_mean': 1.0,
            'ratio_std': 0.0,
            'advantage_mean': 0.0,
            'advantage_std': 0.0,
            'unforced_fraction': 0.0,
            'unforced_count': 0,
            'forced_count': 0,
        }
    
    def _log_warnings(self):
        """Log warnings for problematic diagnostic values."""
        
        # Low active fraction
        if self.metrics['active_fraction'] < 0.5:
            logger.warning(f"⚠️ Low active fraction: {self.metrics['active_fraction']:.1%} "
                          f"(many tokens clipped, poor gradient signal)")
        
        # Token misalignment
        if self.metrics['token_alignment'] < 0.99:
            logger.error(f"❌ Token misalignment: {self.metrics['token_alignment']:.1%} "
                        f"(OLD-NEW tokens don't match)")
        
        # Extreme ratios
        if self.metrics['ratio_p10'] < 0.1 or self.metrics['ratio_p90'] > 10.0:
            logger.warning(f"⚠️ Extreme ratios detected: p10={self.metrics['ratio_p10']:.3f}, "
                          f"p90={self.metrics['ratio_p90']:.3f}")
        
        # Low unforced fraction
        if self.metrics['unforced_fraction'] < 0.3:
            logger.warning(f"⚠️ Low unforced fraction: {self.metrics['unforced_fraction']:.1%} "
                          f"(training on mostly schema tokens)")
        
        # Degenerate ratios
        if self.metrics['ratio_std'] < 1e-6:
            logger.error(f"❌ Degenerate ratios: std={self.metrics['ratio_std']:.2e} "
                        f"(no policy change detected)")
        
        # High KL schema overhead
        if 'kl_schema_overhead' in self.metrics and self.metrics['kl_schema_overhead'] > 1.0:
            logger.warning(f"⚠️ High KL schema overhead: {self.metrics['kl_schema_overhead']:.3f} "
                          f"(wasting capacity on formatting tokens)")
    
    def log_summary(self):
        """Log a concise summary of key diagnostics."""
        logger.info(f"PPO window — active={self.metrics.get('active_fraction', 0.0):.1%}, "
                   f"clipped={self.metrics.get('clipped_fraction', 1.0):.1%}, "
                   f"unforced={self.metrics.get('unforced_fraction', 0.0):.1%}")
        
        logger.info(f"Alignment — HF action ids == vLLM action ids ? "
                   f"{self.metrics.get('token_alignment', 0.0) >= 0.99}, "
                   f"score={self.metrics.get('token_alignment', 0.0):.1%}")
        
        logger.info(f"Ratios — mean={self.metrics.get('ratio_mean', 1.0):.3f}, "
                   f"std={self.metrics.get('ratio_std', 0.0):.3f}, "
                   f"range=[{self.metrics.get('ratio_p10', 1.0):.3f}, "
                   f"{self.metrics.get('ratio_p90', 1.0):.3f}]")

def normalize_advantages_masked(advantages: torch.Tensor, unforced_mask: torch.Tensor) -> torch.Tensor:
    """
    Normalize advantages using only unforced tokens for statistics.
    This prevents schema tokens from diluting the advantage statistics.
    """
    if not unforced_mask.any():
        logger.warning("No unforced tokens for advantage normalization")
        return advantages
    
    # Compute statistics only on unforced tokens
    adv_masked = advantages[unforced_mask]
    mean_masked = adv_masked.mean()
    std_masked = adv_masked.std()
    
    # Apply normalization to all advantages
    advantages_norm = (advantages - mean_masked) / (std_masked + 1e-8)
    
    logger.debug(f"Advantage normalization: masked_mean={mean_masked:.4f}, "
                f"masked_std={std_masked:.4f}, unforced_frac={unforced_mask.float().mean():.1%}")
    
    return advantages_norm