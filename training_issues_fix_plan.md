# Training Issues Fix Plan - Mathematical Analysis Implementation

## Executive Summary
Based on the mathematical analysis, we have identified critical issues in our PPO/GRPO implementation that are causing training instability. This document outlines a comprehensive fix plan with specific implementation details.

## Issue 1: Forced Token Masking (Critical)

### Current Problem
- We mark **entire actions** as forced/unforced (binary per action)
- We should mask **individual tokens** within actions
- Schema tokens (`<tool_call>`, `{`, `}`, `"name":`, etc.) add zero-mean noise to gradients

### Mathematical Impact
- Forced tokens contribute: `E[g_t] = 0` (no learning signal)
- But add variance: `Var(g_t) = E[r_t² A_t² ||∇log π||²]`
- Result: Poor signal-to-noise ratio, slow learning

### Implementation Fix

#### Step 1: Create Token-Level Masking Function
```python
# In training/utils/token_masking.py (NEW FILE)
def create_token_mask(token_ids: List[int], tokenizer) -> torch.Tensor:
    """
    Create a mask where:
    - 1 = learnable token (model's choice)
    - 0 = forced token (schema/punctuation)
    """
    FORCED_TOKENS = {
        tokenizer.encode("<tool_call>", add_special_tokens=False),
        tokenizer.encode("</tool_call>", add_special_tokens=False),
        tokenizer.encode("{", add_special_tokens=False)[0],
        tokenizer.encode("}", add_special_tokens=False)[0],
        tokenizer.encode("[", add_special_tokens=False)[0],
        tokenizer.encode("]", add_special_tokens=False)[0],
        tokenizer.encode(":", add_special_tokens=False)[0],
        tokenizer.encode(",", add_special_tokens=False)[0],
        tokenizer.encode('"name"', add_special_tokens=False),
        tokenizer.encode('"arguments"', add_special_tokens=False),
    }
    
    mask = torch.ones(len(token_ids), dtype=torch.bool)
    for i, token_id in enumerate(token_ids):
        if token_id in FORCED_TOKENS:
            mask[i] = 0
    
    # Also mask common JSON structure patterns
    # e.g., consecutive punctuation, quotes around keys
    return mask
```

#### Step 2: Modify Policy to Return Token Masks
```python
# In training/scripts/train_qwen3_grpo_real_env_vllm.py
# Modify generate_action() method around line 400:

def generate_action(self, prompts, ...):
    # ... existing code ...
    
    # NEW: Collect per-token forced masks
    token_forced_masks = []
    
    for i, (response, output) in enumerate(zip(responses, outputs)):
        if is_valid_tool_call(response):
            # Get token IDs for the response
            token_ids = output.outputs[0].token_ids
            
            # Create token-level mask
            token_mask = create_token_mask(token_ids, self.tokenizer)
            token_forced_masks.append(token_mask)
            
            # Calculate unforced fraction
            unforced_fraction = token_mask.float().mean().item()
            logger.info(f"Token mask: {unforced_fraction:.1%} unforced tokens")
        else:
            # Fallback - all tokens are forced
            token_forced_masks.append(torch.zeros(len(token_ids), dtype=torch.bool))
    
    return {
        "actions": processed_responses,
        "token_logprobs": token_logprobs,
        "token_forced_masks": token_forced_masks,  # NEW
        # ...
    }
```

#### Step 3: Update Trainer to Use Token Masks
```python
# In training/core/grpo_trainer_gradient_fix.py, line ~265:

# Replace current forced_mask logic with:
def train_step(self, trajectories):
    # Collect token-level masks
    all_token_masks = []
    for traj in trajectories:
        if hasattr(traj, 'token_forced_masks'):
            # Concatenate all token masks for this trajectory
            traj_mask = torch.cat(traj.token_forced_masks)
            all_token_masks.append(traj_mask)
        else:
            # Fallback: assume all unforced
            all_token_masks.append(torch.ones(len(traj.old_log_probs), dtype=torch.bool))
    
    # Concatenate all masks
    unforced_mask = torch.cat(all_token_masks).to(self.device)
    
    # Log statistics
    unforced_fraction = unforced_mask.float().mean().item()
    logger.info(f"Token masking: {unforced_fraction:.1%} tokens are learnable")
```

## Issue 2: Token Alignment (OLD vs NEW)

### Current Problem
- Mean ratios ~0.02-0.10 suggest misalignment
- Expected: `E[r_t] ≈ exp(-H) ≈ 0.05` when tokens don't match
- We may have off-by-one errors or special token mismatches

### Mathematical Impact
- Misaligned ratios: `r̃_t = exp(log π_θ(a_t|s_t) - log π_old(b_t|s_t))` where `a_t ≠ b_t`
- Results in arbitrary ratios, killing gradient signal

### Implementation Fix

#### Step 1: Add Alignment Verification
```python
# In training/data/trajectory_collector.py, add method:

def verify_token_alignment(self, hf_tokens: List[int], vllm_tokens: List[int]) -> float:
    """
    Verify that HF and vLLM tokens are aligned.
    Returns alignment score (1.0 = perfect).
    """
    if len(hf_tokens) != len(vllm_tokens):
        logger.error(f"Token length mismatch: HF={len(hf_tokens)}, vLLM={len(vllm_tokens)}")
        return 0.0
    
    matches = sum(1 for h, v in zip(hf_tokens, vllm_tokens) if h == v)
    alignment = matches / len(hf_tokens)
    
    if alignment < 1.0:
        # Log mismatches for debugging
        for i, (h, v) in enumerate(zip(hf_tokens, vllm_tokens)):
            if h != v:
                h_str = self.tokenizer.decode([h])
                v_str = self.tokenizer.decode([v])
                logger.warning(f"Token mismatch at {i}: HF='{h_str}' vs vLLM='{v_str}'")
    
    return alignment
```

#### Step 2: Fix Tokenization Consistency
```python
# In training/scripts/train_qwen3_grpo_real_env_vllm.py
# Ensure consistent tokenization settings:

def compute_log_probs(self, sequences, ...):
    # CRITICAL: Use same tokenization settings
    inputs = self.tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,  # IMPORTANT: Must match vLLM
        return_attention_mask=True,
    )
    
    # Get action start position correctly
    prompt_length = len(self.tokenizer.encode(
        prompt, 
        add_special_tokens=False  # Must match above
    ))
    
    # Extract ONLY action tokens
    action_start = prompt_length  # No off-by-one!
    action_logprobs = logprobs[action_start:action_start + action_length]
```

## Issue 3: Diagnostic Metrics

### Mathematical Diagnostics Needed
From section 8 of the analysis, we need to track:
1. Active fraction: `ρ_active = #{r_t ∈ [1-ε, 1+ε]} / #{unforced}`
2. Clipped fraction: `f_clip = #{r_t ∉ [1-ε, 1+ε]} / #{unforced}`
3. Mean ratio by advantage sign
4. OLD-NEW alignment score
5. Masked vs unmasked KL

### Implementation

```python
# In training/utils/ppo_diagnostics.py (NEW FILE)

class PPODiagnostics:
    def __init__(self, clip_range=0.2):
        self.clip_range = clip_range
        self.metrics = {}
    
    def compute_diagnostics(self, ratios, advantages, unforced_mask, old_tokens, new_tokens):
        """Compute all diagnostic metrics from section 8."""
        
        # Only consider unforced tokens
        r = ratios[unforced_mask]
        a = advantages[unforced_mask]
        
        # 1. Active fraction (non-clipped)
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
        alignment = sum(1 for o, n in zip(old_tokens, new_tokens) if o == n) / len(old_tokens)
        self.metrics['token_alignment'] = alignment
        
        # 5. Per-token ratio distribution
        self.metrics['ratio_p10'] = torch.quantile(r, 0.1).item()
        self.metrics['ratio_p50'] = torch.quantile(r, 0.5).item()
        self.metrics['ratio_p90'] = torch.quantile(r, 0.9).item()
        
        # Warning thresholds
        if self.metrics['active_fraction'] < 0.5:
            logger.warning(f"⚠️ Low active fraction: {self.metrics['active_fraction']:.1%}")
        
        if self.metrics['token_alignment'] < 0.99:
            logger.error(f"❌ Token misalignment: {self.metrics['token_alignment']:.1%}")
        
        return self.metrics
```

## Issue 4: KL Regularization Optimization

### Current Settings (Too Loose)
- `kl_hard_cap: 100.0` (too high for production)
- `kl_penalty_coef: 0.01` (might be too low)
- Reference update every 100 steps (too infrequent early)

### Target Values (Mathematically Grounded)
- KL per token: 0.3-0.5 nats (normal operation)
- KL hard cap: 3.0 nats (emergency brake)
- Ratio bounds: [0.25, 4.0] → `|log r_t| ≤ 1.386`

### Implementation

```python
# In training/configs/grpo_config_fixed.yaml

# Optimized KL settings
kl_penalty_coef: 0.05              # Increased from 0.01
kl_hard_cap: 3.0                   # Reduced from 100.0 (production value)
kl_target: 0.4                     # Target KL per token

# Ratio gating (re-enable with proper bounds)
max_ratio_per_token: 4.0           # Re-enable (was 0)
min_ratio_per_token: 0.25          # New parameter

# Token requirements
min_unforced_tokens_per_step: 16   # Minimum learnable tokens
min_unforced_fraction_per_step: 0.3 # At least 30% unforced

# Reference policy updates
ref_policy_update_frequency: 20    # More frequent initially
ref_policy_ema_alpha: 0.99         # Slower EMA for stability
```

## Issue 5: Value Function Improvements

### Current Issue
- Value loss trends down but doesn't help policy
- Need better advantage estimation

### Implementation

```python
# In training/core/grpo_trainer_with_value.py

def compute_advantages_with_value(self, rewards, values, dones):
    """Compute GAE with proper value bootstrapping."""
    
    # Use GAE-Lambda for advantages
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0 if dones[t] else values[t + 1]
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + self.gamma * next_value - values[t]
        gae = delta + self.gamma * self.gae_lambda * gae * (1 - dones[t])
        advantages.insert(0, gae)
    
    advantages = torch.tensor(advantages)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages
```

## Issue 6: Bootstrap Mode Improvements

### Current Issue
- Bootstrap mode trains on ALL tokens when all are forced
- Should gradually transition to normal mode

### Implementation

```python
# In training/core/grpo_trainer_gradient_fix.py

def apply_bootstrap_decay(self, step):
    """Gradually reduce bootstrap mode influence."""
    
    # Decay bootstrap weight over first 100 steps
    bootstrap_weight = max(0.0, 1.0 - step / 100)
    
    if bootstrap_weight > 0 and all_tokens_forced:
        # Mix bootstrap loss with exploration bonus
        bootstrap_loss = -0.01 * current_log_probs.mean()  # Small BC term
        exploration_bonus = 0.001 * entropy  # Encourage diversity
        
        total_loss = (
            bootstrap_weight * bootstrap_loss + 
            (1 - bootstrap_weight) * ppo_loss +
            exploration_bonus
        )
```

## Testing Plan

### Phase 1: Unit Tests
1. Test token masking function with known examples
2. Verify alignment checker catches mismatches
3. Test diagnostic metrics computation

### Phase 2: Integration Tests
1. Run 10 training steps with new masking
2. Verify metrics show >30% unforced tokens
3. Check ratio distributions are centered around 1.0
4. Ensure KL stays under 3.0 nats

### Phase 3: Full Training Run
1. Train for 100 steps with all fixes
2. Monitor diagnostic dashboard
3. Verify policy loss decreases steadily
4. Check that model learns tool format

## Success Metrics

1. **Token Masking**: >30% tokens marked as learnable
2. **Alignment**: >99% token alignment score
3. **Ratios**: Mean ratio ∈ [0.8, 1.2], std < 0.5
4. **KL**: Per-token KL < 0.5 nats (normal), < 3.0 (always)
5. **Active Fraction**: >60% tokens in non-clipped region
6. **Policy Loss**: Steady decrease over 100 steps
7. **Tool Success**: Model generates valid tool calls by step 50

## Implementation Priority

1. **Critical (Do First)**:
   - Per-token forced masking
   - Token alignment verification
   - Diagnostic metrics

2. **Important (Do Second)**:
   - KL regularization tuning
   - Reference policy update frequency
   - Value function improvements

3. **Nice to Have (Do Third)**:
   - Bootstrap decay
   - Advanced ratio gating
   - Entropy bonuses

## Rollback Plan

If training becomes unstable:
1. Disable per-token masking (revert to action-level)
2. Increase KL hard cap to 10.0
3. Disable ratio gating
4. Reduce learning rate by 50%

## Monitoring Dashboard

Create real-time dashboard showing:
- Token masking distribution
- Ratio histogram with clip boundaries
- KL divergence over time
- Alignment scores per batch
- Active vs clipped fraction
- Policy and value losses

This comprehensive plan addresses all mathematical issues identified and provides clear implementation steps with fallback options.