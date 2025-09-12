# training/utils/token_masking.py
from __future__ import annotations
import re
from typing import List, Tuple
import torch
from transformers import PreTrainedTokenizerBase
import logging

logger = logging.getLogger(__name__)

JSON_PUNCT_CHARS = set('{}[]:,')
# Regex for JSON keys like "name": or "arguments":
KEY_RE = re.compile(r'"[^"]*"\s*(?=:)')

def _schema_char_spans(text: str) -> List[Tuple[int, int]]:
    """Identify character spans that correspond to schema/forced tokens."""
    spans: List[Tuple[int, int]] = []
    
    # Tool-call tags
    for m in re.finditer(r'</?tool_call>', text):
        spans.append((m.start(), m.end()))
    
    # JSON keys (include the quotes, exclude the colon so the colon is its own span)
    for m in KEY_RE.finditer(text):
        spans.append((m.start(), m.end()))
    
    # JSON punctuation as 1-char spans
    for i, ch in enumerate(text):
        if ch in JSON_PUNCT_CHARS:
            spans.append((i, i + 1))
    
    # Merge overlapping spans
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    
    return merged

def create_token_mask_from_text(
    action_text: str,
    action_token_ids: List[int],
    tokenizer: PreTrainedTokenizerBase,
) -> torch.Tensor:
    """
    Returns a boolean mask of length len(action_token_ids):
      True  = learnable (unforced)
      False = forced (schema/punctuation/tool tags)
    """
    # Prefer fast tokenizers to get offsets
    if not getattr(tokenizer, "is_fast", False):
        # Fallback: approximate by re-encoding and aligning lengths
        # (still safer than raw ID == lookup because of spaces)
        try:
            enc = tokenizer(
                action_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
            if enc.input_ids.shape[-1] != len(action_token_ids):
                # Last-resort: mark everything unforced to avoid training on wrong mask
                logger.warning(f"Token count mismatch in fallback masking: {enc.input_ids.shape[-1]} vs {len(action_token_ids)}")
                return torch.ones(len(action_token_ids), dtype=torch.bool)
            
            # Without offsets, we can only mask obvious 1-char punctuation
            mask = torch.tensor(
                [tokenizer.decode([tid]).strip() not in JSON_PUNCT_CHARS for tid in action_token_ids],
                dtype=torch.bool,
            )
            logger.debug(f"Fallback masking: {mask.sum().item()}/{len(mask)} tokens unforced")
            return mask
        except Exception as e:
            logger.error(f"Fallback masking failed: {e}")
            return torch.ones(len(action_token_ids), dtype=torch.bool)

    try:
        enc = tokenizer(
            action_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        ids = enc.input_ids[0].tolist()
        offs = enc.offset_mapping[0].tolist()  # List[(start_char, end_char)]
        
        # Safety: require identical tokenization to vLLM action ids
        if ids != list(action_token_ids):
            # If mismatch, bail to avoid poisoning the mask
            logger.warning(f"Token ID mismatch: HF={len(ids)} vs vLLM={len(action_token_ids)}")
            logger.debug(f"HF IDs: {ids[:10]}... vs vLLM IDs: {list(action_token_ids)[:10]}...")
            return torch.ones(len(action_token_ids), dtype=torch.bool)

        schema_spans = _schema_char_spans(action_text)
        
        def is_forced_token(start: int, end: int) -> bool:
            # A token is forced if its entire char span is inside ANY schema span
            for s, e in schema_spans:
                if start >= s and end <= e:
                    return True
            return False

        mask = []
        for (start, end) in offs:
            # Some tokenizers may return (0, 0) for special/unk; treat them as forced
            if start == end:
                mask.append(False)
            else:
                mask.append(not is_forced_token(start, end))
        
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        unforced_count = mask_tensor.sum().item()
        total_count = len(mask_tensor)
        
        logger.debug(f"Token masking: {unforced_count}/{total_count} ({unforced_count/total_count:.1%}) tokens unforced")
        logger.debug(f"Schema spans: {schema_spans}")
        
        return mask_tensor
        
    except Exception as e:
        logger.error(f"Token masking failed: {e}")
        # Fallback to all unforced to avoid breaking training
        return torch.ones(len(action_token_ids), dtype=torch.bool)

def log_mask_stats(mask: torch.Tensor, action_text: str, logger_instance=None):
    """Log detailed statistics about the token mask."""
    if logger_instance is None:
        logger_instance = logger
        
    unforced_count = mask.sum().item()
    total_count = len(mask)
    forced_count = total_count - unforced_count
    
    logger_instance.info(f"Mask stats — kept={unforced_count}, dropped={forced_count}, "
                        f"total={total_count} (unforced={unforced_count/total_count:.1%})")
    
    # Log which parts are forced for debugging
    if logger_instance.isEnabledFor(logging.DEBUG):
        schema_spans = _schema_char_spans(action_text)
        logger_instance.debug(f"Forced spans in '{action_text}': {schema_spans}")