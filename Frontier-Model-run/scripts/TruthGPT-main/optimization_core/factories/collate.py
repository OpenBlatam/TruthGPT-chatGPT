"""
Enhanced collate functions for efficient data processing.

Follows best practices for:
- Dynamic padding for efficient batch processing
- Proper tokenization handling
- Error handling and validation
"""
from typing import List, Dict, Any, Optional
import torch
import logging

from optimization_core.factories.registry import Registry

logger = logging.getLogger(__name__)

COLLATE = Registry()


@COLLATE.register("lm")
def build_lm_collate(tokenizer, max_length: int):
    """
    Build language modeling collate function with dynamic padding.
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Collate function that processes batches of text
    """
    def collate_fn(batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Collate function for language modeling with dynamic padding.
        
        Args:
            batch_texts: List of text strings to tokenize and batch
        
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        try:
            # Tokenize with truncation but no padding (we'll pad dynamically)
            tokens = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=False,  # We'll pad manually for efficiency
                return_tensors=None,  # Return lists, not tensors
                add_special_tokens=True,
            )
            
            # Get the actual lengths
            lengths = [len(t["input_ids"]) for t in tokens]
            
            # Find the maximum length in this batch (capped at max_length)
            batch_max_length = min(max(lengths), max_length)
            
            # Pad sequences dynamically
            padded_input_ids = []
            padded_attention_mask = []
            
            for i, tokenized in enumerate(tokens):
                input_ids = tokenized["input_ids"]
                
                # Truncate if needed (shouldn't happen, but safety check)
                if len(input_ids) > batch_max_length:
                    input_ids = input_ids[:batch_max_length]
                
                # Create attention mask
                attention_mask = [1] * len(input_ids)
                
                # Pad to batch_max_length
                pad_length = batch_max_length - len(input_ids)
                if pad_length > 0:
                    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                    input_ids = input_ids + [pad_token_id] * pad_length
                    attention_mask = attention_mask + [0] * pad_length
                
                padded_input_ids.append(input_ids)
                padded_attention_mask.append(attention_mask)
            
            # Convert to tensors
            input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
            attention_mask_tensor = torch.tensor(padded_attention_mask, dtype=torch.long)
            
            # Labels are same as input_ids for causal LM
            labels_tensor = input_ids_tensor.clone()
            
            return {
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor,
                "labels": labels_tensor,
            }
            
        except Exception as e:
            logger.error(f"Error in collate_fn: {e}", exc_info=True)
            # Return empty batch as fallback
            return {
                "input_ids": torch.empty((0, max_length), dtype=torch.long),
                "attention_mask": torch.empty((0, max_length), dtype=torch.long),
                "labels": torch.empty((0, max_length), dtype=torch.long),
            }
    
    return collate_fn


@COLLATE.register("cv")
def build_cv_collate():
    def collate_fn(samples):
        return samples
    return collate_fn




