"""
Collator implementations for data batching.
"""
from typing import Dict, List, Any
import torch
from transformers import PreTrainedTokenizer


class LMCollator:
    """
    Language modeling collator with dynamic padding.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        """
        Initialize collator.
        
        Args:
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of texts.
        
        Args:
            batch: List of text strings
        
        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        # Tokenize with dynamic padding
        encoded = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Create labels (same as input_ids for LM)
        labels = encoded["input_ids"].clone()
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }



