"""
Relative Positional Embeddings for TruthGPT
Implements relative positional encoding as described in "Self-Attention with Relative Position Representations"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding implementation.
    
    This implementation follows the "Self-Attention with Relative Position Representations"
    paper and provides relative positional information to the attention mechanism.
    """
    
    def __init__(
        self, 
        d_model: int,
        max_relative_position: int = 32,
        num_buckets: int = 32,
        bidirectional: bool = True
    ):
        """
        Initialize relative positional encoding.
        
        Args:
            d_model: Model dimension
            max_relative_position: Maximum relative position
            num_buckets: Number of buckets for relative positions
            bidirectional: Whether to use bidirectional relative positions
        """
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        self.num_buckets = num_buckets
        self.bidirectional = bidirectional
        
        # Create relative position embeddings
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)
        
        # Initialize weights
        nn.init.normal_(self.relative_attention_bias.weight, std=0.02)
    
    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """
        Compute relative position buckets.
        
        Args:
            relative_position: Relative position tensor
            
        Returns:
            Bucket indices for relative positions
        """
        num_buckets = self.num_buckets
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        # For small relative positions, use exact buckets
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) / 
            math.log(self.max_relative_position / max_exact) * 
            (num_buckets - max_exact)
        ).int()
        
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        return torch.where(is_small, relative_position, relative_position_if_large)
    
    def forward(
        self, 
        query_length: int, 
        key_length: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute relative position bias.
        
        Args:
            query_length: Length of query sequence
            key_length: Length of key sequence
            device: Device to place tensors on
            
        Returns:
            Relative position bias tensor of shape (1, 1, query_length, key_length)
        """
        # Create relative position matrix
        context_position = torch.arange(query_length, device=device)[:, None]
        memory_position = torch.arange(key_length, device=device)[None, :]
        relative_position = memory_position - context_position
        
        # Compute relative position buckets
        relative_position_bucket = self._relative_position_bucket(relative_position)
        
        # Get relative attention bias
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        
        return values

class T5RelativePositionalEncoding(RelativePositionalEncoding):
    """
    T5-style relative positional encoding.
    
    This follows the T5 model's implementation of relative positional encoding.
    """
    
    def __init__(
        self, 
        d_model: int,
        max_relative_position: int = 32,
        num_buckets: int = 32,
        bidirectional: bool = True
    ):
        """Initialize T5-style relative positional encoding."""
        super().__init__(d_model, max_relative_position, num_buckets, bidirectional)
    
    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """T5-style relative position bucket computation."""
        num_buckets = self.num_buckets
        max_exact = num_buckets // 2
        
        # For small relative positions, use exact buckets
        is_small = relative_position < max_exact
        
        # For large relative positions, use logarithmic buckets
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) / 
            math.log(self.max_relative_position / max_exact) * 
            (num_buckets - max_exact)
        ).int()
        
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        return torch.where(is_small, relative_position, relative_position_if_large)

class DeBERTaRelativePositionalEncoding(RelativePositionalEncoding):
    """
    DeBERTa-style relative positional encoding.
    
    This follows the DeBERTa model's implementation of relative positional encoding.
    """
    
    def __init__(
        self, 
        d_model: int,
        max_relative_position: int = 32,
        num_buckets: int = 32,
        bidirectional: bool = True
    ):
        """Initialize DeBERTa-style relative positional encoding."""
        super().__init__(d_model, max_relative_position, num_buckets, bidirectional)
    
    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """DeBERTa-style relative position bucket computation."""
        num_buckets = self.num_buckets
        max_exact = num_buckets // 2
        
        # For small relative positions, use exact buckets
        is_small = relative_position < max_exact
        
        # For large relative positions, use logarithmic buckets
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) / 
            math.log(self.max_relative_position / max_exact) * 
            (num_buckets - max_exact)
        ).int()
        
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        return torch.where(is_small, relative_position, relative_position_if_large)

# Factory functions
def create_relative_embedding(
    d_model: int,
    max_relative_position: int = 32,
    num_buckets: int = 32,
    bidirectional: bool = True,
    style: str = "standard"
) -> RelativePositionalEncoding:
    """
    Create a relative positional encoding instance.
    
    Args:
        d_model: Model dimension
        max_relative_position: Maximum relative position
        num_buckets: Number of buckets for relative positions
        bidirectional: Whether to use bidirectional relative positions
        style: Style of relative encoding ("standard", "t5", "deberta")
        
    Returns:
        Relative positional encoding instance
    """
    if style == "standard":
        return RelativePositionalEncoding(d_model, max_relative_position, num_buckets, bidirectional)
    elif style == "t5":
        return T5RelativePositionalEncoding(d_model, max_relative_position, num_buckets, bidirectional)
    elif style == "deberta":
        return DeBERTaRelativePositionalEncoding(d_model, max_relative_position, num_buckets, bidirectional)
    else:
        raise ValueError(f"Unsupported relative encoding style: {style}")

def create_t5_relative_embedding(
    d_model: int,
    max_relative_position: int = 32,
    num_buckets: int = 32,
    bidirectional: bool = True
) -> T5RelativePositionalEncoding:
    """Create a T5-style relative positional encoding instance."""
    return T5RelativePositionalEncoding(d_model, max_relative_position, num_buckets, bidirectional)

def create_deberta_relative_embedding(
    d_model: int,
    max_relative_position: int = 32,
    num_buckets: int = 32,
    bidirectional: bool = True
) -> DeBERTaRelativePositionalEncoding:
    """Create a DeBERTa-style relative positional encoding instance."""
    return DeBERTaRelativePositionalEncoding(d_model, max_relative_position, num_buckets, bidirectional)





