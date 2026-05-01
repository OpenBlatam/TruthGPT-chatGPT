"""
Rotary Embeddings (RoPE) for TruthGPT
Implements Rotary Position Embedding as described in "RoFormer: Enhanced Transformer with Rotary Position Embedding"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    
    This implementation follows the original RoPE paper and provides
    efficient rotary embeddings for transformer models.
    """
    
    def __init__(
        self, 
        dim: int, 
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize rotary embedding.
        
        Args:
            dim: Dimension of the embedding
            max_position_embeddings: Maximum number of position embeddings
            base: Base for the frequency computation
            device: Device to place tensors on
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = device or torch.device('cpu')
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=self.device).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(
        self, 
        x: torch.Tensor, 
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            seq_len: Sequence length (if None, inferred from x)
            position_ids: Position indices (if None, created automatically)
            
        Returns:
            Tuple of (cos, sin) tensors for rotary embedding
        """
        if seq_len is None:
            seq_len = x.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device)
        
        # Ensure position_ids are within bounds
        position_ids = position_ids.clamp(0, self.max_position_embeddings - 1)
        
        # Compute frequencies
        freqs = torch.outer(position_ids.float(), self.inv_freq)
        
        # Create cos and sin tensors
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        # Expand to match input dimensions
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim//2)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim//2)
        
        return cos, sin

class LlamaRotaryEmbedding(RotaryEmbedding):
    """
    LLaMA-style rotary embedding implementation.
    
    This follows the LLaMA model's specific implementation of RoPE.
    """
    
    def __init__(
        self, 
        dim: int, 
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None
    ):
        """Initialize LLaMA rotary embedding."""
        super().__init__(dim, max_position_embeddings, base, device)
    
    def forward(
        self, 
        x: torch.Tensor, 
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply LLaMA-style rotary embeddings."""
        return super().forward(x, seq_len, position_ids)

class FixedLlamaRotaryEmbedding(nn.Module):
    """
    Fixed LLaMA rotary embedding that doesn't require position_ids.
    
    This is useful for inference where we want to avoid recomputing
    the rotary embeddings for each forward pass.
    """
    
    def __init__(
        self, 
        dim: int, 
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None
    ):
        """Initialize fixed LLaMA rotary embedding."""
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = device or torch.device('cpu')
        
        # Precompute all possible cos and sin values
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=self.device).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for all positions
        t = torch.arange(max_position_embeddings, device=self.device)
        freqs = torch.outer(t.float(), inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)
    
    def forward(
        self, 
        x: torch.Tensor, 
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply fixed rotary embeddings.
        
        Args:
            x: Input tensor
            seq_len: Sequence length
            position_ids: Position indices (ignored for fixed embedding)
            
        Returns:
            Tuple of (cos, sin) tensors
        """
        if seq_len is None:
            seq_len = x.size(1)
        
        # Get cached cos and sin values
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # Expand dimensions to match input
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim//2)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim//2)
        
        return cos, sin

def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.
    
    Args:
        q: Query tensor of shape (batch_size, n_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, n_heads, seq_len, head_dim)
        cos: Cosine values of shape (1, 1, seq_len, head_dim//2)
        sin: Sine values of shape (1, 1, seq_len, head_dim//2)
        
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    def rotate_half(x):
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    # Apply rotary embedding
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

def apply_rotary_pos_emb_single(
    x: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary position embedding to a single tensor.
    
    Args:
        x: Input tensor of shape (batch_size, n_heads, seq_len, head_dim)
        cos: Cosine values of shape (1, 1, seq_len, head_dim//2)
        sin: Sine values of shape (1, 1, seq_len, head_dim//2)
        
    Returns:
        Rotated tensor
    """
    def rotate_half(x):
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    return (x * cos) + (rotate_half(x) * sin)

# Factory functions
def create_rotary_embedding(
    dim: int,
    max_position_embeddings: int = 2048,
    base: float = 10000.0,
    device: Optional[torch.device] = None
) -> RotaryEmbedding:
    """Create a rotary embedding instance."""
    return RotaryEmbedding(dim, max_position_embeddings, base, device)

def create_llama_rotary_embedding(
    dim: int,
    max_position_embeddings: int = 2048,
    base: float = 10000.0,
    device: Optional[torch.device] = None
) -> LlamaRotaryEmbedding:
    """Create a LLaMA-style rotary embedding instance."""
    return LlamaRotaryEmbedding(dim, max_position_embeddings, base, device)

def create_fixed_llama_rotary_embedding(
    dim: int,
    max_position_embeddings: int = 2048,
    base: float = 10000.0,
    device: Optional[torch.device] = None
) -> FixedLlamaRotaryEmbedding:
    """Create a fixed LLaMA rotary embedding instance."""
    return FixedLlamaRotaryEmbedding(dim, max_position_embeddings, base, device)





