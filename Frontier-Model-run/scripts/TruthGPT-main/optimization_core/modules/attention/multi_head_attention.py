"""
Multi-Head Attention implementation for TruthGPT
Provides efficient multi-head attention with various optimizations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Union, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class AttentionBase(ABC, nn.Module):
    """Abstract base class for attention mechanisms."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize attention mechanism.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        
        # Validate dimensions
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
    
    @abstractmethod
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights)
        """
        pass

class ScaledDotProductAttention(AttentionBase):
    """
    Scaled Dot-Product Attention implementation.
    
    This is the standard attention mechanism described in "Attention Is All You Need".
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """Initialize scaled dot-product attention."""
        super().__init__(d_model, n_heads, dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = query.size()
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        # Reshape back to original format
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention implementation.
    
    This combines multiple attention heads and provides a complete
    multi-head attention mechanism with linear transformations.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.1,
        bias: bool = True,
        attention_type: str = "scaled_dot_product"
    ):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias in linear layers
            attention_type: Type of attention mechanism
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        
        # Validate dimensions
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        # Linear transformations
        self.query_linear = nn.Linear(d_model, d_model, bias=bias)
        self.key_linear = nn.Linear(d_model, d_model, bias=bias)
        self.value_linear = nn.Linear(d_model, d_model, bias=bias)
        self.output_linear = nn.Linear(d_model, d_model, bias=bias)
        
        # Attention mechanism
        if attention_type == "scaled_dot_product":
            self.attention = ScaledDotProductAttention(d_model, n_heads, dropout)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = query.size()
        
        # Apply linear transformations
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        # Compute attention
        output, attention_weights = self.attention(query, key, value, mask, **kwargs)
        
        # Apply output linear transformation
        output = self.output_linear(output)
        
        return output, attention_weights

class CausalMultiHeadAttention(MultiHeadAttention):
    """
    Causal Multi-Head Attention for autoregressive models.
    
    This applies a causal mask to prevent attention to future positions.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.1,
        bias: bool = True
    ):
        """Initialize causal multi-head attention."""
        super().__init__(d_model, n_heads, dropout, bias)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute causal multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask (will be combined with causal mask)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = query.size()
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # Combine with provided mask
        if mask is not None:
            combined_mask = mask * causal_mask
        else:
            combined_mask = causal_mask
        
        return super().forward(query, key, value, combined_mask, **kwargs)

class CrossAttention(MultiHeadAttention):
    """
    Cross-Attention for encoder-decoder models.
    
    This allows the decoder to attend to encoder outputs.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.1,
        bias: bool = True
    ):
        """Initialize cross-attention."""
        super().__init__(d_model, n_heads, dropout, bias)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-attention.
        
        Args:
            query: Query tensor from decoder (batch_size, tgt_len, d_model)
            key: Key tensor from encoder (batch_size, src_len, d_model)
            value: Value tensor from encoder (batch_size, src_len, d_model)
            mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights)
        """
        return super().forward(query, key, value, mask, **kwargs)

# Factory functions
def create_multi_head_attention(
    d_model: int,
    n_heads: int,
    dropout: float = 0.1,
    bias: bool = True,
    attention_type: str = "scaled_dot_product"
) -> MultiHeadAttention:
    """Create a multi-head attention instance."""
    return MultiHeadAttention(d_model, n_heads, dropout, bias, attention_type)

def create_causal_attention(
    d_model: int,
    n_heads: int,
    dropout: float = 0.1,
    bias: bool = True
) -> CausalMultiHeadAttention:
    """Create a causal multi-head attention instance."""
    return CausalMultiHeadAttention(d_model, n_heads, dropout, bias)

def create_cross_attention(
    d_model: int,
    n_heads: int,
    dropout: float = 0.1,
    bias: bool = True
) -> CrossAttention:
    """Create a cross-attention instance."""
    return CrossAttention(d_model, n_heads, dropout, bias)





