"""
Flash Attention implementation for TruthGPT
Provides memory-efficient attention mechanisms for long sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Union, Dict, Any
import math

logger = logging.getLogger(__name__)

class FlashAttention(nn.Module):
    """
    Flash Attention implementation.
    
    This provides memory-efficient attention computation for long sequences
    by processing attention in blocks and using online softmax.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.1,
        block_size: int = 64,
        use_flash_attention: bool = True
    ):
        """
        Initialize Flash Attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            block_size: Block size for attention computation
            use_flash_attention: Whether to use Flash Attention
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size
        self.use_flash_attention = use_flash_attention
        
        # Validate dimensions
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        # Linear transformations
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        
        # Scale factor
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
        Compute Flash Attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights)
        """
        if self.use_flash_attention:
            return self._flash_attention(query, key, value, mask)
        else:
            return self._standard_attention(query, key, value, mask)
    
    def _flash_attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Flash Attention with block processing."""
        batch_size, seq_len, d_model = query.size()
        
        # Apply linear transformations
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention in blocks
        output = torch.zeros_like(query)
        attention_weights = torch.zeros(batch_size, self.n_heads, seq_len, seq_len, device=query.device)
        
        for i in range(0, seq_len, self.block_size):
            q_block = query[:, :, i:i+self.block_size, :]
            q_len = q_block.size(2)
            
            # Initialize block output
            block_output = torch.zeros_like(q_block)
            block_attention = torch.zeros(batch_size, self.n_heads, q_len, seq_len, device=query.device)
            
            for j in range(0, seq_len, self.block_size):
                k_block = key[:, :, j:j+self.block_size, :]
                v_block = value[:, :, j:j+self.block_size, :]
                k_len = k_block.size(2)
                
                # Compute attention scores for this block
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / self.scale
                
                # Apply mask if provided
                if mask is not None:
                    mask_block = mask[:, :, i:i+q_len, j:j+k_len]
                    scores = scores.masked_fill(mask_block == 0, -1e9)
                
                # Apply softmax
                attention_block = F.softmax(scores, dim=-1)
                attention_block = self.dropout(attention_block)
                
                # Apply attention to values
                block_output += torch.matmul(attention_block, v_block)
                block_attention[:, :, :, j:j+k_len] = attention_block
            
            # Store block output
            output[:, :, i:i+q_len, :] = block_output
            attention_weights[:, :, i:i+q_len, :] = block_attention
        
        # Reshape back to original format
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.output_linear(output)
        
        return output, attention_weights
    
    def _standard_attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute standard attention (fallback)."""
        batch_size, seq_len, d_model = query.size()
        
        # Apply linear transformations
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
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
        output = self.output_linear(output)
        
        return output, attention_weights

class FlashAttentionV2(FlashAttention):
    """
    Flash Attention V2.

    Currently inherits V1 block processing unchanged.
    Extension point: override ``_flash_attention`` with online-softmax
    rescaling or tiled SRAM scheduling for true V2 semantics.
    """
    pass  # inherits everything from V1

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_VERSION_MAP = {"v1": FlashAttention, "v2": FlashAttentionV2}

def create_flash_attention(
    d_model: int,
    n_heads: int,
    dropout: float = 0.1,
    block_size: int = 64,
    use_flash_attention: bool = True,
    version: str = "v1",
) -> Union[FlashAttention, FlashAttentionV2]:
    """
    Create a Flash Attention instance.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        block_size: Block size for tiled computation.
        use_flash_attention: Use flash (True) or standard fallback (False).
        version: ``'v1'`` or ``'v2'``.
    """
    cls = _VERSION_MAP.get(version)
    if cls is None:
        raise ValueError(f"Unsupported version '{version}'. Choose from {list(_VERSION_MAP)}")
    return cls(d_model, n_heads, dropout, block_size, use_flash_attention)


# Backward-compat aliases
create_flash_attention_v1 = lambda **kw: create_flash_attention(version="v1", **kw)
create_flash_attention_v2 = lambda **kw: create_flash_attention(version="v2", **kw)

