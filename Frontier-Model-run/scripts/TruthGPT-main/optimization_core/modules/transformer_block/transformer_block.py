"""
Transformer Block implementations for TruthGPT
Provides various transformer block architectures with different normalization strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Union, Dict, Any, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class TransformerBlockBase(ABC, nn.Module):
    """Abstract base class for transformer blocks."""
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        bias: bool = True
    ):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.bias = bias
    
    @abstractmethod
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        pass

class TransformerBlock(TransformerBlockBase):
    """
    Standard transformer block.
    
    This implements the standard transformer block from "Attention Is All You Need":
    x = x + MultiHeadAttention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        bias: bool = True,
        attention_type: str = "scaled_dot_product"
    ):
        """Initialize standard transformer block."""
        super().__init__(d_model, n_heads, d_ff, dropout, activation, bias)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Multi-head attention
        from ..attention.multi_head_attention import create_multi_head_attention
        self.attention = create_multi_head_attention(
            d_model, n_heads, dropout, bias, attention_type
        )
        
        # Feed-forward network
        from ..feed_forward.feed_forward import create_feed_forward
        self.feed_forward = create_feed_forward(
            d_model, d_ff, dropout, "standard", bias
        )
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through the transformer block."""
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = x + self.dropout_layer(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout_layer(ff_output)
        x = self.norm2(x)
        
        return x

class PreNormTransformerBlock(TransformerBlockBase):
    """
    Pre-normalization transformer block.
    
    This applies layer normalization before the attention and feed-forward layers:
    x = x + MultiHeadAttention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        bias: bool = True,
        attention_type: str = "scaled_dot_product"
    ):
        """Initialize pre-normalization transformer block."""
        super().__init__(d_model, n_heads, d_ff, dropout, activation, bias)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Multi-head attention
        from ..attention.multi_head_attention import create_multi_head_attention
        self.attention = create_multi_head_attention(
            d_model, n_heads, dropout, bias, attention_type
        )
        
        # Feed-forward network
        from ..feed_forward.feed_forward import create_feed_forward
        self.feed_forward = create_feed_forward(
            d_model, d_ff, dropout, "standard", bias
        )
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through the pre-normalization transformer block."""
        # Self-attention with pre-norm and residual connection
        attn_output, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout_layer(attn_output)
        
        # Feed-forward with pre-norm and residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout_layer(ff_output)
        
        return x

class PostNormTransformerBlock(TransformerBlockBase):
    """
    Post-normalization transformer block.
    
    This applies layer normalization after the attention and feed-forward layers:
    x = LayerNorm(x + MultiHeadAttention(x))
    x = LayerNorm(x + FeedForward(x))
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        bias: bool = True,
        attention_type: str = "scaled_dot_product"
    ):
        """Initialize post-normalization transformer block."""
        super().__init__(d_model, n_heads, d_ff, dropout, activation, bias)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Multi-head attention
        from ..attention.multi_head_attention import create_multi_head_attention
        self.attention = create_multi_head_attention(
            d_model, n_heads, dropout, bias, attention_type
        )
        
        # Feed-forward network
        from ..feed_forward.feed_forward import create_feed_forward
        self.feed_forward = create_feed_forward(
            d_model, d_ff, dropout, "standard", bias
        )
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through the post-normalization transformer block."""
        # Self-attention with post-norm and residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout_layer(attn_output))
        
        # Feed-forward with post-norm and residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout_layer(ff_output))
        
        return x

class FlashTransformerBlock(TransformerBlockBase):
    """
    Flash transformer block with Flash Attention.
    
    This uses Flash Attention for memory-efficient attention computation.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        bias: bool = True,
        use_flash_attention: bool = True
    ):
        """Initialize flash transformer block."""
        super().__init__(d_model, n_heads, d_ff, dropout, activation, bias)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Flash attention
        from ..attention.flash_attention import create_flash_attention
        self.attention = create_flash_attention(
            d_model, n_heads, dropout, use_flash_attention=use_flash_attention
        )
        
        # Feed-forward network
        from ..feed_forward.feed_forward import create_feed_forward
        self.feed_forward = create_feed_forward(
            d_model, d_ff, dropout, "standard", bias
        )
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through the flash transformer block."""
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = x + self.dropout_layer(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout_layer(ff_output)
        x = self.norm2(x)
        
        return x

class AdaptiveTransformerBlock(TransformerBlockBase):
    """
    Adaptive transformer block that can switch between different architectures.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        bias: bool = True,
        architecture: str = "standard",
        **kwargs
    ):
        """Initialize adaptive transformer block."""
        super().__init__(d_model, n_heads, d_ff, dropout, activation, bias)
        self.architecture = architecture
        
        if architecture == "standard":
            self.block = TransformerBlock(d_model, n_heads, d_ff, dropout, activation, bias, **kwargs)
        elif architecture == "pre_norm":
            self.block = PreNormTransformerBlock(d_model, n_heads, d_ff, dropout, activation, bias, **kwargs)
        elif architecture == "post_norm":
            self.block = PostNormTransformerBlock(d_model, n_heads, d_ff, dropout, activation, bias, **kwargs)
        elif architecture == "flash":
            self.block = FlashTransformerBlock(d_model, n_heads, d_ff, dropout, activation, bias, **kwargs)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through the adaptive transformer block."""
        return self.block(x, mask, **kwargs)
    
    def switch_architecture(self, new_architecture: str) -> None:
        """Switch to a different architecture."""
        if new_architecture == self.architecture:
            return
        
        self.architecture = new_architecture
        if new_architecture == "standard":
            self.block = TransformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout, self.activation, self.bias)
        elif new_architecture == "pre_norm":
            self.block = PreNormTransformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout, self.activation, self.bias)
        elif new_architecture == "post_norm":
            self.block = PostNormTransformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout, self.activation, self.bias)
        elif new_architecture == "flash":
            self.block = FlashTransformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout, self.activation, self.bias)
        else:
            raise ValueError(f"Unsupported architecture: {new_architecture}")

# Factory functions
def create_transformer_block(
    d_model: int,
    n_heads: int,
    d_ff: int,
    dropout: float = 0.1,
    activation: str = "relu",
    bias: bool = True,
    architecture: str = "standard",
    **kwargs
) -> TransformerBlockBase:
    """
    Create a transformer block instance.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
        bias: Whether to use bias in linear layers
        architecture: Architecture type
        **kwargs: Additional arguments
        
    Returns:
        Transformer block instance
    """
    if architecture == "standard":
        return TransformerBlock(d_model, n_heads, d_ff, dropout, activation, bias, **kwargs)
    elif architecture == "pre_norm":
        return PreNormTransformerBlock(d_model, n_heads, d_ff, dropout, activation, bias, **kwargs)
    elif architecture == "post_norm":
        return PostNormTransformerBlock(d_model, n_heads, d_ff, dropout, activation, bias, **kwargs)
    elif architecture == "flash":
        return FlashTransformerBlock(d_model, n_heads, d_ff, dropout, activation, bias, **kwargs)
    elif architecture == "adaptive":
        return AdaptiveTransformerBlock(d_model, n_heads, d_ff, dropout, activation, bias, **kwargs)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

def create_pre_norm_block(
    d_model: int,
    n_heads: int,
    d_ff: int,
    dropout: float = 0.1,
    activation: str = "relu",
    bias: bool = True
) -> PreNormTransformerBlock:
    """Create a pre-normalization transformer block."""
    return PreNormTransformerBlock(d_model, n_heads, d_ff, dropout, activation, bias)

def create_post_norm_block(
    d_model: int,
    n_heads: int,
    d_ff: int,
    dropout: float = 0.1,
    activation: str = "relu",
    bias: bool = True
) -> PostNormTransformerBlock:
    """Create a post-normalization transformer block."""
    return PostNormTransformerBlock(d_model, n_heads, d_ff, dropout, activation, bias)

def create_flash_block(
    d_model: int,
    n_heads: int,
    d_ff: int,
    dropout: float = 0.1,
    activation: str = "relu",
    bias: bool = True,
    use_flash_attention: bool = True
) -> FlashTransformerBlock:
    """Create a flash transformer block."""
    return FlashTransformerBlock(d_model, n_heads, d_ff, dropout, activation, bias, use_flash_attention)





