"""
Ultra-fast attention optimizations
Following deep learning best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """Attention configuration"""
    num_heads: int = 12
    head_dim: int = 64
    dropout: float = 0.1
    use_flash_attention: bool = True
    use_rotary_embeddings: bool = True


class FlashAttention(nn.Module):
    """Ultra-fast Flash Attention implementation"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.scale = 1.0 / math.sqrt(config.head_dim)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with Flash Attention"""
        if self.config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.config.dropout)
        else:
            return self._standard_attention(q, k, v, mask)
    
    def _standard_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard attention implementation"""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.config.dropout, training=self.training)
        
        return torch.matmul(attn_weights, v)


class MultiHeadAttention(nn.Module):
    """Optimized Multi-Head Attention"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.num_heads * config.head_dim
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        
        self.flash_attention = FlashAttention(config)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        
        # Apply attention
        attn_output = self.flash_attention(q, k, v, mask)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class AttentionOptimizer:
    """Attention optimization utilities"""
    
    @staticmethod
    def optimize_attention_patterns(attention_weights: torch.Tensor) -> torch.Tensor:
        """Optimize attention patterns for better performance"""
        # Apply attention dropout
        if attention_weights.requires_grad:
            attention_weights = F.dropout(attention_weights, p=0.1, training=True)
        
        # Apply attention scaling
        attention_weights = attention_weights / math.sqrt(attention_weights.size(-1))
        
        return attention_weights
    
    @staticmethod
    def get_attention_visualization(attention_weights: torch.Tensor) -> torch.Tensor:
        """Get attention weights for visualization"""
        return attention_weights.mean(dim=1)  # Average over heads
    
    @staticmethod
    def apply_attention_caching(attention_weights: torch.Tensor, 
                              cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention caching for faster inference"""
        if cache is not None:
            # Concatenate with cache
            attention_weights = torch.cat([cache, attention_weights], dim=-2)
        
        # Update cache
        new_cache = attention_weights
        
        return attention_weights, new_cache



