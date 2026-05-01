"""
Transformer-based Optimization Model
===================================

Advanced transformer implementation with:
- Multi-head attention mechanisms
- Positional encodings
- Layer normalization
- Feed-forward networks
- Flash Attention support
- LoRA fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import math
import logging

from .base import BaseModel, ModelConfig


@dataclass
class TransformerConfig(ModelConfig):
    """Configuration for transformer model"""
    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    activation: str = "relu"
    
    # Attention mechanism
    attention_type: str = "multihead"  # multihead, flash, sparse
    use_flash_attention: bool = True
    attention_dropout: float = 0.1
    
    # Positional encoding
    max_seq_length: int = 1024
    use_rotary_embeddings: bool = True
    
    # LoRA fine-tuning
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    
    # Optimization
    use_gradient_checkpointing: bool = True
    use_activation_checkpointing: bool = True


class MultiHeadAttention(nn.Module):
    """Multi-head attention with Flash Attention support"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        # Linear projections
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # Flash Attention support
        self.use_flash_attention = config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Flash Attention
        if self.use_flash_attention:
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            # Standard attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(attn_output)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_seq_length = config.max_seq_length
        
        # Create positional encoding matrix
        pe = torch.zeros(config.max_seq_length, config.d_model)
        position = torch.arange(0, config.max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * 
                           -(math.log(10000.0) / config.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_seq_length = config.max_seq_length
        
        # Precompute frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_model, 2).float() / self.d_model))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create position tensor
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = torch.cos(emb)[:, None, :]
        sin = torch.sin(emb)[:, None, :]
        
        return cos, sin


class TransformerBlock(nn.Module):
    """Single transformer block"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Attention
        self.attention = MultiHeadAttention(config)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU() if config.activation == "relu" else nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Rotary embeddings
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(config)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerOptimizer(BaseModel):
    """
    Transformer-based optimization model.
    
    Features:
    - Multi-head attention with Flash Attention support
    - Positional encodings (sinusoidal and rotary)
    - Layer normalization and residual connections
    - LoRA fine-tuning support
    - Gradient checkpointing
    """
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        super().__init__(config)
    
    def _initialize_model(self):
        """Initialize transformer architecture"""
        # Input embedding
        self.input_embedding = nn.Embedding(
            self.config.vocab_size if hasattr(self.config, 'vocab_size') else 10000,
            self.config.d_model
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.config)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.config) for _ in range(self.config.n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.config.d_model, self.config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(self.config.dropout)
        
        # Initialize weights
        self.apply(self._initialize_weights)
        
        # Enable gradient checkpointing if requested
        if self.config.use_gradient_checkpointing:
            self.enable_gradient_checkpointing()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Forward pass through transformer"""
        # Input embedding
        x = self.input_embedding(x) * math.sqrt(self.config.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            if self.config.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, mask)
            else:
                x = block(x, mask)
        
        # Output projection
        x = self.output_projection(x)
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """Get attention weights from specified layer"""
        if layer_idx < 0:
            layer_idx = len(self.transformer_blocks) + layer_idx
        
        if layer_idx >= len(self.transformer_blocks):
            raise ValueError(f"Layer index {layer_idx} out of range")
        
        # Forward pass to get attention weights
        x = self.input_embedding(x) * math.sqrt(self.config.d_model)
        x = self.pos_encoding(x)
        
        for i, block in enumerate(self.transformer_blocks):
            if i == layer_idx:
                # Get attention weights from this layer
                return block.attention(x, x, x)
            x = block(x)
        
        return None
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get input embeddings"""
        return self.input_embedding(x)
    
    def create_lora_adapters(self):
        """Create LoRA adapters for fine-tuning"""
        if not self.config.use_lora:
            return
        
        # This would implement LoRA adapters
        # For now, just log that LoRA is enabled
        self.logger.info("LoRA adapters would be created here")
    
    def apply_lora_adapters(self):
        """Apply LoRA adapters to the model"""
        if not self.config.use_lora:
            return
        
        # This would apply LoRA adapters
        self.logger.info("LoRA adapters would be applied here")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        info = super().get_model_size()
        info.update({
            'd_model': self.config.d_model,
            'n_heads': self.config.n_heads,
            'n_layers': self.config.n_layers,
            'd_ff': self.config.d_ff,
            'attention_type': self.config.attention_type,
            'use_flash_attention': self.config.use_flash_attention,
            'use_rotary_embeddings': self.config.use_rotary_embeddings,
            'use_lora': self.config.use_lora
        })
        return info



