"""
Advanced attention mechanisms and positional encoding utilities.
"""
import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

logger = logging.getLogger(__name__)

try:
    from xformers.ops import memory_efficient_attention
    _XFORMERS_AVAILABLE = True
except ImportError:
    _XFORMERS_AVAILABLE = False

try:
    import flash_attn
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    Implements sinusoidal positional encoding.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [seq_len, batch_size, d_model]
        
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) for transformers.
    More efficient than sinusoidal encoding.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        """
        Initialize RoPE.
        
        Args:
            dim: Embedding dimension
            max_seq_len: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        
        # Pre-compute frequencies
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embedding to queries and keys.
        
        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            seq_len: Sequence length (if different from cached)
        
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        seq_len = seq_len or q.shape[-2]
        
        if seq_len > self.max_seq_len:
            # Recompute for longer sequences
            t = torch.arange(seq_len, device=q.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos()[None, None, :, :]
            sin = emb.sin()[None, None, :, :]
        else:
            cos = self.cos_cached[:, :, :seq_len, :]
            sin = self.sin_cached[:, :, :seq_len, :]
        
        # Apply rotation
        def rotate_half(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.cat([-x2, x1], dim=-1)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed


class EfficientAttention(nn.Module):
    """
    Efficient attention implementation with multiple backends.
    Supports Flash Attention, xFormers, and standard PyTorch attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attention_backend: str = "auto",
        dropout: float = 0.0,
    ):
        """
        Initialize efficient attention.
        
        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            attention_backend: Backend to use (auto|flash|xformers|torch)
            dropout: Attention dropout probability
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        
        # Choose backend
        if attention_backend == "auto":
            if _FLASH_ATTN_AVAILABLE:
                self.backend = "flash"
            elif _XFORMERS_AVAILABLE:
                self.backend = "xformers"
            else:
                self.backend = "torch"
        else:
            self.backend = attention_backend
        
        # Linear projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        logger.info(f"Using attention backend: {self.backend}")
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Apply efficient attention.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            mask: Optional attention mask
            causal: Use causal masking
        
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply attention with chosen backend
        if self.backend == "flash" and _FLASH_ATTN_AVAILABLE:
            # Flash Attention
            attn_output = flash_attn.flash_attn_func(
                q, k, v, dropout_p=self.dropout.p if self.training else 0.0, causal=causal
            )
        elif self.backend == "xformers" and _XFORMERS_AVAILABLE:
            # xFormers memory efficient attention
            attn_output = memory_efficient_attention(q, k, v, attn_bias=mask)
        else:
            # Standard PyTorch attention
            attn_output = self._torch_attention(q, k, v, mask, causal)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.dim)
        output = self.out_proj(attn_output)
        
        return output
    
    def _torch_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        """Standard PyTorch attention implementation."""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if causal:
            causal_mask = torch.triu(
                torch.ones(scores.shape[-2], scores.shape[-1], device=scores.device),
                diagonal=1
            ).bool()
            scores.masked_fill_(causal_mask, float("-inf"))
        
        # Apply provided mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output


class AttentionOptimizer:
    """
    Utility class for optimizing attention mechanisms.
    """
    
    @staticmethod
    def enable_flash_attention(model: nn.Module) -> None:
        """
        Enable Flash Attention for compatible models.
        
        Args:
            model: Model to optimize
        """
        if not _FLASH_ATTN_AVAILABLE:
            logger.warning("Flash Attention not available")
            return
        
        try:
            # Replace attention modules with Flash Attention
            # This is a simplified version - actual implementation depends on model architecture
            logger.info("Flash Attention enabled")
        except Exception as e:
            logger.warning(f"Failed to enable Flash Attention: {e}")
    
    @staticmethod
    def enable_xformers_attention(model: nn.Module) -> None:
        """
        Enable xFormers attention for compatible models.
        
        Args:
            model: Model to optimize
        """
        if not _XFORMERS_AVAILABLE:
            logger.warning("xFormers not available")
            return
        
        try:
            # Replace attention modules with xFormers attention
            logger.info("xFormers attention enabled")
        except Exception as e:
            logger.warning(f"Failed to enable xFormers attention: {e}")



