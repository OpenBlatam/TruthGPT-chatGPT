"""
ALiBi (Attention with Linear Biases) embeddings for TruthGPT
Implements the ALiBi positional encoding as described in "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

class AliBi(nn.Module):
    """
    ALiBi (Attention with Linear Biases) implementation.
    
    ALiBi adds linear biases to attention scores instead of using positional embeddings,
    allowing models to extrapolate to longer sequences than seen during training.
    """
    
    def __init__(
        self, 
        n_heads: int,
        max_length: int = 2048,
        slope: Optional[float] = None
    ):
        """
        Initialize ALiBi.
        
        Args:
            n_heads: Number of attention heads
            max_length: Maximum sequence length
            slope: Slope for the linear bias (if None, computed automatically)
        """
        super().__init__()
        self.n_heads = n_heads
        self.max_length = max_length
        
        # Compute slopes for each head
        if slope is None:
            # Use the standard ALiBi slope computation
            slopes = self._compute_slopes(n_heads)
        else:
            slopes = torch.full((n_heads,), slope)
        
        self.register_buffer("slopes", slopes)
        
        # Precompute bias matrix
        self._precompute_bias()
    
    def _compute_slopes(self, n_heads: int) -> torch.Tensor:
        """
        Compute slopes for ALiBi as described in the paper.
        
        Args:
            n_heads: Number of attention heads
            
        Returns:
            Tensor of slopes for each head
        """
        def get_slopes(n):
            """Get slopes for n heads."""
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]
            
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                # Find closest power of 2
                closest_power_of_2 = 2**math.floor(math.log2(n))
                slopes = get_slopes_power_of_2(closest_power_of_2)
                slopes.extend(get_slopes_power_of_2(2*closest_power_of_2)[0::2][:n-closest_power_of_2])
                return slopes
        
        slopes = get_slopes(n_heads)
        return torch.tensor(slopes, dtype=torch.float32)
    
    def _precompute_bias(self) -> None:
        """Precompute the bias matrix for efficiency."""
        # Create position matrix
        positions = torch.arange(self.max_length)
        
        # Create distance matrix
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Apply slopes to create bias matrix
        bias = distances.unsqueeze(0) * self.slopes.unsqueeze(-1).unsqueeze(-1)
        
        self.register_buffer("bias", bias)
    
    def forward(
        self, 
        attention_scores: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply ALiBi bias to attention scores.
        
        Args:
            attention_scores: Attention scores of shape (batch_size, n_heads, seq_len, seq_len)
            seq_len: Sequence length (if None, inferred from attention_scores)
            
        Returns:
            Attention scores with ALiBi bias applied
        """
        if seq_len is None:
            seq_len = attention_scores.size(-1)
        
        # Ensure we don't exceed max_length
        if seq_len > self.max_length:
            logger.warning(f"Sequence length {seq_len} exceeds max_length {self.max_length}")
            # Dynamically compute bias for longer sequences
            return self._compute_dynamic_bias(attention_scores, seq_len)
        
        # Get precomputed bias
        bias = self.bias[:, :, :seq_len, :seq_len]
        
        # Add bias to attention scores
        return attention_scores + bias
    
    def _compute_dynamic_bias(self, attention_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Compute bias dynamically for sequences longer than max_length.
        
        Args:
            attention_scores: Attention scores
            seq_len: Sequence length
            
        Returns:
            Attention scores with dynamic ALiBi bias
        """
        # Create position matrix
        positions = torch.arange(seq_len, device=attention_scores.device)
        
        # Create distance matrix
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Apply slopes to create bias matrix
        bias = distances.unsqueeze(0) * self.slopes.unsqueeze(-1).unsqueeze(-1)
        bias = bias.to(attention_scores.device)
        
        # Add bias to attention scores
        return attention_scores + bias
    
    def get_bias_matrix(self, seq_len: int) -> torch.Tensor:
        """
        Get the ALiBi bias matrix for a given sequence length.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Bias matrix of shape (n_heads, seq_len, seq_len)
        """
        if seq_len <= self.max_length:
            return self.bias[:, :, :seq_len, :seq_len]
        else:
            # Compute dynamically
            positions = torch.arange(seq_len, device=self.bias.device)
            distances = positions.unsqueeze(0) - positions.unsqueeze(1)
            bias = distances.unsqueeze(0) * self.slopes.unsqueeze(-1).unsqueeze(-1)
            return bias.to(self.bias.device)

class AdaptiveAliBi(AliBi):
    """
    Adaptive ALiBi that can adjust slopes based on sequence length.
    
    This variant can modify the slope values based on the input sequence length,
    potentially improving performance on very long sequences.
    """
    
    def __init__(
        self, 
        n_heads: int,
        max_length: int = 2048,
        slope: Optional[float] = None,
        adaptive_factor: float = 1.0
    ):
        """
        Initialize adaptive ALiBi.
        
        Args:
            n_heads: Number of attention heads
            max_length: Maximum sequence length
            slope: Base slope for the linear bias
            adaptive_factor: Factor to adjust slopes based on sequence length
        """
        super().__init__(n_heads, max_length, slope)
        self.adaptive_factor = adaptive_factor
    
    def forward(
        self, 
        attention_scores: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """Apply adaptive ALiBi bias to attention scores."""
        if seq_len is None:
            seq_len = attention_scores.size(-1)
        
        # Adjust slopes based on sequence length
        adjusted_slopes = self.slopes * (self.adaptive_factor ** (seq_len / self.max_length))
        
        # Compute bias with adjusted slopes
        positions = torch.arange(seq_len, device=attention_scores.device)
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)
        bias = distances.unsqueeze(0) * adjusted_slopes.unsqueeze(-1).unsqueeze(-1)
        
        return attention_scores + bias

# Factory functions
def create_alibi_embedding(
    n_heads: int,
    max_length: int = 2048,
    slope: Optional[float] = None
) -> AliBi:
    """Create an ALiBi embedding instance."""
    return AliBi(n_heads, max_length, slope)

def create_adaptive_alibi_embedding(
    n_heads: int,
    max_length: int = 2048,
    slope: Optional[float] = None,
    adaptive_factor: float = 1.0
) -> AdaptiveAliBi:
    """Create an adaptive ALiBi embedding instance."""
    return AdaptiveAliBi(n_heads, max_length, slope, adaptive_factor)





