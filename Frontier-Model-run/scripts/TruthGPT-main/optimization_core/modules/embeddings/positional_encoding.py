"""
Positional Encoding implementations for TruthGPT
Provides various positional encoding strategies for transformer models
"""

import math
import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class PositionalEncoding(ABC, nn.Module):
    """Abstract base class for positional encodings."""
    
    def __init__(self, d_model: int, max_length: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_length: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
    
    @abstractmethod
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            positions: Optional position tensor
            
        Returns:
            Tensor with positional encoding applied
        """
        pass

class SinusoidalPositionalEncoding(PositionalEncoding):
    """
    Sinusoidal positional encoding as described in "Attention Is All You Need".
    
    This implementation follows the original paper's formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_length: int = 5000, dropout: float = 0.1):
        """
        Initialize sinusoidal positional encoding.
        
        Args:
            d_model: Model dimension
            max_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__(d_model, max_length)
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        # Calculate div_term for the sinusoidal functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer so it's not considered a parameter
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply sinusoidal positional encoding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            positions: Optional position tensor (not used for sinusoidal encoding)
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        
        # Ensure we don't exceed max_length
        if seq_len > self.max_length:
            logger.warning(f"Sequence length {seq_len} exceeds max_length {self.max_length}")
            # Truncate or extend pe as needed
            if seq_len > self.pe.size(1):
                self._extend_pe(seq_len)
        
        # Add positional encoding
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)
    
    def _extend_pe(self, new_length: int) -> None:
        """Extend positional encoding to new length."""
        if new_length <= self.pe.size(1):
            return
        
        # Create new positional encoding
        pe = torch.zeros(new_length, self.d_model, device=self.pe.device)
        position = torch.arange(0, new_length, dtype=torch.float, device=self.pe.device).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=self.pe.device).float() * 
                           (-math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.pe = pe.unsqueeze(0)
        self.max_length = new_length

class LearnedPositionalEncoding(PositionalEncoding):
    """
    Learned positional encoding.
    
    This uses learnable embeddings for each position, which can be more flexible
    than sinusoidal encoding but requires more parameters.
    """
    
    def __init__(self, d_model: int, max_length: int = 5000, dropout: float = 0.1):
        """
        Initialize learned positional encoding.
        
        Args:
            d_model: Model dimension
            max_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__(d_model, max_length)
        self.dropout = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(max_length, d_model)
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply learned positional encoding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            positions: Optional position tensor
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        batch_size = x.size(0)
        
        # Create position indices if not provided
        if positions is None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Ensure positions are within bounds
        positions = positions.clamp(0, self.max_length - 1)
        
        # Get positional embeddings
        pos_embeddings = self.embeddings(positions)
        
        # Add positional encoding
        x = x + pos_embeddings
        return self.dropout(x)

class AdaptivePositionalEncoding(PositionalEncoding):
    """
    Adaptive positional encoding that can switch between different encoding types.
    """
    
    def __init__(
        self, 
        d_model: int, 
        max_length: int = 5000,
        encoding_type: str = "sinusoidal",
        dropout: float = 0.1
    ):
        """
        Initialize adaptive positional encoding.
        
        Args:
            d_model: Model dimension
            max_length: Maximum sequence length
            encoding_type: Type of encoding ("sinusoidal" or "learned")
            dropout: Dropout rate
        """
        super().__init__(d_model, max_length)
        self.encoding_type = encoding_type
        
        if encoding_type == "sinusoidal":
            self.encoding = SinusoidalPositionalEncoding(d_model, max_length, dropout)
        elif encoding_type == "learned":
            self.encoding = LearnedPositionalEncoding(d_model, max_length, dropout)
        else:
            raise ValueError(f"Unsupported encoding type: {encoding_type}")
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply adaptive positional encoding."""
        return self.encoding(x, positions)
    
    def switch_encoding(self, new_type: str) -> None:
        """Switch to a different encoding type."""
        if new_type == self.encoding_type:
            return
        
        self.encoding_type = new_type
        if new_type == "sinusoidal":
            self.encoding = SinusoidalPositionalEncoding(
                self.d_model, self.max_length, self.encoding.dropout.p
            )
        elif new_type == "learned":
            self.encoding = LearnedPositionalEncoding(
                self.d_model, self.max_length, self.encoding.dropout.p
            )
        else:
            raise ValueError(f"Unsupported encoding type: {new_type}")

# Factory functions
def create_positional_encoding(
    encoding_type: str = "sinusoidal",
    d_model: int = 512,
    max_length: int = 5000,
    dropout: float = 0.1,
    **kwargs
) -> PositionalEncoding:
    """
    Create a positional encoding instance.
    
    Args:
        encoding_type: Type of encoding ("sinusoidal", "learned", "adaptive")
        d_model: Model dimension
        max_length: Maximum sequence length
        dropout: Dropout rate
        **kwargs: Additional arguments
        
    Returns:
        Positional encoding instance
    """
    if encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_length, dropout)
    elif encoding_type == "learned":
        return LearnedPositionalEncoding(d_model, max_length, dropout)
    elif encoding_type == "adaptive":
        return AdaptivePositionalEncoding(d_model, max_length, dropout=dropout, **kwargs)
    else:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")

def create_sinusoidal_encoding(
    d_model: int = 512,
    max_length: int = 5000,
    dropout: float = 0.1
) -> SinusoidalPositionalEncoding:
    """Create sinusoidal positional encoding."""
    return SinusoidalPositionalEncoding(d_model, max_length, dropout)

def create_learned_encoding(
    d_model: int = 512,
    max_length: int = 5000,
    dropout: float = 0.1
) -> LearnedPositionalEncoding:
    """Create learned positional encoding."""
    return LearnedPositionalEncoding(d_model, max_length, dropout)





