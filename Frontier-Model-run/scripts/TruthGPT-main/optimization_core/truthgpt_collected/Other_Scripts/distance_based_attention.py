#!/usr/bin/env python3
"""
Distance-Based Attention Mechanism for TruthGPT Optimization Core
================================================================

This module implements a novel distance-based attention mechanism that replaces
the traditional scaled dot-product attention with distance-based computations.

Mathematical Foundation:
- Traditional attention: α = softmax(QK^T / √d_k)
- Distance-based attention: α_new = softmax(λL / √d_k)
- Where L_ij = -distance(Q_i, K_j) and λ is a tuning parameter

The module supports multiple distance metrics including L1, L2, and general Lp distances.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any
from enum import Enum
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistanceType(Enum):
    """Supported distance metrics for the attention mechanism."""
    L1 = "l1"  # Manhattan distance
    L2 = "l2"  # Euclidean distance
    LP = "lp"  # General Lp distance
    COSINE = "cosine"  # Cosine distance
    CHEBYSHEV = "chebyshev"  # Chebyshev distance

class DistanceBasedAttention(nn.Module):
    """
    Distance-Based Attention Mechanism
    
    Implements the novel attention mechanism based on distance computations
    between queries and keys, as described in the mathematical formulation.
    
    Args:
        hidden_size (int): The hidden size of the model
        num_heads (int): Number of attention heads
        distance_type (DistanceType): Type of distance metric to use
        lambda_param (float): Tuning parameter λ for attention scaling
        p_norm (float): P value for Lp distance (when distance_type is LP)
        dropout (float): Dropout probability
        bias (bool): Whether to use bias in linear projections
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        distance_type: DistanceType = DistanceType.L1,
        lambda_param: float = 1.0,
        p_norm: float = 2.0,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.distance_type = distance_type
        self.lambda_param = lambda_param
        self.p_norm = p_norm
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Learnable lambda parameter (optional)
        self.learnable_lambda = nn.Parameter(torch.tensor(lambda_param))
        self.use_learnable_lambda = False
        
        logger.info(f"Initialized DistanceBasedAttention with {distance_type.value} distance")
    
    def set_learnable_lambda(self, use_learnable: bool = True):
        """Enable or disable learnable lambda parameter."""
        self.use_learnable_lambda = use_learnable
        if use_learnable:
            logger.info("Enabled learnable lambda parameter")
        else:
            logger.info("Using fixed lambda parameter")
    
    def compute_distance_matrix(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Compute the distance matrix L where L_ij = -distance(Q_i, K_j)
        
        Args:
            Q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            K: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            
        Returns:
            Distance matrix L of shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape
        
        if self.distance_type == DistanceType.L1:
            # L1 (Manhattan) distance: L_ij = -||Q_i - K_j||_1
            Q_expanded = Q.unsqueeze(-2)  # (batch, heads, seq_len, 1, head_dim)
            K_expanded = K.unsqueeze(-3)  # (batch, heads, 1, seq_len, head_dim)
            distances = torch.sum(torch.abs(Q_expanded - K_expanded), dim=-1)
            
        elif self.distance_type == DistanceType.L2:
            # L2 (Euclidean) distance: L_ij = -||Q_i - K_j||_2
            Q_expanded = Q.unsqueeze(-2)  # (batch, heads, seq_len, 1, head_dim)
            K_expanded = K.unsqueeze(-3)  # (batch, heads, 1, seq_len, head_dim)
            distances = torch.sqrt(torch.sum((Q_expanded - K_expanded) ** 2, dim=-1) + 1e-8)
            
        elif self.distance_type == DistanceType.LP:
            # Lp distance: L_ij = -||Q_i - K_j||_p
            Q_expanded = Q.unsqueeze(-2)  # (batch, heads, seq_len, 1, head_dim)
            K_expanded = K.unsqueeze(-3)  # (batch, heads, 1, seq_len, head_dim)
            distances = torch.pow(torch.sum(torch.pow(torch.abs(Q_expanded - K_expanded), self.p_norm), dim=-1), 1.0/self.p_norm)
            
        elif self.distance_type == DistanceType.COSINE:
            # Cosine distance: L_ij = -cosine_distance(Q_i, K_j)
            Q_norm = F.normalize(Q, p=2, dim=-1)
            K_norm = F.normalize(K, p=2, dim=-1)
            cosine_sim = torch.matmul(Q_norm, K_norm.transpose(-2, -1))
            distances = 1.0 - cosine_sim
            
        elif self.distance_type == DistanceType.CHEBYSHEV:
            # Chebyshev distance: L_ij = -max|Q_i - K_j|
            Q_expanded = Q.unsqueeze(-2)  # (batch, heads, seq_len, 1, head_dim)
            K_expanded = K.unsqueeze(-3)  # (batch, heads, 1, seq_len, head_dim)
            distances = torch.max(torch.abs(Q_expanded - K_expanded), dim=-1)[0]
            
        else:
            raise ValueError(f"Unsupported distance type: {self.distance_type}")
        
        # Return negative distances as per the mathematical formulation
        return -distances
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the distance-based attention mechanism.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            head_mask: Optional head mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (output_tensor, attention_weights)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Linear projections
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute distance matrix L
        L = self.compute_distance_matrix(Q, K)
        
        # Get lambda parameter
        lambda_val = self.learnable_lambda if self.use_learnable_lambda else self.lambda_param
        
        # Compute attention scores: α_new = softmax(λL / √d_k)
        attention_scores = lambda_val * L * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match attention_scores shape
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply head mask if provided
        if head_mask is not None:
            attention_scores = attention_scores * head_mask
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        attention_output = self.out_proj(attention_output)
        
        outputs = (attention_output, attention_weights if output_attentions else None)
        return outputs

class MultiHeadDistanceAttention(nn.Module):
    """
    Multi-Head Distance-Based Attention with advanced features.
    
    This class provides a more advanced implementation with additional features
    like layer normalization, residual connections, and configurable distance types.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        distance_type: DistanceType = DistanceType.L1,
        lambda_param: float = 1.0,
        p_norm: float = 2.0,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # Distance-based attention
        self.attention = DistanceBasedAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            distance_type=distance_type,
            lambda_param=lambda_param,
            p_norm=p_norm,
            dropout=dropout
        )
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        else:
            self.layer_norm = None
        
        # Feed-forward network (optional)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        logger.info(f"Initialized MultiHeadDistanceAttention with {distance_type.value} distance")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with residual connections and layer normalization.
        """
        residual = hidden_states
        
        # Apply layer normalization before attention (pre-norm)
        if self.use_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        
        # Distance-based attention
        attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions
        )
        attention_output = attention_outputs[0]
        attention_weights = attention_outputs[1]
        
        # Residual connection
        if self.use_residual:
            attention_output = attention_output + residual
        
        # Feed-forward network
        ff_output = self.feed_forward(attention_output)
        
        # Final residual connection
        if self.use_residual:
            output = ff_output + attention_output
        else:
            output = ff_output
        
        return (output, attention_weights)

class DistanceAttentionConfig:
    """
    Configuration class for distance-based attention mechanisms.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        distance_type: str = "l1",
        lambda_param: float = 1.0,
        p_norm: float = 2.0,
        dropout: float = 0.1,
        use_learnable_lambda: bool = False,
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.distance_type = DistanceType(distance_type)
        self.lambda_param = lambda_param
        self.p_norm = p_norm
        self.dropout = dropout
        self.use_learnable_lambda = use_learnable_lambda
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'distance_type': self.distance_type.value,
            'lambda_param': self.lambda_param,
            'p_norm': self.p_norm,
            'dropout': self.dropout,
            'use_learnable_lambda': self.use_learnable_lambda,
            'use_residual': self.use_residual,
            'use_layer_norm': self.use_layer_norm
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DistanceAttentionConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

def create_distance_attention(config: DistanceAttentionConfig) -> nn.Module:
    """
    Factory function to create distance-based attention mechanism.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured attention mechanism
    """
    if config.use_residual or config.use_layer_norm:
        return MultiHeadDistanceAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            distance_type=config.distance_type,
            lambda_param=config.lambda_param,
            p_norm=config.p_norm,
            dropout=config.dropout
        )
    else:
        attention = DistanceBasedAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            distance_type=config.distance_type,
            lambda_param=config.lambda_param,
            p_norm=config.p_norm,
            dropout=config.dropout
        )
        if config.use_learnable_lambda:
            attention.set_learnable_lambda(True)
        return attention

# Example usage and testing
if __name__ == "__main__":
    # Test the distance-based attention mechanism
    batch_size, seq_len, hidden_size = 2, 10, 768
    num_heads = 12
    
    # Create test input
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Test different distance types
    for distance_type in [DistanceType.L1, DistanceType.L2, DistanceType.LP]:
        print(f"\nTesting {distance_type.value} distance attention...")
        
        config = DistanceAttentionConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            distance_type=distance_type.value,
            lambda_param=1.0
        )
        
        attention = create_distance_attention(config)
        
        # Forward pass
        output, attn_weights = attention(x, output_attentions=True)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Attention weights shape: {attn_weights.shape}")
        print(f"Attention weights sum (should be 1.0): {attn_weights.sum(dim=-1).mean():.4f}")
    
    print("\nDistance-based attention mechanism test completed successfully!")






