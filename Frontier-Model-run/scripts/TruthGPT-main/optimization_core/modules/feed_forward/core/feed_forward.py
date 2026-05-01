"""
Feed-Forward Network implementations for TruthGPT
=================================================

Provides a unified, modular feed-forward architecture that supports multiple
activation functions and gating variants (GLU) through a single engine.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logger = logging.getLogger(__name__)

class FeedForwardBase(nn.Module, ABC):
    """Abstract base class for feed-forward networks."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward network."""
        pass

class ModularFeedForward(FeedForwardBase):
    """
    Unified Feed-Forward Network engine.
    
    Supports standard, gated, and GLU variants (SwiGLU, ReGLU, GeGLU)
    through a single, highly configurable implementation.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        gated: bool = False,
        bias: bool = True
    ) -> None:
        """
        Initialize the modular feed-forward network.

        Args:
            d_model: Model dimension.
            d_ff: Feed-forward dimension.
            dropout: Dropout rate.
            activation: Activation function ('relu', 'gelu', 'swish', 'silu', etc.).
            gated: Whether to use gating (GLU variants).
            bias: Whether to use bias in linear layers.
        """
        super().__init__(d_model, d_ff, dropout)
        self.activation_type = activation.lower()
        self.is_gated = gated
        self.use_bias = bias

        # Define activation function
        if self.activation_type == "relu":
            self.activation = nn.ReLU()
        elif self.activation_type in ("gelu", "geglu"):
            self.activation = nn.GELU()
        elif self.activation_type in ("swish", "silu", "swiglu"):
            self.activation = nn.SiLU()
        elif self.activation_type == "reglu":
            self.activation = nn.ReLU() # ReGLU uses ReLU as activation for the gate
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Linear layers - Naming maintained for backward state_dict compatibility
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        if self.is_gated:
            self.linear2 = nn.Linear(d_model, d_ff, bias=bias)
            self.linear3 = nn.Linear(d_ff, d_model, bias=bias)
        else:
            self.linear2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using the configured architecture."""
        if self.is_gated:
            # Gated Logic: activation(linear1(x)) * linear2(x) -> linear3
            gate = self.activation(self.linear1(x))
            val = self.linear2(x)
            x = gate * val
            x = self.dropout(x)
            x = self.linear3(x)
        else:
            # Standard Logic: activation(linear1(x)) -> linear2
            x = self.activation(self.linear1(x))
            x = self.dropout(x)
            x = self.linear2(x)
        return x

# --- Backward Compatibility Shims ---

class FeedForward(ModularFeedForward):
    """Legacy FeedForward alias."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, 
                 activation: str = "relu", bias: bool = True):
        super().__init__(d_model, d_ff, dropout, activation, gated=False, bias=bias)

class GatedFeedForward(ModularFeedForward):
    """Legacy GatedFeedForward alias."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, 
                 activation: str = "gelu", bias: bool = True):
        super().__init__(d_model, d_ff, dropout, activation, gated=True, bias=bias)

class SwiGLU(ModularFeedForward):
    """Legacy SwiGLU alias."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = True):
        super().__init__(d_model, d_ff, dropout, activation="swish", gated=True, bias=bias)

class ReGLU(ModularFeedForward):
    """Legacy ReGLU alias."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = True):
        super().__init__(d_model, d_ff, dropout, activation="reglu", gated=True, bias=bias)

class GeGLU(ModularFeedForward):
    """Legacy GeGLU alias."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = True):
        super().__init__(d_model, d_ff, dropout, activation="geglu", gated=True, bias=bias)

class AdaptiveFeedForward(ModularFeedForward):
    """Legacy AdaptiveFeedForward alias."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, 
                 architecture: str = "standard", bias: bool = True):
        gated = architecture.lower() in ("gated", "swiglu", "reglu", "geglu")
        activation = architecture.lower()
        if activation == "standard": activation = "relu"
        super().__init__(d_model, d_ff, dropout, activation, gated=gated, bias=bias)
        self.architecture = architecture # Maintain property access

    def switch_architecture(self, new_architecture: str) -> None:
        """Shim for switching architecture (re-initializes internal state if needed)."""
        logger.warning("switch_architecture is deprecated and may not preserve weights.")
        self.__init__(self.d_model, self.d_ff, self.dropout_rate, new_architecture, self.use_bias)

# --- Factory Functions ---

def create_feed_forward(
    d_model: int,
    d_ff: int,
    dropout: float = 0.1,
    architecture: str = "standard",
    bias: bool = True,
    **kwargs: Any
) -> ModularFeedForward:
    """Create a modular feed-forward network instance."""
    arch = architecture.lower()
    if arch == "standard":
        return ModularFeedForward(d_model, d_ff, dropout, activation=kwargs.get("activation", "relu"), gated=False, bias=bias)
    elif arch == "gated":
        return ModularFeedForward(d_model, d_ff, dropout, activation=kwargs.get("activation", "gelu"), gated=True, bias=bias)
    elif arch == "swiglu":
        return SwiGLU(d_model, d_ff, dropout, bias)
    elif arch == "reglu":
        return ReGLU(d_model, d_ff, dropout, bias)
    elif arch == "geglu":
        return GeGLU(d_model, d_ff, dropout, bias)
    elif arch == "adaptive":
        return AdaptiveFeedForward(d_model, d_ff, dropout, architecture, bias)
    else:
        # Generic case
        return ModularFeedForward(d_model, d_ff, dropout, activation=architecture, gated=True, bias=bias)

def create_swiglu(d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = True) -> SwiGLU:
    return SwiGLU(d_model, d_ff, dropout, bias)

def create_gated_ffn(d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = True) -> GatedFeedForward:
    return GatedFeedForward(d_model, d_ff, dropout, bias=bias)

