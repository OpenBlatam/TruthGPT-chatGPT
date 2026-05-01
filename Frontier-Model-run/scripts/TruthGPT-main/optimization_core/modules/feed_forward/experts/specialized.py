"""
Specialized Experts
===================

Specialized expert implementations for different computation tasks.
"""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from .base import BaseExpert, ExpertType


class SpecializedExpert(BaseExpert):
    """
    Consolidated specialized expert that can be configured for various tasks.
    Migrated from PiMoEExpert and ModularExperts.
    """

    def __init__(
        self,
        hidden_size: int,
        expert_type: ExpertType,
        intermediate_size: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        **kwargs: Any
    ) -> None:
        """
        Initialize the specialized expert.

        Args:
            hidden_size: Input hidden size.
            expert_type: Type of expert.
            intermediate_size: Intermediate dimension for FFN.
            activation: Activation function name.
            dropout: Dropout probability.
        """
        super().__init__(hidden_size, expert_type, **kwargs)
        
        self.intermediate_size = intermediate_size or hidden_size * 4
        
        if expert_type == ExpertType.REASONING:
            self.expert = self._build_reasoning_expert(activation, dropout)
        elif expert_type == ExpertType.COMPUTATION:
            self.expert = self._build_computation_expert(activation, dropout)
        elif expert_type == ExpertType.MATHEMATICAL:
            self.expert = self._build_mathematical_expert(activation, dropout)
        elif expert_type == ExpertType.LOGICAL:
            self.expert = self._build_logical_expert(activation, dropout)
        else:
            self.expert = self._build_general_expert(activation, dropout)

    def _build_reasoning_expert(self, activation: str, dropout: float) -> nn.Module:
        """Build expert specialized for reasoning tasks."""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size * 2),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(self.intermediate_size * 2, self.hidden_size)
        )

    def _build_computation_expert(self, activation: str, dropout: float) -> nn.Module:
        """Build expert specialized for high-precision computation."""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size),
            self._get_activation(activation),
            nn.Linear(self.intermediate_size, self.intermediate_size),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(self.intermediate_size, self.hidden_size)
        )

    def _build_mathematical_expert(self, activation: str, dropout: float) -> nn.Module:
        """Build expert specialized for mathematical operations."""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size),
            nn.ReLU(), # Math often benefits from ReLU
            nn.Linear(self.intermediate_size, self.hidden_size)
        )

    def _build_logical_expert(self, activation: str, dropout: float) -> nn.Module:
        """Build expert specialized for logical reasoning."""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.intermediate_size, self.hidden_size)
        )

    def _build_general_expert(self, activation: str, dropout: float) -> nn.Module:
        """Build a generic expert."""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(self.intermediate_size, self.hidden_size)
        )

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        if name.lower() == "relu":
            return nn.ReLU()
        elif name.lower() == "gelu":
            return nn.GELU()
        elif name.lower() in ("silu", "swish"):
            return nn.SiLU()
        return nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert."""
        return self.expert(x)

