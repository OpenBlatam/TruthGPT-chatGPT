"""
Base Expert Definition
======================

Unified base class and types for modular experts.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


class ExpertType(Enum):
    """Types of experts for different computation tasks."""
    REASONING = "reasoning"
    COMPUTATION = "computation"
    LANGUAGE = "language"
    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    GENERAL = "general"


class BaseExpert(nn.Module, ABC):
    """
    Abstract base class for all experts in the MoE system.
    """

    def __init__(
        self,
        hidden_size: int,
        expert_type: ExpertType = ExpertType.GENERAL,
        **kwargs: Any
    ) -> None:
        """
        Initialize the expert.

        Args:
            hidden_size: Input hidden size.
            expert_type: Specialized type of this expert.
            kwargs: Additional configuration parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_type = expert_type
        self.config = kwargs

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the expert.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get information about the expert."""
        return {
            "type": self.expert_type.value,
            "hidden_size": self.hidden_size,
            "params": sum(p.numel() for p in self.parameters()),
            "config": self.config
        }

