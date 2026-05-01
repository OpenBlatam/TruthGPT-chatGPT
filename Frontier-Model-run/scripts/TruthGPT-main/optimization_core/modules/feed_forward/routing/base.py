"""
Base Router Definition
======================

Unified base class and types for MoE routers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..experts.base import ExpertType


@dataclass
class RoutingResult:
    """Result of a routing decision."""
    # Indices of selected experts for each token: [batch_size * seq_len, top_k]
    expert_indices: torch.Tensor
    # Scores/probabilities for selected experts: [batch_size * seq_len, top_k]
    expert_scores: torch.Tensor
    # Raw routing scores (before softmax/top-k): [batch_size * seq_len, num_experts]
    raw_scores: torch.Tensor
    # Metadata about the routing process
    metadata: Dict[str, Any]


class BaseRouter(nn.Module, ABC):
    """
    Abstract base class for all routers in the MoE system.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        **kwargs: Any
    ) -> None:
        """
        Initialize the router.

        Args:
            hidden_size: Input hidden size.
            num_experts: Number of experts to route to.
            kwargs: Additional configuration parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.config = kwargs

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any
    ) -> RoutingResult:
        """
        Forward pass through the router.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).
            kwargs: Additional context for routing.

        Returns:
            RoutingResult object.
        """
        pass

