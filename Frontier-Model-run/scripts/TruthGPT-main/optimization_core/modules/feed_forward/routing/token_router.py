"""
Token-Level Router Implementation
=================================

Router that makes decisions at the individual token level.
Migrated from PiMoE TokenLevelRouter.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseRouter, RoutingResult
from ..experts.base import ExpertType


class TokenLevelRouter(BaseRouter):
    """
    Router that assigns each token to one or more experts.
    Supports gating, load balancing, and auxiliary losses.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        expert_types: List[ExpertType],
        router_hidden_size: Optional[int] = None,
        top_k: int = 1,
        temperature: float = 1.0,
        load_balance_weight: float = 0.1,
        use_gating: bool = True,
        use_auxiliary_loss: bool = True,
        dropout: float = 0.1,
        **kwargs: Any
    ) -> None:
        """
        Initialize the TokenLevelRouter.

        Args:
            hidden_size: Input hidden size.
            num_experts: Number of experts to route to.
            expert_types: List of available expert types.
            router_hidden_size: Hidden size for the router network.
            top_k: Number of experts to select per token.
            temperature: Temperature for softmax scaling.
            load_balance_weight: Weight for load balancing loss.
            use_gating: Whether to use a gating mechanism.
            use_auxiliary_loss: Whether to calculate auxiliary losses.
            dropout: Dropout probability.
        """
        super().__init__(hidden_size, num_experts, **kwargs)
        
        self.expert_types = expert_types
        self.router_hidden_size = router_hidden_size or hidden_size // 2
        self.top_k = top_k
        self.temperature = temperature
        self.load_balance_weight = load_balance_weight
        self.use_gating = use_gating
        self.use_auxiliary_loss = use_auxiliary_loss

        # Core router network
        self.router_network = nn.Sequential(
            nn.Linear(hidden_size, self.router_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.router_hidden_size, num_experts)
        )

        # Expert type classifier (for auxiliary consistency loss)
        self.expert_type_classifier = nn.Linear(hidden_size, len(expert_types))

        # Gating mechanism
        if self.use_gating:
            self.gate_network = nn.Sequential(
                nn.Linear(hidden_size, self.router_hidden_size),
                nn.Tanh(),
                nn.Linear(self.router_hidden_size, 1),
                nn.Sigmoid()
            )

        # Statistics tracking
        self.register_buffer('expert_usage_count', torch.zeros(num_experts))

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any
    ) -> RoutingResult:
        """
        Forward pass through the token-level router.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size].

        Returns:
            RoutingResult object.
        """
        batch_size, seq_len, hidden_size = x.shape
        flat_x = x.reshape(-1, hidden_size)

        # 1. Get routing scores
        logits = self.router_network(flat_x)  # [N, num_experts]
        logits = logits / self.temperature

        # 2. Apply gating
        if self.use_gating:
            gate_scores = self.gate_network(flat_x)
            logits = logits * gate_scores

        # 3. Expert Type Prediction (for metadata/loss)
        type_logits = self.expert_type_classifier(flat_x)
        type_probs = F.softmax(type_logits, dim=-1)

        # 4. Selection (Top-k)
        probs = F.softmax(logits, dim=-1)
        expert_scores, expert_indices = torch.topk(probs, k=self.top_k, dim=-1)

        # 5. Load Balancing Update (if training)
        if self.training:
            # Simple tracking for now
            with torch.no_grad():
                self.expert_usage_count.scatter_add_(0, expert_indices.view(-1), torch.ones_like(expert_indices.view(-1)).float())

        # 6. Auxiliary Losses
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.use_auxiliary_loss:
            # Load balance loss
            usage_freq = torch.mean(probs, dim=0)
            target_freq = torch.ones_like(usage_freq) / self.num_experts
            lb_loss = F.mse_loss(usage_freq, target_freq)
            aux_loss = aux_loss + self.load_balance_weight * lb_loss

        # Collect metadata
        metadata = {
            "type_probs": type_probs,
            "auxiliary_loss": aux_loss,
            "probabilities": probs
        }

        return RoutingResult(
            expert_indices=expert_indices,
            expert_scores=expert_scores,
            raw_scores=logits,
            metadata=metadata
        )

