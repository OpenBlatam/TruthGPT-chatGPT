"""
Unified MoE Block
=================

High-level block that integrates routing and expert specialized computation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..routing.base import BaseRouter, RoutingResult
from ..experts.base import BaseExpert


class MoEBlock(nn.Module):
    """
    Modular Mixture of Experts Block.
    Dispatches input tokens to a set of experts based on router decisions.
    """

    def __init__(
        self,
        router: BaseRouter,
        experts: Union[nn.ModuleList, List[BaseExpert]],
        expert_capacity_factor: float = 1.25,
        use_residual: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize the MoE Block.

        Args:
            router: Router instance.
            experts: Collection of experts.
            expert_capacity_factor: Factor for expert capacity buffer.
            use_residual: Whether to use a residual connection around the MoE block.
        """
        super().__init__()
        self.router = router
        if isinstance(experts, list):
            self.experts = nn.ModuleList(experts)
        else:
            self.experts = experts
            
        self.num_experts = len(self.experts)
        self.expert_capacity_factor = expert_capacity_factor
        self.use_residual = use_residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass through the MoE Block.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size].
            kwargs: Additional context for routing and experts.

        Returns:
            Output tensor [batch_size, seq_len, hidden_size].
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. Get routing decisions
        routing_result = self.router(hidden_states, **kwargs)
        
        # 2. Dispatch and aggregate expert outputs
        # For simplicity in this unified implementation, we use a more efficient
        # approach than the original token-by-token loop.
        
        flat_hidden = hidden_states.view(-1, hidden_size)
        expert_indices = routing_result.expert_indices  # [N, top_k]
        expert_scores = routing_result.expert_scores    # [N, top_k]
        
        final_output = torch.zeros_like(flat_hidden)
        
        # Process each expert's assigned tokens in batches
        for expert_id in range(self.num_experts):
            # Find tokens assigned to this expert
            # (mask shape: [N, top_k])
            mask = (expert_indices == expert_id)
            
            # If no tokens for this expert, skip
            if not mask.any():
                continue
                
            # Get flat indices of tokens corresponding to TRUE in mask
            # N_i is number of tokens for this expert
            token_indices, k_indices = torch.where(mask)
            
            # Extract inputs for this expert
            expert_inputs = flat_hidden[token_indices]
            
            # Execute expert
            expert_outputs = self.experts[expert_id](expert_inputs)
            
            # Scale by routing scores and add to final output
            # (expert_scores shape: [N, top_k])
            scores = expert_scores[token_indices, k_indices].unsqueeze(-1)
            final_output.index_add_(0, token_indices, expert_outputs * scores)
            
        # Reshape back
        output = final_output.view(batch_size, seq_len, hidden_size)
        
        # 3. Handle residual connection
        if self.use_residual:
            output = output + hidden_states
            
        if kwargs.get("return_routing_info", False):
            return output, routing_result.metadata
            
        return output

    def get_stats(self) -> Dict[str, Any]:
        """Get block statistics."""
        return {
            "num_experts": self.num_experts,
            "experts_info": [e.get_info() if hasattr(e, 'get_info') else str(type(e)) for e in self.experts],
            "router_config": self.router.config
        }

