import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class MoEShard(nn.Module):
    def __init__(self, dim, hidden_dim, n_local_experts):
        super().__init__()
        self.n_local_experts = n_local_experts
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim) for _ in range(n_local_experts)
        ])

    def forward(self, x, expert_mask, weights, local_indices):
        """
        Each process computes outputs for its local experts only.
        
        Args:
            x: (B, D) input
            expert_mask: (B,) bool mask for tokens routed to this shard
            weights: (B, k) softmax weights
            local_indices: (B,) int indices of local experts

        Returns:
            Local output tensor (B, D), filled only for matching tokens
        """
        B, D = x.shape
        y = torch.zeros_like(x)

        # Only work on inputs routed to this rank
        x_local = x[expert_mask]
        weights_local = weights[expert_mask]

        for i in range(self.n_local_experts):
            token_mask = (local_indices[expert_mask] == i)
            if not token_mask.any():
                continue
            x_i = x_local[token_mask]
            w_i = weights_local[token_mask]

            out = self.experts[i](x_i)
            y[expert_mask][token_mask] += out * w_i.unsqueeze(-1)

        return y
