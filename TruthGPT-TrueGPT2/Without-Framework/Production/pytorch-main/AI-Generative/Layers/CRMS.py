import torch
import torch.nn as nn

class CRMSNorm(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int, epsilon: float = 1e-8):
        super().__init__()
        self.input_dim = input_dim
        self.epsilon = epsilon
        self.condition_proj = nn.Linear(cond_dim, input_dim)
        self.scale = nn.Parameter(torch.ones(input_dim)) # type: ignore

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            cond (torch.Tensor): Conditioning tensor of shape (batch_size, cond_dim)

        Returns:
            torch.Tensor: Conditioned and normalized tensor of shape (batch_size, seq_len, input_dim)
        """
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.epsilon)
        cond_scale = self.condition_proj(cond).unsqueeze(1)  # shape (batch_size, 1, input_dim)
        return (x / norm) * (self.scale + cond_scale)
