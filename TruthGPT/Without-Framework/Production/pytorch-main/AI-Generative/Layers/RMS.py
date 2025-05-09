import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, input_dim: int, epsilon: float = 1e-8, scale_init: float = 1.0):
        """
        Implements the RMS Normalization layer as a variant of Layer Normalization.
        
        Args:
            input_dim (int): The dimensionality of the input tensor.
            epsilon (float): A small constant added to the denominator for numerical stability.
            scale_init (float): Initial scale value for the learned parameter.
        """
        super(RMSNorm, self).__init__()
        self.input_dim = input_dim
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(input_dim) * scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMS Normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        
        Returns:
            torch.Tensor: Normalized tensor of the same shape as the input.
        """
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.epsilon)
        return self.scale * (x / norm)
