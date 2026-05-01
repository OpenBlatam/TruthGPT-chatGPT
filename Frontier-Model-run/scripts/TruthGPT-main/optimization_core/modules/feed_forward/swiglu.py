import torch
import torch.nn as nn
import torch.nn.functional as F

from optimization_core.factories.registry import Registry

# We might not have a global FEED_FORWARD registry yet, but we'll define the class 
# so it can be used/imported.

class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function from 'GLU Variants Improve Transformer'.
    
    SwiGLU(x) = Swish(xW) * (xV)
    
    Where W and V are learnable weight matrices.
    In many implementations (like Llama), this implies 3 linear projections:
    1. gate_proj (W)
    2. up_proj   (V)
    3. down_proj (Projection back to embedding dim)
    
    This module implements the Activation part, or the full MLP block depending on usage.
    Here we implement the Full FeedForward Block variants.
    """
    def __init__(self, hidden_dim: int, intermediate_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=bias) # Gate
        self.w2 = nn.Linear(hidden_dim, intermediate_dim, bias=bias) # Up
        self.w3 = nn.Linear(intermediate_dim, hidden_dim, bias=bias) # Down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu is Swish
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class SwigluMLP(nn.Module):
    """
    Drop-in replacement for standard FeedForward, using SwiGLU.
    """
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.n_embd
        # Typically intermediate is 4*h, but for SwiGLU it's often 8/3 * h or similar to keep param count same
        # We'll assume config provides intermediate_size or we default to 4x
        intermediate_dim = getattr(config, "intermediate_size", 4 * hidden_dim)
        
        self.swiglu = SwiGLU(hidden_dim, intermediate_dim, bias=getattr(config, "bias", False))
        
    def forward(self, x):
        return self.swiglu(x)

