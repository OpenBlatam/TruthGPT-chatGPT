import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def _layer_norm_fwd_kernel(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    N, D,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
    **meta
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute mean
    x = tl.load(x_ptr + pid * D + tl.arange(0, D), mask=tl.arange(0, D) < D)
    mean = tl.sum(x, axis=0) / D
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / D
    
    # Compute normalized output
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * inv_std
    
    # Apply affine transformation
    gamma = tl.load(gamma_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D)
    beta = tl.load(beta_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D)
    y = x_norm * gamma + beta
    
    # Store output
    tl.store(y_ptr + pid * D + tl.arange(0, D), y, mask=tl.arange(0, D) < D)

@triton.jit
def _layer_norm_bwd_kernel(
    dy_ptr, x_ptr, gamma_ptr, beta_ptr,
    dx_ptr, dgamma_ptr, dbeta_ptr,
    N, D,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
    **meta
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Load input
    x = tl.load(x_ptr + pid * D + tl.arange(0, D), mask=tl.arange(0, D) < D)
    dy = tl.load(dy_ptr + pid * D + tl.arange(0, D), mask=tl.arange(0, D) < D)
    gamma = tl.load(gamma_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D)
    
    # Compute mean and variance
    mean = tl.sum(x, axis=0) / D
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Compute gradients
    x_norm = x_centered * inv_std
    dgamma = tl.sum(dy * x_norm, axis=0)
    dbeta = tl.sum(dy, axis=0)
    
    # Compute dx
    dx = (dy * gamma) * inv_std
    dx = dx - (tl.sum(dx, axis=0) / D)
    dx = dx - (x_centered * tl.sum(dx * x_centered, axis=0) / (D * (var + eps)))
    
    # Store gradients
    tl.store(dx_ptr + pid * D + tl.arange(0, D), dx, mask=tl.arange(0, D) < D)
    tl.store(dgamma_ptr + tl.arange(0, D), dgamma, mask=tl.arange(0, D) < D)
    tl.store(dbeta_ptr + tl.arange(0, D), dbeta, mask=tl.arange(0, D) < D)

class DeepSeekLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        # Save for backward
        ctx.save_for_backward(x, gamma, beta)
        ctx.eps = eps
        
        # Get dimensions
        N, D = x.shape
        
        # Allocate output
        y = torch.empty_like(x)
        
        # Launch kernel
        grid = (N,)
        _layer_norm_fwd_kernel[grid](
            x, gamma, beta, y,
            N, D, eps,
            BLOCK_SIZE=1024,
            num_warps=8
        )
        
        return y
    
    @staticmethod
    def backward(ctx, dy):
        # Get saved tensors
        x, gamma, beta = ctx.saved_tensors
        eps = ctx.eps
        
        # Get dimensions
        N, D = x.shape
        
        # Allocate gradients
        dx = torch.empty_like(x)
        dgamma = torch.zeros_like(gamma)
        dbeta = torch.zeros_like(beta)
        
        # Launch kernel
        grid = (N,)
        _layer_norm_bwd_kernel[grid](
            dy, x, gamma, beta,
            dx, dgamma, dbeta,
            N, D, eps,
            BLOCK_SIZE=1024,
            num_warps=8
        )
        
        return dx, dgamma, dbeta, None

class DeepSeekLayerNormModule(torch.nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        return DeepSeekLayerNorm.apply(x, self.gamma, self.beta, self.eps) 