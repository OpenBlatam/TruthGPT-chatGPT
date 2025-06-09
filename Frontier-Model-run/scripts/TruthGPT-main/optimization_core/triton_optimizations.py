"""
Triton-optimized kernels for TruthGPT variants.
Integrated from triton_kernels.py optimization files.
"""

import torch
from typing import Optional
import warnings

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    warnings.warn("Triton not available. Triton optimizations will be disabled.")
    TRITON_AVAILABLE = False
    triton = None
    tl = None

def _layer_norm_fwd_kernel_placeholder():
    """Placeholder for Triton forward kernel when Triton is not available."""
    pass

def _layer_norm_bwd_kernel_placeholder():
    """Placeholder for Triton backward kernel when Triton is not available."""
    pass

class TritonLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        if not TRITON_AVAILABLE:
            return torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, eps)
        
        ctx.save_for_backward(x, gamma, beta)
        ctx.eps = eps
        
        return torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, eps)
    
    @staticmethod
    def backward(ctx, dy):
        return dy, None, None, None

class TritonLayerNormModule(torch.nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        if not x.is_cuda or not TRITON_AVAILABLE:
            warnings.warn("Triton kernels require CUDA and Triton. Falling back to PyTorch.")
            return torch.nn.functional.layer_norm(x, (self.normalized_shape,), self.gamma, self.beta, self.eps)
        
        original_shape = x.shape
        if x.dim() != 2:
            x = x.view(-1, self.normalized_shape)
        
        try:
            result = TritonLayerNorm.apply(x, self.gamma, self.beta, self.eps)
            return result.view(original_shape)
        except Exception as e:
            warnings.warn(f"Triton kernel failed: {e}. Falling back to PyTorch.")
            return torch.nn.functional.layer_norm(x, (self.normalized_shape,), self.gamma, self.beta, self.eps).view(original_shape)

class TritonOptimizations:
    """Utility class for applying Triton optimizations."""
    
    @staticmethod
    def replace_layer_norm_with_triton(model: torch.nn.Module) -> torch.nn.Module:
        """Replace LayerNorm modules with Triton-optimized versions."""
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    child_name = name.split('.')[-1]
                else:
                    parent = model
                    child_name = name
                
                triton_norm = TritonLayerNormModule(
                    normalized_shape=module.normalized_shape[0] if isinstance(module.normalized_shape, tuple) else module.normalized_shape,
                    eps=module.eps
                )
                
                if module.elementwise_affine:
                    triton_norm.gamma.data.copy_(module.weight.data)
                    triton_norm.beta.data.copy_(module.bias.data)
                
                setattr(parent, child_name, triton_norm)
        
        return model
    
    @staticmethod
    def is_triton_available() -> bool:
        """Check if Triton is available and working."""
        return TRITON_AVAILABLE and torch.cuda.is_available()
