"""
Advanced normalization techniques for TruthGPT optimization.
Integrates layernorm.py, RMS.py, LlamaRSMNorm.py, and CRMS.py optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
import warnings
import math

class AdvancedRMSNorm(nn.Module):
    """Advanced RMS Normalization with enhanced features."""
    
    def __init__(self, input_dim: int, epsilon: float = 1e-8, scale_init: float = 1.0):
        super(AdvancedRMSNorm, self).__init__()
        self.input_dim = input_dim
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(input_dim) * scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.epsilon)
        return self.scale * (x / norm)

class LlamaRMSNorm(nn.Module):
    """LlamaRMSNorm implementation with FusedRMSNorm fallback."""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

try:
    from functools import partial
    from apex.normalization import FusedRMSNorm
    
    LlamaRMSNormFused = partial(FusedRMSNorm, eps=1e-6)
    FUSED_RMS_AVAILABLE = True
    print('Discovered apex.normalization.FusedRMSNorm - will use it for enhanced performance')
except ImportError:
    LlamaRMSNormFused = LlamaRMSNorm
    FUSED_RMS_AVAILABLE = False
except Exception:
    print('discovered apex but it failed to load, falling back to LlamaRMSNorm')
    LlamaRMSNormFused = LlamaRMSNorm
    FUSED_RMS_AVAILABLE = False

class CRMSNorm(nn.Module):
    """Conditional RMS Normalization with conditioning input."""
    
    def __init__(self, input_dim: int, cond_dim: int, epsilon: float = 1e-8):
        super().__init__()
        self.input_dim = input_dim
        self.epsilon = epsilon
        try:
            from optimization_core.cuda_kernels import OptimizedLinear
            self.condition_proj = OptimizedLinear(cond_dim, input_dim)
        except ImportError:
            try:
                from optimization_core.enhanced_mlp import EnhancedLinear
                self.condition_proj = EnhancedLinear(cond_dim, input_dim)
            except ImportError:
                self.condition_proj = nn.Linear(cond_dim, input_dim)
        self.scale = nn.Parameter(torch.ones(input_dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.epsilon)
        cond_scale = self.condition_proj(cond).unsqueeze(1)
        return (x / norm) * (self.scale + cond_scale)

class ONNXLayerNormOptimizer:
    """ONNX LayerNorm optimization utilities."""
    
    @staticmethod
    def exclude_layer_norm_nodes(graph, model):
        """Exclude LayerNorm nodes from ONNX optimization."""
        layer_norm_subgraphs = []
        for add_node in model.get_nodes_by_op_type("Add"):
            layer_norm_components = model.match_parent_path(
                add_node,
                ["Mul", "Div", "Sqrt", "Add", "ReduceMean", "Pow", "Sub", "ReduceMean"],
                [0, 0, 1, 0, 0, 0, 0, 1],
            )

            if layer_norm_components is not None:
                layer_norm_components.append(add_node)
                layer_norm_subgraphs.append(layer_norm_components)

        ln_components = (node.name for ln in layer_norm_subgraphs for node in ln)
        return set(), set(ln_components)

class AdvancedNormalizationOptimizations:
    """Utility class for applying advanced normalization optimizations."""
    
    @staticmethod
    def replace_rms_norm_with_advanced(model: nn.Module) -> nn.Module:
        """Replace RMSNorm modules with AdvancedRMSNorm."""
        for name, module in model.named_modules():
            if hasattr(module, '__class__') and 'RMSNorm' in module.__class__.__name__:
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    child_name = name.split('.')[-1]
                else:
                    parent = model
                    child_name = name
                
                if hasattr(module, 'weight') and hasattr(module, 'eps'):
                    advanced_norm = AdvancedRMSNorm(
                        input_dim=module.weight.shape[0],
                        epsilon=getattr(module, 'eps', 1e-8),
                        scale_init=1.0
                    )
                    advanced_norm.scale.data.copy_(module.weight.data)
                    setattr(parent, child_name, advanced_norm)
        
        return model
    
    @staticmethod
    def replace_with_llama_rms_norm(model: nn.Module) -> nn.Module:
        """Replace RMSNorm modules with LlamaRMSNorm (with FusedRMSNorm if available)."""
        for name, module in model.named_modules():
            if hasattr(module, '__class__') and 'RMSNorm' in module.__class__.__name__:
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    child_name = name.split('.')[-1]
                else:
                    parent = model
                    child_name = name
                
                if hasattr(module, 'weight'):
                    if FUSED_RMS_AVAILABLE:
                        llama_norm = LlamaRMSNormFused(
                            normalized_shape=module.weight.shape[0]
                        )
                    else:
                        llama_norm = LlamaRMSNorm(
                            hidden_size=module.weight.shape[0],
                            eps=getattr(module, 'eps', 1e-6)
                        )
                    
                    if hasattr(llama_norm, 'weight'):
                        llama_norm.weight.data.copy_(module.weight.data)
                    setattr(parent, child_name, llama_norm)
        
        return model
    
    @staticmethod
    def add_conditional_normalization(model: nn.Module, cond_dim: int) -> nn.Module:
        """Add conditional normalization layers where appropriate."""
        return model
    
    @staticmethod
    def get_normalization_report(model: nn.Module) -> dict:
        """Get a report of normalization optimization status."""
        total_norm_modules = 0
        advanced_norm_modules = 0
        llama_norm_modules = 0
        crms_norm_modules = 0
        
        for module in model.modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                total_norm_modules += 1
            elif isinstance(module, AdvancedRMSNorm):
                advanced_norm_modules += 1
                total_norm_modules += 1
            elif isinstance(module, LlamaRMSNorm):
                llama_norm_modules += 1
                total_norm_modules += 1
            elif isinstance(module, CRMSNorm):
                crms_norm_modules += 1
                total_norm_modules += 1
        
        return {
            'total_normalization_modules': total_norm_modules,
            'advanced_rms_norm_modules': advanced_norm_modules,
            'llama_rms_norm_modules': llama_norm_modules,
            'crms_norm_modules': crms_norm_modules,
            'fused_rms_available': FUSED_RMS_AVAILABLE,
            'optimization_ratio': (advanced_norm_modules + llama_norm_modules + crms_norm_modules) / total_norm_modules if total_norm_modules > 0 else 0
        }

def create_advanced_rms_norm(input_dim: int, **kwargs) -> AdvancedRMSNorm:
    """Factory function to create AdvancedRMSNorm."""
    return AdvancedRMSNorm(input_dim, **kwargs)

def create_llama_rms_norm(hidden_size: int, **kwargs) -> Union[LlamaRMSNorm, 'FusedRMSNorm']:
    """Factory function to create LlamaRMSNorm (with FusedRMSNorm if available)."""
    if FUSED_RMS_AVAILABLE:
        return LlamaRMSNormFused(normalized_shape=hidden_size, **kwargs)
    else:
        return LlamaRMSNorm(hidden_size, **kwargs)

def create_crms_norm(input_dim: int, cond_dim: int, **kwargs) -> CRMSNorm:
    """Factory function to create CRMSNorm."""
    return CRMSNorm(input_dim, cond_dim, **kwargs)
