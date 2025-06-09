"""
Advanced Kernel Fusion Optimizations for TruthGPT
Implements sophisticated kernel fusion techniques for enhanced performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import warnings

class FusedLayerNormLinear(nn.Module):
    """Fused LayerNorm + Linear operation for better performance."""
    
    def __init__(self, normalized_shape: int, linear_in_features: int, linear_out_features: int, 
                 eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
        self.linear_weight = nn.Parameter(torch.randn(linear_out_features, linear_in_features))
        if bias:
            self.linear_bias = nn.Parameter(torch.zeros(linear_out_features))
        else:
            self.register_parameter('linear_bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        normalized = normalized * self.weight + self.bias
        
        output = F.linear(normalized, self.linear_weight, self.linear_bias)
        return output

class FusedAttentionMLP(nn.Module):
    """Fused Attention + MLP for transformer blocks."""
    
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.mlp_hidden_size = hidden_size * mlp_ratio
        
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.mlp_gate = nn.Linear(hidden_size, self.mlp_hidden_size * 2, bias=False)
        self.mlp_down = nn.Linear(self.mlp_hidden_size, hidden_size)
        
        try:
            from optimization_core import OptimizedLayerNorm
            self.norm1 = OptimizedLayerNorm(hidden_size)
            self.norm2 = OptimizedLayerNorm(hidden_size)
        except ImportError:
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        residual = x
        x = self.norm1(x)
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        attn_output = self.out_proj(attn_output)
        x = residual + attn_output
        
        residual = x
        x = self.norm2(x)
        
        gate_proj = self.mlp_gate(x)
        gate, up = gate_proj.chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.mlp_down(x)
        x = residual + x
        
        return x

class KernelFusionOptimizer:
    """Advanced kernel fusion optimization utilities."""
    
    @staticmethod
    def fuse_layernorm_linear(model: nn.Module) -> nn.Module:
        """Fuse LayerNorm + Linear operations where possible."""
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Sequential) and len(module) >= 2:
                if (isinstance(module[0], (nn.LayerNorm, )) and 
                    isinstance(module[1], nn.Linear)):
                    
                    layernorm = module[0]
                    linear = module[1]
                    
                    fused = FusedLayerNormLinear(
                        normalized_shape=layernorm.normalized_shape[0],
                        linear_in_features=linear.in_features,
                        linear_out_features=linear.out_features,
                        eps=layernorm.eps,
                        bias=linear.bias is not None
                    )
                    
                    fused.weight.data.copy_(layernorm.weight.data)
                    fused.bias.data.copy_(layernorm.bias.data)
                    fused.linear_weight.data.copy_(linear.weight.data)
                    if linear.bias is not None:
                        fused.linear_bias.data.copy_(linear.bias.data)
                    
                    parent_name = '.'.join(name.split('.')[:-1])
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                        child_name = name.split('.')[-1]
                        setattr(parent, child_name, fused)
        
        return model
    
    @staticmethod
    def fuse_attention_mlp(model: nn.Module) -> nn.Module:
        """Fuse attention and MLP operations in transformer blocks."""
        for name, module in model.named_modules():
            if hasattr(module, 'attention') and hasattr(module, 'mlp'):
                hidden_size = getattr(module.attention, 'hidden_size', 512)
                num_heads = getattr(module.attention, 'num_heads', 8)
                
                fused_block = FusedAttentionMLP(hidden_size, num_heads)
                
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    child_name = name.split('.')[-1]
                    setattr(parent, child_name, fused_block)
        
        return model
    
    @staticmethod
    def apply_kernel_fusion(model: nn.Module, fusion_config: Dict[str, bool]) -> nn.Module:
        """Apply various kernel fusion optimizations."""
        if fusion_config.get('fuse_layernorm_linear', True):
            model = KernelFusionOptimizer.fuse_layernorm_linear(model)
        
        if fusion_config.get('fuse_attention_mlp', True):
            model = KernelFusionOptimizer.fuse_attention_mlp(model)
        
        return model
    
    @staticmethod
    def get_fusion_report(model: nn.Module) -> Dict[str, Any]:
        """Get report of kernel fusion optimizations applied."""
        fused_modules = 0
        total_modules = 0
        fusion_types = {}
        
        for module in model.modules():
            total_modules += 1
            if isinstance(module, (FusedLayerNormLinear, FusedAttentionMLP)):
                fused_modules += 1
                fusion_type = type(module).__name__
                fusion_types[fusion_type] = fusion_types.get(fusion_type, 0) + 1
        
        return {
            'total_modules': total_modules,
            'fused_modules': fused_modules,
            'fusion_ratio': fused_modules / total_modules if total_modules > 0 else 0,
            'fusion_types': fusion_types
        }

def create_kernel_fusion_optimizer(config: Dict[str, Any]) -> KernelFusionOptimizer:
    """Create kernel fusion optimizer from configuration."""
    return KernelFusionOptimizer()
