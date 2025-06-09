"""
Enhanced MLP optimizations for TruthGPT.
Integrates Multi-Layer_Perceptron.py optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
import math

class OptimizedLinear(nn.Module):
    """Optimized linear layer with advanced features."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x):
        return self.linear(x)

class SwiGLU(nn.Module):
    """SwiGLU activation function for enhanced MLP performance."""
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, bias: bool = False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * dim / 3)
            hidden_dim = int(2 * hidden_dim / 3) * 3
        
        try:
            from optimization_core import OptimizedLinear
            self.w1 = OptimizedLinear(dim, hidden_dim, bias=bias)
            self.w2 = OptimizedLinear(hidden_dim, dim, bias=bias)
            self.w3 = OptimizedLinear(dim, hidden_dim, bias=bias)
        except ImportError:
            self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
            self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
            self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class GatedMLP(nn.Module):
    """Gated MLP with configurable activation functions."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        activation: str = "swiglu",
        bias: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        
        if activation == "swiglu":
            self.gate = SwiGLU(input_dim, hidden_dim, bias)
        elif activation == "gelu":
            try:
                from optimization_core import OptimizedLinear
                self.gate_proj = OptimizedLinear(input_dim, hidden_dim, bias=bias)
                self.up_proj = OptimizedLinear(input_dim, hidden_dim, bias=bias)
                self.down_proj = OptimizedLinear(hidden_dim, output_dim, bias=bias)
            except ImportError:
                self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
                self.up_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
                self.down_proj = nn.Linear(hidden_dim, output_dim, bias=bias)
        else:
            try:
                from optimization_core import OptimizedLinear
                self.linear1 = OptimizedLinear(input_dim, hidden_dim, bias=bias)
                self.linear2 = OptimizedLinear(hidden_dim, output_dim, bias=bias)
            except ImportError:
                self.linear1 = nn.Linear(input_dim, hidden_dim, bias=bias)
                self.linear2 = nn.Linear(hidden_dim, output_dim, bias=bias)
            
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        if self.activation == "swiglu":
            output = self.gate(x)
        elif self.activation == "gelu":
            gate = F.gelu(self.gate_proj(x))
            up = self.up_proj(x)
            output = self.down_proj(gate * up)
        else:
            hidden = F.relu(self.linear1(x))
            output = self.linear2(hidden)
        
        if self.dropout is not None:
            output = self.dropout(output)
        
        return output

class ExpertMLP(nn.Module):
    """Expert MLP for Mixture of Experts architectures."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        expert_id: int = 0,
        activation: str = "swiglu",
        bias: bool = False
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
            
        self.expert_id = expert_id
        self.mlp = GatedMLP(input_dim, hidden_dim, output_dim, activation, bias)

    def forward(self, x):
        return self.mlp(x)

class MixtureOfExperts(nn.Module):
    """Mixture of Experts with top-k routing."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        output_dim: Optional[int] = None,
        activation: str = "swiglu",
        bias: bool = False,
        load_balancing_loss_coeff: float = 0.01
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
            
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.load_balancing_loss_coeff = load_balancing_loss_coeff
        
        try:
            from optimization_core import OptimizedLinear
            self.gate = OptimizedLinear(input_dim, num_experts, bias=False)
        except ImportError:
            self.gate = nn.Linear(input_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertMLP(input_dim, hidden_dim, output_dim, i, activation, bias)
            for i in range(num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(x_flat)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i:i+1]
            
            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_probs[mask] * expert_output
        
        output = output.view(batch_size, seq_len, self.output_dim)
        
        if self.training:
            load_balancing_loss = self._compute_load_balancing_loss(gate_probs)
            return output, load_balancing_loss
        else:
            return output

    def _compute_load_balancing_loss(self, gate_probs):
        """Compute load balancing loss to encourage equal expert usage."""
        expert_usage = gate_probs.mean(dim=0)
        target_usage = 1.0 / self.num_experts
        load_balancing_loss = torch.sum((expert_usage - target_usage) ** 2)
        return self.load_balancing_loss_coeff * load_balancing_loss

class AdaptiveMLP(nn.Module):
    """Adaptive MLP that adjusts capacity based on input complexity."""
    
    def __init__(
        self,
        input_dim: int,
        base_hidden_dim: int,
        max_hidden_dim: int,
        output_dim: Optional[int] = None,
        activation: str = "swiglu",
        bias: bool = False,
        complexity_threshold: float = 0.5
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
            
        self.input_dim = input_dim
        self.base_hidden_dim = base_hidden_dim
        self.max_hidden_dim = max_hidden_dim
        self.output_dim = output_dim
        self.complexity_threshold = complexity_threshold
        
        try:
            from optimization_core import OptimizedLinear
            self.complexity_predictor = OptimizedLinear(input_dim, 1)
        except ImportError:
            self.complexity_predictor = nn.Linear(input_dim, 1)
        
        self.base_mlp = GatedMLP(input_dim, base_hidden_dim, output_dim, activation, bias)
        self.extended_mlp = GatedMLP(input_dim, max_hidden_dim, output_dim, activation, bias)

    def forward(self, x):
        complexity_score = torch.sigmoid(self.complexity_predictor(x)).mean()
        
        if complexity_score > self.complexity_threshold:
            return self.extended_mlp(x)
        else:
            return self.base_mlp(x)

class EnhancedMLPOptimizations:
    """Utility class for applying enhanced MLP optimizations."""
    
    @staticmethod
    def replace_mlp_with_swiglu(model: nn.Module) -> nn.Module:
        """Replace standard MLP modules with SwiGLU variants."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential) and len(module) >= 2:
                if (isinstance(module[0], nn.Linear) and 
                    len(module) >= 3 and isinstance(module[2], nn.Linear)):
                    
                    parent_name = '.'.join(name.split('.')[:-1])
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                        child_name = name.split('.')[-1]
                    else:
                        parent = model
                        child_name = name
                    
                    input_dim = module[0].in_features
                    hidden_dim = module[0].out_features
                    output_dim = module[2].out_features
                    
                    swiglu_mlp = SwiGLU(input_dim, hidden_dim)
                    setattr(parent, child_name, swiglu_mlp)
        
        return model
    
    @staticmethod
    def add_mixture_of_experts(model: nn.Module, num_experts: int = 8, top_k: int = 2) -> nn.Module:
        """Add Mixture of Experts to appropriate MLP layers."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, SwiGLU, GatedMLP)):
                if hasattr(module, 'in_features') or hasattr(module, 'input_dim'):
                    parent_name = '.'.join(name.split('.')[:-1])
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                        child_name = name.split('.')[-1]
                    else:
                        parent = model
                        child_name = name
                    
                    if hasattr(module, 'in_features'):
                        input_dim = module.in_features
                        output_dim = module.out_features
                        hidden_dim = max(input_dim * 2, 512)
                    else:
                        input_dim = getattr(module, 'input_dim', 512)
                        output_dim = getattr(module, 'output_dim', input_dim)
                        hidden_dim = getattr(module, 'hidden_dim', input_dim * 2)
                    
                    moe = MixtureOfExperts(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        output_dim=output_dim
                    )
                    setattr(parent, child_name, moe)
        
        return model
    
    @staticmethod
    def get_mlp_optimization_report(model: nn.Module) -> dict:
        """Get a report of MLP optimization status."""
        total_mlp_modules = 0
        swiglu_modules = 0
        gated_mlp_modules = 0
        moe_modules = 0
        adaptive_mlp_modules = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total_mlp_modules += 1
            elif isinstance(module, SwiGLU):
                swiglu_modules += 1
                total_mlp_modules += 1
            elif isinstance(module, GatedMLP):
                gated_mlp_modules += 1
                total_mlp_modules += 1
            elif isinstance(module, MixtureOfExperts):
                moe_modules += 1
                total_mlp_modules += 1
            elif isinstance(module, AdaptiveMLP):
                adaptive_mlp_modules += 1
                total_mlp_modules += 1
        
        return {
            'total_mlp_modules': total_mlp_modules,
            'swiglu_modules': swiglu_modules,
            'gated_mlp_modules': gated_mlp_modules,
            'mixture_of_experts_modules': moe_modules,
            'adaptive_mlp_modules': adaptive_mlp_modules,
            'optimization_ratio': (swiglu_modules + gated_mlp_modules + moe_modules + adaptive_mlp_modules) / total_mlp_modules if total_mlp_modules > 0 else 0
        }

def create_swiglu(dim: int, hidden_dim: Optional[int] = None, **kwargs) -> SwiGLU:
    """Factory function to create SwiGLU."""
    return SwiGLU(dim, hidden_dim, **kwargs)

def create_gated_mlp(input_dim: int, hidden_dim: int, **kwargs) -> GatedMLP:
    """Factory function to create GatedMLP."""
    return GatedMLP(input_dim, hidden_dim, **kwargs)

def create_mixture_of_experts(input_dim: int, hidden_dim: int, **kwargs) -> MixtureOfExperts:
    """Factory function to create MixtureOfExperts."""
    return MixtureOfExperts(input_dim, hidden_dim, **kwargs)

def create_adaptive_mlp(input_dim: int, base_hidden_dim: int, max_hidden_dim: int, **kwargs) -> AdaptiveMLP:
    """Factory function to create AdaptiveMLP."""
    return AdaptiveMLP(input_dim, base_hidden_dim, max_hidden_dim, **kwargs)
