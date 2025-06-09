"""
Advanced Quantization Optimizations for TruthGPT
Implements sophisticated quantization techniques for enhanced performance and memory efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union
import warnings
import math

class QuantizedLinear(nn.Module):
    """Advanced quantized linear layer with dynamic quantization."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 quantization_bits: int = 8, dynamic_quantization: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_bits = quantization_bits
        self.dynamic_quantization = dynamic_quantization
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('weight_zero_point', torch.zeros(1, dtype=torch.int8))
        
    def quantize_weight(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize weights to specified bit precision."""
        if self.quantization_bits == 8:
            qmin, qmax = -128, 127
        elif self.quantization_bits == 4:
            qmin, qmax = -8, 7
        else:
            qmin, qmax = -(2**(self.quantization_bits-1)), 2**(self.quantization_bits-1) - 1
        
        min_val = weight.min()
        max_val = weight.max()
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        zero_point = torch.clamp(zero_point, qmin, qmax).round()
        
        quantized_weight = torch.clamp(
            (weight / scale + zero_point).round(), qmin, qmax
        )
        
        return quantized_weight, scale, zero_point
    
    def dequantize_weight(self, quantized_weight: torch.Tensor, 
                         scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """Dequantize weights back to float."""
        return scale * (quantized_weight - zero_point)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_quantization and self.training:
            quantized_weight, scale, zero_point = self.quantize_weight(self.weight)
            dequantized_weight = self.dequantize_weight(quantized_weight, scale, zero_point)
            return F.linear(x, dequantized_weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)

class QuantizedLayerNorm(nn.Module):
    """Quantized LayerNorm with reduced precision computations."""
    
    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], 
                 eps: float = 1e-5, elementwise_affine: bool = True,
                 quantization_bits: int = 8):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.quantization_bits = quantization_bits
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantization_bits < 32:
            x_quantized = self.quantize_tensor(x)
            mean = x_quantized.mean(-1, keepdim=True)
            var = ((x_quantized - mean) ** 2).mean(-1, keepdim=True)
            normalized = (x_quantized - mean) / torch.sqrt(var + self.eps)
        else:
            mean = x.mean(-1, keepdim=True)
            var = ((x - mean) ** 2).mean(-1, keepdim=True)
            normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.elementwise_affine:
            normalized = normalized * self.weight + self.bias
        
        return normalized
    
    def quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to specified bit precision."""
        if self.quantization_bits >= 32:
            return tensor
        
        if self.quantization_bits == 8:
            qmin, qmax = -128, 127
        elif self.quantization_bits == 4:
            qmin, qmax = -8, 7
        else:
            qmin, qmax = -(2**(self.quantization_bits-1)), 2**(self.quantization_bits-1) - 1
        
        scale = (tensor.max() - tensor.min()) / (qmax - qmin)
        zero_point = qmin - tensor.min() / scale
        
        quantized = torch.clamp(
            (tensor / scale + zero_point).round(), qmin, qmax
        )
        
        return scale * (quantized - zero_point)

class MixedPrecisionOptimizer:
    """Mixed precision optimization utilities."""
    
    @staticmethod
    def convert_to_fp16(model: nn.Module) -> nn.Module:
        """Convert model to FP16 precision."""
        return model.half()
    
    @staticmethod
    def convert_to_bf16(model: nn.Module) -> nn.Module:
        """Convert model to BF16 precision."""
        return model.to(torch.bfloat16)
    
    @staticmethod
    def apply_mixed_precision(model: nn.Module, precision_config: Dict[str, str]) -> nn.Module:
        """Apply mixed precision based on layer types."""
        for name, module in model.named_modules():
            layer_type = type(module).__name__
            if layer_type in precision_config:
                precision = precision_config[layer_type]
                if precision == 'fp16':
                    module.half()
                elif precision == 'bf16':
                    module.to(torch.bfloat16)
                elif precision == 'fp32':
                    module.float()
        
        return model

class AdvancedQuantizationOptimizer:
    """Advanced quantization optimization coordinator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quantization_bits = config.get('quantization_bits', 8)
        self.dynamic_quantization = config.get('dynamic_quantization', True)
        self.mixed_precision = config.get('mixed_precision', True)
        self.quantize_weights = config.get('quantize_weights', True)
        self.quantize_activations = config.get('quantize_activations', False)
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply comprehensive quantization optimizations."""
        if self.quantize_weights:
            model = self.replace_linear_with_quantized(model)
        
        if self.quantize_activations:
            model = self.replace_layernorm_with_quantized(model)
        
        if self.mixed_precision:
            precision_config = self.config.get('precision_config', {
                'Linear': 'fp16',
                'LayerNorm': 'fp32',
                'Embedding': 'fp16'
            })
            model = MixedPrecisionOptimizer.apply_mixed_precision(model, precision_config)
        
        return model
    
    def replace_linear_with_quantized(self, model: nn.Module) -> nn.Module:
        """Replace Linear layers with quantized versions."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    child_name = name.split('.')[-1]
                else:
                    parent = model
                    child_name = name
                
                quantized_linear = QuantizedLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    quantization_bits=self.quantization_bits,
                    dynamic_quantization=self.dynamic_quantization
                )
                
                quantized_linear.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    quantized_linear.bias.data.copy_(module.bias.data)
                
                setattr(parent, child_name, quantized_linear)
        
        return model
    
    def replace_layernorm_with_quantized(self, model: nn.Module) -> nn.Module:
        """Replace LayerNorm layers with quantized versions."""
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    child_name = name.split('.')[-1]
                else:
                    parent = model
                    child_name = name
                
                quantized_layernorm = QuantizedLayerNorm(
                    normalized_shape=module.normalized_shape,
                    eps=module.eps,
                    elementwise_affine=module.elementwise_affine,
                    quantization_bits=self.quantization_bits
                )
                
                if module.elementwise_affine:
                    quantized_layernorm.weight.data.copy_(module.weight.data)
                    quantized_layernorm.bias.data.copy_(module.bias.data)
                
                setattr(parent, child_name, quantized_layernorm)
        
        return model
    
    def get_quantization_report(self, model: nn.Module) -> Dict[str, Any]:
        """Get quantization optimization report."""
        total_params = sum(p.numel() for p in model.parameters())
        quantized_params = 0
        quantized_modules = 0
        total_modules = 0
        
        for module in model.modules():
            total_modules += 1
            if isinstance(module, (QuantizedLinear, QuantizedLayerNorm)):
                quantized_modules += 1
                quantized_params += sum(p.numel() for p in module.parameters())
        
        memory_savings = 1 - (self.quantization_bits / 32)
        
        return {
            'total_parameters': total_params,
            'quantized_parameters': quantized_params,
            'quantized_modules': quantized_modules,
            'total_modules': total_modules,
            'quantization_ratio': quantized_params / total_params if total_params > 0 else 0,
            'quantization_bits': self.quantization_bits,
            'estimated_memory_savings': memory_savings,
            'dynamic_quantization_enabled': self.dynamic_quantization
        }

def create_quantization_optimizer(config: Dict[str, Any]) -> AdvancedQuantizationOptimizer:
    """Create advanced quantization optimizer from configuration."""
    return AdvancedQuantizationOptimizer(config)
