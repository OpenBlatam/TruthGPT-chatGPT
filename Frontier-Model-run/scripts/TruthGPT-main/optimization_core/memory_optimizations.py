"""
Advanced Memory and Quantization Optimizations for TruthGPT Models
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimizations."""
    enable_fp16: bool = True
    enable_gradient_checkpointing: bool = True
    enable_quantization: bool = True
    quantization_bits: int = 8
    enable_pruning: bool = True
    pruning_ratio: float = 0.1

class MemoryOptimizer:
    """Advanced memory optimization utilities."""
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply comprehensive memory optimizations to model."""
        if self.config.enable_fp16:
            model = model.half()
        
        if self.config.enable_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
        
        if self.config.enable_quantization:
            model = self.apply_quantization(model)
        
        if self.config.enable_pruning:
            model = self.apply_pruning(model)
        
        return model
    
    def apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to model."""
        try:
            return torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        except Exception:
            return model
    
    def apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning to model."""
        try:
            import torch.nn.utils.prune as prune
            
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=self.config.pruning_ratio)
        except ImportError:
            pass
        
        return model
    
    def get_memory_stats(self, model: nn.Module) -> Dict[str, float]:
        """Get detailed memory statistics for model."""
        stats = {
            'model_size_mb': 0.0,
            'parameter_count': 0,
            'buffer_size_mb': 0.0
        }
        
        param_size = 0
        buffer_size = 0
        param_count = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            param_count += param.nelement()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        stats['model_size_mb'] = (param_size + buffer_size) / 1024 / 1024
        stats['parameter_count'] = param_count
        stats['buffer_size_mb'] = buffer_size / 1024 / 1024
        
        return stats

def create_memory_optimizer(config: Dict[str, Any]) -> MemoryOptimizer:
    """Create memory optimizer from configuration."""
    opt_config = MemoryOptimizationConfig(
        enable_fp16=config.get('enable_fp16', True),
        enable_gradient_checkpointing=config.get('enable_gradient_checkpointing', True),
        enable_quantization=config.get('enable_quantization', True),
        quantization_bits=config.get('quantization_bits', 8),
        enable_pruning=config.get('enable_pruning', True),
        pruning_ratio=config.get('pruning_ratio', 0.1)
    )
    
    return MemoryOptimizer(opt_config)
