"""
Ultra-fast memory optimizations
Following deep learning best practices
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from dataclasses import dataclass
import gc


@dataclass
class MemoryConfig:
    """Memory optimization configuration"""
    use_gradient_checkpointing: bool = True
    use_activation_checkpointing: bool = True
    use_memory_efficient_attention: bool = True
    max_memory_usage: float = 0.8  # 80% of available memory
    cleanup_frequency: int = 100  # Cleanup every N steps


class MemoryOptimizer:
    """Ultra-fast memory optimizer"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.memory_usage = 0.0
        self.step_count = 0
    
    def optimize_memory_usage(self, model: nn.Module) -> nn.Module:
        """Apply memory optimizations to model"""
        if self.config.use_gradient_checkpointing:
            model = self._apply_gradient_checkpointing(model)
        
        if self.config.use_activation_checkpointing:
            model = self._apply_activation_checkpointing(model)
        
        return model
    
    def _apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        return model
    
    def _apply_activation_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply activation checkpointing"""
        # Enable activation checkpointing for transformer layers
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
        return model
    
    def cleanup_memory(self):
        """Clean up memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info['cuda_allocated'] = torch.cuda.memory_allocated() / 1e9  # GB
            memory_info['cuda_reserved'] = torch.cuda.memory_reserved() / 1e9  # GB
            memory_info['cuda_max_allocated'] = torch.cuda.max_memory_allocated() / 1e9  # GB
        
        return memory_info


class GradientCheckpointing:
    """Gradient checkpointing utilities"""
    
    @staticmethod
    def enable_checkpointing(model: nn.Module):
        """Enable gradient checkpointing"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    @staticmethod
    def disable_checkpointing(model: nn.Module):
        """Disable gradient checkpointing"""
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()


class MemoryEfficientAttention:
    """Memory-efficient attention implementation"""
    
    @staticmethod
    def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None,
                                   dropout_p: float = 0.0) -> torch.Tensor:
        """Memory-efficient scaled dot product attention"""
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p
            )
        else:
            # Fallback implementation
            return MemoryEfficientAttention._fallback_attention(q, k, v, mask, dropout_p)
    
    @staticmethod
    def _fallback_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          mask: Optional[torch.Tensor] = None,
                          dropout_p: float = 0.0) -> torch.Tensor:
        """Fallback attention implementation"""
        scale = 1.0 / (q.size(-1) ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.dropout(attn_weights, p=dropout_p, training=True)
        
        return torch.matmul(attn_weights, v)



