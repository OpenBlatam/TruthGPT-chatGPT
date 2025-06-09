"""
Optimization Profiles for Different Use Cases
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class OptimizationProfile:
    """Optimization profile configuration."""
    name: str
    memory_optimizations: Dict[str, Any]
    computational_optimizations: Dict[str, Any]
    accuracy_trade_offs: Dict[str, Any]

def get_optimization_profiles() -> Dict[str, OptimizationProfile]:
    """Get predefined optimization profiles."""
    return {
        'speed_optimized': OptimizationProfile(
            name='Speed Optimized',
            memory_optimizations={
                'enable_fp16': True,
                'enable_gradient_checkpointing': True,
                'enable_quantization': True,
                'quantization_bits': 8,
                'enable_pruning': True,
                'pruning_ratio': 0.2
            },
            computational_optimizations={
                'use_fused_attention': True,
                'enable_kernel_fusion': True,
                'optimize_batch_size': True,
                'use_flash_attention': True
            },
            accuracy_trade_offs={
                'acceptable_accuracy_loss': 0.05,
                'prioritize_speed': True
            }
        ),
        'accuracy_optimized': OptimizationProfile(
            name='Accuracy Optimized',
            memory_optimizations={
                'enable_fp16': False,
                'enable_gradient_checkpointing': False,
                'enable_quantization': False,
                'enable_pruning': False
            },
            computational_optimizations={
                'use_fused_attention': False,
                'enable_kernel_fusion': False,
                'optimize_batch_size': False,
                'use_flash_attention': False
            },
            accuracy_trade_offs={
                'acceptable_accuracy_loss': 0.0,
                'prioritize_speed': False
            }
        ),
        'balanced': OptimizationProfile(
            name='Balanced',
            memory_optimizations={
                'enable_fp16': True,
                'enable_gradient_checkpointing': True,
                'enable_quantization': True,
                'quantization_bits': 16,
                'enable_pruning': True,
                'pruning_ratio': 0.1
            },
            computational_optimizations={
                'use_fused_attention': True,
                'enable_kernel_fusion': True,
                'optimize_batch_size': True,
                'use_flash_attention': True
            },
            accuracy_trade_offs={
                'acceptable_accuracy_loss': 0.02,
                'prioritize_speed': False
            }
        )
    }

def apply_optimization_profile(model, profile_name: str = 'balanced'):
    """Apply optimization profile to model."""
    profiles = get_optimization_profiles()
    
    if profile_name not in profiles:
        profile_name = 'balanced'
    
    profile = profiles[profile_name]
    
    from .memory_optimizations import create_memory_optimizer
    from .computational_optimizations import create_computational_optimizer
    
    memory_optimizer = create_memory_optimizer(profile.memory_optimizations)
    computational_optimizer = create_computational_optimizer(profile.computational_optimizations)
    
    model = memory_optimizer.optimize_model(model)
    model = computational_optimizer.optimize_model(model)
    
    return model, profile
