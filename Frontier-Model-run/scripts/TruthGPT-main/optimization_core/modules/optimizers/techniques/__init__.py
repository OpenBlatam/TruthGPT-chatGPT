"""
Optimization Techniques Module

This module contains optimization techniques and computational optimizations.
Includes advanced normalization, positional encodings, enhanced MLP, RL pruning,
computational optimizations, and Triton support.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

__all__ = [
    # Computational optimizations
    'ComputationalOptimizer',
    'FusedAttention',
    'BatchOptimizer',
    'create_computational_optimizer',
    # Triton optimizations
    'TritonOptimizations',
    'TritonLayerNorm',
    'TritonLayerNormModule',
    'rotary_embed',
    'block_copy',
    # Advanced Normalization
    'AdvancedRMSNorm',
    'LlamaRMSNorm',
    'CRMSNorm',
    'AdvancedNormalizationOptimizations',
    'create_advanced_rms_norm',
    'create_llama_rms_norm',
    'create_crms_norm',
    # Positional Encodings
    'SinusoidalPositionalEmbedding',
    'RotaryEmbedding',
    'LlamaRotaryEmbedding',
    'FixedLlamaRotaryEmbedding',
    'AliBi',
    'PositionalEncodingOptimizations',
    'create_rotary_embedding',
    'create_llama_rotary_embedding',
    'create_alibi',
    'create_sinusoidal_embedding',
    # Enhanced MLP
    'OptimizedLinear',
    'SwiGLU',
    'GatedMLP',
    'MixtureOfExperts',
    'AdaptiveMLP',
    'EnhancedMLPOptimizations',
    'create_swiglu',
    'create_gated_mlp',
    'create_mixture_of_experts',
    'create_adaptive_mlp',
    # RL Pruning
    'RLPruningAgent',
    'RLPruning',
    'RLPruningOptimizations',
    'create_rl_pruning',
    'create_rl_pruning_agent',
]

_LAZY_IMPORTS = {
    # Computational optimizations
    'ComputationalOptimizer': '.computational_optimizations',
    'FusedAttention': '.computational_optimizations',
    'BatchOptimizer': '.computational_optimizations',
    'create_computational_optimizer': '.computational_optimizations',
    
    # Triton optimizations
    'TritonOptimizations': '.triton_optimizations',
    'TritonLayerNorm': '.triton_optimizations',
    'TritonLayerNormModule': '.triton_optimizations',
    'rotary_embed': '.triton_optimizations',
    'block_copy': '.triton_optimizations',
    
    # Advanced Normalization
    'AdvancedRMSNorm': '.advanced_normalization',
    'LlamaRMSNorm': '.advanced_normalization',
    'CRMSNorm': '.advanced_normalization',
    'AdvancedNormalizationOptimizations': '.advanced_normalization',
    'create_advanced_rms_norm': '.advanced_normalization',
    'create_llama_rms_norm': '.advanced_normalization',
    'create_crms_norm': '.advanced_normalization',
    
    # Positional Encodings
    'SinusoidalPositionalEmbedding': '.positional_encodings',
    'RotaryEmbedding': '.positional_encodings',
    'LlamaRotaryEmbedding': '.positional_encodings',
    'FixedLlamaRotaryEmbedding': '.positional_encodings',
    'AliBi': '.positional_encodings',
    'PositionalEncodingOptimizations': '.positional_encodings',
    'create_rotary_embedding': '.positional_encodings',
    'create_llama_rotary_embedding': '.positional_encodings',
    'create_alibi': '.positional_encodings',
    'create_sinusoidal_embedding': '.positional_encodings',
    
    # Enhanced MLP
    'OptimizedLinear': '.enhanced_mlp',
    'SwiGLU': '.enhanced_mlp',
    'GatedMLP': '.enhanced_mlp',
    'MixtureOfExperts': '.enhanced_mlp',
    'AdaptiveMLP': '.enhanced_mlp',
    'EnhancedMLPOptimizations': '.enhanced_mlp',
    'create_swiglu': '.enhanced_mlp',
    'create_gated_mlp': '.enhanced_mlp',
    'create_mixture_of_experts': '.enhanced_mlp',
    'create_adaptive_mlp': '.enhanced_mlp',
    
    # RL Pruning
    'RLPruningAgent': '.rl_pruning',
    'RLPruning': '.rl_pruning',
    'RLPruningOptimizations': '.rl_pruning',
    'create_rl_pruning': '.rl_pruning',
    'create_rl_pruning_agent': '.rl_pruning',
}

_import_cache = {}


def __getattr__(name: str) -> Any:
    """Lazy import system for optimization techniques."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name in _import_cache:
        return _import_cache[name]
    
    module_path = _LAZY_IMPORTS[name]
    try:
        module = importlib.import_module(module_path, package=__package__)
        obj = getattr(module, name)
        _import_cache[name] = obj
        return obj
    except (ImportError, AttributeError) as e:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Failed to import: {e}"
        ) from e


def list_available_techniques() -> List[str]:
    """List all available optimization techniques."""
    return sorted(list(_LAZY_IMPORTS.keys()))

