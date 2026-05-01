"""
Backward-compatibility shim for enhanced_mlp.
This module has been moved to optimization_core.utils.enhanced_mlp.
"""

import warnings
from optimization_core.utils.enhanced_mlp import (
    OptimizedLinear,
    SwiGLU,
    GatedMLP,
    ExpertMLP,
    MixtureOfExperts,
    AdaptiveMLP,
    EnhancedMLPOptimizations,
    create_swiglu,
    create_gated_mlp,
    create_mixture_of_experts,
    create_adaptive_mlp
)

warnings.warn(
    "optimization_core.optimizers.techniques.enhanced_mlp is deprecated. "
    "Please use optimization_core.utils.enhanced_mlp instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'OptimizedLinear',
    'SwiGLU',
    'GatedMLP',
    'ExpertMLP',
    'MixtureOfExperts',
    'AdaptiveMLP',
    'EnhancedMLPOptimizations',
    'create_swiglu',
    'create_gated_mlp',
    'create_mixture_of_experts',
    'create_adaptive_mlp'
]

