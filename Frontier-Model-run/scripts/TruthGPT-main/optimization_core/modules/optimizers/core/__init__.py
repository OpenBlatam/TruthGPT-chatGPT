"""
Core Optimizers Module

This module contains the foundational optimizer classes and utilities:
- Base optimizer classes
- Component optimizers
- Generic optimizer system
- Optimization techniques
- Metrics
- Optimization pipelines
"""

from __future__ import annotations

import importlib

__all__ = [
    'BaseTruthGPTOptimizer',
    'UnifiedTruthGPTOptimizer',
    'OptimizationLevel',
    'OptimizationResult',
    'ComponentOptimizer',
    'get_component_optimizer',
    'OptimizationTechnique',
    'GradientCheckpointingTechnique',
]

# Lazy imports for better startup performance
_LAZY_IMPORTS = {
    'BaseTruthGPTOptimizer': '.base_truthgpt_optimizer',
    'UnifiedTruthGPTOptimizer': '.base_truthgpt_optimizer',
    'OptimizationLevel': '.base_truthgpt_optimizer',
    'OptimizationResult': '.base_truthgpt_optimizer',
    'ComponentOptimizer': '.component_optimizers',
    'get_component_optimizer': '.component_optimizers',
    'OptimizationTechnique': '.techniques',
    'GradientCheckpointingTechnique': '.techniques',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for core optimizer modules."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        available = sorted(_LAZY_IMPORTS.keys())[:10]
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Available: {', '.join(available)}..."
        )
    
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



