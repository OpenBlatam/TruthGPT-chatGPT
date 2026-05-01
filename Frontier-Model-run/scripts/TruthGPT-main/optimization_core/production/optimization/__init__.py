"""
Production Optimization Module

This module contains production optimization components.
"""

from __future__ import annotations

import importlib

__all__ = [
    'ProductionOptimizer',
    'ProductionOptimizationConfig',
    'OptimizationLevel',
    'PerformanceProfile',
    'PerformanceMetrics',
    'CircuitBreaker',
    'create_production_optimizer',
]

_LAZY_IMPORTS = {
    'ProductionOptimizer': '..production_optimizer',
    'ProductionOptimizationConfig': '..production_optimizer',
    'OptimizationLevel': '..production_optimizer',
    'PerformanceProfile': '..production_optimizer',
    'PerformanceMetrics': '..production_optimizer',
    'CircuitBreaker': '..production_optimizer',
    'create_production_optimizer': '..production_optimizer',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for production optimization components."""
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


def list_available_optimization_components() -> list[str]:
    """List all available production optimization components."""
    return list(_LAZY_IMPORTS.keys())


