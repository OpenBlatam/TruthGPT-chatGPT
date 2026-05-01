"""
Memory Utilities Module

This module contains memory optimization and management utilities.
"""

from __future__ import annotations

import importlib

__all__ = [
    'MemoryOptimizer',
    'MemoryOptimizationConfig',
    'create_memory_optimizer',
    'TensorPool',
    'ActivationCache',
    'MemoryPoolingOptimizer',
    'MemoryUtils',
]

_LAZY_IMPORTS = {
    'MemoryOptimizer': '..memory_optimizations',
    'MemoryOptimizationConfig': '..memory_optimizations',
    'create_memory_optimizer': '..memory_optimizations',
    'TensorPool': '..memory_pooling',
    'ActivationCache': '..memory_pooling',
    'MemoryPoolingOptimizer': '..memory_pooling',
    'MemoryUtils': '..memory_utils',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for memory utility modules."""
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



