"""
KV Cache Optimizers Module

This module contains optimizers specifically for KV cache optimization.
"""

from __future__ import annotations

import importlib

__all__ = [
    'KVCacheOptimizer',
    'UltraKVCacheOptimizer',
]

_LAZY_IMPORTS = {
    'KVCacheOptimizer': '..kv_cache_optimizer',
    'UltraKVCacheOptimizer': '..ultra_kv_cache_optimizer',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for KV cache optimizer modules."""
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



