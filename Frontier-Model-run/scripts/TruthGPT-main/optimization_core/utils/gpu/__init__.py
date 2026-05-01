"""
GPU Utilities Module

This module contains GPU-specific utilities and CUDA kernel optimizations.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Dict, List

__all__ = [
    'GPUUtils',
    'CUDAOptimizations',
    'OptimizedLayerNorm',
    'OptimizedRMSNorm',
    'EnhancedCUDAOptimizations',
]

_LAZY_IMPORTS = {
    'GPUUtils': 'optimization_core.modules.acceleration.gpu.gpu_utils',
    'CUDAOptimizations': 'optimization_core.modules.acceleration.gpu.cuda_kernels',
    'OptimizedLayerNorm': 'optimization_core.modules.acceleration.gpu.cuda_kernels',
    'OptimizedRMSNorm': 'optimization_core.modules.acceleration.gpu.cuda_kernels',
    'EnhancedCUDAOptimizations': 'optimization_core.modules.acceleration.gpu.enhanced_kernels',
}

# Thread-safe cache for loaded modules
_import_cache: Dict[str, Any] = {}
_cache_lock = threading.RLock()


def __getattr__(name: str):
    """Lazy import system for GPU utility modules."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    with _cache_lock:
        if name in _import_cache:
            return _import_cache[name]
        
        if name not in _LAZY_IMPORTS:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
        
        module_path = _LAZY_IMPORTS[name]
        try:
            # Use absolute imports
            module = importlib.import_module(module_path)
            obj = getattr(module, name)
            _import_cache[name] = obj
            return obj
        except (ImportError, AttributeError) as e:
            raise AttributeError(
                f"module '{__name__}' has no attribute '{name}'. "
                f"Failed to import from '{module_path}': {e}"
            ) from e



