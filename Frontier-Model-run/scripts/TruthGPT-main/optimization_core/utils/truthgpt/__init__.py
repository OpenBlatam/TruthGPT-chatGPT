"""
TruthGPT Utilities Module

This module contains TruthGPT-specific utilities and components.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Dict, List

__all__ = [
    'OptimizationLevel',
    'DeviceType',
    'PrecisionType',
    'TruthGPTConfig',
    'BaseTruthGPTOptimizer',
    'TruthGPTDeviceManager',
    'TruthGPTPrecisionManager',
    'TruthGPTMemoryManager',
    'TruthGPTPerformanceManager',
    'TruthGPTAttentionOptimizer',
    'TruthGPTQuantizationOptimizer',
    'TruthGPTPruningOptimizer',
    'TruthGPTIntegratedOptimizer',
    'create_truthgpt_config',
    'create_truthgpt_optimizer',
    'quick_truthgpt_optimization',
    'truthgpt_optimization_context',
]

_LAZY_IMPORTS = {
    # Core components from truthgpt_core
    'OptimizationLevel': 'optimization_core.modules.truthgpt.core',
    'DeviceType': 'optimization_core.modules.truthgpt.core',
    'PrecisionType': 'optimization_core.modules.truthgpt.core',
    'TruthGPTConfig': 'optimization_core.modules.truthgpt.core',
    'BaseTruthGPTOptimizer': 'optimization_core.modules.truthgpt.core',
    'TruthGPTDeviceManager': 'optimization_core.modules.truthgpt.core',
    'TruthGPTPrecisionManager': 'optimization_core.modules.truthgpt.core',
    'TruthGPTMemoryManager': 'optimization_core.modules.truthgpt.core',
    'TruthGPTPerformanceManager': 'optimization_core.modules.truthgpt.core',
    'TruthGPTAttentionOptimizer': 'optimization_core.modules.truthgpt.core',
    'TruthGPTQuantizationOptimizer': 'optimization_core.modules.truthgpt.core',
    'TruthGPTPruningOptimizer': 'optimization_core.modules.truthgpt.core',
    'TruthGPTIntegratedOptimizer': 'optimization_core.modules.truthgpt.core',
    'create_truthgpt_config': 'optimization_core.modules.truthgpt.core',
    'create_truthgpt_optimizer': 'optimization_core.modules.truthgpt.core',
    'quick_truthgpt_optimization': 'optimization_core.modules.truthgpt.core',
    'truthgpt_optimization_context': 'optimization_core.modules.truthgpt.core',
}

# Thread-safe cache for loaded modules
_import_cache: Dict[str, Any] = {}
_cache_lock = threading.RLock()


def __getattr__(name: str):
    """Lazy import system for TruthGPT utility modules."""
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


def list_available_truthgpt_components() -> list[str]:
    """List all available TruthGPT components."""
    return list(_LAZY_IMPORTS.keys())


def get_truthgpt_component_info(component_name: str) -> dict[str, any]:
    """Get information about a TruthGPT component."""
    if component_name not in _LAZY_IMPORTS:
        raise ValueError(f"Unknown TruthGPT component: {component_name}")
    
    return {
        'name': component_name,
        'module': _LAZY_IMPORTS[component_name],
        'available': component_name in _import_cache or True,
    }


