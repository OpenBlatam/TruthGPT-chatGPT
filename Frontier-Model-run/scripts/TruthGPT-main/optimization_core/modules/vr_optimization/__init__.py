"""
VR/AR Optimization Package API
"""
from typing import Dict, Any
from optimization_core.utils.dependency_manager import resolve_lazy_import

_LAZY_IMPORTS = {
    'VROptimizationLevel': '.enums',
    'ImmersiveTechnology': '.enums',
    'VROptimizationConfig': '.config',
    'VROptimizationResult': '.results',
    'UltraVROptimizationEngine': '.system'
}

def __getattr__(name: str) -> Any:
    return resolve_lazy_import(name, __package__ or 'vr_optimization', _LAZY_IMPORTS)

__all__ = list(_LAZY_IMPORTS.keys())

