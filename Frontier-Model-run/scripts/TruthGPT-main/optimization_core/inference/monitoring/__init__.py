"""
Inference Monitoring Components

This module contains monitoring components: metrics collector and observability.
"""

from __future__ import annotations

import importlib

__all__ = [
    'MetricsCollector',
    'MetricsSnapshot',
    'Observability',
]

_LAZY_IMPORTS = {
    'MetricsCollector': '..metrics',
    'MetricsSnapshot': '..metrics',
    'Observability': '..observability',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for inference monitoring components."""
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


def list_available_monitoring_components() -> list[str]:
    """List all available monitoring components."""
    return list(_LAZY_IMPORTS.keys())


def get_monitoring_component_info(component_name: str) -> dict[str, any]:
    """Get information about a monitoring component."""
    if component_name not in _LAZY_IMPORTS:
        raise ValueError(f"Unknown monitoring component: {component_name}")
    
    return {
        'name': component_name,
        'module': _LAZY_IMPORTS[component_name],
        'available': component_name in _import_cache or True,
    }


