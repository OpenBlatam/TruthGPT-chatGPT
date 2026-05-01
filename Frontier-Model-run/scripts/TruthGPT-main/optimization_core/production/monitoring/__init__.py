"""
Production Monitoring Module

This module contains production monitoring components.
"""

from __future__ import annotations

import importlib

__all__ = [
    'ProductionMonitor',
    'AlertLevel',
    'MetricType',
    'Alert',
    'Metric',
    'PerformanceSnapshot',
    'create_production_monitor',
    'production_monitoring_context',
    'setup_monitoring_for_optimizer',
]

_LAZY_IMPORTS = {
    'ProductionMonitor': '..production_monitoring',
    'AlertLevel': '..production_monitoring',
    'MetricType': '..production_monitoring',
    'Alert': '..production_monitoring',
    'Metric': '..production_monitoring',
    'PerformanceSnapshot': '..production_monitoring',
    'create_production_monitor': '..production_monitoring',
    'production_monitoring_context': '..production_monitoring',
    'setup_monitoring_for_optimizer': '..production_monitoring',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for production monitoring components."""
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
    """List all available production monitoring components."""
    return list(_LAZY_IMPORTS.keys())


