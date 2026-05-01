"""
Production Configuration Module

This module contains production configuration components.
"""

from __future__ import annotations

import importlib

__all__ = [
    'ProductionConfig',
    'Environment',
    'ConfigSource',
    'ConfigValidationRule',
    'ConfigMetadata',
    'create_production_config',
    'load_config_from_file',
    'create_environment_config',
    'production_config_context',
    'create_optimization_validation_rules',
    'create_monitoring_validation_rules',
]

_LAZY_IMPORTS = {
    'ProductionConfig': '..production_config',
    'Environment': '..production_config',
    'ConfigSource': '..production_config',
    'ConfigValidationRule': '..production_config',
    'ConfigMetadata': '..production_config',
    'create_production_config': '..production_config',
    'load_config_from_file': '..production_config',
    'create_environment_config': '..production_config',
    'production_config_context': '..production_config',
    'create_optimization_validation_rules': '..production_config',
    'create_monitoring_validation_rules': '..production_config',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for production config components."""
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


def list_available_config_components() -> list[str]:
    """List all available production config components."""
    return list(_LAZY_IMPORTS.keys())


