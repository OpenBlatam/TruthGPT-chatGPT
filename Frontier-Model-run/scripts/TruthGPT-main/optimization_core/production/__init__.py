"""
Production Module

This module provides organized access to production components:
- config: Production configuration
- optimization: Production optimization
- monitoring: Production monitoring
- testing: Production testing
"""

from __future__ import annotations

import importlib

# Lazy imports for organized submodules
_LAZY_IMPORTS = {
    'config': '.config',
    'optimization': '.optimization',
    'monitoring': '.monitoring',
    'testing': '.testing',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for production submodules."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name in _import_cache:
        return _import_cache[name]
    
    module_path = _LAZY_IMPORTS[name]
    try:
        module = importlib.import_module(module_path, package=__package__)
        _import_cache[name] = module
        return module
    except (ImportError, AttributeError) as e:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Failed to import: {e}"
        ) from e


def list_available_production_modules() -> list[str]:
    """List all available production submodules."""
    return list(_LAZY_IMPORTS.keys())


__all__ = [
    "config",
    "optimization",
    "monitoring",
    "testing",
    "list_available_production_modules",
]


