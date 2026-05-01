"""
Configs Module

This module provides configuration loading and schema validation.
It works alongside the config/ module which provides configuration classes.
"""

from __future__ import annotations

import importlib

# Direct imports
from .loader import (
    load_config,
    parse_overrides,
    deep_merge,
)
from .schema import (
    AppCfg,
    ModelCfg,
    TrainingCfg,
)

# Lazy imports for presets
_LAZY_IMPORTS = {
    'presets': '.presets',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for configs submodules."""
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


def list_available_config_modules() -> list[str]:
    """List all available config submodules."""
    return list(_LAZY_IMPORTS.keys())


__all__ = [
    "load_config",
    "parse_overrides",
    "deep_merge",
    "AppCfg",
    "ModelCfg",
    "TrainingCfg",
    "presets",
    "list_available_config_modules",
]


