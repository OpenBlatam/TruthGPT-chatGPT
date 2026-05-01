"""
Scripts Module

This module provides access to utility scripts and command-line tools.
"""

from __future__ import annotations

import importlib

__all__ = [
    'cli',
    'build',
    'build_trainer',
    'train_llm',
    'init_project',
    'install_extras',
    'migration_helper',
    'validate_config',
]

_LAZY_IMPORTS = {
    'cli': '..cli',
    'build': '..build',
    'build_trainer': '..build_trainer',
    'train_llm': '..train_llm',
    'init_project': '..init_project',
    'install_extras': '..install_extras',
    'migration_helper': '..migration_helper',
    'validate_config': '..validate_config',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for script modules."""
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


def list_available_scripts() -> list[str]:
    """List all available scripts."""
    return list(_LAZY_IMPORTS.keys())


def get_script_info(script_name: str) -> dict[str, any]:
    """Get information about a script."""
    if script_name not in _LAZY_IMPORTS:
        raise ValueError(f"Unknown script: {script_name}")
    
    return {
        'name': script_name,
        'module': _LAZY_IMPORTS[script_name],
        'available': script_name in _import_cache or True,
    }


