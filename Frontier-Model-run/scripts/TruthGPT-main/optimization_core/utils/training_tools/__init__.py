"""
Training Tools Module

This module contains utilities for training monitoring, visualization, and analysis.
"""

from __future__ import annotations

import importlib

__all__ = [
    'visualize_checkpoints',
    'summarize_run',
    'compare_runs',
    'get_run_info',
    'monitor_training',
    'cleanup_runs',
]

_LAZY_IMPORTS = {
    'visualize_checkpoints': '..visualize_training',
    'summarize_run': '..visualize_training',
    'compare_runs': '..compare_runs',
    'get_run_info': '..compare_runs',
    'monitor_training': '..monitor_training',
    'cleanup_runs': '..cleanup_runs',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for training tool modules."""
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


def list_available_training_tools() -> list[str]:
    """List all available training tools."""
    return list(_LAZY_IMPORTS.keys())


def get_training_tool_info(tool_name: str) -> dict[str, any]:
    """Get information about a training tool."""
    if tool_name not in _LAZY_IMPORTS:
        raise ValueError(f"Unknown training tool: {tool_name}")
    
    return {
        'name': tool_name,
        'module': _LAZY_IMPORTS[tool_name],
        'available': tool_name in _import_cache or True,
    }


