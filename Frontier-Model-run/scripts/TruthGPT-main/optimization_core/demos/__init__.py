"""
Demos Module

This module provides access to demonstration scripts and examples.
"""

from __future__ import annotations

import importlib

__all__ = [
    'compiler_demo',
    'enhanced_compiler_demo',
    'demo_gradio_llm',
]

_LAZY_IMPORTS = {
    'compiler_demo': '..compiler_demo',
    'enhanced_compiler_demo': '..enhanced_compiler_demo',
    'demo_gradio_llm': '..demo_gradio_llm',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for demo modules."""
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


def list_available_demos() -> list[str]:
    """List all available demos."""
    return list(_LAZY_IMPORTS.keys())


def get_demo_info(demo_name: str) -> dict[str, any]:
    """Get information about a demo."""
    if demo_name not in _LAZY_IMPORTS:
        raise ValueError(f"Unknown demo: {demo_name}")
    
    return {
        'name': demo_name,
        'module': _LAZY_IMPORTS[demo_name],
        'available': demo_name in _import_cache or True,
    }


