
"""
Tools Module

This module provides access to integration tests and utility tools.
"""

from __future__ import annotations

import importlib

__all__ = [
    'test_compiler_integration',
    'test_kv_cache',
]

_LAZY_IMPORTS = {
    'test_compiler_integration': '..test_compiler_integration',
    'test_kv_cache': '..test_kv_cache',
    'sota_research_injector': '.sota_injector',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for tool modules."""
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


def list_available_tools() -> list[str]:
    """List all available tools."""
    return list(_LAZY_IMPORTS.keys())


def get_tool_info(tool_name: str) -> dict[str, any]:
    """Get information about a tool."""
    if tool_name not in _LAZY_IMPORTS:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    return {
        'name': tool_name,
        'module': _LAZY_IMPORTS[tool_name],
        'available': tool_name in _import_cache or True,
    }


