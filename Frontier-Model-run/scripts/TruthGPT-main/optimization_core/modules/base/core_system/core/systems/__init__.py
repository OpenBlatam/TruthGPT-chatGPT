"""
Core Systems Module

This module contains core infrastructure systems: factory, event system, service registry, plugin system, and module loader.
"""

from __future__ import annotations

import importlib

__all__ = [
    # Dynamic Factory
    'DynamicFactory',
    'factory',
    'register_component',
    'create_factory',
    # Event System
    'EventEmitter',
    'EventType',
    'Event',
    'get_event_emitter',
    'emit_event',
    'on_event',
    # Service Registry
    'ServiceRegistry',
    'ServiceContainer',
    'register_service',
    'get_service',
    # Plugin System
    'Plugin',
    'PluginManager',
    'get_plugin_manager',
    # Module Loader
    'ModuleLoader',
    'get_module_loader',
    'lazy_load',
]

_LAZY_IMPORTS = {
    # Dynamic Factory
    'DynamicFactory': '..dynamic_factory',
    'factory': '..dynamic_factory',
    'register_component': '..dynamic_factory',
    'create_factory': '..dynamic_factory',
    # Event System
    'EventEmitter': '..event_system',
    'EventType': '..event_system',
    'Event': '..event_system',
    'get_event_emitter': '..event_system',
    'emit_event': '..event_system',
    'on_event': '..event_system',
    # Service Registry
    'ServiceRegistry': '..service_registry',
    'ServiceContainer': '..service_registry',
    'register_service': '..service_registry',
    'get_service': '..service_registry',
    # Plugin System
    'Plugin': '..plugin_system',
    'PluginManager': '..plugin_system',
    'get_plugin_manager': '..plugin_system',
    # Module Loader
    'ModuleLoader': '..module_loader',
    'get_module_loader': '..module_loader',
    'lazy_load': '..module_loader',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for core systems components."""
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


def list_available_systems() -> list[str]:
    """List all available core systems."""
    return list(_LAZY_IMPORTS.keys())


def get_system_info(system_name: str) -> dict[str, any]:
    """Get information about a core system."""
    if system_name not in _LAZY_IMPORTS:
        raise ValueError(f"Unknown system: {system_name}")
    
    return {
        'name': system_name,
        'module': _LAZY_IMPORTS[system_name],
        'available': system_name in _import_cache or True,
    }


