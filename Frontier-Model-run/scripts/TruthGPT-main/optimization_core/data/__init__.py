"""
Data loading and processing modules.

This module provides organized access to data components:
- Dataset management
- Data loader factories
- Data collators
- Text processing
- High-performance processors (Polars)
"""
from __future__ import annotations

import importlib

# Direct imports for backward compatibility
from .dataset_manager import DatasetManager
from .data_loader_factory import DataLoaderFactory
from .collators import LMCollator

# New high-performance processors
try:
    from .polars_processor import PolarsProcessor
    from .processor_factory import (
        create_data_processor,
        ProcessorType,
        list_available_processors,
    )
    _NEW_PROCESSORS_AVAILABLE = True
except ImportError:
    _NEW_PROCESSORS_AVAILABLE = False

# Lazy imports for additional components
_LAZY_IMPORTS = {
    'text_hf': '.text_hf',
    'registry': '.registry',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for data submodules."""
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


def create_data_component(component_type: str = "dataset_manager", config: dict = None):
    """
    Unified factory function to create data components.
    
    Args:
        component_type: Type of component. Options: "dataset_manager", "data_loader_factory", "collator"
        config: Optional configuration dictionary
    
    Returns:
        The requested component instance
    """
    if config is None:
        config = {}
    
    component_type = component_type.lower()
    
    factory_map = {
        "dataset_manager": lambda cfg: DatasetManager(**cfg),
        "data_loader_factory": lambda cfg: DataLoaderFactory(**cfg),
        "collator": lambda cfg: LMCollator(**cfg),
    }
    
    if component_type not in factory_map:
        available = ", ".join(factory_map.keys())
        raise ValueError(
            f"Unknown data component type: '{component_type}'. "
            f"Available types: {available}"
        )
    
    factory = factory_map[component_type]
    return factory(config)


DATA_COMPONENT_REGISTRY = {
    "dataset_manager": {
        "class": DatasetManager,
        "module": "data.dataset_manager",
    },
    "data_loader_factory": {
        "class": DataLoaderFactory,
        "module": "data.data_loader_factory",
    },
    "collator": {
        "class": LMCollator,
        "module": "data.collators",
    },
}


def list_available_data_components() -> list[str]:
    """List all available data component types."""
    return list(DATA_COMPONENT_REGISTRY.keys())


def get_data_component_info(component_type: str) -> dict[str, any]:
    """Get information about a data component."""
    if component_type not in DATA_COMPONENT_REGISTRY:
        raise ValueError(f"Unknown data component: {component_type}")
    
    return {
        'name': component_type,
        'class': DATA_COMPONENT_REGISTRY[component_type]['class'].__name__,
        'module': DATA_COMPONENT_REGISTRY[component_type]['module'],
    }


__all__ = [
    # Core components (backward compatible)
    "DatasetManager",
    "DataLoaderFactory",
    "LMCollator",
    "create_data_component",
    "list_available_data_components",
    "get_data_component_info",
    "DATA_COMPONENT_REGISTRY",
    # New high-performance processors
    "PolarsProcessor",
    "create_data_processor",
    "ProcessorType",
    "list_available_processors",
]

