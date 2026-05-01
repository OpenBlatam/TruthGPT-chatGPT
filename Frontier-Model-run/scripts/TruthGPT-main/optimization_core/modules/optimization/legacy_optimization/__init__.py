"""
Performance optimization utilities.

This module provides organized access to optimization components:
- Performance optimization
- Memory optimization
- Model profiling
"""

from __future__ import annotations

# Direct imports for backward compatibility
from .performance_optimizer import PerformanceOptimizer
from .memory_optimizer import MemoryOptimizer
from .profiler import ModelProfiler


def create_optimization_component(component_type: str = "performance", config: dict = None):
    """
    Unified factory function to create optimization components.
    
    Args:
        component_type: Type of component. Options: "performance", "memory", "profiler"
        config: Optional configuration dictionary
    
    Returns:
        The requested component instance
    """
    if config is None:
        config = {}
    
    component_type = component_type.lower()
    
    factory_map = {
        "performance": lambda cfg: PerformanceOptimizer(**cfg),
        "memory": lambda cfg: MemoryOptimizer(**cfg),
        "profiler": lambda cfg: ModelProfiler(**cfg),
    }
    
    if component_type not in factory_map:
        available = ", ".join(factory_map.keys())
        raise ValueError(
            f"Unknown optimization component type: '{component_type}'. "
            f"Available types: {available}"
        )
    
    factory = factory_map[component_type]
    return factory(config)


OPTIMIZATION_COMPONENT_REGISTRY = {
    "performance": {
        "class": PerformanceOptimizer,
        "module": "optimization.performance_optimizer",
    },
    "memory": {
        "class": MemoryOptimizer,
        "module": "optimization.memory_optimizer",
    },
    "profiler": {
        "class": ModelProfiler,
        "module": "optimization.profiler",
    },
}


def list_available_optimization_components() -> list[str]:
    """List all available optimization component types."""
    return list(OPTIMIZATION_COMPONENT_REGISTRY.keys())


def get_optimization_component_info(component_type: str) -> dict[str, any]:
    """Get information about an optimization component."""
    if component_type not in OPTIMIZATION_COMPONENT_REGISTRY:
        raise ValueError(f"Unknown optimization component: {component_type}")
    
    return {
        'name': component_type,
        'class': OPTIMIZATION_COMPONENT_REGISTRY[component_type]['class'].__name__,
        'module': OPTIMIZATION_COMPONENT_REGISTRY[component_type]['module'],
    }


__all__ = [
    "PerformanceOptimizer",
    "MemoryOptimizer",
    "ModelProfiler",
    "create_optimization_component",
    "list_available_optimization_components",
    "get_optimization_component_info",
    "OPTIMIZATION_COMPONENT_REGISTRY",
]



