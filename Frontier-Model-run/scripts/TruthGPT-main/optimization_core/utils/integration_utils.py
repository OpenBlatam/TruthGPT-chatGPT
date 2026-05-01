"""
Integration utilities for optimization_core.

Provides utilities for integrating different modules and components.
"""
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path

from .shared_validators import validate_not_none, validate_type
from .error_handling import OptimizationCoreError, handle_error

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """Registry for components."""
    
    def __init__(self):
        """Initialize component registry."""
        self._components: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
    
    def register(
        self,
        name: str,
        component: Any = None,
        factory: Optional[Callable] = None
    ):
        """
        Register a component or factory.
        
        Args:
            name: Component name
            component: Component instance (optional)
            factory: Factory function (optional)
        """
        validate_not_none(name, "name")
        
        if component is not None:
            self._components[name] = component
        elif factory is not None:
            self._factories[name] = factory
        else:
            raise ValueError("Either component or factory must be provided")
    
    def get(self, name: str, **kwargs) -> Any:
        """
        Get a component.
        
        Args:
            name: Component name
            **kwargs: Arguments for factory (if factory is used)
        
        Returns:
            Component instance
        """
        validate_not_none(name, "name")
        
        if name in self._components:
            return self._components[name]
        
        if name in self._factories:
            return self._factories[name](**kwargs)
        
        raise ValueError(f"Component '{name}' not found in registry")
    
    def list_components(self) -> List[str]:
        """List all registered component names."""
        return list(set(list(self._components.keys()) + list(self._factories.keys())))
    
    def unregister(self, name: str):
        """Unregister a component."""
        if name in self._components:
            del self._components[name]
        if name in self._factories:
            del self._factories[name]


# Global registry instance
_global_registry = ComponentRegistry()


def register_component(name: str, component: Any = None, factory: Optional[Callable] = None):
    """Register a component in the global registry."""
    _global_registry.register(name, component, factory)


def get_component(name: str, **kwargs) -> Any:
    """Get a component from the global registry."""
    return _global_registry.get(name, **kwargs)


def list_components() -> List[str]:
    """List all components in the global registry."""
    return _global_registry.list_components()


class Pipeline:
    """Pipeline for chaining operations."""
    
    def __init__(self, name: str = "pipeline"):
        """
        Initialize pipeline.
        
        Args:
            name: Pipeline name
        """
        self.name = name
        self.steps: List[Callable] = []
    
    def add_step(self, step: Callable, name: Optional[str] = None):
        """
        Add a step to the pipeline.
        
        Args:
            step: Step function
            name: Optional step name
        """
        validate_not_none(step, "step")
        validate_type(step, Callable, "step")
        
        if name:
            step.__name__ = name
        
        self.steps.append(step)
        logger.debug(f"Added step '{step.__name__}' to pipeline '{self.name}'")
    
    def run(self, data: Any, **kwargs) -> Any:
        """
        Run the pipeline.
        
        Args:
            data: Input data
            **kwargs: Additional arguments
        
        Returns:
            Pipeline result
        """
        result = data
        
        for i, step in enumerate(self.steps):
            try:
                logger.debug(f"Running step {i+1}/{len(self.steps)}: {step.__name__}")
                result = step(result, **kwargs)
            except Exception as e:
                handle_error(
                    e,
                    context={
                        "pipeline": self.name,
                        "step": step.__name__,
                        "step_index": i,
                    },
                    reraise=True
                )
        
        return result
    
    def __call__(self, data: Any, **kwargs) -> Any:
        """Make pipeline callable."""
        return self.run(data, **kwargs)


def create_pipeline(name: str = "pipeline") -> Pipeline:
    """
    Create a new pipeline.
    
    Args:
        name: Pipeline name
    
    Returns:
        Pipeline instance
    """
    return Pipeline(name=name)













