"""
Dynamic factory system with automatic registration.
Enables dynamic plugin-based component registration.
"""
import logging
from typing import Dict, Type, TypeVar, Callable, Any, Optional
from abc import ABC
import inspect

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DynamicFactory:
    """
    Dynamic factory with automatic registration based on naming conventions
    and decorators.
    """
    
    def __init__(self, base_class: Optional[Type] = None):
        """
        Initialize factory.
        
        Args:
            base_class: Optional base class that all registered types must inherit from
        """
        self.base_class = base_class
        self._registry: Dict[str, Type] = {}
        self._factories: Dict[str, Callable] = {}
    
    def register(
        self,
        name: str,
        component: Type[T],
        override: bool = False
    ) -> None:
        """
        Register a component.
        
        Args:
            name: Component name
            component: Component class or factory function
            override: Allow overriding existing registrations
        """
        # Validate base class if specified
        if self.base_class and not issubclass(component, self.base_class):
            raise ValueError(
                f"Component '{name}' must inherit from {self.base_class.__name__}"
            )
        
        # Check for existing registration
        if name in self._registry and not override:
            raise ValueError(f"Component '{name}' already registered")
        
        self._registry[name] = component
        logger.debug(f"Component '{name}' registered")
    
    def register_factory(
        self,
        name: str,
        factory: Callable,
        override: bool = False
    ) -> None:
        """
        Register a factory function.
        
        Args:
            name: Factory name
            factory: Factory function
            override: Allow overriding
        """
        if name in self._factories and not override:
            raise ValueError(f"Factory '{name}' already registered")
        
        self._factories[name] = factory
        logger.debug(f"Factory '{name}' registered")
    
    def create(
        self,
        name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Create a component instance.
        
        Args:
            name: Component name
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Component instance
        """
        # Try factory first
        if name in self._factories:
            factory = self._factories[name]
            return factory(*args, **kwargs)
        
        # Try registered class
        if name in self._registry:
            component_class = self._registry[name]
            
            # Check if it's a class or instance
            if inspect.isclass(component_class):
                return component_class(*args, **kwargs)
            else:
                return component_class
        
        raise ValueError(f"Component '{name}' not found in factory")
    
    def list_components(self) -> list[str]:
        """List all registered component names."""
        components = set(self._registry.keys())
        components.update(self._factories.keys())
        return sorted(components)
    
    def auto_register_from_module(
        self,
        module: Any,
        name_pattern: Optional[str] = None
    ) -> None:
        """
        Automatically register components from a module.
        
        Looks for classes that:
        1. Inherit from base_class (if specified)
        2. Match name_pattern (if specified)
        
        Args:
            module: Module to scan
            name_pattern: Optional name pattern (e.g., "*Optimizer")
        """
        for name, obj in inspect.getmembers(module):
            # Check if it's a class
            if not inspect.isclass(obj):
                continue
            
            # Check base class
            if self.base_class and not issubclass(obj, self.base_class):
                continue
            
            # Check name pattern
            if name_pattern:
                import fnmatch
                if not fnmatch.fnmatch(name, name_pattern):
                    continue
            
            # Register with lowercase name
            self.register(name.lower(), obj)
            logger.debug(f"Auto-registered component: {name}")


def factory(base_class: Optional[Type] = None):
    """
    Decorator to create a factory instance.
    
    Usage:
        @factory(BaseOptimizer)
        class OptimizerFactory(DynamicFactory):
            pass
    """
    return DynamicFactory(base_class=base_class)


def register_component(name: Optional[str] = None):
    """
    Decorator to register a component with a factory.
    
    Usage:
        @register_component("my_optimizer")
        class MyOptimizer(BaseOptimizer):
            pass
    """
    def decorator(cls):
        # Auto-register with module-level factories
        # This is a simplified version - actual implementation would
        # need to track which factory to register with
        return cls
    
    return decorator


class AutoRegisterMeta(type):
    """
    Metaclass for automatic factory registration.
    """
    
    def __new__(mcs, name, bases, namespace, factory_instance=None, register_name=None):
        cls = super().__new__(mcs, name, bases, namespace)
        
        if factory_instance and register_name:
            factory_instance.register(register_name or name.lower(), cls)
        
        return cls


# Example usage pattern
class BaseComponent(ABC):
    """Base class for components."""
    pass


class ComponentFactory(DynamicFactory):
    """Factory for components."""
    pass


# Global factories (can be extended)
_global_factories: Dict[str, DynamicFactory] = {}


def get_factory(name: str) -> Optional[DynamicFactory]:
    """Get a global factory by name."""
    return _global_factories.get(name)


def create_factory(
    name: str,
    base_class: Optional[Type] = None
) -> DynamicFactory:
    """
    Create and register a global factory.
    
    Args:
        name: Factory name
        base_class: Optional base class
    
    Returns:
        Created factory
    """
    factory_instance = DynamicFactory(base_class=base_class)
    _global_factories[name] = factory_instance
    return factory_instance



