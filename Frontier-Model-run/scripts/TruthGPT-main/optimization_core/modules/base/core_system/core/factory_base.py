"""
Base factory classes for optimization_core.

Provides common factory patterns to reduce duplication across modules.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Type, TypeVar, Optional, Any, Callable, List
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class FactoryError(Exception):
    """Base exception for factory errors."""
    pass


class ComponentNotFoundError(FactoryError):
    """Raised when a component is not found in factory."""
    pass


class ComponentNotAvailableError(FactoryError):
    """Raised when a component is registered but not available."""
    pass


class BaseFactory(ABC):
    """
    Base factory class with common functionality.
    
    Provides:
    - Component registration
    - Auto-selection based on availability
    - Availability checking
    - Type validation
    
    Example:
        class MyFactory(BaseFactory):
            def _check_availability(self, component_type: str) -> bool:
                if component_type == "polars":
                    try:
                        import polars
                        return True
                    except ImportError:
                        return False
                return False
            
            def _create_component(self, component_type: str, **kwargs):
                if component_type == "polars":
                    from .polars_processor import PolarsProcessor
                    return PolarsProcessor(**kwargs)
                raise ComponentNotFoundError(f"Unknown type: {component_type}")
    """
    
    def __init__(self, default_type: str = "auto"):
        """
        Initialize factory.
        
        Args:
            default_type: Default component type to use
        """
        self._registry: Dict[str, Type] = {}
        self._availability_cache: Dict[str, bool] = {}
        self._default_type = default_type
        self._creation_count = 0
        self._error_count = 0
    
    @abstractmethod
    def _check_availability(self, component_type: str) -> bool:
        """
        Check if a component type is available.
        
        Args:
            component_type: Type to check
        
        Returns:
            True if available
        """
        pass
    
    @abstractmethod
    def _create_component(self, component_type: str, **kwargs) -> Any:
        """
        Create a component instance.
        
        Args:
            component_type: Type of component to create
            **kwargs: Component-specific arguments
        
        Returns:
            Component instance
        
        Raises:
            ComponentNotFoundError: If type is not registered
        """
        pass
    
    def register(self, component_type: str, component_class: Type) -> None:
        """
        Register a component type.
        
        Args:
            component_type: Type identifier
            component_class: Class to instantiate
        """
        self._registry[component_type] = component_class
        logger.debug(f"Registered {component_type} in {self.__class__.__name__}")
    
    def is_available(self, component_type: str) -> bool:
        """
        Check if component type is available.
        
        Args:
            component_type: Type to check
        
        Returns:
            True if available
        """
        if component_type not in self._availability_cache:
            self._availability_cache[component_type] = self._check_availability(component_type)
        return self._availability_cache[component_type]
    
    def create(
        self,
        component_type: Optional[str] = None,
        auto_select: bool = True,
        **kwargs
    ) -> Any:
        """
        Create a component instance.
        
        Args:
            component_type: Type to create (None uses default)
            auto_select: Auto-select best available if type is None
            **kwargs: Component-specific arguments
        
        Returns:
            Component instance
        
        Raises:
            ComponentNotFoundError: If type is not found
            ComponentNotAvailableError: If type is not available
        """
        if component_type is None:
            if auto_select:
                component_type = self.select_best()
            else:
                component_type = self._default_type
        
        if component_type not in self._registry:
            raise ComponentNotFoundError(
                f"Component type '{component_type}' not registered in {self.__class__.__name__}. "
                f"Available: {list(self._registry.keys())}"
            )
        
        if not self.is_available(component_type):
            raise ComponentNotAvailableError(
                f"Component type '{component_type}' is not available. "
                f"Install required dependencies."
            )
        
        try:
            instance = self._create_component(component_type, **kwargs)
            self._creation_count += 1
            logger.debug(f"Created {component_type} via {self.__class__.__name__}")
            return instance
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to create {component_type}: {e}")
            raise
    
    def select_best(self) -> str:
        """
        Select best available component type.
        
        Returns:
            Best available type
        
        Raises:
            ComponentNotAvailableError: If no components are available
        """
        for component_type in self._registry.keys():
            if self.is_available(component_type):
                logger.debug(f"Selected {component_type} as best available")
                return component_type
        
        raise ComponentNotAvailableError(
            f"No components available in {self.__class__.__name__}. "
            f"Registered types: {list(self._registry.keys())}"
        )
    
    def list_available(self) -> Dict[str, bool]:
        """
        List all registered components and their availability.
        
        Returns:
            Dictionary mapping component types to availability
        """
        return {
            component_type: self.is_available(component_type)
            for component_type in self._registry.keys()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get factory statistics.
        
        Returns:
            Dictionary with stats
        """
        return {
            "creation_count": self._creation_count,
            "error_count": self._error_count,
            "registered_types": list(self._registry.keys()),
            "available_types": [
                t for t in self._registry.keys() if self.is_available(t)
            ],
        }


class SimpleFactory(BaseFactory):
    """
    Simple factory that uses registered classes directly.
    
    Example:
        factory = SimpleFactory()
        factory.register("polars", PolarsProcessor)
        factory._check_availability = lambda t: t == "polars" and POLARS_AVAILABLE
        processor = factory.create("polars", lazy=True)
    """
    
    def _check_availability(self, component_type: str) -> bool:
        """Override in subclass or set directly."""
        return True
    
    def _create_component(self, component_type: str, **kwargs) -> Any:
        """Create component from registered class."""
        component_class = self._registry[component_type]
        return component_class(**kwargs)


class CallableFactory(BaseFactory):
    """
    Factory that uses callable creators.
    
    Example:
        factory = CallableFactory()
        factory.register("polars", lambda **kw: PolarsProcessor(**kw))
        factory._check_availability = lambda t: POLARS_AVAILABLE
        processor = factory.create("polars", lazy=True)
    """
    
    def __init__(self, default_type: str = "auto"):
        super().__init__(default_type)
        self._creators: Dict[str, Callable] = {}
    
    def register_creator(self, component_type: str, creator: Callable) -> None:
        """
        Register a creator function.
        
        Args:
            component_type: Type identifier
            creator: Callable that creates the component
        """
        self._creators[component_type] = creator
    
    def _check_availability(self, component_type: str) -> bool:
        """Override in subclass."""
        return True
    
    def _create_component(self, component_type: str, **kwargs) -> Any:
        """Create component using creator function."""
        if component_type not in self._creators:
            raise ComponentNotFoundError(f"Creator not registered: {component_type}")
        creator = self._creators[component_type]
        return creator(**kwargs)


class FactoryRegistry:
    """
    Global registry for all factories.
    
    Allows centralized management and discovery of factories.
    """
    
    _factories: Dict[str, BaseFactory] = {}
    
    @classmethod
    def register_factory(cls, name: str, factory: BaseFactory) -> None:
        """
        Register a factory.
        
        Args:
            name: Factory name
            factory: Factory instance
        """
        cls._factories[name] = factory
        logger.info(f"Registered factory: {name}")
    
    @classmethod
    def get_factory(cls, name: str) -> Optional[BaseFactory]:
        """
        Get a factory by name.
        
        Args:
            name: Factory name
        
        Returns:
            Factory instance or None
        """
        return cls._factories.get(name)
    
    @classmethod
    def list_factories(cls) -> List[str]:
        """List all registered factory names."""
        return list(cls._factories.keys())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered factories."""
        cls._factories.clear()













