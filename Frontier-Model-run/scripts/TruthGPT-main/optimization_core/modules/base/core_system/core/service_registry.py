"""
Service Registry with Dependency Injection for maximum modularity.
Enables loose coupling between components through service registration.
"""
import logging
from typing import Dict, Any, Optional, Type, Callable, TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceRegistry:
    """
    Central registry for services with dependency injection.
    Implements singleton pattern for global access.
    """
    
    _instance: Optional['ServiceRegistry'] = None
    _services: Dict[str, Any] = {}
    _factories: Dict[str, Callable] = {}
    _singletons: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(
        self,
        name: str,
        service: Any,
        singleton: bool = False
    ) -> None:
        """
        Register a service.
        
        Args:
            name: Service name
            service: Service instance or class
            singleton: If True, service will be instantiated once
        """
        if singleton and callable(service) and not isinstance(service, type):
            # It's a factory function
            self._factories[name] = service
        elif singleton:
            # Store for lazy initialization
            self._factories[name] = service
        else:
            self._services[name] = service
        
        logger.debug(f"Service registered: {name} (singleton={singleton})")
    
    def get(self, name: str, **kwargs) -> Any:
        """
        Get a service instance.
        
        Args:
            name: Service name
            **kwargs: Arguments for factory functions
        
        Returns:
            Service instance
        """
        # Check singletons first
        if name in self._singletons:
            return self._singletons[name]
        
        # Check factories (singleton services)
        if name in self._factories:
            factory = self._factories[name]
            
            # Lazy initialization for singletons
            if name not in self._singletons:
                if callable(factory):
                    instance = factory(**kwargs) if kwargs else factory()
                else:
                    instance = factory
                
                self._singletons[name] = instance
                return instance
        
        # Check regular services
        if name in self._services:
            service = self._services[name]
            
            # If it's a class, instantiate it
            if isinstance(service, type):
                return service(**kwargs) if kwargs else service()
            
            return service
        
        raise ValueError(f"Service '{name}' not found in registry")
    
    def unregister(self, name: str) -> None:
        """Unregister a service."""
        self._services.pop(name, None)
        self._factories.pop(name, None)
        self._singletons.pop(name, None)
        logger.debug(f"Service unregistered: {name}")
    
    def clear(self) -> None:
        """Clear all services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        logger.info("Service registry cleared")
    
    def list_services(self) -> list[str]:
        """List all registered service names."""
        all_services = set(self._services.keys())
        all_services.update(self._factories.keys())
        return sorted(all_services)


# Global registry instance
registry = ServiceRegistry()


def register_service(name: str, singleton: bool = False):
    """
    Decorator to register a service.
    
    Usage:
        @register_service("my_service", singleton=True)
        class MyService:
            pass
    """
    def decorator(cls_or_func):
        registry.register(name, cls_or_func, singleton=singleton)
        return cls_or_func
    return decorator


def get_service(name: str, **kwargs) -> Any:
    """Get a service from the registry."""
    return registry.get(name, **kwargs)


@dataclass
class ServiceDescriptor:
    """Descriptor for service metadata."""
    name: str
    service_type: Type
    singleton: bool = False
    dependencies: list[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class ServiceProvider(ABC):
    """Base class for service providers."""
    
    @abstractmethod
    def register_services(self, registry: ServiceRegistry) -> None:
        """Register services with the registry."""
        pass


class ServiceContainer:
    """
    Container for managing services with dependency injection.
    """
    
    def __init__(self):
        self._registry = ServiceRegistry()
        self._providers: list[ServiceProvider] = []
    
    def add_provider(self, provider: ServiceProvider) -> None:
        """Add a service provider."""
        self._providers.append(provider)
        provider.register_services(self._registry)
    
    def register(
        self,
        name: str,
        service: Any,
        singleton: bool = False
    ) -> None:
        """Register a service."""
        self._registry.register(name, service, singleton)
    
    def get(self, name: str, **kwargs) -> Any:
        """Get a service."""
        return self._registry.get(name, **kwargs)
    
    def build(self, service_class: Type[T], **kwargs) -> T:
        """
        Build a service instance with dependency injection.
        
        Args:
            service_class: Service class to instantiate
            **kwargs: Additional arguments
        
        Returns:
            Service instance
        """
        # Get dependencies from annotations (simplified)
        import inspect
        sig = inspect.signature(service_class.__init__)
        
        resolved_kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            # Try to resolve from registry
            if param_name in kwargs:
                resolved_kwargs[param_name] = kwargs[param_name]
            elif param.annotation != inspect.Parameter.empty:
                # Try to get service by type name
                type_name = param.annotation.__name__ if hasattr(param.annotation, "__name__") else None
                if type_name and type_name in self._registry.list_services():
                    resolved_kwargs[param_name] = self._registry.get(type_name)
        
        # Add remaining kwargs
        resolved_kwargs.update(kwargs)
        
        return service_class(**resolved_kwargs)



