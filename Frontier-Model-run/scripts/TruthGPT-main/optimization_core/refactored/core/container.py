"""
Dependency Injection Container
==============================

Advanced dependency injection container with:
- Service registration and resolution
- Lifecycle management
- Scoped dependencies
- Circular dependency detection
- Auto-wiring
"""

import logging
from typing import Dict, Any, Optional, Type, Callable, List, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import inspect
import threading
import weakref


class ServiceLifetime(Enum):
    """Service lifetime options"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class ServiceRegistration:
    """Service registration metadata"""
    service_type: Type
    implementation: Any
    lifetime: ServiceLifetime
    factory: Optional[Callable] = None
    dependencies: List[Type] = None
    instance: Any = None


class DependencyContainer:
    """
    Advanced dependency injection container.
    
    Features:
    - Service registration with different lifetimes
    - Automatic dependency resolution
    - Circular dependency detection
    - Scoped services
    - Factory pattern support
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.services: Dict[Type, ServiceRegistration] = {}
        self.scoped_services: Dict[Type, Any] = {}
        self.resolution_stack: Set[Type] = set()
        self.lock = threading.RLock()
        
        # Register container itself
        self.register_instance(DependencyContainer, self)
    
    def register_singleton(self, service_type: Type, implementation: Type = None, instance: Any = None):
        """Register singleton service"""
        if instance is not None:
            self.register_instance(service_type, instance)
        else:
            self._register_service(service_type, implementation, ServiceLifetime.SINGLETON)
    
    def register_transient(self, service_type: Type, implementation: Type = None):
        """Register transient service"""
        self._register_service(service_type, implementation, ServiceLifetime.TRANSIENT)
    
    def register_scoped(self, service_type: Type, implementation: Type = None):
        """Register scoped service"""
        self._register_service(service_type, implementation, ServiceLifetime.SCOPED)
    
    def register_factory(self, service_type: Type, factory: Callable, lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT):
        """Register service with factory function"""
        registration = ServiceRegistration(
            service_type=service_type,
            implementation=None,
            lifetime=lifetime,
            factory=factory,
            dependencies=self._get_factory_dependencies(factory)
        )
        
        with self.lock:
            self.services[service_type] = registration
    
    def register_instance(self, service_type: Type, instance: Any):
        """Register service instance"""
        registration = ServiceRegistration(
            service_type=service_type,
            implementation=type(instance),
            lifetime=ServiceLifetime.SINGLETON,
            instance=instance
        )
        
        with self.lock:
            self.services[service_type] = registration
    
    def _register_service(self, service_type: Type, implementation: Type, lifetime: ServiceLifetime):
        """Internal service registration"""
        if implementation is None:
            implementation = service_type
        
        dependencies = self._get_constructor_dependencies(implementation)
        
        registration = ServiceRegistration(
            service_type=service_type,
            implementation=implementation,
            lifetime=lifetime,
            dependencies=dependencies
        )
        
        with self.lock:
            self.services[service_type] = registration
    
    def _get_constructor_dependencies(self, implementation: Type) -> List[Type]:
        """Get constructor dependencies using type hints"""
        try:
            signature = inspect.signature(implementation.__init__)
            dependencies = []
            
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                
                if param.annotation != inspect.Parameter.empty:
                    dependencies.append(param.annotation)
            
            return dependencies
        except Exception:
            return []
    
    def _get_factory_dependencies(self, factory: Callable) -> List[Type]:
        """Get factory function dependencies"""
        try:
            signature = inspect.signature(factory)
            dependencies = []
            
            for param_name, param in signature.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    dependencies.append(param.annotation)
            
            return dependencies
        except Exception:
            return []
    
    def resolve(self, service_type: Type) -> Any:
        """Resolve service instance"""
        with self.lock:
            return self._resolve_service(service_type)
    
    def _resolve_service(self, service_type: Type) -> Any:
        """Internal service resolution with circular dependency detection"""
        # Check for circular dependencies
        if service_type in self.resolution_stack:
            raise ValueError(f"Circular dependency detected: {service_type}")
        
        # Get registration
        registration = self.services.get(service_type)
        if not registration:
            raise ValueError(f"Service not registered: {service_type}")
        
        # Handle different lifetimes
        if registration.lifetime == ServiceLifetime.SINGLETON:
            if registration.instance is None:
                registration.instance = self._create_instance(registration)
            return registration.instance
        
        elif registration.lifetime == ServiceLifetime.SCOPED:
            if service_type not in self.scoped_services:
                self.scoped_services[service_type] = self._create_instance(registration)
            return self.scoped_services[service_type]
        
        else:  # TRANSIENT
            return self._create_instance(registration)
    
    def _create_instance(self, registration: ServiceRegistration) -> Any:
        """Create service instance with dependency injection"""
        # Add to resolution stack
        self.resolution_stack.add(registration.service_type)
        
        try:
            if registration.factory:
                # Use factory function
                dependencies = self._resolve_dependencies(registration.dependencies)
                return registration.factory(*dependencies)
            else:
                # Use constructor
                dependencies = self._resolve_dependencies(registration.dependencies)
                return registration.implementation(*dependencies)
        
        finally:
            # Remove from resolution stack
            self.resolution_stack.discard(registration.service_type)
    
    def _resolve_dependencies(self, dependencies: List[Type]) -> List[Any]:
        """Resolve list of dependencies"""
        resolved = []
        for dep_type in dependencies:
            resolved.append(self._resolve_service(dep_type))
        return resolved
    
    def try_resolve(self, service_type: Type) -> Optional[Any]:
        """Try to resolve service, return None if not registered"""
        try:
            return self.resolve(service_type)
        except ValueError:
            return None
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if service is registered"""
        return service_type in self.services
    
    def get_registered_services(self) -> List[Type]:
        """Get list of registered service types"""
        return list(self.services.keys())
    
    def clear_scoped_services(self):
        """Clear scoped services (call at end of scope)"""
        with self.lock:
            self.scoped_services.clear()
    
    def unregister(self, service_type: Type):
        """Unregister service"""
        with self.lock:
            if service_type in self.services:
                del self.services[service_type]
                self.logger.info(f"Unregistered service: {service_type}")
    
    def get_service_info(self, service_type: Type) -> Optional[Dict[str, Any]]:
        """Get service registration info"""
        registration = self.services.get(service_type)
        if not registration:
            return None
        
        return {
            'service_type': service_type.__name__,
            'implementation': registration.implementation.__name__ if registration.implementation else None,
            'lifetime': registration.lifetime.value,
            'has_factory': registration.factory is not None,
            'dependencies': [dep.__name__ for dep in (registration.dependencies or [])],
            'has_instance': registration.instance is not None
        }
    
    def get_container_metrics(self) -> Dict[str, Any]:
        """Get container metrics"""
        return {
            'registered_services': len(self.services),
            'scoped_services': len(self.scoped_services),
            'resolution_stack_size': len(self.resolution_stack)
        }
    
    def validate_registrations(self) -> List[str]:
        """Validate all service registrations"""
        errors = []
        
        for service_type, registration in self.services.items():
            # Check if implementation exists
            if not registration.implementation and not registration.factory:
                errors.append(f"Service {service_type} has no implementation or factory")
                continue
            
            # Check dependencies
            if registration.dependencies:
                for dep_type in registration.dependencies:
                    if not self.is_registered(dep_type):
                        errors.append(f"Service {service_type} depends on unregistered service {dep_type}")
        
        return errors
    
    def create_scope(self):
        """Create new dependency scope"""
        return DependencyScope(self)


class DependencyScope:
    """Dependency scope for managing scoped services"""
    
    def __init__(self, container: DependencyContainer):
        self.container = container
        self.scoped_services = {}
    
    def resolve(self, service_type: Type) -> Any:
        """Resolve service within scope"""
        if service_type in self.scoped_services:
            return self.scoped_services[service_type]
        
        # Get registration
        registration = self.container.services.get(service_type)
        if not registration:
            raise ValueError(f"Service not registered: {service_type}")
        
        if registration.lifetime == ServiceLifetime.SCOPED:
            instance = self.container._create_instance(registration)
            self.scoped_services[service_type] = instance
            return instance
        else:
            return self.container.resolve(service_type)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup scoped services
        for service in self.scoped_services.values():
            if hasattr(service, 'cleanup'):
                service.cleanup()
        self.scoped_services.clear()



