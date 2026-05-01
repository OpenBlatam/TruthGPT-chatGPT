"""
Registry pattern for polyglot_core.

Provides component registration and discovery.
"""

from typing import Dict, Type, TypeVar, Optional, Any, Callable, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


T = TypeVar('T')


class RegistryType(str, Enum):
    """Registry types."""
    BACKEND = "backend"
    COMPONENT = "component"
    PLUGIN = "plugin"
    STRATEGY = "strategy"
    ALGORITHM = "algorithm"


@dataclass
class RegistryEntry:
    """Registry entry."""
    name: str
    component: Any
    component_type: RegistryType
    metadata: Dict[str, Any]
    priority: int = 0


class ComponentRegistry:
    """
    Component registry for polyglot_core.
    
    Manages registration and discovery of components.
    """
    
    def __init__(self):
        self._registries: Dict[RegistryType, Dict[str, RegistryEntry]] = {
            RegistryType.BACKEND: {},
            RegistryType.COMPONENT: {},
            RegistryType.PLUGIN: {},
            RegistryType.STRATEGY: {},
            RegistryType.ALGORITHM: {},
        }
    
    def register(
        self,
        name: str,
        component: Any,
        component_type: RegistryType = RegistryType.COMPONENT,
        priority: int = 0,
        **metadata
    ) -> str:
        """
        Register component.
        
        Args:
            name: Component name
            component: Component instance or class
            component_type: Component type
            priority: Registration priority (higher = preferred)
            **metadata: Additional metadata
            
        Returns:
            Registration ID
        """
        entry = RegistryEntry(
            name=name,
            component=component,
            component_type=component_type,
            metadata=metadata,
            priority=priority
        )
        
        registry = self._registries[component_type]
        registry[name] = entry
        
        return name
    
    def unregister(self, name: str, component_type: RegistryType = RegistryType.COMPONENT):
        """Unregister component."""
        registry = self._registries[component_type]
        registry.pop(name, None)
    
    def get(
        self,
        name: str,
        component_type: RegistryType = RegistryType.COMPONENT
    ) -> Optional[Any]:
        """
        Get component by name.
        
        Args:
            name: Component name
            component_type: Component type
            
        Returns:
            Component instance or None
        """
        registry = self._registries[component_type]
        entry = registry.get(name)
        return entry.component if entry else None
    
    def list(
        self,
        component_type: RegistryType = RegistryType.COMPONENT
    ) -> List[str]:
        """
        List registered components.
        
        Args:
            component_type: Component type
            
        Returns:
            List of component names
        """
        registry = self._registries[component_type]
        return list(registry.keys())
    
    def get_all(
        self,
        component_type: RegistryType = RegistryType.COMPONENT
    ) -> Dict[str, Any]:
        """
        Get all components of type.
        
        Args:
            component_type: Component type
            
        Returns:
            Dictionary of name -> component
        """
        registry = self._registries[component_type]
        return {name: entry.component for name, entry in registry.items()}
    
    def get_best(
        self,
        component_type: RegistryType = RegistryType.COMPONENT
    ) -> Optional[Any]:
        """
        Get best component by priority.
        
        Args:
            component_type: Component type
            
        Returns:
            Best component or None
        """
        registry = self._registries[component_type]
        if not registry:
            return None
        
        best_entry = max(registry.values(), key=lambda e: e.priority)
        return best_entry.component
    
    def clear(self, component_type: Optional[RegistryType] = None):
        """
        Clear registry.
        
        Args:
            component_type: Optional component type to clear (None = all)
        """
        if component_type:
            self._registries[component_type].clear()
        else:
            for registry in self._registries.values():
                registry.clear()


# Global registry
_global_registry = ComponentRegistry()


def get_registry() -> ComponentRegistry:
    """Get global component registry."""
    return _global_registry


def register_component(
    name: str,
    component: Any,
    component_type: RegistryType = RegistryType.COMPONENT,
    priority: int = 0,
    **metadata
) -> str:
    """Convenience function to register component."""
    return _global_registry.register(name, component, component_type, priority, **metadata)


def get_component(
    name: str,
    component_type: RegistryType = RegistryType.COMPONENT
) -> Optional[Any]:
    """Convenience function to get component."""
    return _global_registry.get(name, component_type)


