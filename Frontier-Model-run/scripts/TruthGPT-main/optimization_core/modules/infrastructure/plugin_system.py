"""
Plugin system for optimization_core.

Provides a plugin architecture for extending functionality.
"""
import logging
import importlib
from typing import Dict, Any, List, Optional, Type, Callable
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field

# Use relative imports for infrastructure internal components if they were here,
# but these are standard utils or internal to this file.
from ...utils.shared_validators import validate_not_none, validate_type
from ...utils.error_handling import OptimizationCoreError, handle_error

logger = logging.getLogger(__name__)

class PluginInfo(BaseModel):
    """Information about a plugin."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    version: str
    description: str
    author: Optional[str] = None
    plugin_class: Optional[Type] = None
    module_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (Pydantic model_dump)."""
        return self.model_dump()

class PluginRegistry:
    """Registry for plugins."""
    
    def __init__(self):
        """Initialize plugin registry."""
        self._plugins: Dict[str, PluginInfo] = {}
        self._instances: Dict[str, Any] = {}
    
    def register(
        self,
        plugin_info: PluginInfo,
        instance: Optional[Any] = None
    ):
        """
        Register a plugin.
        
        Args:
            plugin_info: Plugin information
            instance: Optional plugin instance
        """
        validate_not_none(plugin_info.name, "plugin name")
        self._plugins[plugin_info.name] = plugin_info
        
        if instance:
            self._instances[plugin_info.name] = instance
        
        logger.info(f"Registered plugin: {plugin_info.name} v{plugin_info.version}")
    
    def load_plugin(
        self,
        module_path: str,
        plugin_class_name: str,
        plugin_name: Optional[str] = None
    ) -> PluginInfo:
        """
        Load plugin from module.
        
        Args:
            module_path: Path to module
            plugin_class_name: Name of plugin class
            plugin_name: Optional plugin name (defaults to class name)
        
        Returns:
            PluginInfo
        """
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, plugin_class_name)
            
            # Create instance
            instance = plugin_class()
            
            # Get plugin info
            name = plugin_name or plugin_class_name
            version = getattr(instance, "version", "1.0.0")
            description = getattr(instance, "description", "")
            author = getattr(instance, "author", None)
            
            plugin_info = PluginInfo(
                name=name,
                version=version,
                description=description,
                author=author,
                plugin_class=plugin_class,
                module_path=module_path
            )
            
            self.register(plugin_info, instance)
            return plugin_info
            
        except Exception as e:
            handle_error(
                e,
                context={"module_path": module_path, "class_name": plugin_class_name},
                reraise=True
            )
    
    def get_plugin(self, name: str) -> Optional[Any]:
        """
        Get plugin instance.
        
        Args:
            name: Plugin name
        
        Returns:
            Plugin instance or None
        """
        if name in self._instances:
            return self._instances[name]
        
        if name in self._plugins:
            plugin_info = self._plugins[name]
            if plugin_info.plugin_class:
                instance = plugin_info.plugin_class()
                self._instances[name] = instance
                return instance
        
        return None
    
    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())
    
    def get_plugin_info(self, name: str) -> Optional[PluginInfo]:
        """Get plugin information."""
        return self._plugins.get(name)
    
    def unregister(self, name: str):
        """Unregister a plugin."""
        if name in self._plugins:
            del self._plugins[name]
        if name in self._instances:
            del self._instances[name]

class BasePlugin:
    """Base class for plugins."""
    
    name: str = "base_plugin"
    version: str = "1.0.0"
    description: str = "Base plugin"
    author: Optional[str] = None
    
    def initialize(self, **kwargs):
        """
        Initialize plugin.
        
        Args:
            **kwargs: Initialization arguments
        """
        pass
    
    def cleanup(self):
        """Cleanup plugin resources."""
        pass
    
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute plugin functionality.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Plugin result
        """
        raise NotImplementedError("Plugin must implement execute method")

# Global plugin registry
_global_plugin_registry = PluginRegistry()

def register_plugin(plugin_info: PluginInfo, instance: Optional[Any] = None):
    """Register plugin in global registry."""
    _global_plugin_registry.register(plugin_info, instance)

def get_plugin(name: str) -> Optional[Any]:
    """Get plugin from global registry."""
    return _global_plugin_registry.get_plugin(name)

def list_plugins() -> List[str]:
    """List all plugins in global registry."""
    return _global_plugin_registry.list_plugins()
