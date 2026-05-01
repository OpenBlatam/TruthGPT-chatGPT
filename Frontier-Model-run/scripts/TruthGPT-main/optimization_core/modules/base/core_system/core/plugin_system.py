"""
Plugin system for extensible modular architecture.
Allows dynamic loading and registration of plugins.
"""
import logging
import importlib
from typing import Dict, List, Type, Optional, Any
from pathlib import Path
from abc import ABC, abstractmethod

from .service_registry import ServiceRegistry

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """Base class for all plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass
    
    @abstractmethod
    def initialize(self, registry: ServiceRegistry) -> None:
        """
        Initialize plugin and register services.
        
        Args:
            registry: Service registry
        """
        pass
    
    def activate(self) -> None:
        """Called when plugin is activated."""
        logger.info(f"Plugin '{self.name}' activated")
    
    def deactivate(self) -> None:
        """Called when plugin is deactivated."""
        logger.info(f"Plugin '{self.name}' deactivated")
    
    def get_dependencies(self) -> List[str]:
        """
        Get list of plugin dependencies.
        
        Returns:
            List of required plugin names
        """
        return []


class PluginManager:
    """
    Manages plugin loading, activation, and lifecycle.
    """
    
    def __init__(self, registry: Optional[ServiceRegistry] = None):
        """
        Initialize plugin manager.
        
        Args:
            registry: Service registry (uses global if None)
        """
        self.registry = registry or ServiceRegistry()
        self._plugins: Dict[str, Plugin] = {}
        self._active_plugins: List[str] = []
        self._plugin_paths: Dict[str, Path] = {}
    
    def register_plugin(
        self,
        plugin: Plugin,
        auto_activate: bool = True
    ) -> None:
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance
            auto_activate: Automatically activate plugin
        """
        if plugin.name in self._plugins:
            logger.warning(f"Plugin '{plugin.name}' already registered")
            return
        
        # Check dependencies
        dependencies = plugin.get_dependencies()
        missing_deps = [dep for dep in dependencies if dep not in self._plugins]
        if missing_deps:
            raise ValueError(
                f"Plugin '{plugin.name}' has unmet dependencies: {missing_deps}"
            )
        
        self._plugins[plugin.name] = plugin
        
        if auto_activate:
            self.activate_plugin(plugin.name)
        else:
            logger.info(f"Plugin '{plugin.name}' registered (not activated)")
    
    def load_plugin_from_module(
        self,
        module_path: str,
        plugin_class_name: str = "Plugin",
        auto_activate: bool = True
    ) -> Plugin:
        """
        Load plugin from a Python module.
        
        Args:
            module_path: Module path (e.g., "plugins.my_plugin")
            plugin_class_name: Name of plugin class
            auto_activate: Automatically activate plugin
        
        Returns:
            Loaded plugin instance
        """
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, plugin_class_name)
            plugin = plugin_class()
            
            self.register_plugin(plugin, auto_activate=auto_activate)
            return plugin
            
        except Exception as e:
            logger.error(f"Failed to load plugin from {module_path}: {e}", exc_info=True)
            raise
    
    def load_plugins_from_directory(
        self,
        directory: str,
        auto_activate: bool = True
    ) -> List[Plugin]:
        """
        Load all plugins from a directory.
        
        Args:
            directory: Directory containing plugin modules
            auto_activate: Automatically activate plugins
        
        Returns:
            List of loaded plugins
        """
        plugin_dir = Path(directory)
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory not found: {directory}")
            return []
        
        loaded_plugins = []
        
        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            
            module_name = plugin_file.stem
            module_path = f"{plugin_dir.name}.{module_name}"
            
            try:
                plugin = self.load_plugin_from_module(
                    module_path,
                    auto_activate=auto_activate
                )
                loaded_plugins.append(plugin)
                self._plugin_paths[plugin.name] = plugin_file
            except Exception as e:
                logger.warning(f"Failed to load plugin from {plugin_file}: {e}")
        
        return loaded_plugins
    
    def activate_plugin(self, plugin_name: str) -> None:
        """
        Activate a plugin.
        
        Args:
            plugin_name: Name of plugin to activate
        """
        if plugin_name not in self._plugins:
            raise ValueError(f"Plugin '{plugin_name}' not found")
        
        if plugin_name in self._active_plugins:
            logger.warning(f"Plugin '{plugin_name}' already active")
            return
        
        plugin = self._plugins[plugin_name]
        
        # Initialize plugin (registers services)
        plugin.initialize(self.registry)
        
        # Activate plugin
        plugin.activate()
        self._active_plugins.append(plugin_name)
        
        logger.info(f"Plugin '{plugin_name}' activated")
    
    def deactivate_plugin(self, plugin_name: str) -> None:
        """
        Deactivate a plugin.
        
        Args:
            plugin_name: Name of plugin to deactivate
        """
        if plugin_name not in self._active_plugins:
            logger.warning(f"Plugin '{plugin_name}' not active")
            return
        
        plugin = self._plugins[plugin_name]
        plugin.deactivate()
        self._active_plugins.remove(plugin_name)
        
        logger.info(f"Plugin '{plugin_name}' deactivated")
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self._plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins."""
        return list(self._plugins.keys())
    
    def list_active_plugins(self) -> List[str]:
        """List all active plugins."""
        return self._active_plugins.copy()


# Global plugin manager
_plugin_manager = PluginManager()


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager."""
    return _plugin_manager


# Example plugin implementation
class ExamplePlugin(Plugin):
    """Example plugin implementation."""
    
    @property
    def name(self) -> str:
        return "example_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def initialize(self, registry: ServiceRegistry) -> None:
        """Initialize the plugin."""
        # Register services
        registry.register("example_service", self._create_service, singleton=True)
        logger.info("Example plugin initialized")
    
    def _create_service(self):
        """Create example service."""
        return {"status": "ready"}



