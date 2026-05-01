"""
Plugin system for polyglot_core.

Provides extensibility through plugins.
"""

from typing import Dict, List, Optional, Any, Callable, Type
from abc import ABC, abstractmethod
import importlib
import importlib.util
import inspect
from pathlib import Path


class Plugin(ABC):
    """Base class for polyglot_core plugins."""
    
    @abstractmethod
    def name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def version(self) -> str:
        """Get plugin version."""
        pass
    
    def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin."""
        pass
    
    def cleanup(self):
        """Cleanup plugin."""
        pass


class PluginManager:
    """
    Plugin manager for polyglot_core.
    
    Manages plugin registration, loading, and lifecycle.
    """
    
    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._hooks: Dict[str, List[Callable]] = {}
    
    def register(self, plugin: Plugin):
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance
        """
        plugin_name = plugin.name()
        self._plugins[plugin_name] = plugin
        plugin.initialize()
    
    def unregister(self, plugin_name: str):
        """
        Unregister a plugin.
        
        Args:
            plugin_name: Plugin name
        """
        if plugin_name in self._plugins:
            self._plugins[plugin_name].cleanup()
            del self._plugins[plugin_name]
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """
        Get plugin by name.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self._plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins."""
        return list(self._plugins.keys())
    
    def load_from_module(self, module_path: str, class_name: str):
        """
        Load plugin from module.
        
        Args:
            module_path: Module path (e.g., "my_plugin.plugin")
            class_name: Class name
        """
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            
            if not issubclass(plugin_class, Plugin):
                raise ValueError(f"{class_name} is not a Plugin")
            
            plugin = plugin_class()
            self.register(plugin)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load plugin from {module_path}.{class_name}: {e}")
    
    def load_from_file(self, filepath: str):
        """
        Load plugin from file.
        
        Args:
            filepath: Path to plugin file
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Plugin file not found: {filepath}")
        
        # Load as module
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load plugin from {filepath}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find Plugin subclass
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, Plugin) and 
                obj != Plugin):
                plugin = obj()
                self.register(plugin)
                return
        
        raise ValueError(f"No Plugin class found in {filepath}")
    
    def register_hook(self, hook_name: str, callback: Callable):
        """
        Register hook callback.
        
        Args:
            hook_name: Hook name
            callback: Callback function
        """
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)
    
    def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Call hook callbacks.
        
        Args:
            hook_name: Hook name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            List of return values from callbacks
        """
        results = []
        for callback in self._hooks.get(hook_name, []):
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                # Log error but continue
                print(f"Error in hook {hook_name}: {e}")
        
        return results


# Global plugin manager
_global_plugin_manager = PluginManager()


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager."""
    return _global_plugin_manager


def register_plugin(plugin: Plugin):
    """Convenience function to register plugin."""
    _global_plugin_manager.register(plugin)


def get_plugin(plugin_name: str) -> Optional[Plugin]:
    """Convenience function to get plugin."""
    return _global_plugin_manager.get_plugin(plugin_name)


