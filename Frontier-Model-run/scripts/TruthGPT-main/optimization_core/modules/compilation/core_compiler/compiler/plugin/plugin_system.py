"""
Plugin System for TruthGPT Compiler
Extensible plugin architecture for compiler components
"""

import enum
import logging
import time
import importlib
import inspect
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
import numpy as np

logger = logging.getLogger(__name__)

class PluginStatus(enum.Enum):
    """Plugin status"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"

class PluginResult:
    """Result of plugin operation"""
    
    def __init__(self, success: bool, data: Any = None, error: str = None):
        self.success = success
        self.data = data
        self.error = error
        self.timestamp = time.time()

@dataclass
class PluginConfig:
    """Configuration for plugin"""
    name: str
    version: str
    description: str
    enabled: bool = True
    priority: int = 0
    dependencies: List[str] = None
    parameters: Dict[str, Any] = None
    auto_load: bool = True

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.parameters is None:
            self.parameters = {}

class PluginInterface(ABC):
    """Base interface for all plugins"""
    
    @abstractmethod
    def initialize(self, config: PluginConfig) -> PluginResult:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    def execute(self, data: Any, **kwargs) -> PluginResult:
        """Execute plugin operation"""
        pass
    
    @abstractmethod
    def cleanup(self) -> PluginResult:
        """Cleanup plugin resources"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information"""
        pass

class CompilerPlugin(PluginInterface):
    """Base class for compiler plugins"""
    
    def __init__(self, config: PluginConfig):
        self.config = config
        self.status = PluginStatus.UNLOADED
        self.logger = logging.getLogger(f"Plugin.{config.name}")
        
    def initialize(self, config: PluginConfig) -> PluginResult:
        """Initialize the plugin"""
        try:
            self.status = PluginStatus.LOADING
            self.logger.info(f"Initializing plugin: {config.name}")
            
            # Plugin-specific initialization
            result = self._initialize_plugin(config)
            
            if result.success:
                self.status = PluginStatus.LOADED
                self.logger.info(f"Plugin {config.name} initialized successfully")
            else:
                self.status = PluginStatus.ERROR
                self.logger.error(f"Plugin {config.name} initialization failed: {result.error}")
            
            return result
            
        except Exception as e:
            self.status = PluginStatus.ERROR
            self.logger.error(f"Plugin {config.name} initialization error: {str(e)}")
            return PluginResult(success=False, error=str(e))
    
    def execute(self, data: Any, **kwargs) -> PluginResult:
        """Execute plugin operation"""
        try:
            if self.status != PluginStatus.LOADED and self.status != PluginStatus.ACTIVE:
                return PluginResult(success=False, error="Plugin not loaded")
            
            self.status = PluginStatus.ACTIVE
            start_time = time.time()
            
            # Execute plugin-specific operation
            result = self._execute_plugin(data, **kwargs)
            
            execution_time = time.time() - start_time
            self.logger.debug(f"Plugin {self.config.name} executed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.status = PluginStatus.ERROR
            self.logger.error(f"Plugin {self.config.name} execution error: {str(e)}")
            return PluginResult(success=False, error=str(e))
    
    def cleanup(self) -> PluginResult:
        """Cleanup plugin resources"""
        try:
            self.logger.info(f"Cleaning up plugin: {self.config.name}")
            
            # Plugin-specific cleanup
            result = self._cleanup_plugin()
            
            self.status = PluginStatus.UNLOADED
            self.logger.info(f"Plugin {self.config.name} cleaned up")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Plugin {self.config.name} cleanup error: {str(e)}")
            return PluginResult(success=False, error=str(e))
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information"""
        return {
            "name": self.config.name,
            "version": self.config.version,
            "description": self.config.description,
            "status": self.status.value,
            "enabled": self.config.enabled,
            "priority": self.config.priority,
            "dependencies": self.config.dependencies
        }
    
    @abstractmethod
    def _initialize_plugin(self, config: PluginConfig) -> PluginResult:
        """Plugin-specific initialization"""
        pass
    
    @abstractmethod
    def _execute_plugin(self, data: Any, **kwargs) -> PluginResult:
        """Plugin-specific execution"""
        pass
    
    @abstractmethod
    def _cleanup_plugin(self) -> PluginResult:
        """Plugin-specific cleanup"""
        pass

class PluginRegistry:
    """Registry for managing plugins"""
    
    def __init__(self):
        self.plugins = {}
        self.plugin_types = {}
        self.dependencies = {}
        
    def register_plugin(self, plugin_class: Type[CompilerPlugin], config: PluginConfig):
        """Register a plugin class"""
        self.plugin_types[config.name] = plugin_class
        self.dependencies[config.name] = config.dependencies
        self.logger.info(f"Registered plugin: {config.name}")
    
    def unregister_plugin(self, name: str):
        """Unregister a plugin"""
        if name in self.plugin_types:
            del self.plugin_types[name]
            del self.dependencies[name]
            self.logger.info(f"Unregistered plugin: {name}")
    
    def get_plugin_class(self, name: str) -> Optional[Type[CompilerPlugin]]:
        """Get plugin class by name"""
        return self.plugin_types.get(name)
    
    def get_available_plugins(self) -> List[str]:
        """Get list of available plugins"""
        return list(self.plugin_types.keys())
    
    def get_plugin_dependencies(self, name: str) -> List[str]:
        """Get plugin dependencies"""
        return self.dependencies.get(name, [])

class PluginManager:
    """Manager for plugin lifecycle"""
    
    def __init__(self):
        self.registry = PluginRegistry()
        self.active_plugins = {}
        self.plugin_configs = {}
        self.logger = logging.getLogger("PluginManager")
        
    def register_plugin(self, plugin_class: Type[CompilerPlugin], config: PluginConfig):
        """Register a plugin"""
        self.registry.register_plugin(plugin_class, config)
        self.plugin_configs[config.name] = config
        
    def load_plugin(self, name: str) -> PluginResult:
        """Load a plugin"""
        if name not in self.plugin_configs:
            return PluginResult(success=False, error=f"Plugin {name} not registered")
        
        if name in self.active_plugins:
            return PluginResult(success=True, data=self.active_plugins[name])
        
        config = self.plugin_configs[name]
        plugin_class = self.registry.get_plugin_class(name)
        
        if not plugin_class:
            return PluginResult(success=False, error=f"Plugin class for {name} not found")
        
        try:
            plugin = plugin_class(config)
            result = plugin.initialize(config)
            
            if result.success:
                self.active_plugins[name] = plugin
                self.logger.info(f"Plugin {name} loaded successfully")
            else:
                self.logger.error(f"Plugin {name} load failed: {result.error}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Plugin {name} load error: {str(e)}")
            return PluginResult(success=False, error=str(e))
    
    def unload_plugin(self, name: str) -> PluginResult:
        """Unload a plugin"""
        if name not in self.active_plugins:
            return PluginResult(success=False, error=f"Plugin {name} not loaded")
        
        try:
            plugin = self.active_plugins[name]
            result = plugin.cleanup()
            
            if result.success:
                del self.active_plugins[name]
                self.logger.info(f"Plugin {name} unloaded successfully")
            else:
                self.logger.error(f"Plugin {name} unload failed: {result.error}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Plugin {name} unload error: {str(e)}")
            return PluginResult(success=False, error=str(e))
    
    def execute_plugin(self, name: str, data: Any, **kwargs) -> PluginResult:
        """Execute a plugin"""
        if name not in self.active_plugins:
            return PluginResult(success=False, error=f"Plugin {name} not loaded")
        
        plugin = self.active_plugins[name]
        return plugin.execute(data, **kwargs)
    
    def get_plugin(self, name: str) -> Optional[CompilerPlugin]:
        """Get a plugin instance"""
        return self.active_plugins.get(name)
    
    def get_active_plugins(self) -> List[str]:
        """Get list of active plugins"""
        return list(self.active_plugins.keys())
    
    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get plugin information"""
        if name in self.active_plugins:
            return self.active_plugins[name].get_info()
        return None
    
    def load_all_plugins(self) -> Dict[str, PluginResult]:
        """Load all registered plugins"""
        results = {}
        
        for name, config in self.plugin_configs.items():
            if config.auto_load and config.enabled:
                results[name] = self.load_plugin(name)
        
        return results
    
    def unload_all_plugins(self) -> Dict[str, PluginResult]:
        """Unload all active plugins"""
        results = {}
        
        for name in list(self.active_plugins.keys()):
            results[name] = self.unload_plugin(name)
        
        return results
    
    def get_plugin_statistics(self) -> Dict[str, Any]:
        """Get plugin statistics"""
        return {
            "total_registered": len(self.plugin_configs),
            "total_active": len(self.active_plugins),
            "enabled_plugins": sum(1 for config in self.plugin_configs.values() if config.enabled),
            "disabled_plugins": sum(1 for config in self.plugin_configs.values() if not config.enabled)
        }

# Example plugin implementations
class OptimizationPlugin(CompilerPlugin):
    """Example optimization plugin"""
    
    def _initialize_plugin(self, config: PluginConfig) -> PluginResult:
        """Initialize optimization plugin"""
        self.logger.info("Initializing optimization plugin")
        return PluginResult(success=True)
    
    def _execute_plugin(self, data: Any, **kwargs) -> PluginResult:
        """Execute optimization"""
        self.logger.info("Executing optimization plugin")
        # Plugin-specific optimization logic
        return PluginResult(success=True, data=data)
    
    def _cleanup_plugin(self) -> PluginResult:
        """Cleanup optimization plugin"""
        self.logger.info("Cleaning up optimization plugin")
        return PluginResult(success=True)

class AnalysisPlugin(CompilerPlugin):
    """Example analysis plugin"""
    
    def _initialize_plugin(self, config: PluginConfig) -> PluginResult:
        """Initialize analysis plugin"""
        self.logger.info("Initializing analysis plugin")
        return PluginResult(success=True)
    
    def _execute_plugin(self, data: Any, **kwargs) -> PluginResult:
        """Execute analysis"""
        self.logger.info("Executing analysis plugin")
        # Plugin-specific analysis logic
        return PluginResult(success=True, data=data)
    
    def _cleanup_plugin(self) -> PluginResult:
        """Cleanup analysis plugin"""
        self.logger.info("Cleaning up analysis plugin")
        return PluginResult(success=True)

def create_plugin_manager() -> PluginManager:
    """Create a plugin manager instance"""
    return PluginManager()

def plugin_compilation_context(plugin_manager: PluginManager):
    """Create a plugin compilation context"""
    class PluginCompilationContext:
        def __init__(self, pm: PluginManager):
            self.plugin_manager = pm
            
        def __enter__(self):
            # Load all plugins
            self.plugin_manager.load_all_plugins()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Plugin cleanup is handled by individual plugins
            pass
    
    return PluginCompilationContext(plugin_manager)






