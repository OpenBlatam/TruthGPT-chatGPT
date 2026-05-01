"""
Lazy module loading system for better modularity.
Allows loading modules on-demand to reduce memory footprint.
"""
import logging
import importlib
from typing import Dict, Any, Optional, Type
from pathlib import Path

logger = logging.getLogger(__name__)


class ModuleLoader:
    """
    Lazy loader for modules.
    Allows deferring module imports until needed.
    """
    
    def __init__(self):
        """Initialize module loader."""
        self._loaded_modules: Dict[str, Any] = {}
        self._module_paths: Dict[str, str] = {}
    
    def register_module(self, name: str, module_path: str) -> None:
        """
        Register a module for lazy loading.
        
        Args:
            name: Module name (alias)
            module_path: Full module path (e.g., "models.model_manager")
        """
        self._module_paths[name] = module_path
        logger.debug(f"Module registered: {name} -> {module_path}")
    
    def load_module(self, name: str) -> Any:
        """
        Load a module (lazy load if not already loaded).
        
        Args:
            name: Module name
        
        Returns:
            Loaded module
        """
        # Return if already loaded
        if name in self._loaded_modules:
            return self._loaded_modules[name]
        
        # Load from registered path
        if name not in self._module_paths:
            raise ValueError(f"Module '{name}' not registered")
        
        module_path = self._module_paths[name]
        
        try:
            module = importlib.import_module(module_path)
            self._loaded_modules[name] = module
            logger.debug(f"Module loaded: {name}")
            return module
        except Exception as e:
            logger.error(f"Failed to load module '{name}': {e}", exc_info=True)
            raise
    
    def get_class(self, module_name: str, class_name: str) -> Type:
        """
        Get a class from a module.
        
        Args:
            module_name: Module name
            class_name: Class name
        
        Returns:
            Class type
        """
        module = self.load_module(module_name)
        return getattr(module, class_name)
    
    def create_instance(
        self,
        module_name: str,
        class_name: str,
        **kwargs
    ) -> Any:
        """
        Create an instance of a class from a module.
        
        Args:
            module_name: Module name
            class_name: Class name
            **kwargs: Constructor arguments
        
        Returns:
            Instance
        """
        cls = self.get_class(module_name, class_name)
        return cls(**kwargs)
    
    def load_from_directory(
        self,
        directory: str,
        package_name: str
    ) -> Dict[str, str]:
        """
        Auto-register modules from a directory.
        
        Args:
            directory: Directory path
            package_name: Package name prefix
        
        Returns:
            Dictionary of registered modules
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return {}
        
        registered = {}
        
        for py_file in dir_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            
            module_name = py_file.stem
            full_path = f"{package_name}.{module_name}"
            
            self.register_module(module_name, full_path)
            registered[module_name] = full_path
        
        logger.info(f"Registered {len(registered)} modules from {directory}")
        return registered
    
    def unload_module(self, name: str) -> None:
        """
        Unload a module (remove from cache).
        
        Args:
            name: Module name
        """
        self._loaded_modules.pop(name, None)
        logger.debug(f"Module unloaded: {name}")
    
    def clear_cache(self) -> None:
        """Clear all loaded modules."""
        self._loaded_modules.clear()
        logger.info("Module cache cleared")


# Global module loader
_module_loader = ModuleLoader()


def get_module_loader() -> ModuleLoader:
    """Get the global module loader."""
    return _module_loader


def lazy_load(module_path: str):
    """
    Decorator for lazy module loading.
    
    Usage:
        @lazy_load("models.model_manager")
        def get_model_manager():
            from models.model_manager import ModelManager
            return ModelManager
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            module = _module_loader.load_module(module_path)
            return func(*args, **kwargs)
        return wrapper
    return decorator



