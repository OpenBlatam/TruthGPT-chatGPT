"""
Dependency management utilities for optimization_core.

Provides utilities for managing dependencies and versions.
"""
import logging
import importlib
import sys
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


from pydantic import BaseModel


class Dependency(BaseModel):
    """Dependency information."""
    name: str
    version: str
    required: bool = True
    installed: bool = False
    installed_version: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation."""
        status = "✓" if self.installed else "✗"
        if self.installed and self.installed_version:
            return f"{status} {self.name}=={self.version} (installed: {self.installed_version})"
        return f"{status} {self.name}=={self.version}"


class DependencyManager:
    """Manager for dependencies."""
    
    def __init__(self):
        """Initialize dependency manager."""
        self.dependencies: Dict[str, Dependency] = {}
        self._check_installed()
    
    def register(
        self,
        name: str,
        version: str,
        required: bool = True
    ):
        """
        Register a dependency.
        
        Args:
            name: Package name
            version: Required version
            required: Whether dependency is required
        """
        self.dependencies[name] = Dependency(
            name=name,
            version=version,
            required=required
        )
        self._check_installed()
    
    def _check_installed(self):
        """Check which dependencies are installed."""
        for dep in self.dependencies.values():
            try:
                # Use importlib.metadata for cleaner dependency checking
                if sys.version_info >= (3, 8):
                    from importlib.metadata import version, PackageNotFoundError
                else:
                    # Fallback for older python
                    from importlib_metadata import version, PackageNotFoundError

                dep.installed_version = version(dep.name)
                dep.installed = True
            except (ImportError, Exception):
                # Fallback or package not found
                dep.installed = False
                dep.installed_version = None
    
    def check_all(self) -> Tuple[bool, List[str]]:
        """
        Check all dependencies.
        
        Returns:
            Tuple of (all_ok, missing_deps)
        """
        missing = []
        
        for dep in self.dependencies.values():
            if dep.required and not dep.installed:
                missing.append(f"{dep.name}=={dep.version}")
        
        return len(missing) == 0, missing
    
    def get_missing(self) -> List[Dependency]:
        """
        Get missing required dependencies.
        
        Returns:
            List of missing dependencies
        """
        return [dep for dep in self.dependencies.values() if dep.required and not dep.installed]
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get dependency status.
        
        Returns:
            Status dictionary
        """
        all_deps = list(self.dependencies.values())
        installed = [dep for dep in all_deps if dep.installed]
        missing = [dep for dep in all_deps if not dep.installed and dep.required]
        
        return {
            "total": len(all_deps),
            "installed": len(installed),
            "missing": len(missing),
            "dependencies": {dep.name: {
                "version": dep.version,
                "installed": dep.installed,
                "installed_version": dep.installed_version,
                "required": dep.required,
            } for dep in all_deps}
        }
    
    def import_safe(
        self,
        module_name: str,
        package_name: Optional[str] = None
    ) -> Optional[Any]:
        """
        Safely import a module.
        
        Args:
            module_name: Module name
            package_name: Package name (for checking dependency)
        
        Returns:
            Imported module or None
        """
        if package_name:
            dep = self.dependencies.get(package_name)
            if dep and dep.required and not dep.installed:
                logger.warning(f"Required dependency '{package_name}' not installed")
                return None
        
        try:
            return importlib.import_module(module_name)
        except ImportError as e:
            logger.warning(f"Failed to import '{module_name}': {e}")
            return None


# Global dependency manager
_global_dependency_manager = DependencyManager()


def get_dependency_manager() -> DependencyManager:
    """Get global dependency manager."""
    return _global_dependency_manager


def register_dependency(name: str, version: str, required: bool = True):
    """
    Register a dependency.
    
    Args:
        name: Package name
        version: Required version
        required: Whether dependency is required
    """
    _global_dependency_manager.register(name, version, required)


# --- Lazy Import Utilities ---

_import_cache = {}
_cache_lock = threading.RLock()

def resolve_lazy_import(name: str, package: str, lazy_imports: Dict[str, str]) -> Any:
    """
    Resolve a lazy import.

    Args:
        name: The name of the attribute relative to the package.
        package: The name of the package (e.g., 'optimization_core.optimizers').
        lazy_imports: A dictionary mapping names to module paths.

    Returns:
        The imported module or attribute.

    Raises:
        AttributeError: If the name is not in lazy_imports or import fails.
    """
    if name.startswith('_'):
         raise AttributeError(f"module '{package}' has no attribute '{name}'")

    with _cache_lock:
        cache_key = f"{package}.{name}"
        if cache_key in _import_cache:
            return _import_cache[cache_key]

        if name not in lazy_imports:
             raise AttributeError(f"module '{package}' has no attribute '{name}'")

        module_path = lazy_imports[name]

        try:
            if module_path.startswith('.'):
                 # Relative import
                 module = importlib.import_module(module_path, package=package)
            else:
                 module = importlib.import_module(module_path)

            # If the attribute exists in the module, return it (Class/Function)
            # Otherwise return the module itself (Submodule)
            if hasattr(module, name):
                obj = getattr(module, name)
            else:
                obj = module

            _import_cache[cache_key] = obj
            return obj
        except Exception as e:
            raise AttributeError(
                f"module '{package}' has no attribute '{name}'. "
                f"Failed to import: {e}"
            ) from e

