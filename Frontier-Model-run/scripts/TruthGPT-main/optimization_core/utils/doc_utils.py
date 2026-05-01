"""
Documentation utilities for optimization_core.

Provides utilities for automatic documentation generation.
"""
import logging
import inspect
from typing import Dict, Any, List, Optional, Type
from pathlib import Path
import ast

logger = logging.getLogger(__name__)


class DocGenerator:
    """Generator for automatic documentation."""
    
    def __init__(self):
        """Initialize documentation generator."""
        pass
    
    def generate_module_doc(
        self,
        module: Type,
        include_private: bool = False
    ) -> Dict[str, Any]:
        """
        Generate documentation for a module.
        
        Args:
            module: Module to document
            include_private: Whether to include private members
        
        Returns:
            Documentation dictionary
        """
        doc = {
            "name": module.__name__,
            "docstring": inspect.getdoc(module) or "",
            "classes": [],
            "functions": [],
            "constants": [],
        }
        
        # Get classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if not include_private and name.startswith('_'):
                continue
            if obj.__module__ == module.__name__:
                doc["classes"].append(self._document_class(obj))
        
        # Get functions
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if not include_private and name.startswith('_'):
                continue
            if obj.__module__ == module.__name__:
                doc["functions"].append(self._document_function(obj))
        
        return doc
    
    def _document_class(self, cls: Type) -> Dict[str, Any]:
        """Document a class."""
        return {
            "name": cls.__name__,
            "docstring": inspect.getdoc(cls) or "",
            "methods": [
                {
                    "name": name,
                    "docstring": inspect.getdoc(method) or "",
                    "signature": str(inspect.signature(method)),
                }
                for name, method in inspect.getmembers(cls, inspect.ismethod)
                if not name.startswith('_')
            ],
        }
    
    def _document_function(self, func: callable) -> Dict[str, Any]:
        """Document a function."""
        sig = inspect.signature(func)
        return {
            "name": func.__name__,
            "docstring": inspect.getdoc(func) or "",
            "signature": str(sig),
            "parameters": [
                {
                    "name": param.name,
                    "annotation": str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                    "default": str(param.default) if param.default != inspect.Parameter.empty else None,
                }
                for param in sig.parameters.values()
            ],
        }
    
    def generate_api_doc(
        self,
        modules: List[Type],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate API documentation for multiple modules.
        
        Args:
            modules: List of modules to document
            output_path: Optional path to save documentation
        
        Returns:
            API documentation dictionary
        """
        api_doc = {
            "modules": [self.generate_module_doc(module) for module in modules],
        }
        
        if output_path:
            import json
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(api_doc, f, indent=2)
            logger.info(f"API documentation saved to {output_path}")
        
        return api_doc


def generate_documentation(
    modules: List[Type],
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate documentation for modules.
    
    Args:
        modules: List of modules to document
        output_path: Optional path to save documentation
    
    Returns:
        Documentation dictionary
    """
    generator = DocGenerator()
    return generator.generate_api_doc(modules, output_path)













