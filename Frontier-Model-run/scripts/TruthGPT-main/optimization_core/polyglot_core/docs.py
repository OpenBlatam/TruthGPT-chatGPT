"""
Documentation utilities for polyglot_core.

Provides automatic documentation generation.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import inspect
import json


class DocumentationGenerator:
    """
    Documentation generator for polyglot_core.
    
    Generates API documentation from code.
    """
    
    def __init__(self):
        self._modules: Dict[str, Any] = {}
    
    def register_module(self, name: str, module: Any):
        """
        Register module for documentation.
        
        Args:
            name: Module name
            module: Module object
        """
        self._modules[name] = module
    
    def extract_module_docs(self, module: Any) -> Dict[str, Any]:
        """
        Extract documentation from module.
        
        Args:
            module: Module object
            
        Returns:
            Documentation dictionary
        """
        docs = {
            'name': module.__name__,
            'docstring': module.__doc__,
            'classes': [],
            'functions': []
        }
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                class_doc = {
                    'name': name,
                    'docstring': obj.__doc__,
                    'methods': []
                }
                
                for method_name, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                    if method_name.startswith('_'):
                        continue
                    class_doc['methods'].append({
                        'name': method_name,
                        'docstring': method.__doc__,
                        'signature': str(inspect.signature(method))
                    })
                
                docs['classes'].append(class_doc)
            
            elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
                docs['functions'].append({
                    'name': name,
                    'docstring': obj.__doc__,
                    'signature': str(inspect.signature(obj))
                })
        
        return docs
    
    def generate_markdown(self, output_file: Optional[Path] = None) -> str:
        """
        Generate Markdown documentation.
        
        Args:
            output_file: Optional output file path
            
        Returns:
            Markdown string
        """
        lines = ["# Polyglot Core API Documentation\n"]
        
        for module_name, module in self._modules.items():
            docs = self.extract_module_docs(module)
            
            lines.append(f"## {docs['name']}\n")
            
            if docs['docstring']:
                lines.append(f"{docs['docstring']}\n")
            
            # Classes
            if docs['classes']:
                lines.append("### Classes\n\n")
                for class_doc in docs['classes']:
                    lines.append(f"#### {class_doc['name']}\n\n")
                    if class_doc['docstring']:
                        lines.append(f"{class_doc['docstring']}\n\n")
                    
                    if class_doc['methods']:
                        lines.append("**Methods:**\n\n")
                        for method in class_doc['methods']:
                            lines.append(f"- `{method['name']}{method['signature']}`\n")
                            if method['docstring']:
                                lines.append(f"  {method['docstring']}\n")
                        lines.append("\n")
            
            # Functions
            if docs['functions']:
                lines.append("### Functions\n\n")
                for func_doc in docs['functions']:
                    lines.append(f"#### {func_doc['name']}\n\n")
                    lines.append(f"```python\n{func_doc['name']}{func_doc['signature']}\n```\n\n")
                    if func_doc['docstring']:
                        lines.append(f"{func_doc['docstring']}\n\n")
            
            lines.append("\n---\n\n")
        
        markdown = "".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown)
        
        return markdown
    
    def generate_json(self, output_file: Optional[Path] = None) -> str:
        """
        Generate JSON documentation.
        
        Args:
            output_file: Optional output file path
            
        Returns:
            JSON string
        """
        docs = {}
        
        for module_name, module in self._modules.items():
            docs[module_name] = self.extract_module_docs(module)
        
        json_str = json.dumps(docs, indent=2)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        return json_str


# Global documentation generator
_global_doc_generator = DocumentationGenerator()


def get_documentation_generator() -> DocumentationGenerator:
    """Get global documentation generator."""
    return _global_doc_generator


def generate_docs(output_file: Optional[Path] = None, format: str = "markdown") -> str:
    """
    Generate documentation.
    
    Args:
        output_file: Optional output file path
        format: Output format ("markdown" or "json")
        
    Returns:
        Documentation string
    """
    generator = get_documentation_generator()
    
    # Register all polyglot_core modules
    try:
        from . import backend, cache, attention, compression, inference
        generator.register_module("backend", backend)
        generator.register_module("cache", cache)
        generator.register_module("attention", attention)
        generator.register_module("compression", compression)
        generator.register_module("inference", inference)
    except ImportError:
        pass
    
    if format == "markdown":
        return generator.generate_markdown(output_file)
    else:
        return generator.generate_json(output_file)













