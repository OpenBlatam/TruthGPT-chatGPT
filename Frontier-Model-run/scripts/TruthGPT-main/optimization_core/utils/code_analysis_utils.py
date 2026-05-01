"""
Code analysis utilities for optimization_core.

Provides utilities for analyzing code quality and complexity.
"""
import logging
import ast
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """Analyzer for Python code."""
    
    def __init__(self):
        """Initialize code analyzer."""
        pass
    
    def analyze_file(
        self,
        file_path: Path
    ) -> Dict[str, Any]:
        """
        Analyze a Python file.
        
        Args:
            file_path: Path to Python file
        
        Returns:
            Analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            return {
                "file": str(file_path),
                "lines": len(source.splitlines()),
                "functions": self._count_functions(tree),
                "classes": self._count_classes(tree),
                "imports": self._count_imports(tree),
                "complexity": self._calculate_complexity(tree),
            }
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}", exc_info=True)
            return {
                "file": str(file_path),
                "error": str(e)
            }
    
    def _count_functions(self, tree: ast.AST) -> int:
        """Count functions in AST."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
    
    def _count_classes(self, tree: ast.AST) -> int:
        """Count classes in AST."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
    
    def _count_imports(self, tree: ast.AST) -> int:
        """Count imports in AST."""
        return len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))])
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity (simplified)."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Add complexity for control flow statements
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def analyze_directory(
        self,
        directory: Path,
        pattern: str = "*.py"
    ) -> Dict[str, Any]:
        """
        Analyze all Python files in directory.
        
        Args:
            directory: Directory to analyze
            pattern: File pattern
        
        Returns:
            Analysis results
        """
        results = []
        total_lines = 0
        total_functions = 0
        total_classes = 0
        
        for file_path in directory.rglob(pattern):
            if file_path.is_file():
                analysis = self.analyze_file(file_path)
                if "error" not in analysis:
                    results.append(analysis)
                    total_lines += analysis.get("lines", 0)
                    total_functions += analysis.get("functions", 0)
                    total_classes += analysis.get("classes", 0)
        
        return {
            "files_analyzed": len(results),
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "files": results,
        }


def analyze_codebase(
    root_path: Path,
    pattern: str = "*.py"
) -> Dict[str, Any]:
    """
    Analyze entire codebase.
    
    Args:
        root_path: Root path of codebase
        pattern: File pattern
    
    Returns:
        Analysis results
    """
    analyzer = CodeAnalyzer()
    return analyzer.analyze_directory(root_path, pattern)













