"""
Production Testing Module

This module contains production testing components.
"""

from __future__ import annotations

import importlib

__all__ = [
    'ProductionTestSuite',
    'TestType',
    'TestStatus',
    'TestResult',
    'BenchmarkResult',
    'create_production_test_suite',
    'production_testing_context',
    'test_optimization_basic',
    'benchmark_optimization_performance',
]

_LAZY_IMPORTS = {
    'ProductionTestSuite': '..production_testing',
    'TestType': '..production_testing',
    'TestStatus': '..production_testing',
    'TestResult': '..production_testing',
    'BenchmarkResult': '..production_testing',
    'create_production_test_suite': '..production_testing',
    'production_testing_context': '..production_testing',
    'test_optimization_basic': '..production_testing',
    'benchmark_optimization_performance': '..production_testing',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for production testing components."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name in _import_cache:
        return _import_cache[name]
    
    module_path = _LAZY_IMPORTS[name]
    try:
        module = importlib.import_module(module_path, package=__package__)
        obj = getattr(module, name)
        _import_cache[name] = obj
        return obj
    except (ImportError, AttributeError) as e:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Failed to import: {e}"
        ) from e


def list_available_testing_components() -> list[str]:
    """List all available production testing components."""
    return list(_LAZY_IMPORTS.keys())


