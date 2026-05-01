"""
Quantum Utilities Module

This module contains quantum computing utilities and quantum-based optimization systems.
"""

from __future__ import annotations

import importlib

__all__ = [
    'QuantumUtils',
    'QuantumDeepLearningSystem',
    'QuantumHybridAISystem',
    'QuantumNeuralOptimizationEngine',
    'UniversalQuantumOptimizer',
]

_LAZY_IMPORTS = {
    'QuantumUtils': '..quantum_utils',
    'QuantumDeepLearningSystem': '..quantum_deep_learning_system',
    'QuantumHybridAISystem': '..quantum_hybrid_ai_system',
    'QuantumNeuralOptimizationEngine': '..quantum_neural_optimization_engine',
    'UniversalQuantumOptimizer': '..universal_quantum_optimizer',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for quantum utility modules."""
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



