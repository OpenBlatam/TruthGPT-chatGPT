"""
Optimizer Utilities Module

This module contains various optimizer utilities and optimization engines.
"""

from __future__ import annotations

import importlib

__all__ = [
    'HyperSpeedOptimizer',
    'CuttingEdgeUniversalQuantumOptimizer',
    'UniversalQuantumOptimizer',
    'NeuralEvolutionaryOptimizer',
    'AdvancedAIOptimizer',
    'AutoPerformanceOptimizer',
    'UltraNeuralNetworkOptimizer',
    'UltraAIOptimizer',
    'UltraMachineLearningOptimizer',
    'NextGenOptimizationEngine',
    'NextGenQuantumNeuralOptimizationEngine',
    'RevolutionaryQuantumDeepLearningSystem',
    'UltraQuantumOptimization',
]

_LAZY_IMPORTS = {
    'HyperSpeedOptimizer': '..hyper_speed_optimizer',
    'CuttingEdgeUniversalQuantumOptimizer': '..cutting_edge_universal_quantum_optimizer',
    'UniversalQuantumOptimizer': '..universal_quantum_optimizer',
    'NeuralEvolutionaryOptimizer': '..neural_evolutionary_optimizer',
    'AdvancedAIOptimizer': '..advanced_ai_optimizer',
    'AutoPerformanceOptimizer': '..auto_performance_optimizer',
    'UltraNeuralNetworkOptimizer': '..ultra_neural_network_optimizer',
    'UltraAIOptimizer': '..ultra_ai_optimizer',
    'UltraMachineLearningOptimizer': '..ultra_machine_learning_optimizer',
    'NextGenOptimizationEngine': '..next_gen_optimization_engine',
    'NextGenQuantumNeuralOptimizationEngine': '..next_gen_quantum_neural_optimization_engine',
    'RevolutionaryQuantumDeepLearningSystem': '..revolutionary_quantum_deep_learning_system',
    'UltraQuantumOptimization': '..ultra_quantum_optimization',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for optimizer utility modules."""
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


def list_available_optimizers() -> list[str]:
    """List all available optimizer utilities."""
    return list(_LAZY_IMPORTS.keys())


def get_optimizer_info(optimizer_name: str) -> dict[str, any]:
    """Get information about an optimizer utility."""
    if optimizer_name not in _LAZY_IMPORTS:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return {
        'name': optimizer_name,
        'module': _LAZY_IMPORTS[optimizer_name],
        'available': optimizer_name in _import_cache or True,
    }


