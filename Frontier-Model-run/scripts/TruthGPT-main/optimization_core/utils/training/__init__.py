"""
Training Utilities Module

This module contains utilities for training, evaluation, and optimization.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Dict, List

__all__ = [
    'TruthGPTTrainingUtils',
    'TruthGPTAdvancedTraining',
    'TruthGPTOptimizationUtils',
    'TruthGPTEvaluationUtils',
    'TruthGPTAdvancedEvaluation',
]

_LAZY_IMPORTS = {
    'TruthGPTTrainingUtils': 'optimization_core.modules.truthgpt.training',
    'TruthGPTAdvancedTraining': 'optimization_core.modules.truthgpt.advanced_training',
    'TruthGPTOptimizationUtils': 'optimization_core.modules.truthgpt.optimization_utils',
    'TruthGPTEvaluationUtils': 'optimization_core.modules.truthgpt.evaluation',
    'TruthGPTAdvancedEvaluation': 'optimization_core.modules.truthgpt.advanced_evaluation',
}

# Thread-safe cache for loaded modules
_import_cache: Dict[str, Any] = {}
_cache_lock = threading.RLock()


def __getattr__(name: str):
    """Lazy import system for training utility modules."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    with _cache_lock:
        if name in _import_cache:
            return _import_cache[name]
        
        if name not in _LAZY_IMPORTS:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
        
        module_path = _LAZY_IMPORTS[name]
        try:
            # Use absolute imports
            module = importlib.import_module(module_path)
            obj = getattr(module, name)
            _import_cache[name] = obj
            return obj
        except (ImportError, AttributeError) as e:
            raise AttributeError(
                f"module '{__name__}' has no attribute '{name}'. "
                f"Failed to import from '{module_path}': {e}"
            ) from e



