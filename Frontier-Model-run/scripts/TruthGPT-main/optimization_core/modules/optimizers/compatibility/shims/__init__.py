"""
Compatibility Shims
===================
Backward compatibility shims for deprecated optimization core classes.
These redirect to UnifiedOptimizer with appropriate strategies.
"""

from .enhanced_optimization_core import EnhancedOptimizationCore
from .hybrid_optimization_core import HybridOptimizationCore

__all__ = [
    'EnhancedOptimizationCore',
    'HybridOptimizationCore',
]





