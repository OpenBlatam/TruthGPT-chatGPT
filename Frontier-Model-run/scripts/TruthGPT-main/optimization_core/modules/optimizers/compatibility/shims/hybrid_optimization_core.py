"""
Compatibility Shim for HybridOptimizationCore
==============================================
DEPRECATED: Use UnifiedOptimizer with OptimizationLevel.EXPERT instead.

This shim provides backward compatibility for code using HybridOptimizationCore.
It redirects to UnifiedOptimizer with HybridStrategy.
"""
import warnings
from typing import Dict, Any
import torch.nn as nn

# Import the new unified optimizer
from ...core.unified_optimizer import UnifiedOptimizer
from ...core.base_truthgpt_optimizer import OptimizationLevel, OptimizationResult


class HybridOptimizationCore:
    """
    Compatibility shim for hybrid_optimization_core.
    
    DEPRECATED: Use UnifiedOptimizer with OptimizationLevel.EXPERT instead.
    
    This class will be removed in a future version.
    
    Migration:
        # Old code
        from optimizers.hybrid_optimization_core import HybridOptimizationCore
        optimizer = HybridOptimizationCore(config)
        
        # New code
        from optimizers.core.unified_optimizer import UnifiedOptimizer
        from optimizers.core.base_truthgpt_optimizer import OptimizationLevel
        optimizer = UnifiedOptimizer(level=OptimizationLevel.EXPERT, config=config)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize compatibility shim.
        
        Args:
            config: Configuration dictionary
        """
        warnings.warn(
            "HybridOptimizationCore is deprecated. "
            "Use UnifiedOptimizer(level=OptimizationLevel.EXPERT) instead. "
            "See migration guide for details.",
            DeprecationWarning,
            stacklevel=2
        )
        self._optimizer = UnifiedOptimizer(
            level=OptimizationLevel.EXPERT,
            config=config
        )
        self.config = config or {}
    
    def optimize(self, model: nn.Module, **kwargs) -> OptimizationResult:
        """
        Optimize model using hybrid strategy.
        
        Args:
            model: Model to optimize
            **kwargs: Additional parameters
            
        Returns:
            OptimizationResult
        """
        return self._optimizer.optimize(model, **kwargs)
    
    def __getattr__(self, name):
        """
        Delegate attribute access to underlying optimizer.
        
        This allows access to methods and properties of the underlying
        UnifiedOptimizer for backward compatibility.
        """
        return getattr(self._optimizer, name)


# Export for backward compatibility
__all__ = ['HybridOptimizationCore']





