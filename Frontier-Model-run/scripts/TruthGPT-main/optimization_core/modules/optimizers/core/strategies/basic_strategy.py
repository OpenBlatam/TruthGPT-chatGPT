"""
Basic Optimization Strategy
===========================
Basic optimization strategy with minimal techniques.
Replaces basic optimization core implementations.
"""
import torch.nn as nn
from typing import Dict, Any, List
from .base_strategy import OptimizationStrategy
from ..base_truthgpt_optimizer import OptimizationLevel
from ..optimization_pipeline import OptimizationPipeline, OptimizationStep


class BasicStrategy(OptimizationStrategy):
    """
    Basic optimization strategy.
    
    Applies minimal optimizations suitable for basic use cases.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(OptimizationLevel.BASIC, config)
        self._build_pipeline()
    
    def _build_pipeline(self):
        """Build optimization pipeline for basic strategy."""
        steps = [
            OptimizationStep('gradient_checkpointing', {}, required=False),
        ]
        self.pipeline = OptimizationPipeline(steps)
    
    def apply(self, model: nn.Module, **kwargs) -> nn.Module:
        """Apply basic optimizations."""
        if not self.can_apply(model):
            return model
        
        try:
            optimized, applied, failed = self.pipeline.apply(model)
            self.applied_techniques = applied
            if failed:
                self.logger.warning(f"Some techniques failed: {failed}")
            return optimized
        except Exception as e:
            self.logger.error(f"Error applying basic strategy: {e}")
            return model
    
    def get_techniques(self) -> List[str]:
        """Return techniques for basic optimization."""
        return [
            "gradient_checkpointing",
        ]


