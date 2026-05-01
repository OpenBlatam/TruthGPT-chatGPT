"""
Base Optimization Strategy
==========================
Base class for optimization strategies using Strategy Pattern.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import torch.nn as nn
import logging
from ..base_truthgpt_optimizer import OptimizationLevel

logger = logging.getLogger(__name__)


class OptimizationStrategy(ABC):
    """
    Base class for optimization strategies.
    
    Each strategy encapsulates a set of optimization techniques
    that can be applied to a model. Strategies can be composed
    together for more complex optimization pipelines.
    """
    
    def __init__(self, level: OptimizationLevel, config: Dict[str, Any] = None):
        """
        Initialize strategy.
        
        Args:
            level: Optimization level for this strategy
            config: Configuration dictionary
        """
        self.level = level
        self.config = config or {}
        self.applied_techniques: List[str] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def apply(self, model: nn.Module, **kwargs) -> nn.Module:
        """
        Apply optimization strategy to model.
        
        Args:
            model: Model to optimize
            **kwargs: Additional parameters
            
        Returns:
            Optimized model
        """
        pass
    
    @abstractmethod
    def get_techniques(self) -> List[str]:
        """
        Return list of techniques this strategy applies.
        
        Returns:
            List of technique names
        """
        pass
    
    def calculate_metrics(
        self, 
        original: nn.Module, 
        optimized: nn.Module
    ) -> Dict[str, float]:
        """
        Calculate performance metrics between original and optimized model.
        
        Args:
            original: Original model
            optimized: Optimized model
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics calculation
        # Can be overridden by subclasses for more sophisticated metrics
        return {
            'speed_improvement': 1.0,
            'memory_reduction': 0.0,
            'accuracy_preservation': 1.0,
            'energy_efficiency': 1.0,
        }
    
    def can_apply(self, model: nn.Module) -> bool:
        """
        Check if strategy can be applied to model.
        
        Args:
            model: Model to check
            
        Returns:
            True if strategy can be applied
        """
        return True





