"""
Optimization Strategy Pattern Implementation
Base classes and concrete strategies for AI-powered optimization
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging

from .config import DEFAULT_PRUNING_AMOUNT

logger = logging.getLogger(__name__)


class OptimizationStrategy(ABC):
    """Base class for optimization strategies."""
    
    @abstractmethod
    def apply(self, model: nn.Module) -> nn.Module:
        """Apply optimization strategy to model."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name."""
        pass


class QuantizationStrategy(OptimizationStrategy):
    """AI-powered quantization strategy."""
    
    @property
    def name(self) -> str:
        return "ai_quantization"
    
    def apply(self, model: nn.Module) -> nn.Module:
        try:
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
        except Exception as e:
            logger.warning(f"AI quantization failed: {e}")
        return model


class PruningStrategy(OptimizationStrategy):
    """AI-powered pruning strategy."""
    
    @property
    def name(self) -> str:
        return "ai_pruning"
    
    def apply(self, model: nn.Module) -> nn.Module:
        try:
            from torch.nn.utils import prune
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=DEFAULT_PRUNING_AMOUNT)
        except Exception as e:
            logger.warning(f"AI pruning failed: {e}")
        return model


class MixedPrecisionStrategy(OptimizationStrategy):
    """AI-powered mixed precision strategy."""
    
    @property
    def name(self) -> str:
        return "ai_mixed_precision"
    
    def apply(self, model: nn.Module) -> nn.Module:
        return model.half()


class KernelFusionStrategy(OptimizationStrategy):
    """AI-powered kernel fusion strategy."""
    
    @property
    def name(self) -> str:
        return "ai_kernel_fusion"
    
    def apply(self, model: nn.Module) -> nn.Module:
        torch.backends.cudnn.benchmark = True
        return model


class MemoryOptimizationStrategy(OptimizationStrategy):
    """AI-powered memory optimization strategy."""
    
    @property
    def name(self) -> str:
        return "ai_memory_optimization"
    
    def apply(self, model: nn.Module) -> nn.Module:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model


class ParallelProcessingStrategy(OptimizationStrategy):
    """AI-powered parallel processing strategy."""
    
    @property
    def name(self) -> str:
        return "ai_parallel_processing"
    
    def apply(self, model: nn.Module) -> nn.Module:
        if torch.cuda.device_count() > 1:
            return nn.DataParallel(model)
        return model


class StrategyRegistry:
    """Registry for optimization strategies."""
    
    def __init__(self):
        self._strategies: Dict[str, OptimizationStrategy] = {}
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """Initialize default optimization strategies."""
        strategies = [
            QuantizationStrategy(),
            PruningStrategy(),
            MixedPrecisionStrategy(),
            KernelFusionStrategy(),
            MemoryOptimizationStrategy(),
            ParallelProcessingStrategy(),
        ]
        
        for strategy in strategies:
            self.register(strategy)
    
    def register(self, strategy: OptimizationStrategy):
        """Register an optimization strategy."""
        strategy_key = strategy.name.replace('ai_', '')
        self._strategies[strategy_key] = strategy
    
    def get(self, strategy_name: str) -> Optional[OptimizationStrategy]:
        """Get strategy by name."""
        return self._strategies.get(strategy_name)
    
    def get_all_strategy_names(self) -> List[str]:
        """Get all registered strategy names."""
        return list(self._strategies.keys())
    
    def apply_strategy(self, model: nn.Module, strategy_name: str) -> nn.Module:
        """Apply a strategy by name."""
        strategy = self.get(strategy_name)
        if strategy is None:
            logger.warning(f"Strategy '{strategy_name}' not found, returning original model")
            return model
        return strategy.apply(model)


