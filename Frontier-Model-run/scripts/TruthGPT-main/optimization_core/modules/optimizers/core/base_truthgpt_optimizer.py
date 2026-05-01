"""
Unified TruthGPT Optimizer Base Classes
========================================
Base classes for all TruthGPT optimizers to eliminate code duplication.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from collections import defaultdict, deque
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Unified optimization levels for TruthGPT."""
    BASIC = "basic"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"
    LEGENDARY = "legendary"
    TRANSCENDENT = "transcendent"
    DIVINE = "divine"
    OMNIPOTENT = "omnipotent"
    INFINITE = "infinite"
    ULTIMATE = "ultimate"
    SUPREME = "supreme"
    ENTERPRISE = "enterprise"
    ULTRA_FAST = "ultra_fast"
    ULTRA_SPEED = "ultra_speed"
    HYPER_SPEED = "hyper_speed"
    LIGHTNING_SPEED = "lightning_speed"
    LIBRARY = "library"


@dataclass
class OptimizationResult:
    """Result of optimization."""
    optimized_model: nn.Module
    speed_improvement: float
    memory_reduction: float
    accuracy_preservation: float
    energy_efficiency: float
    optimization_time: float
    level: OptimizationLevel
    techniques_applied: List[str]
    performance_metrics: Dict[str, float]
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)


class BaseTruthGPTOptimizer(ABC):
    """Base class for all TruthGPT optimizers."""
    
    def __init__(self, config: Dict[str, Any] = None, level: OptimizationLevel = None):
        self.config = config or {}
        self.level = level or OptimizationLevel.BASIC
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance tracking
        self.optimization_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        
        # Initialize sub-optimizers (lazy loading)
        self._sub_optimizers = {}
        
    @abstractmethod
    def optimize(self, model: nn.Module, **kwargs) -> OptimizationResult:
        """Apply optimization to model."""
        pass
    
    def _calculate_metrics(self, original_model: nn.Module, optimized_model: nn.Module) -> Dict[str, float]:
        """Calculate performance metrics."""
        from optimization_core.modules.optimizers.metrics import calculate_optimization_metrics
        
        techniques = getattr(self, '_last_applied_techniques', [])
        opt_time = getattr(self, '_last_optimization_time', 0.0)
        
        return calculate_optimization_metrics(
            original_model, optimized_model, techniques, opt_time
        )
    
    def _get_sub_optimizer(self, name: str):
        """Get or create a sub-optimizer (lazy loading)."""
        if name not in self._sub_optimizers:
            # Lazy import to avoid circular dependencies
            from optimization_core.modules.optimizers.component_optimizers import get_component_optimizer
            self._sub_optimizers[name] = get_component_optimizer(name, self.config.get(name, {}))
        return self._sub_optimizers[name]


class UnifiedTruthGPTOptimizer(BaseTruthGPTOptimizer):
    """Unified TruthGPT optimizer that consolidates all optimization levels."""
    
    def __init__(self, config: Dict[str, Any] = None, level: OptimizationLevel = None):
        super().__init__(config, level)
        
        from .optimization_pipeline import build_optimization_pipeline
        self.pipeline = build_optimization_pipeline(self.level.value, config=self.config)
    
    def optimize(self, model: nn.Module, **kwargs) -> OptimizationResult:
        """Apply optimization based on level."""
        start_time = time.perf_counter()
        
        self.logger.info(f"🚀 TruthGPT optimization started (level: {self.level.value})")
        
        optimized_model, techniques_applied, failed_techniques = self.pipeline.apply(model)
        
        if failed_techniques:
            self.logger.warning(f"Some techniques failed: {failed_techniques}")
        
        optimization_time = (time.perf_counter() - start_time) * 1000
        self._last_applied_techniques = techniques_applied
        self._last_optimization_time = optimization_time
        performance_metrics = self._calculate_metrics(model, optimized_model)
        
        result = OptimizationResult(
            optimized_model=optimized_model,
            speed_improvement=performance_metrics['speed_improvement'],
            memory_reduction=performance_metrics['memory_reduction'],
            accuracy_preservation=performance_metrics['accuracy_preservation'],
            energy_efficiency=performance_metrics['energy_efficiency'],
            optimization_time=optimization_time,
            level=self.level,
            techniques_applied=techniques_applied,
            performance_metrics=performance_metrics
        )
        
        self.optimization_history.append(result)
        self.logger.info(f"✅ Optimization completed: {len(techniques_applied)} techniques applied")
        
        return result
    


