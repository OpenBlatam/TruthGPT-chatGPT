"""
Generic Optimizer System
=========================
Unified system for generic optimizers (speed, master, etc.) to eliminate duplication.
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


class GenericOptimizationLevel(Enum):
    """Generic optimization levels."""
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
    
    # Speed-specific levels
    ULTRA_FAST = "ultra_fast"
    ULTRA_SPEED = "ultra_speed"
    SUPER_SPEED = "super_speed"
    LIGHTNING_SPEED = "lightning_speed"
    HYPER_SPEED = "hyper_speed"
    EXTREME = "extreme"


@dataclass
class GenericOptimizationResult:
    """Result of generic optimization."""
    optimized_model: nn.Module
    speed_improvement: float
    memory_reduction: float
    accuracy_preservation: float
    energy_efficiency: float
    optimization_time: float
    level: GenericOptimizationLevel
    techniques_applied: List[str]
    performance_metrics: Dict[str, float]
    additional_benefits: Dict[str, float] = field(default_factory=dict)


class BaseGenericOptimizer(ABC):
    """Base class for generic optimizers."""
    
    def __init__(self, config: Dict[str, Any] = None, level: GenericOptimizationLevel = None):
        self.config = config or {}
        self.level = level or GenericOptimizationLevel.BASIC
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance tracking
        self.optimization_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        
        # Sub-optimizers (lazy loading)
        self._sub_optimizers = {}
        
    @abstractmethod
    def optimize(self, model: nn.Module, **kwargs) -> GenericOptimizationResult:
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


class UnifiedGenericOptimizer(BaseGenericOptimizer):
    """Unified generic optimizer that handles all optimization types."""
    
    def __init__(self, config: Dict[str, Any] = None, level: GenericOptimizationLevel = None, optimizer_type: str = "speed"):
        super().__init__(config, level)
        self.optimizer_type = optimizer_type
        
        from optimization_core.modules.optimizers.optimization_pipeline import build_optimization_pipeline
        self.pipeline = build_optimization_pipeline(self.level.value, config=self.config)
    
    def optimize(self, model: nn.Module, **kwargs) -> GenericOptimizationResult:
        """Apply optimization based on level and type."""
        start_time = time.perf_counter()
        
        self.logger.info(f"🚀 Generic {self.optimizer_type} optimization started (level: {self.level.value})")
        
        optimized_model, techniques_applied, failed_techniques = self.pipeline.apply(model)
        
        if failed_techniques:
            self.logger.warning(f"Some techniques failed: {failed_techniques}")
        
        optimization_time = (time.perf_counter() - start_time) * 1000
        self._last_applied_techniques = techniques_applied
        self._last_optimization_time = optimization_time
        performance_metrics = self._calculate_metrics(model, optimized_model)
        
        result = GenericOptimizationResult(
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
    


def create_generic_optimizer(level: str = "basic", optimizer_type: str = "speed", config: Dict[str, Any] = None):
    """
    Create a generic optimizer with the specified level and type.
    
    Args:
        level: Optimization level (basic, advanced, expert, ultra_speed, etc.)
        optimizer_type: Type of optimizer (speed, master, extreme, etc.)
        config: Optional configuration dictionary
    
    Returns:
        UnifiedGenericOptimizer instance
    """
    # Map string level to enum
    level_map = {
        'basic': GenericOptimizationLevel.BASIC,
        'advanced': GenericOptimizationLevel.ADVANCED,
        'expert': GenericOptimizationLevel.EXPERT,
        'master': GenericOptimizationLevel.MASTER,
        'legendary': GenericOptimizationLevel.LEGENDARY,
        'transcendent': GenericOptimizationLevel.TRANSCENDENT,
        'divine': GenericOptimizationLevel.DIVINE,
        'omnipotent': GenericOptimizationLevel.OMNIPOTENT,
        'infinite': GenericOptimizationLevel.INFINITE,
        'ultimate': GenericOptimizationLevel.ULTIMATE,
        'ultra_fast': GenericOptimizationLevel.ULTRA_FAST,
        'ultra_speed': GenericOptimizationLevel.ULTRA_SPEED,
        'super_speed': GenericOptimizationLevel.SUPER_SPEED,
        'lightning_speed': GenericOptimizationLevel.LIGHTNING_SPEED,
        'hyper_speed': GenericOptimizationLevel.HYPER_SPEED,
        'extreme': GenericOptimizationLevel.EXTREME,
    }
    
    opt_level = level_map.get(level.lower(), GenericOptimizationLevel.BASIC)
    return UnifiedGenericOptimizer(config=config or {}, level=opt_level, optimizer_type=optimizer_type)


