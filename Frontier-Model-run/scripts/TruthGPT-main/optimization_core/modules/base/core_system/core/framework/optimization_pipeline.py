"""
Optimization Pipeline Module
Orchestrates the complete optimization workflow
"""

import torch.nn as nn
import time
import logging
from typing import Tuple

from .model_features import ModelFeatureExtractor
from .strategy_selector import StrategySelector
from .optimization_strategies import StrategyRegistry
from .learning_mechanism import LearningMechanism
from .metrics_calculator import MetricsCalculator
from .insights_generator import InsightsGenerator
from .result_builder import ResultBuilder
from .models import AIOptimizationResult
from .metrics_calculator import AIOptimizationLevel
from .model_utils import calculate_memory_reduction
from .error_handler import StrategyErrorHandler
from .config import MILLISECONDS_PER_SECOND

logger = logging.getLogger(__name__)


class OptimizationPipeline:
    """Orchestrates the complete optimization workflow."""
    
    def __init__(
        self,
        feature_extractor: ModelFeatureExtractor,
        strategy_selector: StrategySelector,
        strategy_registry: StrategyRegistry,
        learning_mechanism: LearningMechanism,
        metrics_calculator: MetricsCalculator,
        insights_generator: InsightsGenerator,
        optimization_level: AIOptimizationLevel
    ):
        self.feature_extractor = feature_extractor
        self.strategy_selector = strategy_selector
        self.strategy_registry = strategy_registry
        self.learning_mechanism = learning_mechanism
        self.metrics_calculator = metrics_calculator
        self.insights_generator = insights_generator
        self.optimization_level = optimization_level
    
    def execute(
        self,
        model: nn.Module,
        start_time: float
    ) -> AIOptimizationResult:
        """Execute the complete optimization pipeline."""
        model_features = self.feature_extractor.extract(model)
        strategy, confidence = self.strategy_selector.select_strategy(model_features)
        optimized_model, techniques_applied = self._apply_strategy(model, strategy)
        
        memory_reduction = calculate_memory_reduction(model, optimized_model)
        self.learning_mechanism.record_experience(strategy, confidence, memory_reduction)
        
        optimization_time = (time.perf_counter() - start_time) * MILLISECONDS_PER_SECOND
        performance_metrics = self.metrics_calculator.calculate(
            model, optimized_model, len(self.learning_mechanism.get_learning_history())
        )
        ai_insights = self.insights_generator.generate_insights(model, optimized_model)
        
        return ResultBuilder.build(
            optimized_model=optimized_model,
            performance_metrics=performance_metrics,
            optimization_time=optimization_time,
            optimization_level=self.optimization_level,
            techniques_applied=techniques_applied,
            ai_insights=ai_insights
        )
    
    def _apply_strategy(
        self,
        model: nn.Module,
        strategy: str
    ) -> Tuple[nn.Module, list]:
        """Apply optimization strategy with error handling."""
        strategy_func = lambda m: self.strategy_registry.apply_strategy(m, strategy)
        return StrategyErrorHandler.apply_with_fallback(
            strategy_func, 'quantization', strategy, model, self.strategy_registry
        )


