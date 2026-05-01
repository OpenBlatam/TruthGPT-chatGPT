"""
Metrics Calculation Module
Calculates AI optimization metrics
"""

import torch.nn as nn
from typing import Dict
from enum import Enum
import logging

from .config import (
    DEFAULT_LEARNING_HISTORY_SIZE,
    DEFAULT_ACCURACY_PRESERVATION_HIGH,
    DEFAULT_ACCURACY_PRESERVATION_LOW,
    DEFAULT_MEMORY_REDUCTION_THRESHOLD,
    DEFAULT_SPEED_IMPROVEMENT_BASE,
    DEFAULT_SPEED_IMPROVEMENT_INTELLIGENT,
    DEFAULT_SPEED_IMPROVEMENT_GENIUS,
    DEFAULT_SPEED_IMPROVEMENT_SUPERINTELLIGENT,
    DEFAULT_SPEED_IMPROVEMENT_TRANSHUMAN,
    DEFAULT_SPEED_IMPROVEMENT_POSTHUMAN,
    DEFAULT_INTELLIGENCE_SCORE_NORMALIZATION,
    DEFAULT_NEURAL_ADAPTATION_MULTIPLIER,
    DEFAULT_COGNITIVE_ENHANCEMENT_MULTIPLIER,
    DEFAULT_METRIC_MAX_VALUE,
    DEFAULT_METRIC_DIVISOR
)
from .model_utils import calculate_memory_reduction

logger = logging.getLogger(__name__)


class AIOptimizationLevel(Enum):
    """AI optimization levels."""
    INTELLIGENT = "intelligent"
    GENIUS = "genius"
    SUPERINTELLIGENT = "superintelligent"
    TRANSHUMAN = "transhuman"
    POSTHUMAN = "posthuman"


class MetricsCalculator:
    """Calculates AI optimization metrics."""
    
    SPEED_IMPROVEMENTS = {
        AIOptimizationLevel.INTELLIGENT: DEFAULT_SPEED_IMPROVEMENT_INTELLIGENT,
        AIOptimizationLevel.GENIUS: DEFAULT_SPEED_IMPROVEMENT_GENIUS,
        AIOptimizationLevel.SUPERINTELLIGENT: DEFAULT_SPEED_IMPROVEMENT_SUPERINTELLIGENT,
        AIOptimizationLevel.TRANSHUMAN: DEFAULT_SPEED_IMPROVEMENT_TRANSHUMAN,
        AIOptimizationLevel.POSTHUMAN: DEFAULT_SPEED_IMPROVEMENT_POSTHUMAN
    }
    
    def __init__(self, optimization_level: AIOptimizationLevel):
        self.optimization_level = optimization_level
    
    def calculate(
        self,
        original_model: nn.Module,
        optimized_model: nn.Module,
        learning_history_size: int = 0
    ) -> Dict[str, float]:
        """Calculate AI optimization metrics."""
        memory_reduction = calculate_memory_reduction(original_model, optimized_model)
        
        speed_improvement = self.SPEED_IMPROVEMENTS.get(
            self.optimization_level, DEFAULT_SPEED_IMPROVEMENT_BASE
        )
        
        intelligence_score = min(
            DEFAULT_METRIC_MAX_VALUE,
            speed_improvement / DEFAULT_INTELLIGENCE_SCORE_NORMALIZATION
        )
        learning_efficiency = min(
            DEFAULT_METRIC_MAX_VALUE,
            learning_history_size / DEFAULT_LEARNING_HISTORY_SIZE
        )
        neural_adaptation = min(
            DEFAULT_METRIC_MAX_VALUE,
            memory_reduction * DEFAULT_NEURAL_ADAPTATION_MULTIPLIER
        )
        cognitive_enhancement = min(
            DEFAULT_METRIC_MAX_VALUE,
            intelligence_score * DEFAULT_COGNITIVE_ENHANCEMENT_MULTIPLIER
        )
        artificial_wisdom = min(
            DEFAULT_METRIC_MAX_VALUE,
            (intelligence_score + learning_efficiency) / DEFAULT_METRIC_DIVISOR
        )
        
        accuracy_preservation = (
            DEFAULT_ACCURACY_PRESERVATION_HIGH
            if memory_reduction < DEFAULT_MEMORY_REDUCTION_THRESHOLD
            else DEFAULT_ACCURACY_PRESERVATION_LOW
        )
        
        return {
            'speed_improvement': speed_improvement,
            'memory_reduction': memory_reduction,
            'accuracy_preservation': accuracy_preservation,
            'intelligence_score': intelligence_score,
            'learning_efficiency': learning_efficiency,
            'neural_adaptation': neural_adaptation,
            'cognitive_enhancement': cognitive_enhancement,
            'artificial_wisdom': artificial_wisdom,
            'parameter_reduction': memory_reduction,
            'compression_ratio': 1.0 - memory_reduction
        }


