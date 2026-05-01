"""
Result Builder Module
Constructs AIOptimizationResult objects
"""

import torch.nn as nn
from typing import Dict, Any

from .models import AIOptimizationResult
from .metrics_calculator import AIOptimizationLevel


class ResultBuilder:
    """Builds AIOptimizationResult objects."""
    
    @staticmethod
    def build(
        optimized_model: nn.Module,
        performance_metrics: Dict[str, float],
        optimization_time: float,
        optimization_level: AIOptimizationLevel,
        techniques_applied: list,
        ai_insights: Dict[str, Any]
    ) -> AIOptimizationResult:
        """Build AIOptimizationResult from components."""
        return AIOptimizationResult(
            optimized_model=optimized_model,
            speed_improvement=performance_metrics['speed_improvement'],
            memory_reduction=performance_metrics['memory_reduction'],
            accuracy_preservation=performance_metrics['accuracy_preservation'],
            intelligence_score=performance_metrics['intelligence_score'],
            learning_efficiency=performance_metrics['learning_efficiency'],
            optimization_time=optimization_time,
            level=optimization_level,
            techniques_applied=techniques_applied,
            performance_metrics=performance_metrics,
            ai_insights=ai_insights,
            neural_adaptation=performance_metrics.get('neural_adaptation', 0.0),
            cognitive_enhancement=performance_metrics.get('cognitive_enhancement', 0.0),
            artificial_wisdom=performance_metrics.get('artificial_wisdom', 0.0)
        )







