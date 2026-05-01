"""
Statistics Calculator Module — Enhanced with percentile distributions,
trend analysis, and Pydantic computed_field derived metrics.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict, computed_field

from .models import AIOptimizationResult
from .learning_mechanism import LearningMechanism
from .metrics_calculator import AIOptimizationLevel


class MetricDistribution(BaseModel):
    """Statistical distribution of a single metric."""
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0

    @classmethod
    def from_values(cls, values: List[float]) -> "MetricDistribution":
        if not values:
            return cls()
        arr = np.array(values)
        return cls(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            p25=float(np.percentile(arr, 25)),
            p75=float(np.percentile(arr, 75)),
            p95=float(np.percentile(arr, 95)),
        )


class TrendAnalysis(BaseModel):
    """Linear trend analysis of a metric over time."""
    slope: float = 0.0
    is_improving: bool = False
    improvement_rate_pct: float = 0.0


class OptimizationStatistics(BaseModel):
    """Pydantic model representing calculated optimization statistics with rich analytics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_optimizations: int
    # Core averages
    avg_speed_improvement: float
    max_speed_improvement: float
    avg_memory_reduction: float
    avg_intelligence_score: float
    avg_learning_efficiency: float
    avg_neural_adaptation: float
    avg_cognitive_enhancement: float
    avg_artificial_wisdom: float
    # Meta
    optimization_level: int
    learning_history_length: int
    experience_buffer_size: int
    exploration_rate: float
    # Enhanced analytics
    speed_distribution: MetricDistribution = Field(default_factory=MetricDistribution)
    intelligence_distribution: MetricDistribution = Field(default_factory=MetricDistribution)
    speed_trend: TrendAnalysis = Field(default_factory=TrendAnalysis)

    @computed_field  # type: ignore[misc]
    @property
    def overall_health_score(self) -> float:
        """Composite 0-100 score combining key metrics."""
        if self.total_optimizations == 0:
            return 0.0
        weights = {
            "speed": 0.25,
            "memory": 0.15,
            "intelligence": 0.20,
            "learning": 0.15,
            "adaptation": 0.10,
            "cognition": 0.10,
            "wisdom": 0.05,
        }
        raw = (
            self.avg_speed_improvement * weights["speed"]
            + self.avg_memory_reduction * weights["memory"]
            + self.avg_intelligence_score * weights["intelligence"]
            + self.avg_learning_efficiency * weights["learning"]
            + self.avg_neural_adaptation * weights["adaptation"]
            + self.avg_cognitive_enhancement * weights["cognition"]
            + self.avg_artificial_wisdom * weights["wisdom"]
        )
        return round(min(100.0, max(0.0, raw)), 2)


class StatisticsCalculator:
    """Calculates statistics from optimization results with percentile and trend support."""

    @staticmethod
    def _compute_trend(values: List[float]) -> TrendAnalysis:
        """Compute linear trend (slope) across sequential optimization results."""
        if len(values) < 2:
            return TrendAnalysis()
        x = np.arange(len(values), dtype=float)
        y = np.array(values, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        first_val = values[0] if values[0] != 0 else 1e-10
        rate = (slope / abs(first_val)) * 100
        return TrendAnalysis(
            slope=round(slope, 6),
            is_improving=slope > 0,
            improvement_rate_pct=round(rate, 2),
        )

    @staticmethod
    def calculate(
        optimization_history: List[AIOptimizationResult],
        optimization_level: AIOptimizationLevel,
        learning_mechanism: LearningMechanism,
    ) -> OptimizationStatistics:
        """Calculate statistics from optimization history and return a validated model."""
        if not optimization_history:
            return OptimizationStatistics(
                total_optimizations=0,
                avg_speed_improvement=0.0,
                max_speed_improvement=0.0,
                avg_memory_reduction=0.0,
                avg_intelligence_score=0.0,
                avg_learning_efficiency=0.0,
                avg_neural_adaptation=0.0,
                avg_cognitive_enhancement=0.0,
                avg_artificial_wisdom=0.0,
                optimization_level=optimization_level.value,
                learning_history_length=0,
                experience_buffer_size=0,
                exploration_rate=0.0,
            )

        results = list(optimization_history)

        metrics = {
            "speed_improvement": [r.speed_improvement for r in results],
            "memory_reduction": [r.memory_reduction for r in results],
            "intelligence_score": [r.intelligence_score for r in results],
            "learning_efficiency": [r.learning_efficiency for r in results],
            "neural_adaptation": [r.neural_adaptation for r in results],
            "cognitive_enhancement": [r.cognitive_enhancement for r in results],
            "artificial_wisdom": [r.artificial_wisdom for r in results],
        }

        return OptimizationStatistics(
            total_optimizations=len(results),
            avg_speed_improvement=float(np.mean(metrics["speed_improvement"])),
            max_speed_improvement=float(max(metrics["speed_improvement"])),
            avg_memory_reduction=float(np.mean(metrics["memory_reduction"])),
            avg_intelligence_score=float(np.mean(metrics["intelligence_score"])),
            avg_learning_efficiency=float(np.mean(metrics["learning_efficiency"])),
            avg_neural_adaptation=float(np.mean(metrics["neural_adaptation"])),
            avg_cognitive_enhancement=float(np.mean(metrics["cognitive_enhancement"])),
            avg_artificial_wisdom=float(np.mean(metrics["artificial_wisdom"])),
            optimization_level=int(optimization_level.value),
            learning_history_length=len(learning_mechanism.get_learning_history()),
            experience_buffer_size=len(learning_mechanism.get_experience_buffer()),
            exploration_rate=float(learning_mechanism.get_exploration_rate()),
            # Enhanced analytics
            speed_distribution=MetricDistribution.from_values(metrics["speed_improvement"]),
            intelligence_distribution=MetricDistribution.from_values(metrics["intelligence_score"]),
            speed_trend=StatisticsCalculator._compute_trend(metrics["speed_improvement"]),
        )


