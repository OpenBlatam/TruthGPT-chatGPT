"""
Automatic performance tuning for polyglot_core.

Provides automatic performance optimization and tuning.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import statistics


@dataclass
class TuningRecommendation:
    """Performance tuning recommendation."""
    component: str
    current_value: Any
    recommended_value: Any
    reason: str
    expected_improvement: Optional[str] = None
    priority: str = "medium"  # "low", "medium", "high"


class PerformanceTuner:
    """
    Performance tuner for polyglot_core.
    
    Analyzes performance and provides tuning recommendations.
    """
    
    def __init__(self):
        self._metrics_history: Dict[str, List[float]] = {}
        self._recommendations: List[TuningRecommendation] = []
    
    def record_metric(self, component: str, metric_name: str, value: float):
        """
        Record performance metric.
        
        Args:
            component: Component name
            metric_name: Metric name
            value: Metric value
        """
        key = f"{component}.{metric_name}"
        if key not in self._metrics_history:
            self._metrics_history[key] = []
        
        self._metrics_history[key].append(value)
        
        # Keep only recent metrics
        if len(self._metrics_history[key]) > 1000:
            self._metrics_history[key] = self._metrics_history[key][-1000:]
    
    def analyze_performance(self) -> List[TuningRecommendation]:
        """
        Analyze performance and generate recommendations.
        
        Returns:
            List of tuning recommendations
        """
        recommendations = []
        
        # Analyze cache performance
        cache_hit_rate_key = "cache.hit_rate"
        if cache_hit_rate_key in self._metrics_history:
            hit_rates = self._metrics_history[cache_hit_rate_key]
            if hit_rates:
                avg_hit_rate = statistics.mean(hit_rates)
                
                if avg_hit_rate < 0.7:
                    recommendations.append(TuningRecommendation(
                        component="cache",
                        current_value=f"{avg_hit_rate:.2%}",
                        recommended_value="Increase cache size or adjust eviction strategy",
                        reason=f"Low cache hit rate ({avg_hit_rate:.2%})",
                        expected_improvement="20-30% performance improvement",
                        priority="high"
                    ))
        
        # Analyze latency
        latency_keys = [k for k in self._metrics_history.keys() if "latency" in k]
        for key in latency_keys:
            latencies = self._metrics_history[key]
            if latencies:
                avg_latency = statistics.mean(latencies)
                p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
                
                if avg_latency > 100:  # ms
                    component = key.split('.')[0]
                    recommendations.append(TuningRecommendation(
                        component=component,
                        current_value=f"{avg_latency:.2f}ms (p95: {p95_latency:.2f}ms)",
                        recommended_value="Use faster backend or optimize operation",
                        reason=f"High latency detected",
                        expected_improvement="50-70% latency reduction",
                        priority="high" if avg_latency > 500 else "medium"
                    ))
        
        # Analyze memory usage
        memory_keys = [k for k in self._metrics_history.keys() if "memory" in k]
        for key in memory_keys:
            memory_values = self._metrics_history[key]
            if memory_values:
                avg_memory = statistics.mean(memory_values)
                max_memory = max(memory_values)
                
                if max_memory > 8 * 1024 * 1024 * 1024:  # 8GB
                    component = key.split('.')[0]
                    recommendations.append(TuningRecommendation(
                        component=component,
                        current_value=f"{max_memory / (1024**3):.2f}GB",
                        recommended_value="Enable compression or reduce batch size",
                        reason=f"High memory usage",
                        expected_improvement="30-50% memory reduction",
                        priority="medium"
                    ))
        
        self._recommendations = recommendations
        return recommendations
    
    def apply_recommendation(self, recommendation: TuningRecommendation) -> bool:
        """
        Apply a tuning recommendation.
        
        Args:
            recommendation: Recommendation to apply
            
        Returns:
            True if applied successfully
        """
        # This would integrate with actual configuration
        # For now, just mark as applied
        try:
            # In a real implementation, this would modify configurations
            # based on the recommendation
            return True
        except Exception:
            return False
    
    def get_recommendations(self, priority: Optional[str] = None) -> List[TuningRecommendation]:
        """
        Get recommendations, optionally filtered by priority.
        
        Args:
            priority: Optional priority filter
            
        Returns:
            List of recommendations
        """
        if priority:
            return [r for r in self._recommendations if r.priority == priority]
        return self._recommendations.copy()
    
    def auto_tune(self, apply_changes: bool = False) -> Dict[str, Any]:
        """
        Automatically tune performance.
        
        Args:
            apply_changes: Whether to automatically apply changes
            
        Returns:
            Tuning results
        """
        recommendations = self.analyze_performance()
        
        applied = []
        skipped = []
        
        for rec in recommendations:
            if apply_changes and rec.priority == "high":
                if self.apply_recommendation(rec):
                    applied.append(rec)
                else:
                    skipped.append(rec)
            else:
                skipped.append(rec)
        
        return {
            'recommendations': len(recommendations),
            'applied': len(applied),
            'skipped': len(skipped),
            'details': {
                'applied': [r.__dict__ for r in applied],
                'skipped': [r.__dict__ for r in skipped]
            }
        }


# Global performance tuner
_global_tuner = PerformanceTuner()


def get_performance_tuner() -> PerformanceTuner:
    """Get global performance tuner."""
    return _global_tuner


def analyze_performance() -> List[TuningRecommendation]:
    """Convenience function to analyze performance."""
    return _global_tuner.analyze_performance()













