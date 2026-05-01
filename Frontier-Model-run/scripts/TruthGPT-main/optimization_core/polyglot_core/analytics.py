"""
Advanced analytics for polyglot_core.

Provides analytics, insights, and data analysis capabilities.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


@dataclass
class AnalyticsInsight:
    """Analytics insight."""
    type: str
    title: str
    description: str
    severity: str  # "info", "warning", "critical"
    recommendation: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


class AnalyticsEngine:
    """
    Analytics engine for polyglot_core.
    
    Analyzes metrics and generates insights.
    """
    
    def __init__(self):
        self._data_points: Dict[str, List[float]] = defaultdict(list)
        self._max_data_points: int = 10000
    
    def record_data_point(self, metric_name: str, value: float):
        """
        Record data point.
        
        Args:
            metric_name: Metric name
            value: Metric value
        """
        self._data_points[metric_name].append(value)
        
        # Keep only recent data points
        if len(self._data_points[metric_name]) > self._max_data_points:
            self._data_points[metric_name] = self._data_points[metric_name][-self._max_data_points:]
    
    def analyze_metric(self, metric_name: str) -> Dict[str, Any]:
        """
        Analyze a metric.
        
        Args:
            metric_name: Metric name
            
        Returns:
            Analysis results
        """
        if metric_name not in self._data_points or not self._data_points[metric_name]:
            return {}
        
        values = self._data_points[metric_name]
        sorted_values = sorted(values)
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'p25': sorted_values[len(sorted_values) // 4] if sorted_values else 0,
            'p75': sorted_values[len(sorted_values) * 3 // 4] if sorted_values else 0,
            'p95': sorted_values[int(len(sorted_values) * 0.95)] if sorted_values else 0,
            'p99': sorted_values[int(len(sorted_values) * 0.99)] if sorted_values else 0
        }
    
    def generate_insights(self) -> List[AnalyticsInsight]:
        """
        Generate insights from data.
        
        Returns:
            List of insights
        """
        insights = []
        
        for metric_name, values in self._data_points.items():
            if not values:
                continue
            
            analysis = self.analyze_metric(metric_name)
            
            # Detect anomalies
            mean = analysis['mean']
            stdev = analysis['stdev']
            
            if stdev > 0:
                recent_values = values[-100:] if len(values) > 100 else values
                recent_mean = statistics.mean(recent_values)
                
                # Check for significant deviation
                if abs(recent_mean - mean) > 2 * stdev:
                    insights.append(AnalyticsInsight(
                        type="anomaly",
                        title=f"Anomaly detected in {metric_name}",
                        description=f"Recent values ({recent_mean:.2f}) deviate significantly from mean ({mean:.2f})",
                        severity="warning",
                        recommendation="Investigate recent changes",
                        data={'metric': metric_name, 'recent_mean': recent_mean, 'overall_mean': mean}
                    ))
            
            # Check for trends
            if len(values) >= 20:
                first_half = values[:len(values)//2]
                second_half = values[len(values)//2:]
                
                first_mean = statistics.mean(first_half)
                second_mean = statistics.mean(second_half)
                
                change_pct = ((second_mean - first_mean) / first_mean * 100) if first_mean > 0 else 0
                
                if abs(change_pct) > 10:
                    trend = "increasing" if change_pct > 0 else "decreasing"
                    insights.append(AnalyticsInsight(
                        type="trend",
                        title=f"Trend detected in {metric_name}",
                        description=f"Metric is {trend} by {abs(change_pct):.1f}%",
                        severity="info",
                        data={'metric': metric_name, 'change_pct': change_pct, 'trend': trend}
                    ))
        
        return insights
    
    def compare_periods(
        self,
        metric_name: str,
        period1: List[float],
        period2: List[float]
    ) -> Dict[str, Any]:
        """
        Compare two periods.
        
        Args:
            metric_name: Metric name
            period1: First period values
            period2: Second period values
            
        Returns:
            Comparison results
        """
        if not period1 or not period2:
            return {}
        
        mean1 = statistics.mean(period1)
        mean2 = statistics.mean(period2)
        
        change = mean2 - mean1
        change_pct = (change / mean1 * 100) if mean1 > 0 else 0
        
        return {
            'metric': metric_name,
            'period1_mean': mean1,
            'period2_mean': mean2,
            'change': change,
            'change_pct': change_pct,
            'improvement': change_pct < 0 if 'latency' in metric_name or 'error' in metric_name else change_pct > 0
        }


# Global analytics engine
_global_analytics = AnalyticsEngine()


def get_analytics() -> AnalyticsEngine:
    """Get global analytics engine."""
    return _global_analytics


def record_data_point(metric_name: str, value: float):
    """Convenience function to record data point."""
    _global_analytics.record_data_point(metric_name, value)













