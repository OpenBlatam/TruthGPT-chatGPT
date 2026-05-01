"""
Advanced metrics utilities for optimization_core.

Provides advanced metrics collection and analysis.
"""
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from pydantic import BaseModel, Field
from collections import defaultdict, deque
from statistics import mean, median, stdev

logger = logging.getLogger(__name__)


class MetricValue(BaseModel):
    """Metric value with timestamp."""
    value: float
    timestamp: float = Field(default_factory=time.time)
    tags: Dict[str, str] = Field(default_factory=dict)


class MetricStats(BaseModel):
    """Statistics for a metric."""
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    stdev: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None


class AdvancedMetricsCollector:
    """Advanced metrics collector."""
    
    def __init__(self, max_samples: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_samples: Maximum samples per metric
        """
        self.max_samples = max_samples
        self.metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
    
    def record(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        metric_value = MetricValue(
            value=value,
            timestamp=time.time(),
            tags=tags or {}
        )
        self.metrics[name].append(metric_value)
    
    def increment(
        self,
        name: str,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Increment a counter.
        
        Args:
            name: Counter name
            value: Increment value
            tags: Optional tags
        """
        self.counters[name] += value
    
    def set_gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Set a gauge value.
        
        Args:
            name: Gauge name
            value: Gauge value
            tags: Optional tags
        """
        self.gauges[name] = value
    
    def get_stats(
        self,
        name: str,
        window_seconds: Optional[float] = None
    ) -> Optional[MetricStats]:
        """
        Get statistics for a metric.
        
        Args:
            name: Metric name
            window_seconds: Optional time window
        
        Returns:
            Metric statistics
        """
        if name not in self.metrics:
            return None
        
        values = list(self.metrics[name])
        
        # Filter by time window if specified
        if window_seconds:
            cutoff = time.time() - window_seconds
            values = [v for v in values if v.timestamp >= cutoff]
        
        if not values:
            return None
        
        numeric_values = [v.value for v in values]
        sorted_values = sorted(numeric_values)
        
        return MetricStats(
            count=len(numeric_values),
            sum=sum(numeric_values),
            min=min(numeric_values),
            max=max(numeric_values),
            mean=mean(numeric_values),
            median=median(numeric_values),
            stdev=stdev(numeric_values) if len(numeric_values) > 1 else None,
            p50=sorted_values[int(len(sorted_values) * 0.50)] if sorted_values else None,
            p95=sorted_values[int(len(sorted_values) * 0.95)] if sorted_values else None,
            p99=sorted_values[int(len(sorted_values) * 0.99)] if sorted_values else None,
        )
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics.
        
        Returns:
            Dictionary of all metrics
        """
        return {
            "metrics": {
                name: {
                    "count": len(values),
                    "latest": values[-1].value if values else None,
                }
                for name, values in self.metrics.items()
            },
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
        }
    
    def clear(self, name: Optional[str] = None):
        """
        Clear metrics.
        
        Args:
            name: Optional metric name (clears all if None)
        """
        if name:
            if name in self.metrics:
                self.metrics[name].clear()
            if name in self.counters:
                del self.counters[name]
            if name in self.gauges:
                del self.gauges[name]
        else:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()


def create_metrics_collector(max_samples: int = 10000) -> AdvancedMetricsCollector:
    """
    Create an advanced metrics collector.
    
    Args:
        max_samples: Maximum samples per metric
    
    Returns:
        Metrics collector
    """
    return AdvancedMetricsCollector(max_samples)













