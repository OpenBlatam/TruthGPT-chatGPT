"""
Performance metrics collection and analysis.

Provides utilities for collecting and analyzing performance metrics.
"""
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation_name: str
    total_calls: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    durations: List[float] = field(default_factory=list)
    errors: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def avg_duration(self) -> float:
        """Calculate average duration."""
        if self.total_calls == 0:
            return 0.0
        return self.total_duration / self.total_calls
    
    @property
    def throughput(self) -> float:
        """Calculate throughput (operations per second)."""
        if self.avg_duration == 0:
            return 0.0
        return 1.0 / self.avg_duration
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        total = self.total_calls + self.errors
        if total == 0:
            return 0.0
        return self.errors / total
    
    def record_call(self, duration: float, success: bool = True):
        """Record a function call."""
        self.total_calls += 1
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        self.durations.append(duration)
        
        if not success:
            self.errors += 1
    
    def get_percentile(self, percentile: float) -> float:
        """Get duration percentile."""
        if not self.durations:
            return 0.0
        
        sorted_durations = sorted(self.durations)
        index = int(len(sorted_durations) * percentile / 100)
        return sorted_durations[min(index, len(sorted_durations) - 1)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_name": self.operation_name,
            "total_calls": self.total_calls,
            "avg_duration": self.avg_duration,
            "min_duration": self.min_duration if self.min_duration != float('inf') else 0.0,
            "max_duration": self.max_duration,
            "throughput": self.throughput,
            "error_rate": self.error_rate,
            "p50_duration": self.get_percentile(50),
            "p95_duration": self.get_percentile(95),
            "p99_duration": self.get_percentile(99),
            "metadata": self.metadata,
        }


class MetricsCollector:
    """Collector for performance metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, PerformanceMetrics] = {}
    
    def get_or_create(self, operation_name: str) -> PerformanceMetrics:
        """Get or create metrics for an operation."""
        if operation_name not in self.metrics:
            self.metrics[operation_name] = PerformanceMetrics(operation_name=operation_name)
        return self.metrics[operation_name]
    
    def record(
        self,
        operation_name: str,
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric."""
        metrics = self.get_or_create(operation_name)
        metrics.record_call(duration, success)
        
        if metadata:
            metrics.metadata.update(metadata)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            operation: metrics.to_dict()
            for operation, metrics in self.metrics.items()
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()


def collect_metrics(
    operation_name: str,
    func: callable,
    *args,
    collector: Optional[MetricsCollector] = None,
    **kwargs
) -> Any:
    """
    Collect metrics for a function call.
    
    Args:
        operation_name: Name of operation
        func: Function to call
        *args: Positional arguments
        collector: Optional metrics collector
        **kwargs: Keyword arguments
    
    Returns:
        Function result
    """
    if collector is None:
        collector = MetricsCollector()
    
    start_time = time.perf_counter()
    success = True
    
    try:
        result = func(*args, **kwargs)
        return result
    except Exception as e:
        success = False
        logger.error(f"Error in {operation_name}: {e}", exc_info=True)
        raise
    finally:
        duration = time.perf_counter() - start_time
        collector.record(operation_name, duration, success)


def analyze_performance(
    metrics: PerformanceMetrics
) -> Dict[str, Any]:
    """
    Analyze performance metrics.
    
    Args:
        metrics: Performance metrics
    
    Returns:
        Analysis dictionary
    """
    analysis = {
        "summary": metrics.to_dict(),
        "recommendations": [],
    }
    
    # Performance recommendations
    if metrics.avg_duration > 1.0:
        analysis["recommendations"].append(
            "Average duration is high (>1s). Consider optimization."
        )
    
    if metrics.error_rate > 0.1:
        analysis["recommendations"].append(
            f"Error rate is high ({metrics.error_rate:.2%}). Check error handling."
        )
    
    if metrics.max_duration / metrics.avg_duration > 3.0:
        analysis["recommendations"].append(
            "High variance in duration. Consider investigating outliers."
        )
    
    return analysis













