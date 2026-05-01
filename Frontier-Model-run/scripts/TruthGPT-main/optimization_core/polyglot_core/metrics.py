"""
Metrics and monitoring for polyglot_core.

Provides comprehensive metrics collection, aggregation, and reporting.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime
import json
import time


@dataclass
class Metric:
    """Single metric value."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    name: str
    count: int
    sum: float
    min: float
    max: float
    avg: float
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """
    Collects and aggregates metrics from polyglot_core operations.
    
    Example:
        >>> collector = MetricsCollector()
        >>> collector.record("cache_hit", 1.0, tags={"backend": "rust"})
        >>> collector.record("cache_miss", 0.0, tags={"backend": "rust"})
        >>> summary = collector.get_summary("cache_hit")
        >>> print(f"Hit rate: {summary.avg:.2%}")
    """
    
    def __init__(self):
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._lock = None
        try:
            import threading
            self._lock = threading.Lock()
        except ImportError:
            pass
    
    def _safe_append(self, name: str, metric: Metric):
        """Thread-safe append."""
        if self._lock:
            with self._lock:
                self._metrics[name].append(metric)
        else:
            self._metrics[name].append(metric)
    
    def record(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        unit: str = ""
    ):
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
            unit: Unit of measurement
        """
        metric = Metric(
            name=name,
            value=value,
            tags=tags or {},
            unit=unit
        )
        self._safe_append(name, metric)
    
    def record_latency(
        self,
        operation: str,
        duration_ms: float,
        backend: str = "",
        success: bool = True
    ):
        """
        Record operation latency.
        
        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            backend: Backend name
            success: Whether operation succeeded
        """
        tags = {"backend": backend, "success": str(success)}
        self.record(f"{operation}_latency", duration_ms, tags, "ms")
        
        if success:
            self.record(f"{operation}_success", 1.0, tags)
        else:
            self.record(f"{operation}_failure", 1.0, tags)
    
    def record_throughput(
        self,
        operation: str,
        items_per_second: float,
        backend: str = ""
    ):
        """
        Record operation throughput.
        
        Args:
            operation: Operation name
            items_per_second: Throughput
            backend: Backend name
        """
        tags = {"backend": backend}
        self.record(f"{operation}_throughput", items_per_second, tags, "ops/s")
    
    def get_metrics(self, name: str) -> List[Metric]:
        """Get all metrics for a name."""
        if self._lock:
            with self._lock:
                return self._metrics.get(name, []).copy()
        return self._metrics.get(name, []).copy()
    
    def get_summary(self, name: str) -> Optional[MetricSummary]:
        """
        Get summary statistics for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            MetricSummary or None
        """
        metrics = self.get_metrics(name)
        if not metrics:
            return None
        
        values = [m.value for m in metrics]
        values.sort()
        
        count = len(values)
        total = sum(values)
        min_val = min(values)
        max_val = max(values)
        avg = total / count if count > 0 else 0.0
        
        # Percentiles
        p50_idx = int(count * 0.5)
        p95_idx = int(count * 0.95)
        p99_idx = int(count * 0.99)
        
        p50 = values[p50_idx] if p50_idx < count else avg
        p95 = values[p95_idx] if p95_idx < count else max_val
        p99 = values[p99_idx] if p99_idx < count else max_val
        
        return MetricSummary(
            name=name,
            count=count,
            sum=total,
            min=min_val,
            max=max_val,
            avg=avg,
            p50=p50,
            p95=p95,
            p99=p99
        )
    
    def get_all_summaries(self) -> Dict[str, MetricSummary]:
        """Get summaries for all metrics."""
        summaries = {}
        metric_names = list(self._metrics.keys())
        
        if self._lock:
            with self._lock:
                metric_names = list(self._metrics.keys())
        
        for name in metric_names:
            summary = self.get_summary(name)
            if summary:
                summaries[name] = summary
        
        return summaries
    
    def reset(self):
        """Reset all metrics."""
        if self._lock:
            with self._lock:
                self._metrics.clear()
        else:
            self._metrics.clear()
    
    def export_json(self) -> str:
        """Export all metrics as JSON."""
        summaries = self.get_all_summaries()
        data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {name: summary.to_dict() for name, summary in summaries.items()}
        }
        return json.dumps(data, indent=2)
    
    def print_summary(self):
        """Print formatted summary."""
        summaries = self.get_all_summaries()
        
        if not summaries:
            print("No metrics recorded.")
            return
        
        print("\n" + "=" * 100)
        print("Metrics Summary")
        print("=" * 100)
        print(f"{'Metric':<40} {'Count':<10} {'Avg':<12} {'P50':<12} {'P95':<12} {'P99':<12}")
        print("-" * 100)
        
        for name, summary in sorted(summaries.items()):
            print(f"{name:<40} {summary.count:<10} {summary.avg:<12.2f} "
                  f"{summary.p50:<12.2f} {summary.p95:<12.2f} {summary.p99:<12.2f}")
        
        print("=" * 100)


# Global metrics collector
_global_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    return _global_collector


def record_metric(name: str, value: float, **kwargs):
    """Convenience function to record metric."""
    _global_collector.record(name, value, **kwargs)

