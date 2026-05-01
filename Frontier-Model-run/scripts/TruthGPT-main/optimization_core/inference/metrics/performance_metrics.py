"""
Performance Metrics for Inference Engines
==========================================

Advanced metrics collection and analysis for inference engines.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from enum import Enum
import statistics

from ..exceptions import InferenceEngineError


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


@dataclass
class MetricSnapshot:
    """Snapshot of metric values at a point in time."""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HistogramStats:
    """Statistics for histogram metrics."""
    count: int
    min: float
    max: float
    mean: float
    median: float
    p50: float
    p90: float
    p95: float
    p99: float
    std_dev: float


class PerformanceMetrics:
    """
    Advanced performance metrics collector for inference engines.
    
    Features:
    - Multiple metric types (counter, gauge, histogram, timer, rate)
    - Thread-safe operations
    - Automatic aggregation
    - Percentile calculations
    - Rate calculations
    - Metric snapshots
    """
    
    def __init__(self, max_samples: int = 10000):
        """
        Initialize performance metrics collector.
        
        Args:
            max_samples: Maximum number of samples to keep in memory
        """
        self.max_samples = max_samples
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._start_time = time.time()
    
    def counter(self, name: str) -> "CounterMetric":
        """Get or create a counter metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = CounterMetric(name)
            return self._metrics[name]
    
    def gauge(self, name: str) -> "GaugeMetric":
        """Get or create a gauge metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = GaugeMetric(name)
            return self._metrics[name]
    
    def histogram(self, name: str) -> "HistogramMetric":
        """Get or create a histogram metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = HistogramMetric(name, self.max_samples)
            return self._metrics[name]
    
    def timer(self, name: str) -> "TimerMetric":
        """Get or create a timer metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = TimerMetric(name, self.max_samples)
            return self._metrics[name]
    
    def rate(self, name: str) -> "RateMetric":
        """Get or create a rate metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = RateMetric(name)
            return self._metrics[name]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary."""
        with self._lock:
            return {
                name: metric.to_dict()
                for name, metric in self._metrics.items()
            }
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._start_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            uptime = time.time() - self._start_time
            return {
                "uptime_seconds": uptime,
                "metric_count": len(self._metrics),
                "metrics": self.get_all_metrics()
            }


class CounterMetric:
    """Counter metric that only increases."""
    
    def __init__(self, name: str):
        self.name = name
        self._value = 0
        self._lock = threading.Lock()
    
    def inc(self, amount: float = 1.0):
        """Increment counter."""
        with self._lock:
            self._value += amount
    
    def get(self) -> float:
        """Get current value."""
        with self._lock:
            return self._value
    
    def reset(self):
        """Reset counter."""
        with self._lock:
            self._value = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": MetricType.COUNTER.value,
            "value": self.get()
        }


class GaugeMetric:
    """Gauge metric that can go up and down."""
    
    def __init__(self, name: str):
        self.name = name
        self._value = 0.0
        self._lock = threading.Lock()
    
    def set(self, value: float):
        """Set gauge value."""
        with self._lock:
            self._value = value
    
    def inc(self, amount: float = 1.0):
        """Increment gauge."""
        with self._lock:
            self._value += amount
    
    def dec(self, amount: float = 1.0):
        """Decrement gauge."""
        with self._lock:
            self._value -= amount
    
    def get(self) -> float:
        """Get current value."""
        with self._lock:
            return self._value
    
    def reset(self):
        """Reset gauge."""
        with self._lock:
            self._value = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": MetricType.GAUGE.value,
            "value": self.get()
        }


class HistogramMetric:
    """Histogram metric for distribution analysis."""
    
    def __init__(self, name: str, max_samples: int = 10000):
        self.name = name
        self.max_samples = max_samples
        self._samples = deque(maxlen=max_samples)
        self._lock = threading.Lock()
    
    def observe(self, value: float):
        """Record a value."""
        with self._lock:
            self._samples.append(value)
    
    def get_stats(self) -> HistogramStats:
        """Get histogram statistics."""
        with self._lock:
            if not self._samples:
                return HistogramStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            samples = list(self._samples)
            sorted_samples = sorted(samples)
            count = len(samples)
            
            return HistogramStats(
                count=count,
                min=min(samples),
                max=max(samples),
                mean=statistics.mean(samples),
                median=statistics.median(samples),
                p50=sorted_samples[int(count * 0.50)],
                p90=sorted_samples[int(count * 0.90)],
                p95=sorted_samples[int(count * 0.95)],
                p99=sorted_samples[int(count * 0.99)],
                std_dev=statistics.stdev(samples) if count > 1 else 0.0
            )
    
    def reset(self):
        """Reset histogram."""
        with self._lock:
            self._samples.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        stats = self.get_stats()
        return {
            "type": MetricType.HISTOGRAM.value,
            "count": stats.count,
            "min": stats.min,
            "max": stats.max,
            "mean": stats.mean,
            "median": stats.median,
            "p50": stats.p50,
            "p90": stats.p90,
            "p95": stats.p95,
            "p99": stats.p99,
            "std_dev": stats.std_dev
        }


class TimerMetric:
    """Timer metric for measuring durations."""
    
    def __init__(self, name: str, max_samples: int = 10000):
        self.name = name
        self.histogram = HistogramMetric(name, max_samples)
    
    def time(self) -> "TimerContext":
        """Get context manager for timing."""
        return TimerContext(self)
    
    def record(self, duration: float):
        """Record a duration."""
        self.histogram.observe(duration)
    
    def get_stats(self) -> HistogramStats:
        """Get timer statistics."""
        return self.histogram.get_stats()
    
    def reset(self):
        """Reset timer."""
        self.histogram.reset()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": MetricType.TIMER.value,
            **self.histogram.to_dict()
        }


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, timer: TimerMetric):
        self.timer = timer
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.timer.record(duration)


class RateMetric:
    """Rate metric for measuring events per second."""
    
    def __init__(self, name: str):
        self.name = name
        self._count = 0
        self._start_time = time.time()
        self._lock = threading.Lock()
    
    def mark(self, count: int = 1):
        """Mark events."""
        with self._lock:
            self._count += count
    
    def get_rate(self) -> float:
        """Get current rate (events per second)."""
        with self._lock:
            elapsed = time.time() - self._start_time
            if elapsed == 0:
                return 0.0
            return self._count / elapsed
    
    def reset(self):
        """Reset rate metric."""
        with self._lock:
            self._count = 0
            self._start_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": MetricType.RATE.value,
            "rate": self.get_rate(),
            "count": self._count
        }


# Global metrics instance
_global_metrics: Optional[PerformanceMetrics] = None
_metrics_lock = threading.Lock()


def get_metrics() -> PerformanceMetrics:
    """Get global metrics instance."""
    global _global_metrics
    with _metrics_lock:
        if _global_metrics is None:
            _global_metrics = PerformanceMetrics()
        return _global_metrics


def reset_global_metrics():
    """Reset global metrics."""
    global _global_metrics
    with _metrics_lock:
        if _global_metrics is not None:
            _global_metrics.reset()





