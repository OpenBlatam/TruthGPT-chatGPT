"""
Base metrics utilities for optimization_core.

Provides common metrics collection patterns and utilities.
"""
import logging
import time
import math
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class BaseMetrics:
    """
    Base metrics class with common functionality.
    
    Provides:
    - Basic counters
    - Success/failure tracking
    - Latency tracking
    - Dictionary serialization
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests
    
    def to_dict(self, exclude_zero: bool = False) -> Dict[str, Any]:
        """
        Convert metrics to dictionary.
        
        Args:
            exclude_zero: Exclude fields with zero values
        
        Returns:
            Dictionary representation
        """
        result = asdict(self)
        result['success_rate'] = self.success_rate
        result['average_latency_ms'] = self.average_latency_ms
        
        if exclude_zero:
            result = {k: v for k, v in result.items() if v != 0}
        
        return result
    
    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0


class MetricsCollectorBase(ABC):
    """
    Base class for metrics collectors.
    
    Provides common functionality for collecting and aggregating metrics.
    """
    
    def __init__(self, max_samples: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_samples: Maximum number of samples to keep
        """
        self.max_samples = max_samples
        self.request_times: deque = deque(maxlen=max_samples)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self._start_time = time.time()
    
    @abstractmethod
    def record_request(
        self,
        success: bool,
        latency_ms: float = 0.0,
        **kwargs
    ) -> None:
        """
        Record a request.
        
        Args:
            success: Whether request was successful
            latency_ms: Request latency in milliseconds
            **kwargs: Additional request-specific data
        """
        pass
    
    def get_percentile_latency(self, percentile: float) -> float:
        """
        Get latency percentile.
        
        Args:
            percentile: Percentile (0-100)
        
        Returns:
            Latency at percentile
        """
        if not self.request_times:
            return 0.0
        
        sorted_times = sorted(self.request_times)
        index = int(len(sorted_times) * percentile / 100)
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            "p50_latency_ms": self.get_percentile_latency(50),
            "p95_latency_ms": self.get_percentile_latency(95),
            "p99_latency_ms": self.get_percentile_latency(99),
            "error_counts": dict(self.error_counts),
            "total_samples": len(self.request_times),
            "uptime_seconds": time.time() - self._start_time,
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.request_times.clear()
        self.error_counts.clear()
        self._start_time = time.time()


# ════════════════════════════════════════════════════════════════════════════════
# METRIC UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss.
    
    Args:
        loss: Cross-entropy loss
    
    Returns:
        Perplexity value
    """
    try:
        return float(math.exp(loss))
    except (OverflowError, ValueError):
        return float("inf")


def calculate_tokens_per_second(num_tokens: int, seconds: float) -> float:
    """
    Calculate tokens per second.
    
    Args:
        num_tokens: Number of tokens
        seconds: Time in seconds
    
    Returns:
        Tokens per second (inf if seconds <= 0)
    """
    if seconds <= 0:
        return float("inf")
    return float(num_tokens) / float(seconds)


def calculate_throughput(count: int, seconds: float) -> float:
    """
    Calculate throughput (items per second).
    
    Args:
        count: Number of items
        seconds: Time in seconds
    
    Returns:
        Throughput (items per second)
    """
    if seconds <= 0:
        return 0.0
    return float(count) / float(seconds)


def calculate_percentage(part: int, total: int) -> float:
    """
    Calculate percentage.
    
    Args:
        part: Part value
        total: Total value
    
    Returns:
        Percentage (0-100)
    """
    if total == 0:
        return 0.0
    return (float(part) / float(total)) * 100.0


def calculate_rate(numerator: int, denominator: int) -> float:
    """
    Calculate rate (numerator / denominator).
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
    
    Returns:
        Rate (0.0-1.0)
    """
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def format_metric_value(value: float, unit: str = "", precision: int = 2) -> str:
    """
    Format metric value for display.
    
    Args:
        value: Metric value
        unit: Unit string (e.g., "ms", "tokens/s")
        precision: Decimal precision
    
    Returns:
        Formatted string
    """
    if math.isinf(value):
        return f"∞ {unit}".strip()
    if math.isnan(value):
        return f"NaN {unit}".strip()
    
    formatted = f"{value:.{precision}f}"
    if unit:
        formatted += f" {unit}"
    return formatted


__all__ = [
    "BaseMetrics",
    "MetricsCollectorBase",
    "calculate_perplexity",
    "calculate_tokens_per_second",
    "calculate_throughput",
    "calculate_percentage",
    "calculate_rate",
    "format_metric_value",
]


