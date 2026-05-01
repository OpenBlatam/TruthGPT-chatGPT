"""
Logging utilities for inference engines.

Provides structured logging, metrics collection, and performance tracking.
"""
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager


from optimization_core.core.metrics_base import BaseMetrics, calculate_rate


@dataclass
class InferenceMetrics(BaseMetrics):
    """Metrics for inference operations."""
    total_tokens_generated: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        return calculate_rate(self.cache_hits, self.cache_hits + self.cache_misses)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = super().to_dict()
        result.update({
            "total_tokens_generated": self.total_tokens_generated,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
        })
        return result
    
    def reset(self) -> None:
        """Reset all metrics."""
        super().reset()
        self.total_tokens_generated = 0
        self.cache_hits = 0
        self.cache_misses = 0


from optimization_core.core.metrics_base import MetricsCollectorBase


class MetricsCollector(MetricsCollectorBase):
    """Collector for inference metrics."""
    
    def __init__(self, max_samples: int = 10000):
        super().__init__(max_samples=max_samples)
        self.metrics = InferenceMetrics()
    
    def record_request(
        self,
        success: bool,
        tokens_generated: int = 0,
        latency_ms: float = 0.0,
        error_type: Optional[str] = None
    ):
        """Record a request."""
        # Update base metrics
        if success:
            self.request_times.append(latency_ms)
        else:
            if error_type:
                self.error_counts[error_type] += 1
        
        # Update inference-specific metrics
        self.metrics.total_requests += 1
        
        if success:
            self.metrics.successful_requests += 1
            self.metrics.total_tokens_generated += tokens_generated
            self.metrics.total_latency_ms += latency_ms
        else:
            self.metrics.failed_requests += 1
    
    def record_cache_hit(self, hit: bool):
        """Record cache hit/miss."""
        if hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = self.metrics.to_dict()
        base_summary = super().get_summary()
        summary.update(base_summary)
        return summary
    
    def reset(self):
        """Reset all metrics."""
        super().reset()
        self.metrics.reset()


def setup_inference_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging for inference operations.
    
    Args:
        level: Logging level
        format_string: Custom format string
    
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(funcName)s:%(lineno)d - %(message)s"
        )
    
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_string))
    
    logger = logging.getLogger("inference")
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger


@contextmanager
def log_operation(operation_name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager for logging operations.
    
    Args:
        operation_name: Name of the operation
        logger: Logger instance (defaults to inference logger)
    
    Example:
        with log_operation("model_generation"):
            result = engine.generate(prompts)
    """
    if logger is None:
        logger = logging.getLogger("inference")
    
    start_time = time.time()
    logger.info(f"Starting {operation_name}")
    
    try:
        yield
        elapsed = time.time() - start_time
        logger.info(f"Completed {operation_name} in {elapsed:.3f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"Failed {operation_name} after {elapsed:.3f}s: {e}",
            exc_info=True
        )
        raise


def log_metrics(metrics: MetricsCollector, logger: Optional[logging.Logger] = None):
    """
    Log metrics summary.
    
    Args:
        metrics: MetricsCollector instance
        logger: Logger instance (defaults to inference logger)
    """
    if logger is None:
        logger = logging.getLogger("inference")
    
    summary = metrics.get_summary()
    
    logger.info("=== Inference Metrics ===")
    logger.info(f"Total Requests: {summary['total_requests']}")
    logger.info(f"Success Rate: {summary['success_rate']:.2%}")
    logger.info(f"Average Latency: {summary['average_latency_ms']:.2f}ms")
    logger.info(f"P50 Latency: {summary['p50_latency_ms']:.2f}ms")
    logger.info(f"P95 Latency: {summary['p95_latency_ms']:.2f}ms")
    logger.info(f"P99 Latency: {summary['p99_latency_ms']:.2f}ms")
    logger.info(f"Cache Hit Rate: {summary['cache_hit_rate']:.2%}")
    logger.info(f"Total Tokens Generated: {summary['total_tokens_generated']}")
    
    if summary['error_counts']:
        logger.info("Error Counts:")
        for error_type, count in summary['error_counts'].items():
            logger.info(f"  {error_type}: {count}")


