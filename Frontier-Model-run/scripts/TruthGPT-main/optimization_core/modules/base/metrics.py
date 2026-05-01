"""
Metrics shim for modules.base.
"""
from .core_system.core.metrics_base import (
    MetricsCollectorBase as MetricsCollector,
    BaseMetrics as BaseMetric,
)

class MetricConfig:
    """Mock config for backward compatibility."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

__all__ = ['MetricsCollector', 'MetricConfig', 'BaseMetric']
