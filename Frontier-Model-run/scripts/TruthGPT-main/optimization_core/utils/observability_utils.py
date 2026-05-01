"""
Observability utilities for optimization_core.

Provides utilities for monitoring, tracing, and observability.
"""
import logging
import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class TraceLevel(Enum):
    """Trace levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class TraceSpan(BaseModel):
    """Span in a trace."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = Field(default_factory=dict)
    logs: List[Dict[str, Any]] = Field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Get span duration."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def add_log(self, level: TraceLevel, message: str, **kwargs):
        """Add log to span."""
        self.logs.append({
            "level": level.value,
            "message": message,
            "timestamp": time.time(),
            **kwargs
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "tags": self.tags,
            "logs": self.logs,
        }


class Tracer:
    """Tracer for distributed tracing."""
    
    def __init__(self, service_name: str = "optimization_core"):
        """
        Initialize tracer.
        
        Args:
            service_name: Name of service
        """
        self.service_name = service_name
        self.spans: List[TraceSpan] = []
        self.active_spans: List[TraceSpan] = []
    
    def start_span(
        self,
        name: str,
        tags: Optional[Dict[str, Any]] = None
    ) -> TraceSpan:
        """
        Start a new span.
        
        Args:
            name: Span name
            tags: Optional tags
        
        Returns:
            TraceSpan
        """
        span = TraceSpan(
            name=name,
            start_time=time.time(),
            tags=tags or {}
        )
        self.spans.append(span)
        self.active_spans.append(span)
        return span
    
    def finish_span(self, span: TraceSpan):
        """
        Finish a span.
        
        Args:
            span: Span to finish
        """
        span.end_time = time.time()
        if span in self.active_spans:
            self.active_spans.remove(span)
    
    @contextmanager
    def span(
        self,
        name: str,
        tags: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for span.
        
        Args:
            name: Span name
            tags: Optional tags
        
        Example:
            with tracer.span("operation"):
                # Your code
        """
        span = self.start_span(name, tags)
        try:
            yield span
        finally:
            self.finish_span(span)
    
    def get_trace(self) -> Dict[str, Any]:
        """
        Get complete trace.
        
        Returns:
            Trace dictionary
        """
        return {
            "service": self.service_name,
            "spans": [span.to_dict() for span in self.spans],
            "total_spans": len(self.spans),
            "active_spans": len(self.active_spans),
        }
    
    def clear(self):
        """Clear all spans."""
        self.spans.clear()
        self.active_spans.clear()


class MetricsExporter:
    """Exporter for metrics."""
    
    def __init__(self):
        """Initialize metrics exporter."""
        self.metrics: Dict[str, List[float]] = {}
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        key = self._format_key(name, tags)
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)
    
    def _format_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Format metric key."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def get_metric_summary(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """
        Get metric summary.
        
        Args:
            name: Metric name
            tags: Optional tags
        
        Returns:
            Summary dictionary
        """
        key = self._format_key(name, tags)
        values = self.metrics.get(key, [])
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    
    def export(self) -> Dict[str, Any]:
        """
        Export all metrics.
        
        Returns:
            Metrics dictionary
        """
        return {
            key: {
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
            for key, values in self.metrics.items()
        }


# Global tracer instance
_global_tracer = Tracer()


def get_tracer() -> Tracer:
    """Get global tracer."""
    return _global_tracer


def get_metrics_exporter() -> MetricsExporter:
    """Get global metrics exporter."""
    return MetricsExporter()













