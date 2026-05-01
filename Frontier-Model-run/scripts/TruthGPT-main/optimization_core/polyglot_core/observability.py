"""
Observability utilities for polyglot_core.

Provides tracing, monitoring, and observability features.
"""

from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading


@dataclass
class TraceSpan:
    """Trace span for observability."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: list = field(default_factory=list)
    status: str = "ok"
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        end = self.end_time or time.perf_counter()
        return (end - self.start_time) * 1000
    
    def add_attribute(self, key: str, value: Any):
        """Add attribute to span."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span."""
        self.events.append({
            'name': name,
            'time': time.perf_counter(),
            'attributes': attributes or {}
        })


class Tracer:
    """
    Tracer for polyglot_core operations.
    
    Provides distributed tracing capabilities.
    """
    
    def __init__(self):
        self._spans: Dict[str, TraceSpan] = {}
        self._active_spans: list = []
        self._lock = threading.Lock() if threading else None
    
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[str] = None
    ) -> str:
        """
        Start a trace span.
        
        Args:
            name: Span name
            attributes: Initial attributes
            parent: Parent span ID
            
        Returns:
            Span ID
        """
        span_id = f"{name}_{int(time.time() * 1000000)}"
        
        span = TraceSpan(
            name=name,
            start_time=time.perf_counter(),
            attributes=attributes or {}
        )
        
        if parent:
            span.add_attribute("parent_span_id", parent)
        
        if self._lock:
            with self._lock:
                self._spans[span_id] = span
                self._active_spans.append(span_id)
        else:
            self._spans[span_id] = span
            self._active_spans.append(span_id)
        
        return span_id
    
    def end_span(self, span_id: str, status: str = "ok", error: Optional[str] = None):
        """
        End a trace span.
        
        Args:
            span_id: Span ID
            status: Span status
            error: Error message if any
        """
        if span_id not in self._spans:
            return
        
        span = self._spans[span_id]
        span.end_time = time.perf_counter()
        span.status = status
        span.error = error
        
        if self._lock:
            with self._lock:
                if span_id in self._active_spans:
                    self._active_spans.remove(span_id)
        else:
            if span_id in self._active_spans:
                self._active_spans.remove(span_id)
    
    def add_attribute(self, span_id: str, key: str, value: Any):
        """Add attribute to span."""
        if span_id in self._spans:
            self._spans[span_id].add_attribute(key, value)
    
    def add_event(self, span_id: str, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span."""
        if span_id in self._spans:
            self._spans[span_id].add_event(name, attributes)
    
    def get_span(self, span_id: str) -> Optional[TraceSpan]:
        """Get span by ID."""
        return self._spans.get(span_id)
    
    def get_active_spans(self) -> list:
        """Get active span IDs."""
        if self._lock:
            with self._lock:
                return self._active_spans.copy()
        return self._active_spans.copy()
    
    def export_traces(self) -> Dict[str, Any]:
        """Export all traces."""
        return {
            'spans': [
                {
                    'id': span_id,
                    'name': span.name,
                    'duration_ms': span.duration_ms,
                    'status': span.status,
                    'attributes': span.attributes,
                    'events': span.events
                }
                for span_id, span in self._spans.items()
            ],
            'active_spans': self.get_active_spans()
        }


class Observability:
    """
    Observability manager for polyglot_core.
    
    Combines tracing, metrics, and logging.
    """
    
    def __init__(self):
        self.tracer = Tracer()
        self._enabled = True
    
    def trace(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracing.
        
        Example:
            with observability.trace("cache_operation", {"backend": "rust"}):
                cache.get(layer=0, position=0)
        """
        return TraceContext(self, name, attributes)
    
    def enable(self):
        """Enable observability."""
        self._enabled = True
    
    def disable(self):
        """Disable observability."""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if observability is enabled."""
        return self._enabled


class TraceContext:
    """Context manager for tracing."""
    
    def __init__(self, observability: Observability, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.observability = observability
        self.name = name
        self.attributes = attributes or {}
        self.span_id = None
    
    def __enter__(self):
        if self.observability.is_enabled():
            self.span_id = self.observability.tracer.start_span(self.name, self.attributes)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span_id and self.observability.is_enabled():
            status = "error" if exc_type else "ok"
            error = str(exc_val) if exc_val else None
            self.observability.tracer.end_span(self.span_id, status, error)
    
    def add_attribute(self, key: str, value: Any):
        """Add attribute to current span."""
        if self.span_id:
            self.observability.tracer.add_attribute(self.span_id, key, value)
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to current span."""
        if self.span_id:
            self.observability.tracer.add_event(self.span_id, name, attributes)


# Global observability
_global_observability = Observability()


def get_observability() -> Observability:
    """Get global observability instance."""
    return _global_observability


def trace(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Convenience function for tracing."""
    return _global_observability.trace(name, attributes)













