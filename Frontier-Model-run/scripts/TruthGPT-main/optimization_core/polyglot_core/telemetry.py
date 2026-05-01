"""
Telemetry utilities for polyglot_core.

Provides advanced telemetry and analytics.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import time


@dataclass
class TelemetryEvent:
    """Telemetry event."""
    name: str
    timestamp: datetime = field(default_factory=datetime.now)
    properties: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


class TelemetryCollector:
    """
    Telemetry collector for polyglot_core.
    
    Collects and aggregates telemetry data.
    """
    
    def __init__(self):
        self._events: List[TelemetryEvent] = []
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._max_events: int = 10000
    
    def track_event(
        self,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Track an event.
        
        Args:
            name: Event name
            properties: Event properties
            metrics: Event metrics
        """
        event = TelemetryEvent(
            name=name,
            properties=properties or {},
            metrics=metrics or {}
        )
        
        self._events.append(event)
        
        # Keep only recent events
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment counter."""
        self._counters[name] += value
    
    def set_gauge(self, name: str, value: float):
        """Set gauge value."""
        self._gauges[name] = value
    
    def record_histogram(self, name: str, value: float):
        """Record histogram value."""
        self._histograms[name].append(value)
        
        # Keep only recent values
        if len(self._histograms[name]) > 1000:
            self._histograms[name] = self._histograms[name][-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        histograms_summary = {}
        for name, values in self._histograms.items():
            if values:
                histograms_summary[name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'p50': sorted(values)[len(values) // 2] if values else 0,
                    'p95': sorted(values)[int(len(values) * 0.95)] if values else 0,
                    'p99': sorted(values)[int(len(values) * 0.99)] if values else 0
                }
        
        return {
            'counters': dict(self._counters),
            'gauges': self._gauges.copy(),
            'histograms': histograms_summary,
            'event_count': len(self._events)
        }
    
    def get_recent_events(self, limit: int = 100) -> List[TelemetryEvent]:
        """Get recent events."""
        return self._events[-limit:]
    
    def clear(self):
        """Clear all telemetry data."""
        self._events.clear()
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


# Global telemetry collector
_global_telemetry = TelemetryCollector()


def get_telemetry() -> TelemetryCollector:
    """Get global telemetry collector."""
    return _global_telemetry


def track_event(name: str, properties: Optional[Dict[str, Any]] = None, metrics: Optional[Dict[str, float]] = None):
    """Convenience function to track event."""
    _global_telemetry.track_event(name, properties, metrics)













