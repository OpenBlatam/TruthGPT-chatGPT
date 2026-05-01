"""
Monitoring utilities for optimization_core.

Provides utilities for production monitoring and alerting.
"""
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Alert(BaseModel):
    """Alert data structure."""
    level: AlertLevel
    message: str
    component: str
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "message": self.message,
            "component": self.component,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class AlertManager:
    """Manager for alerts."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts: deque = deque(maxlen=1000)  # Keep last 1000 alerts
        self.handlers: Dict[AlertLevel, List[Callable]] = {
            level: [] for level in AlertLevel
        }
    
    def register_handler(
        self,
        level: AlertLevel,
        handler: Callable[[Alert], None]
    ):
        """
        Register alert handler.
        
        Args:
            level: Alert level
            handler: Handler function
        """
        self.handlers[level].append(handler)
        logger.debug(f"Registered handler for {level.value}")
    
    def alert(
        self,
        level: AlertLevel,
        message: str,
        component: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Send an alert.
        
        Args:
            level: Alert level
            message: Alert message
            component: Component name
            metadata: Optional metadata
        """
        alert = Alert(
            level=level,
            message=message,
            component=component,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Call handlers
        for handler in self.handlers[level]:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}", exc_info=True)
        
        # Also call handlers for higher levels
        if level == AlertLevel.WARNING:
            for handler in self.handlers[AlertLevel.ERROR]:
                try:
                    handler(alert)
                except Exception:
                    pass
    
    def get_recent_alerts(
        self,
        level: Optional[AlertLevel] = None,
        limit: int = 100
    ) -> List[Alert]:
        """
        Get recent alerts.
        
        Args:
            level: Filter by level (all if None)
            limit: Maximum number of alerts
        
        Returns:
            List of alerts
        """
        alerts = list(self.alerts)
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts[-limit:]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get alert summary.
        
        Returns:
            Summary dictionary
        """
        alerts = list(self.alerts)
        
        summary = {
            "total": len(alerts),
            "by_level": {},
            "recent_critical": len([a for a in alerts[-100:] if a.level == AlertLevel.CRITICAL]),
            "recent_errors": len([a for a in alerts[-100:] if a.level == AlertLevel.ERROR]),
        }
        
        for level in AlertLevel:
            summary["by_level"][level.value] = len([a for a in alerts if a.level == level])
        
        return summary


class SystemMonitor:
    """Monitor for system metrics."""
    
    def __init__(self):
        """Initialize system monitor."""
        self.metrics: Dict[str, deque] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}
    
    def record_metric(
        self,
        name: str,
        value: float,
        max_history: int = 1000
    ):
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            max_history: Maximum history size
        """
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=max_history)
        
        self.metrics[name].append(value)
    
    def set_threshold(
        self,
        name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ):
        """
        Set threshold for metric.
        
        Args:
            name: Metric name
            min_value: Minimum value
            max_value: Maximum value
        """
        self.thresholds[name] = {
            "min": min_value,
            "max": max_value,
        }
    
    def check_thresholds(
        self,
        alert_manager: Optional[AlertManager] = None
    ) -> List[Dict[str, Any]]:
        """
        Check if metrics exceed thresholds.
        
        Args:
            alert_manager: Optional alert manager
        
        Returns:
            List of violations
        """
        violations = []
        
        for name, values in self.metrics.items():
            if name not in self.thresholds:
                continue
            
            threshold = self.thresholds[name]
            current_value = values[-1] if values else None
            
            if current_value is None:
                continue
            
            if threshold["min"] is not None and current_value < threshold["min"]:
                violation = {
                    "metric": name,
                    "value": current_value,
                    "threshold": threshold["min"],
                    "type": "below_minimum"
                }
                violations.append(violation)
                
                if alert_manager:
                    alert_manager.alert(
                        AlertLevel.WARNING,
                        f"Metric {name} below threshold: {current_value} < {threshold['min']}",
                        component="monitor"
                    )
            
            if threshold["max"] is not None and current_value > threshold["max"]:
                violation = {
                    "metric": name,
                    "value": current_value,
                    "threshold": threshold["max"],
                    "type": "above_maximum"
                }
                violations.append(violation)
                
                if alert_manager:
                    alert_manager.alert(
                        AlertLevel.WARNING,
                        f"Metric {name} above threshold: {current_value} > {threshold['max']}",
                        component="monitor"
                    )
        
        return violations
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Args:
            name: Metric name
        
        Returns:
            Statistics dictionary
        """
        if name not in self.metrics:
            return {}
        
        values = list(self.metrics[name])
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "current": values[-1],
        }


# Global instances
_global_alert_manager = AlertManager()
_global_system_monitor = SystemMonitor()


def get_alert_manager() -> AlertManager:
    """Get global alert manager."""
    return _global_alert_manager


def get_system_monitor() -> SystemMonitor:
    """Get global system monitor."""
    return _global_system_monitor













