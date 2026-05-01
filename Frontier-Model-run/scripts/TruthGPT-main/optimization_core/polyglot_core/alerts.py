"""
Alerting system for polyglot_core.

Provides alerting and notification capabilities.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert definition."""
    id: str
    name: str
    message: str
    severity: AlertSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class AlertRule:
    """Alert rule definition."""
    
    def __init__(
        self,
        name: str,
        condition: Callable[[Any], bool],
        severity: AlertSeverity = AlertSeverity.WARNING,
        message: Optional[str] = None,
        cooldown_seconds: float = 60.0
    ):
        """
        Initialize alert rule.
        
        Args:
            name: Rule name
            condition: Condition function (returns True to trigger alert)
            severity: Alert severity
            message: Alert message template
            cooldown_seconds: Cooldown period between alerts
        """
        self.name = name
        self.condition = condition
        self.severity = severity
        self.message = message or f"Alert: {name}"
        self.cooldown_seconds = cooldown_seconds
        self.last_triggered: Optional[float] = None
    
    def should_trigger(self, data: Any) -> bool:
        """
        Check if alert should trigger.
        
        Args:
            data: Data to check
            
        Returns:
            True if alert should trigger
        """
        if not self.condition(data):
            return False
        
        # Check cooldown
        now = time.time()
        if self.last_triggered and (now - self.last_triggered) < self.cooldown_seconds:
            return False
        
        self.last_triggered = now
        return True


class AlertManager:
    """
    Alert manager for polyglot_core.
    
    Manages alerts and notifications.
    """
    
    def __init__(self):
        self._alerts: Dict[str, Alert] = {}
        self._rules: Dict[str, AlertRule] = {}
        self._handlers: List[Callable[[Alert], None]] = []
    
    def register_rule(self, rule: AlertRule):
        """Register alert rule."""
        self._rules[rule.name] = rule
    
    def register_handler(self, handler: Callable[[Alert], None]):
        """Register alert handler."""
        self._handlers.append(handler)
    
    def check(self, data: Any, source: Optional[str] = None):
        """
        Check all rules against data.
        
        Args:
            data: Data to check
            source: Optional source identifier
        """
        for rule_name, rule in self._rules.items():
            if rule.should_trigger(data):
                alert = Alert(
                    id=f"{rule_name}_{int(time.time())}",
                    name=rule_name,
                    message=rule.message,
                    severity=rule.severity,
                    source=source
                )
                
                self._alerts[alert.id] = alert
                
                # Notify handlers
                for handler in self._handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        print(f"Error in alert handler: {e}")
    
    def create_alert(
        self,
        name: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        source: Optional[str] = None,
        **metadata
    ) -> str:
        """
        Create alert manually.
        
        Args:
            name: Alert name
            message: Alert message
            severity: Alert severity
            source: Optional source
            **metadata: Additional metadata
            
        Returns:
            Alert ID
        """
        alert = Alert(
            id=f"{name}_{int(time.time())}",
            name=name,
            message=message,
            severity=severity,
            source=source,
            metadata=metadata
        )
        
        self._alerts[alert.id] = alert
        
        # Notify handlers
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Error in alert handler: {e}")
        
        return alert.id
    
    def resolve_alert(self, alert_id: str):
        """Resolve alert."""
        if alert_id in self._alerts:
            alert = self._alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
    
    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        resolved: Optional[bool] = None
    ) -> List[Alert]:
        """Get alerts, optionally filtered."""
        alerts = list(self._alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active (unresolved) alerts."""
        return self.get_alerts(resolved=False)


# Global alert manager
_global_alert_manager = AlertManager()


def get_alert_manager() -> AlertManager:
    """Get global alert manager."""
    return _global_alert_manager


def create_alert(
    name: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.WARNING,
    **kwargs
) -> str:
    """Convenience function to create alert."""
    return _global_alert_manager.create_alert(name, message, severity, **kwargs)













