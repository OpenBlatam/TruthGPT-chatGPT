"""
Enterprise TruthGPT Metrics and Alerting System
Advanced metrics collection with intelligent alerting
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import statistics

class MetricType(Enum):
    """Metric type enum."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"

class AlertSeverity(Enum):
    """Alert severity enum."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertCondition(Enum):
    """Alert condition enum."""
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN_OR_EQUAL = "lte"

@dataclass
class MetricPoint:
    """Metric point dataclass."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class AlertRule:
    """Alert rule dataclass."""
    name: str
    metric_name: str
    condition: AlertCondition
    threshold: float
    severity: AlertSeverity
    duration: timedelta = timedelta(minutes=5)
    enabled: bool = True
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert dataclass."""
    rule_name: str
    severity: AlertSeverity
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    """Enterprise metrics collector."""
    
    def __init__(self):
        self.metrics: Dict[str, List[MetricPoint]] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def add_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, metric_type: MetricType = MetricType.GAUGE):
        """Add metric point."""
        with self.lock:
            metric_point = MetricPoint(
                name=name,
                value=value,
                labels=labels or {},
                metric_type=metric_type
            )
            
            if name not in self.metrics:
                self.metrics[name] = []
            
            self.metrics[name].append(metric_point)
            
            # Keep only last 10000 points per metric
            if len(self.metrics[name]) > 10000:
                self.metrics[name] = self.metrics[name][-10000:]
    
    def get_metric(self, name: str, limit: int = 1000) -> List[MetricPoint]:
        """Get metric points."""
        with self.lock:
            return self.metrics.get(name, [])[-limit:]
    
    def get_metric_names(self) -> List[str]:
        """Get all metric names."""
        with self.lock:
            return list(self.metrics.keys())
    
    def get_latest_value(self, name: str) -> Optional[float]:
        """Get latest metric value."""
        with self.lock:
            if name in self.metrics and self.metrics[name]:
                return self.metrics[name][-1].value
            return None
    
    def get_metric_stats(self, name: str, window: timedelta = timedelta(minutes=5)) -> Dict[str, Any]:
        """Get metric statistics."""
        with self.lock:
            if name not in self.metrics:
                return {}
            
            cutoff_time = datetime.now() - window
            recent_points = [
                point for point in self.metrics[name]
                if point.timestamp >= cutoff_time
            ]
            
            if not recent_points:
                return {}
            
            values = [point.value for point in recent_points]
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "latest": values[-1],
                "window_start": cutoff_time,
                "window_end": datetime.now()
            }

class AlertManager:
    """Enterprise alert manager."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: List[Alert] = []
        self.callbacks: Dict[str, List[Callable]] = {}
        self.lock = threading.Lock()
        
        self.running = False
        self.check_thread: Optional[threading.Thread] = None
        self.check_interval = 30  # seconds
        
        self.logger = logging.getLogger(__name__)
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule."""
        with self.lock:
            self.rules[rule.name] = rule
            self.logger.info(f"Alert rule added: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove alert rule."""
        with self.lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                self.logger.info(f"Alert rule removed: {rule_name}")
    
    def start_monitoring(self):
        """Start alert monitoring."""
        if self.running:
            return
        
        self.running = True
        self.check_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.check_thread.start()
        self.logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop alert monitoring."""
        self.running = False
        if self.check_thread:
            self.check_thread.join()
        self.logger.info("Alert monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_rules()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in alert monitoring: {str(e)}")
    
    def _check_rules(self):
        """Check all alert rules."""
        with self.lock:
            for rule_name, rule in self.rules.items():
                if not rule.enabled:
                    continue
                
                self._check_rule(rule)
    
    def _check_rule(self, rule: AlertRule):
        """Check individual alert rule."""
        # Get latest metric value
        latest_value = self.metrics_collector.get_latest_value(rule.metric_name)
        if latest_value is None:
            return
        
        # Check condition
        condition_met = self._evaluate_condition(latest_value, rule.condition, rule.threshold)
        
        if condition_met:
            # Check if alert already exists and is not resolved
            existing_alert = next(
                (alert for alert in self.alerts
                 if alert.rule_name == rule.name and not alert.resolved),
                None
            )
            
            if not existing_alert:
                # Create new alert
                alert = Alert(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=f"{rule.metric_name} {rule.condition.value} {rule.threshold} (current: {latest_value})",
                    metric_name=rule.metric_name,
                    metric_value=latest_value,
                    threshold=rule.threshold,
                    labels=rule.labels
                )
                
                self.alerts.append(alert)
                
                # Trigger callbacks
                self._trigger_callbacks(alert)
                
                self.logger.warning(f"Alert triggered: {alert.message}")
        else:
            # Resolve existing alert if condition is no longer met
            self._resolve_alert_if_exists(rule.name)
    
    def _evaluate_condition(self, value: float, condition: AlertCondition, threshold: float) -> bool:
        """Evaluate alert condition."""
        if condition == AlertCondition.GREATER_THAN:
            return value > threshold
        elif condition == AlertCondition.LESS_THAN:
            return value < threshold
        elif condition == AlertCondition.EQUALS:
            return value == threshold
        elif condition == AlertCondition.NOT_EQUALS:
            return value != threshold
        elif condition == AlertCondition.GREATER_THAN_OR_EQUAL:
            return value >= threshold
        elif condition == AlertCondition.LESS_THAN_OR_EQUAL:
            return value <= threshold
        else:
            return False
    
    def _resolve_alert_if_exists(self, rule_name: str):
        """Resolve alert if it exists."""
        for alert in self.alerts:
            if alert.rule_name == rule_name and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                self.logger.info(f"Alert resolved: {rule_name}")
    
    def _trigger_callbacks(self, alert: Alert):
        """Trigger alert callbacks."""
        if alert.rule_name in self.callbacks:
            for callback in self.callbacks[alert.rule_name]:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {str(e)}")
    
    def add_callback(self, rule_name: str, callback: Callable):
        """Add alert callback."""
        if rule_name not in self.callbacks:
            self.callbacks[rule_name] = []
        self.callbacks[rule_name].append(callback)
    
    def get_alerts(self, unresolved_only: bool = True) -> List[Alert]:
        """Get alerts."""
        with self.lock:
            if unresolved_only:
                return [alert for alert in self.alerts if not alert.resolved]
            return self.alerts.copy()
    
    def resolve_alert(self, alert_id: str):
        """Resolve alert by ID."""
        with self.lock:
            for alert in self.alerts:
                if alert.rule_name == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    self.logger.info(f"Alert resolved: {alert_id}")
                    break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self.lock:
            total_alerts = len(self.alerts)
            unresolved_alerts = len([alert for alert in self.alerts if not alert.resolved])
            
            # Count by severity
            severity_counts = {}
            for alert in self.alerts:
                if not alert.resolved:
                    severity = alert.severity.value
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            return {
                "total_alerts": total_alerts,
                "unresolved_alerts": unresolved_alerts,
                "severity_counts": severity_counts,
                "total_rules": len(self.rules),
                "enabled_rules": len([rule for rule in self.rules.values() if rule.enabled]),
                "monitoring_active": self.running
            }

# Global instances
_metrics_collector: Optional[MetricsCollector] = None
_alert_manager: Optional[AlertManager] = None

def get_metrics_collector() -> MetricsCollector:
    """Get or create metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

def get_alert_manager() -> AlertManager:
    """Get or create alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(get_metrics_collector())
    return _alert_manager

# Example usage
if __name__ == "__main__":
    # Get instances
    metrics = get_metrics_collector()
    alerts = get_alert_manager()
    
    # Add some metrics
    for i in range(100):
        metrics.add_metric("cpu_usage", 50 + i * 0.5)
        metrics.add_metric("memory_usage", 60 + i * 0.3)
        metrics.add_metric("response_time", 100 + i * 2)
        time.sleep(0.1)
    
    # Add alert rules
    cpu_rule = AlertRule(
        name="high_cpu",
        metric_name="cpu_usage",
        condition=AlertCondition.GREATER_THAN,
        threshold=80.0,
        severity=AlertSeverity.HIGH
    )
    alerts.add_rule(cpu_rule)
    
    memory_rule = AlertRule(
        name="high_memory",
        metric_name="memory_usage",
        condition=AlertCondition.GREATER_THAN,
        threshold=90.0,
        severity=AlertSeverity.CRITICAL
    )
    alerts.add_rule(memory_rule)
    
    # Add callback
    def alert_callback(alert: Alert):
        print(f"ALERT: {alert.severity.value.upper()} - {alert.message}")
    
    alerts.add_callback("high_cpu", alert_callback)
    alerts.add_callback("high_memory", alert_callback)
    
    # Start monitoring
    alerts.start_monitoring()
    
    try:
        # Let it run
        time.sleep(5)
        
        # Get stats
        stats = alerts.get_stats()
        print("Alert Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Get active alerts
        active_alerts = alerts.get_alerts()
        print(f"\nActive Alerts: {len(active_alerts)}")
        for alert in active_alerts:
            print(f"  {alert.severity.value}: {alert.message}")
    
    finally:
        alerts.stop_monitoring()
    
    print("\nMonitoring stopped")


