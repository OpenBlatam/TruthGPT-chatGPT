"""
Enterprise TruthGPT Performance Monitor
Advanced monitoring with real-time metrics and alerting
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging

class MetricType(Enum):
    """Metric type enum."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertLevel(Enum):
    """Alert level enum."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Metric dataclass."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class Alert:
    """Alert dataclass."""
    name: str
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    """Enterprise performance monitoring system."""
    
    def __init__(self, check_interval: int = 5):
        self.check_interval = check_interval
        self.metrics: Dict[str, List[Metric]] = {}
        self.alerts: List[Alert] = []
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default thresholds
        self._init_default_thresholds()
    
    def _init_default_thresholds(self):
        """Initialize default thresholds."""
        self.thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "disk_usage": {"warning": 85.0, "critical": 95.0},
            "response_time": {"warning": 1000.0, "critical": 5000.0},
            "error_rate": {"warning": 5.0, "critical": 10.0}
        }
    
    def start_monitoring(self):
        """Start monitoring."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                self._check_thresholds()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.add_metric("cpu_usage", cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.add_metric("memory_usage", memory.percent)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.add_metric("disk_usage", disk_percent)
        
        # Network I/O
        network = psutil.net_io_counters()
        self.add_metric("network_bytes_sent", network.bytes_sent)
        self.add_metric("network_bytes_recv", network.bytes_recv)
    
    def add_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Add metric."""
        metric = Metric(
            name=name,
            value=value,
            labels=labels or {},
            timestamp=datetime.now()
        )
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(metric)
        
        # Keep only last 1000 metrics per name
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
    
    def _check_thresholds(self):
        """Check metric thresholds."""
        for metric_name, metrics in self.metrics.items():
            if not metrics:
                continue
            
            latest_metric = metrics[-1]
            if metric_name not in self.thresholds:
                continue
            
            thresholds = self.thresholds[metric_name]
            
            # Check critical threshold
            if latest_metric.value >= thresholds.get("critical", float('inf')):
                self._create_alert(
                    f"{metric_name}_critical",
                    AlertLevel.CRITICAL,
                    f"{metric_name} is critically high: {latest_metric.value}%"
                )
            
            # Check warning threshold
            elif latest_metric.value >= thresholds.get("warning", float('inf')):
                self._create_alert(
                    f"{metric_name}_warning",
                    AlertLevel.WARNING,
                    f"{metric_name} is high: {latest_metric.value}%"
                )
    
    def _create_alert(self, name: str, level: AlertLevel, message: str):
        """Create alert."""
        # Check if alert already exists and is not resolved
        existing_alert = next(
            (alert for alert in self.alerts 
             if alert.name == name and not alert.resolved),
            None
        )
        
        if existing_alert:
            return
        
        alert = Alert(
            name=name,
            level=level,
            message=message,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        
        # Trigger callbacks
        if name in self.callbacks:
            for callback in self.callbacks[name]:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {str(e)}")
        
        self.logger.warning(f"Alert created: {message}")
    
    def resolve_alert(self, name: str):
        """Resolve alert."""
        for alert in self.alerts:
            if alert.name == name and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                self.logger.info(f"Alert resolved: {name}")
    
    def set_threshold(self, metric_name: str, warning: float, critical: float):
        """Set threshold for metric."""
        self.thresholds[metric_name] = {
            "warning": warning,
            "critical": critical
        }
    
    def add_alert_callback(self, alert_name: str, callback: Callable):
        """Add alert callback."""
        if alert_name not in self.callbacks:
            self.callbacks[alert_name] = []
        self.callbacks[alert_name].append(callback)
    
    def get_metrics(self, name: Optional[str] = None, limit: int = 100) -> Dict[str, List[Metric]]:
        """Get metrics."""
        if name:
            return {name: self.metrics.get(name, [])[-limit:]}
        else:
            return {
                metric_name: metrics[-limit:]
                for metric_name, metrics in self.metrics.items()
            }
    
    def get_alerts(self, unresolved_only: bool = True) -> List[Alert]:
        """Get alerts."""
        if unresolved_only:
            return [alert for alert in self.alerts if not alert.resolved]
        return self.alerts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        total_metrics = sum(len(metrics) for metrics in self.metrics.values())
        unresolved_alerts = len(self.get_alerts(unresolved_only=True))
        
        return {
            "total_metrics": total_metrics,
            "metric_names": list(self.metrics.keys()),
            "unresolved_alerts": unresolved_alerts,
            "monitoring_active": self.running,
            "check_interval": self.check_interval
        }

# Global monitor instance
_monitor: Optional[PerformanceMonitor] = None

def get_monitor() -> PerformanceMonitor:
    """Get or create performance monitor."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor

# Example usage
if __name__ == "__main__":
    # Create monitor
    monitor = PerformanceMonitor(check_interval=2)
    
    # Add alert callback
    def alert_callback(alert: Alert):
        print(f"ALERT: {alert.level.value.upper()} - {alert.message}")
    
    monitor.add_alert_callback("cpu_usage_critical", alert_callback)
    monitor.add_alert_callback("memory_usage_warning", alert_callback)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Simulate some work
        for i in range(10):
            time.sleep(1)
            
            # Get current metrics
            metrics = monitor.get_metrics(limit=5)
            print(f"\nMetrics at {datetime.now()}:")
            for name, metric_list in metrics.items():
                if metric_list:
                    latest = metric_list[-1]
                    print(f"  {name}: {latest.value}")
            
            # Check alerts
            alerts = monitor.get_alerts()
            if alerts:
                print(f"  Active alerts: {len(alerts)}")
    
    finally:
        monitor.stop_monitoring()
    
    # Get final stats
    stats = monitor.get_stats()
    print("\nFinal Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


