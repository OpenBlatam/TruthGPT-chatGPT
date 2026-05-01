"""
Production Monitoring System - Enterprise-grade monitoring and observability
Provides comprehensive monitoring, alerting, and diagnostics for optimization systems
"""

import time
import logging
import threading
import json
import psutil
import torch
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from enum import Enum
import traceback
import signal
import sys
from contextlib import contextmanager

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Alert:
    """Alert definition."""
    id: str
    level: AlertLevel
    message: str
    timestamp: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Metric:
    """Metric definition."""
    name: str
    value: Union[float, int]
    timestamp: float
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class PerformanceSnapshot:
    """System performance snapshot."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_memory_usage: float
    gpu_utilization: float
    disk_usage: float
    network_io: Dict[str, float]
    active_threads: int
    active_processes: int

class ProductionMonitor:
    """Production-grade monitoring system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_buffer = deque(maxlen=10000)
        self.alerts_buffer = deque(maxlen=1000)
        self.performance_snapshots = deque(maxlen=1000)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.alert_handlers = []
        self.metric_handlers = []
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'gpu_memory_usage': 90.0,
            'disk_usage': 90.0,
            'error_rate': 5.0
        }
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info("🔍 Production Monitor initialized")
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        log_dir = Path(self.config.get('log_directory', './logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(log_dir / 'production_monitor.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Error handler for critical issues
        error_handler = logging.FileHandler(log_dir / 'errors.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # Setup logger
        self.logger = logging.getLogger('production_monitor')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(error_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully")
            self.stop_monitoring()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous monitoring."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info(f"🔍 Started monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("🛑 Monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                snapshot = self._collect_performance_snapshot()
                self.performance_snapshots.append(snapshot)
                
                # Check thresholds and generate alerts
                self._check_thresholds(snapshot)
                
                # Process metrics
                self._process_metrics()
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_performance_snapshot(self) -> PerformanceSnapshot:
        """Collect comprehensive system performance snapshot."""
        # CPU and memory
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # GPU metrics
        gpu_memory_usage = 0.0
        gpu_utilization = 0.0
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # Network I/O
        network_io = psutil.net_io_counters()._asdict()
        
        # Process and thread counts
        active_threads = threading.active_count()
        active_processes = len(psutil.pids())
        
        return PerformanceSnapshot(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_memory_usage=gpu_memory_usage,
            gpu_utilization=gpu_utilization,
            disk_usage=disk_usage,
            network_io=network_io,
            active_threads=active_threads,
            active_processes=active_processes
        )
    
    def _check_thresholds(self, snapshot: PerformanceSnapshot):
        """Check performance thresholds and generate alerts."""
        alerts = []
        
        # CPU threshold
        if snapshot.cpu_usage > self.thresholds['cpu_usage']:
            alerts.append(Alert(
                id=f"cpu_high_{int(time.time())}",
                level=AlertLevel.WARNING,
                message=f"High CPU usage: {snapshot.cpu_usage:.1f}%",
                timestamp=time.time(),
                source="system_monitor",
                metadata={'cpu_usage': snapshot.cpu_usage}
            ))
        
        # Memory threshold
        if snapshot.memory_usage > self.thresholds['memory_usage']:
            alerts.append(Alert(
                id=f"memory_high_{int(time.time())}",
                level=AlertLevel.WARNING,
                message=f"High memory usage: {snapshot.memory_usage:.1f}%",
                timestamp=time.time(),
                source="system_monitor",
                metadata={'memory_usage': snapshot.memory_usage}
            ))
        
        # GPU memory threshold
        if snapshot.gpu_memory_usage > self.thresholds['gpu_memory_usage']:
            alerts.append(Alert(
                id=f"gpu_memory_high_{int(time.time())}",
                level=AlertLevel.WARNING,
                message=f"High GPU memory usage: {snapshot.gpu_memory_usage:.1f}%",
                timestamp=time.time(),
                source="system_monitor",
                metadata={'gpu_memory_usage': snapshot.gpu_memory_usage}
            ))
        
        # Disk threshold
        if snapshot.disk_usage > self.thresholds['disk_usage']:
            alerts.append(Alert(
                id=f"disk_high_{int(time.time())}",
                level=AlertLevel.CRITICAL,
                message=f"High disk usage: {snapshot.disk_usage:.1f}%",
                timestamp=time.time(),
                source="system_monitor",
                metadata={'disk_usage': snapshot.disk_usage}
            ))
        
        # Process alerts
        for alert in alerts:
            self._process_alert(alert)
    
    def _process_alert(self, alert: Alert):
        """Process and handle alerts."""
        self.alerts_buffer.append(alert)
        
        # Log alert
        log_level = {
            AlertLevel.INFO: self.logger.info,
            AlertLevel.WARNING: self.logger.warning,
            AlertLevel.ERROR: self.logger.error,
            AlertLevel.CRITICAL: self.logger.critical
        }[alert.level]
        
        log_level(f"🚨 {alert.message} (Source: {alert.source})")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
    
    def _process_metrics(self):
        """Process collected metrics."""
        for handler in self.metric_handlers:
            try:
                handler(self.metrics_buffer)
            except Exception as e:
                self.logger.error(f"Error in metric handler: {e}")
    
    def record_metric(self, name: str, value: Union[float, int], 
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None):
        """Record a custom metric."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            metric_type=metric_type,
            tags=tags or {}
        )
        
        self.metrics_buffer.append(metric)
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add custom alert handler."""
        self.alert_handlers.append(handler)
    
    def add_metric_handler(self, handler: Callable[[List[Metric]], None]):
        """Add custom metric handler."""
        self.metric_handlers.append(handler)
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        # Filter recent snapshots
        recent_snapshots = [
            s for s in self.performance_snapshots 
            if s.timestamp >= cutoff_time
        ]
        
        if not recent_snapshots:
            return {}
        
        # Calculate statistics
        cpu_values = [s.cpu_usage for s in recent_snapshots]
        memory_values = [s.memory_usage for s in recent_snapshots]
        gpu_memory_values = [s.gpu_memory_usage for s in recent_snapshots]
        
        return {
            'time_period_hours': hours,
            'snapshot_count': len(recent_snapshots),
            'cpu_usage': {
                'mean': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'std': np.std(cpu_values)
            },
            'memory_usage': {
                'mean': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values),
                'std': np.std(memory_values)
            },
            'gpu_memory_usage': {
                'mean': np.mean(gpu_memory_values),
                'max': np.max(gpu_memory_values),
                'min': np.min(gpu_memory_values),
                'std': np.std(gpu_memory_values)
            },
            'alerts_count': len([a for a in self.alerts_buffer if a.timestamp >= cutoff_time])
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        if not self.performance_snapshots:
            return {'status': 'unknown', 'message': 'No performance data available'}
        
        latest = self.performance_snapshots[-1]
        
        # Determine health status
        health_issues = []
        
        if latest.cpu_usage > self.thresholds['cpu_usage']:
            health_issues.append(f"High CPU usage: {latest.cpu_usage:.1f}%")
        
        if latest.memory_usage > self.thresholds['memory_usage']:
            health_issues.append(f"High memory usage: {latest.memory_usage:.1f}%")
        
        if latest.gpu_memory_usage > self.thresholds['gpu_memory_usage']:
            health_issues.append(f"High GPU memory usage: {latest.gpu_memory_usage:.1f}%")
        
        if latest.disk_usage > self.thresholds['disk_usage']:
            health_issues.append(f"High disk usage: {latest.disk_usage:.1f}%")
        
        if health_issues:
            status = 'degraded'
            message = '; '.join(health_issues)
        else:
            status = 'healthy'
            message = 'All systems operating normally'
        
        return {
            'status': status,
            'message': message,
            'timestamp': latest.timestamp,
            'metrics': {
                'cpu_usage': latest.cpu_usage,
                'memory_usage': latest.memory_usage,
                'gpu_memory_usage': latest.gpu_memory_usage,
                'disk_usage': latest.disk_usage
            }
        }
    
    def export_metrics(self, filepath: str, hours: int = 24):
        """Export metrics to file."""
        cutoff_time = time.time() - (hours * 3600)
        
        # Filter recent data
        recent_metrics = [
            m for m in self.metrics_buffer 
            if m.timestamp >= cutoff_time
        ]
        recent_snapshots = [
            s for s in self.performance_snapshots 
            if s.timestamp >= cutoff_time
        ]
        recent_alerts = [
            a for a in self.alerts_buffer 
            if a.timestamp >= cutoff_time
        ]
        
        export_data = {
            'export_timestamp': time.time(),
            'time_period_hours': hours,
            'metrics': [
                {
                    'name': m.name,
                    'value': m.value,
                    'timestamp': m.timestamp,
                    'type': m.metric_type.value,
                    'tags': m.tags
                } for m in recent_metrics
            ],
            'performance_snapshots': [
                {
                    'timestamp': s.timestamp,
                    'cpu_usage': s.cpu_usage,
                    'memory_usage': s.memory_usage,
                    'gpu_memory_usage': s.gpu_memory_usage,
                    'gpu_utilization': s.gpu_utilization,
                    'disk_usage': s.disk_usage,
                    'active_threads': s.active_threads,
                    'active_processes': s.active_processes
                } for s in recent_snapshots
            ],
            'alerts': [
                {
                    'id': a.id,
                    'level': a.level.value,
                    'message': a.message,
                    'timestamp': a.timestamp,
                    'source': a.source,
                    'metadata': a.metadata
                } for a in recent_alerts
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"📊 Exported {len(recent_metrics)} metrics to {filepath}")
    
    def cleanup(self):
        """Cleanup monitoring resources."""
        self.stop_monitoring()
        self.logger.info("🧹 Production Monitor cleanup completed")

# Context manager for monitoring
@contextmanager
def production_monitoring_context(config: Optional[Dict[str, Any]] = None):
    """Context manager for production monitoring."""
    monitor = ProductionMonitor(config)
    try:
        monitor.start_monitoring()
        yield monitor
    finally:
        monitor.cleanup()

# Utility functions
def create_production_monitor(config: Optional[Dict[str, Any]] = None) -> ProductionMonitor:
    """Create a production monitor instance."""
    return ProductionMonitor(config)

def setup_monitoring_for_optimizer(optimizer, monitor: ProductionMonitor):
    """Setup monitoring for an optimizer instance."""
    def optimization_metric_handler(metrics):
        for metric in metrics:
            if metric.name.startswith('optimization_'):
                monitor.record_metric(
                    f"optimizer_{metric.name}",
                    metric.value,
                    metric.metric_type,
                    metric.tags
                )
    
    monitor.add_metric_handler(optimization_metric_handler)

if __name__ == "__main__":
    print("🔍 Production Monitoring System")
    print("=" * 40)
    
    # Example usage
    config = {
        'log_directory': './monitoring_logs',
        'thresholds': {
            'cpu_usage': 70.0,
            'memory_usage': 80.0,
            'gpu_memory_usage': 85.0
        }
    }
    
    with production_monitoring_context(config) as monitor:
        print("✅ Production monitor started")
        
        # Record some test metrics
        monitor.record_metric("test_metric", 42.0, MetricType.GAUGE, {"source": "test"})
        
        # Wait a bit to collect some data
        time.sleep(2)
        
        # Get health status
        health = monitor.get_health_status()
        print(f"🏥 System health: {health['status']} - {health['message']}")
        
        # Get performance summary
        summary = monitor.get_performance_summary(hours=1)
        print(f"📊 Performance summary: {summary.get('snapshot_count', 0)} snapshots collected")
        
        print("✅ Production monitoring example completed")

