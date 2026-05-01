"""
Comprehensive Monitoring and Metrics System
==========================================

Advanced monitoring system with:
- Real-time metrics collection
- Performance monitoring
- Resource usage tracking
- Custom metrics
- Export to various backends
"""

import time
import psutil
import threading
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import asyncio
from enum import Enum


class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_usage: float
    network_io: Dict[str, float]
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None


class MetricsCollector:
    """
    Comprehensive metrics collection system.
    
    Features:
    - Real-time metrics collection
    - Performance monitoring
    - Custom metrics
    - Export capabilities
    - Alerting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.custom_metrics: Dict[str, Any] = {}
        
        # Performance monitoring
        self.performance_history: deque = deque(maxlen=100)
        self.start_time = time.time()
        
        # Threading
        self.lock = threading.RLock()
        self.collection_thread = None
        self.running = False
        
        # Export backends
        self.exporters: List[Callable] = []
        
        # Start collection
        self.start_collection()
    
    def start_collection(self):
        """Start metrics collection thread"""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        self.logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        self.logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect custom metrics
                self._collect_custom_metrics()
                
                # Export metrics
                self._export_metrics()
                
                # Sleep
                time.sleep(self.config.get('collection_interval', 10))
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric('system.cpu_usage', cpu_percent, tags={'type': 'system'})
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric('system.memory_usage', memory.percent, tags={'type': 'system'})
            self.record_metric('system.memory_available', memory.available / (1024**3), tags={'type': 'system'})
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric('system.disk_usage', disk_percent, tags={'type': 'system'})
            
            # Network I/O
            network = psutil.net_io_counters()
            self.record_metric('system.network_bytes_sent', network.bytes_sent, tags={'type': 'system'})
            self.record_metric('system.network_bytes_recv', network.bytes_recv, tags={'type': 'system'})
            
            # Process metrics
            process = psutil.Process()
            self.record_metric('process.cpu_percent', process.cpu_percent(), tags={'type': 'process'})
            self.record_metric('process.memory_percent', process.memory_percent(), tags={'type': 'process'})
            self.record_metric('process.num_threads', process.num_threads(), tags={'type': 'process'})
            
            # Store performance metrics
            perf_metrics = PerformanceMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                memory_available=memory.available / (1024**3),
                disk_usage=disk_percent,
                network_io={
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                }
            )
            
            with self.lock:
                self.performance_history.append(perf_metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_custom_metrics(self):
        """Collect custom metrics"""
        for name, metric_func in self.custom_metrics.items():
            try:
                value = metric_func()
                self.record_metric(f'custom.{name}', value, tags={'type': 'custom'})
            except Exception as e:
                self.logger.error(f"Error collecting custom metric {name}: {e}")
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, metric_type: MetricType = MetricType.GAUGE):
        """Record a metric"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metric_type=metric_type
        )
        
        with self.lock:
            self.metrics[name].append(metric)
    
    def get_metric(self, name: str, default: float = 0.0) -> float:
        """Get latest metric value"""
        with self.lock:
            if name in self.metrics and self.metrics[name]:
                return self.metrics[name][-1].value
            return default
    
    def get_metric_history(self, name: str, duration: Optional[timedelta] = None) -> List[Metric]:
        """Get metric history"""
        with self.lock:
            if name not in self.metrics:
                return []
            
            metrics = list(self.metrics[name])
            
            if duration:
                cutoff = datetime.now() - duration
                metrics = [m for m in metrics if m.timestamp >= cutoff]
            
            return metrics
    
    def get_metric_stats(self, name: str, duration: Optional[timedelta] = None) -> Dict[str, float]:
        """Get metric statistics"""
        history = self.get_metric_history(name, duration)
        
        if not history:
            return {}
        
        values = [m.value for m in history]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1]
        }
    
    def register_custom_metric(self, name: str, metric_func: Callable[[], float]):
        """Register custom metric function"""
        self.custom_metrics[name] = metric_func
        self.logger.info(f"Registered custom metric: {name}")
    
    def collect_metrics(self, optimizer, execution_time: float) -> Dict[str, float]:
        """Collect metrics for optimizer execution"""
        metrics = {
            'execution_time': execution_time,
            'cpu_usage': self.get_metric('system.cpu_usage'),
            'memory_usage': self.get_metric('system.memory_usage'),
            'timestamp': time.time()
        }
        
        # Add optimizer-specific metrics
        if hasattr(optimizer, 'get_metrics'):
            optimizer_metrics = optimizer.get_metrics()
            metrics.update(optimizer_metrics)
        
        return metrics
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024**3),
            'available': memory.available / (1024**3),
            'used': memory.used / (1024**3),
            'percent': memory.percent
        }
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        return psutil.cpu_percent(interval=1)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self.lock:
            if not self.performance_history:
                return {}
            
            latest = self.performance_history[-1]
            
            return {
                'cpu_usage': latest.cpu_usage,
                'memory_usage': latest.memory_usage,
                'memory_available': latest.memory_available,
                'disk_usage': latest.disk_usage,
                'uptime': time.time() - self.start_time,
                'gpu_usage': latest.gpu_usage,
                'gpu_memory': latest.gpu_memory
            }
    
    def add_exporter(self, exporter_func: Callable[[List[Metric]], None]):
        """Add metrics exporter"""
        self.exporters.append(exporter_func)
    
    def _export_metrics(self):
        """Export metrics to registered backends"""
        if not self.exporters:
            return
        
        try:
            # Collect all metrics
            all_metrics = []
            with self.lock:
                for metric_list in self.metrics.values():
                    all_metrics.extend(metric_list)
            
            # Export to all backends
            for exporter in self.exporters:
                try:
                    exporter(all_metrics)
                except Exception as e:
                    self.logger.error(f"Error in metrics export: {e}")
        
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
    
    def export_to_json(self, filepath: str):
        """Export metrics to JSON file"""
        def json_exporter(metrics: List[Metric]):
            data = []
            for metric in metrics:
                data.append({
                    'name': metric.name,
                    'value': metric.value,
                    'timestamp': metric.timestamp.isoformat(),
                    'tags': metric.tags,
                    'type': metric.metric_type.value
                })
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        self.add_exporter(json_exporter)
    
    def export_to_prometheus(self, port: int = 9090):
        """Export metrics to Prometheus format"""
        try:
            from prometheus_client import start_http_server, Counter, Gauge, Histogram
            
            # Create Prometheus metrics
            prometheus_metrics = {}
            
            def prometheus_exporter(metrics: List[Metric]):
                for metric in metrics:
                    if metric.name not in prometheus_metrics:
                        if metric.metric_type == MetricType.COUNTER:
                            prometheus_metrics[metric.name] = Counter(metric.name, 'Custom metric')
                        elif metric.metric_type == MetricType.GAUGE:
                            prometheus_metrics[metric.name] = Gauge(metric.name, 'Custom metric')
                        elif metric.metric_type == MetricType.HISTOGRAM:
                            prometheus_metrics[metric.name] = Histogram(metric.name, 'Custom metric')
                    
                    if hasattr(prometheus_metrics[metric.name], 'set'):
                        prometheus_metrics[metric.name].set(metric.value)
                    elif hasattr(prometheus_metrics[metric.name], 'inc'):
                        prometheus_metrics[metric.name].inc(metric.value)
            
            self.add_exporter(prometheus_exporter)
            start_http_server(port)
            self.logger.info(f"Prometheus metrics server started on port {port}")
            
        except ImportError:
            self.logger.warning("Prometheus client not available")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self.lock:
            summary = {
                'total_metrics': sum(len(metrics) for metrics in self.metrics.values()),
                'metric_names': list(self.metrics.keys()),
                'custom_metrics': list(self.custom_metrics.keys()),
                'performance': self.get_performance_summary(),
                'uptime': time.time() - self.start_time
            }
            
            # Add metric statistics
            for name in self.metrics:
                stats = self.get_metric_stats(name)
                if stats:
                    summary[f'{name}_stats'] = stats
            
            return summary
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_collection()
        self.exporters.clear()
        self.custom_metrics.clear()



