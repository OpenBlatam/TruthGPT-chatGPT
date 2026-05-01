"""
Ultra-Advanced Performance Monitoring System
Comprehensive performance monitoring with real-time metrics, profiling, and analytics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import threading
import json
from pathlib import Path
from collections import defaultdict, deque
import psutil
import GPUtil
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class MonitoringLevel(Enum):
    """Performance monitoring levels."""
    BASIC = "basic"                         # Basic metrics only
    ADVANCED = "advanced"                   # Advanced metrics
    EXPERT = "expert"                       # Expert-level metrics
    MASTER = "master"                       # Master-level metrics
    LEGENDARY = "legendary"                 # Legendary metrics

class MetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"                     # Latency metrics
    THROUGHPUT = "throughput"              # Throughput metrics
    MEMORY = "memory"                       # Memory metrics
    CPU = "cpu"                             # CPU metrics
    GPU = "gpu"                             # GPU metrics
    CACHE = "cache"                         # Cache metrics
    NETWORK = "network"                     # Network metrics
    CUSTOM = "custom"                       # Custom metrics

class AlertLevel(Enum):
    """Alert levels for performance issues."""
    INFO = "info"                           # Informational
    WARNING = "warning"                     # Warning
    ERROR = "error"                         # Error
    CRITICAL = "critical"                   # Critical

@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""
    # Basic settings
    monitoring_level: MonitoringLevel = MonitoringLevel.ADVANCED
    enable_real_time: bool = True
    enable_profiling: bool = True
    enable_analytics: bool = True
    
    # Monitoring intervals
    metrics_interval: float = 1.0           # seconds
    profiling_interval: float = 5.0         # seconds
    analytics_interval: float = 60.0        # seconds
    
    # Data retention
    history_size: int = 10000
    retention_days: int = 30
    
    # Alerting
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'latency': 1.0,      # seconds
        'memory_usage': 0.8, # ratio
        'cpu_usage': 0.8,    # ratio
        'gpu_usage': 0.8,    # ratio
        'error_rate': 0.05   # ratio
    })
    
    # Advanced features
    enable_predictive_analytics: bool = True
    enable_anomaly_detection: bool = True
    enable_trend_analysis: bool = True
    enable_correlation_analysis: bool = True
    
    # Export settings
    enable_export: bool = True
    export_formats: List[str] = field(default_factory=lambda: ['json', 'csv', 'plot'])
    export_interval: float = 300.0          # seconds

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Timing metrics
    latency: float = 0.0
    throughput: float = 0.0
    response_time: float = 0.0
    processing_time: float = 0.0
    
    # Resource metrics
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    disk_usage: float = 0.0
    network_usage: float = 0.0
    
    # Model metrics
    model_size: float = 0.0
    parameter_count: int = 0
    flops: float = 0.0
    
    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    cache_size: float = 0.0
    
    # Quality metrics
    accuracy: float = 0.0
    loss: float = 0.0
    error_rate: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class Alert:
    """Performance alert."""
    level: AlertLevel
    metric: str
    value: float
    threshold: float
    message: str
    timestamp: float = field(default_factory=time.time)

class AdvancedPerformanceMonitor:
    """
    Ultra-Advanced Performance Monitoring System.
    
    Features:
    - Real-time performance monitoring
    - Comprehensive metrics collection
    - Advanced profiling and analysis
    - Predictive analytics and anomaly detection
    - Trend analysis and correlation detection
    - Automated alerting and reporting
    - Data export and visualization
    """
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        
        # Metrics storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.history_size))
        self.profiling_data: Dict[str, Any] = {}
        self.analytics_data: Dict[str, Any] = {}
        
        # Alerts
        self.alerts: deque = deque(maxlen=1000)
        self.alert_history: deque = deque(maxlen=10000)
        
        # Advanced components
        self._setup_advanced_components()
        
        # Background monitoring
        self._setup_monitoring()
        
        logger.info(f"Advanced Performance Monitor initialized with level: {config.monitoring_level}")
    
    def _setup_advanced_components(self):
        """Setup advanced monitoring components."""
        # Predictive analytics
        if self.config.enable_predictive_analytics:
            self.predictive_analytics = PredictiveAnalytics()
        
        # Anomaly detection
        if self.config.enable_anomaly_detection:
            self.anomaly_detector = AnomalyDetector()
        
        # Trend analysis
        if self.config.enable_trend_analysis:
            self.trend_analyzer = TrendAnalyzer()
        
        # Correlation analysis
        if self.config.enable_correlation_analysis:
            self.correlation_analyzer = CorrelationAnalyzer()
        
        # Profiler
        if self.config.enable_profiling:
            self.profiler = AdvancedProfiler()
    
    def _setup_monitoring(self):
        """Setup background monitoring threads."""
        if self.config.enable_real_time:
            self.metrics_thread = threading.Thread(target=self._collect_metrics, daemon=True)
            self.metrics_thread.start()
        
        if self.config.enable_profiling:
            self.profiling_thread = threading.Thread(target=self._collect_profiling_data, daemon=True)
            self.profiling_thread.start()
        
        if self.config.enable_analytics:
            self.analytics_thread = threading.Thread(target=self._perform_analytics, daemon=True)
            self.analytics_thread.start()
        
        if self.config.enable_alerts:
            self.alerting_thread = threading.Thread(target=self._check_alerts, daemon=True)
            self.alerting_thread.start()
        
        if self.config.enable_export:
            self.export_thread = threading.Thread(target=self._export_data, daemon=True)
            self.export_thread.start()
    
    def _collect_metrics(self):
        """Background metrics collection."""
        while True:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                
                # Collect GPU metrics
                gpu_metrics = self._collect_gpu_metrics()
                
                # Collect custom metrics
                custom_metrics = self._collect_custom_metrics()
                
                # Combine metrics
                combined_metrics = {**system_metrics, **gpu_metrics, **custom_metrics}
                
                # Store metrics
                timestamp = time.time()
                for metric_name, value in combined_metrics.items():
                    self.metrics_history[metric_name].append({
                        'timestamp': timestamp,
                        'value': value
                    })
                
                time.sleep(self.config.metrics_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                break
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system performance metrics."""
        metrics = {}
        
        # CPU metrics
        metrics['cpu_usage'] = psutil.cpu_percent()
        metrics['cpu_count'] = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics['memory_usage'] = memory.percent / 100.0
        metrics['memory_total'] = memory.total
        metrics['memory_available'] = memory.available
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics['disk_usage'] = disk.percent / 100.0
        metrics['disk_total'] = disk.total
        metrics['disk_available'] = disk.free
        
        # Network metrics
        network = psutil.net_io_counters()
        metrics['network_bytes_sent'] = network.bytes_sent
        metrics['network_bytes_recv'] = network.bytes_recv
        
        return metrics
    
    def _collect_gpu_metrics(self) -> Dict[str, float]:
        """Collect GPU performance metrics."""
        metrics = {}
        
        if torch.cuda.is_available():
            # GPU memory
            metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated()
            metrics['gpu_memory_reserved'] = torch.cuda.memory_reserved()
            metrics['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory
            
            # GPU utilization
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics['gpu_usage'] = gpu.load
                    metrics['gpu_temperature'] = gpu.temperature
                    metrics['gpu_memory_used'] = gpu.memoryUsed
                    metrics['gpu_memory_total_gpu'] = gpu.memoryTotal
            except:
                pass
        
        return metrics
    
    def _collect_custom_metrics(self) -> Dict[str, float]:
        """Collect custom application metrics."""
        metrics = {}
        
        # This would collect custom metrics from the application
        # For now, return empty dict
        return metrics
    
    def _collect_profiling_data(self):
        """Background profiling data collection."""
        while True:
            try:
                if hasattr(self, 'profiler'):
                    profiling_data = self.profiler.collect_profiling_data()
                    self.profiling_data[time.time()] = profiling_data
                
                time.sleep(self.config.profiling_interval)
                
            except Exception as e:
                logger.error(f"Profiling data collection error: {e}")
                break
    
    def _perform_analytics(self):
        """Background analytics processing."""
        while True:
            try:
                # Predictive analytics
                if hasattr(self, 'predictive_analytics'):
                    predictions = self.predictive_analytics.analyze(self.metrics_history)
                    self.analytics_data['predictions'] = predictions
                
                # Anomaly detection
                if hasattr(self, 'anomaly_detector'):
                    anomalies = self.anomaly_detector.detect(self.metrics_history)
                    self.analytics_data['anomalies'] = anomalies
                
                # Trend analysis
                if hasattr(self, 'trend_analyzer'):
                    trends = self.trend_analyzer.analyze(self.metrics_history)
                    self.analytics_data['trends'] = trends
                
                # Correlation analysis
                if hasattr(self, 'correlation_analyzer'):
                    correlations = self.correlation_analyzer.analyze(self.metrics_history)
                    self.analytics_data['correlations'] = correlations
                
                time.sleep(self.config.analytics_interval)
                
            except Exception as e:
                logger.error(f"Analytics processing error: {e}")
                break
    
    def _check_alerts(self):
        """Background alert checking."""
        while True:
            try:
                # Check each metric against thresholds
                for metric_name, threshold in self.config.alert_thresholds.items():
                    if metric_name in self.metrics_history and self.metrics_history[metric_name]:
                        latest_value = self.metrics_history[metric_name][-1]['value']
                        
                        if latest_value > threshold:
                            alert = Alert(
                                level=AlertLevel.WARNING,
                                metric=metric_name,
                                value=latest_value,
                                threshold=threshold,
                                message=f"{metric_name} exceeded threshold: {latest_value:.2f} > {threshold:.2f}"
                            )
                            
                            self.alerts.append(alert)
                            self.alert_history.append(alert)
                            
                            logger.warning(f"Alert: {alert.message}")
                
                time.sleep(self.config.metrics_interval)
                
            except Exception as e:
                logger.error(f"Alert checking error: {e}")
                break
    
    def _export_data(self):
        """Background data export."""
        while True:
            try:
                if self.config.enable_export:
                    self._export_metrics()
                    self._export_analytics()
                    self._export_alerts()
                
                time.sleep(self.config.export_interval)
                
            except Exception as e:
                logger.error(f"Data export error: {e}")
                break
    
    def _export_metrics(self):
        """Export metrics data."""
        timestamp = int(time.time())
        
        for format_type in self.config.export_formats:
            if format_type == 'json':
                self._export_json(timestamp)
            elif format_type == 'csv':
                self._export_csv(timestamp)
            elif format_type == 'plot':
                self._export_plots(timestamp)
    
    def _export_json(self, timestamp: int):
        """Export metrics as JSON."""
        export_data = {
            'timestamp': timestamp,
            'metrics': {name: list(data) for name, data in self.metrics_history.items()},
            'analytics': self.analytics_data,
            'alerts': [alert.__dict__ for alert in self.alert_history]
        }
        
        filename = f"performance_metrics_{timestamp}.json"
        filepath = Path(__file__).parent / "exports" / filename
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def _export_csv(self, timestamp: int):
        """Export metrics as CSV."""
        import csv
        
        filename = f"performance_metrics_{timestamp}.csv"
        filepath = Path(__file__).parent / "exports" / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['timestamp', 'metric_name', 'value'])
            
            # Write data
            for metric_name, data in self.metrics_history.items():
                for entry in data:
                    writer.writerow([entry['timestamp'], metric_name, entry['value']])
    
    def _export_plots(self, timestamp: int):
        """Export performance plots."""
        plt.style.use('seaborn-v0_8')
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Performance Metrics - {timestamp}', fontsize=16)
        
        # Plot CPU usage
        if 'cpu_usage' in self.metrics_history:
            cpu_data = list(self.metrics_history['cpu_usage'])
            timestamps = [entry['timestamp'] for entry in cpu_data]
            values = [entry['value'] for entry in cpu_data]
            
            axes[0, 0].plot(timestamps, values)
            axes[0, 0].set_title('CPU Usage')
            axes[0, 0].set_ylabel('Usage %')
        
        # Plot Memory usage
        if 'memory_usage' in self.metrics_history:
            memory_data = list(self.metrics_history['memory_usage'])
            timestamps = [entry['timestamp'] for entry in memory_data]
            values = [entry['value'] for entry in memory_data]
            
            axes[0, 1].plot(timestamps, values)
            axes[0, 1].set_title('Memory Usage')
            axes[0, 1].set_ylabel('Usage %')
        
        # Plot GPU usage
        if 'gpu_usage' in self.metrics_history:
            gpu_data = list(self.metrics_history['gpu_usage'])
            timestamps = [entry['timestamp'] for entry in gpu_data]
            values = [entry['value'] for entry in gpu_data]
            
            axes[1, 0].plot(timestamps, values)
            axes[1, 0].set_title('GPU Usage')
            axes[1, 0].set_ylabel('Usage %')
        
        # Plot alerts
        if self.alerts:
            alert_timestamps = [alert.timestamp for alert in self.alerts]
            alert_levels = [alert.level.value for alert in self.alerts]
            
            axes[1, 1].scatter(alert_timestamps, alert_levels)
            axes[1, 1].set_title('Alerts')
            axes[1, 1].set_ylabel('Alert Level')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"performance_plots_{timestamp}.png"
        filepath = Path(__file__).parent / "exports" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _export_analytics(self):
        """Export analytics data."""
        timestamp = int(time.time())
        
        analytics_data = {
            'timestamp': timestamp,
            'analytics': self.analytics_data
        }
        
        filename = f"analytics_{timestamp}.json"
        filepath = Path(__file__).parent / "exports" / filename
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(analytics_data, f, indent=2, default=str)
    
    def _export_alerts(self):
        """Export alerts data."""
        timestamp = int(time.time())
        
        alerts_data = {
            'timestamp': timestamp,
            'alerts': [alert.__dict__ for alert in self.alert_history]
        }
        
        filename = f"alerts_{timestamp}.json"
        filepath = Path(__file__).parent / "exports" / filename
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(alerts_data, f, indent=2, default=str)
    
    def record_metric(self, metric_name: str, value: float, metric_type: MetricType = MetricType.CUSTOM):
        """Record a custom metric."""
        timestamp = time.time()
        
        self.metrics_history[metric_name].append({
            'timestamp': timestamp,
            'value': value,
            'type': metric_type.value
        })
    
    def record_performance_metrics(self, metrics: PerformanceMetrics):
        """Record comprehensive performance metrics."""
        timestamp = time.time()
        
        # Record all metrics
        metric_dict = metrics.__dict__
        for metric_name, value in metric_dict.items():
            if isinstance(value, (int, float)):
                self.metrics_history[metric_name].append({
                    'timestamp': timestamp,
                    'value': value
                })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'monitoring_config': self.config.__dict__,
            'metrics_summary': {},
            'analytics_summary': self.analytics_data,
            'alerts_summary': {
                'total_alerts': len(self.alert_history),
                'recent_alerts': list(self.alerts)[-10:],  # Last 10 alerts
                'alert_levels': {
                    level.value: sum(1 for alert in self.alert_history if alert.level == level)
                    for level in AlertLevel
                }
            },
            'performance_trends': self._calculate_performance_trends(),
            'resource_utilization': self._calculate_resource_utilization()
        }
        
        # Calculate metrics summary
        for metric_name, data in self.metrics_history.items():
            if data:
                values = [entry['value'] for entry in data]
                summary['metrics_summary'][metric_name] = {
                    'current': values[-1] if values else 0,
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'count': len(values)
                }
        
        return summary
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends."""
        trends = {}
        
        for metric_name, data in self.metrics_history.items():
            if len(data) > 1:
                values = [entry['value'] for entry in data]
                
                # Calculate trend (simple linear regression)
                x = np.arange(len(values))
                y = np.array(values)
                
                if len(x) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    trends[metric_name] = {
                        'trend': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                        'slope': slope,
                        'change_rate': slope / np.mean(values) if np.mean(values) != 0 else 0
                    }
        
        return trends
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization."""
        utilization = {}
        
        # Calculate average utilization for key metrics
        key_metrics = ['cpu_usage', 'memory_usage', 'gpu_usage', 'disk_usage']
        
        for metric in key_metrics:
            if metric in self.metrics_history and self.metrics_history[metric]:
                values = [entry['value'] for entry in self.metrics_history[metric]]
                utilization[metric] = np.mean(values)
        
        return utilization
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get latest metric values."""
        latest_metrics = {}
        
        for metric_name, data in self.metrics_history.items():
            if data:
                latest_metrics[metric_name] = data[-1]['value']
        
        return latest_metrics
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metric history for specified time period."""
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            entry for entry in self.metrics_history[metric_name]
            if entry['timestamp'] >= cutoff_time
        ]

# Advanced component classes
class PredictiveAnalytics:
    """Predictive analytics engine."""
    
    def __init__(self):
        self.prediction_models = {}
        self.feature_history = []
    
    def analyze(self, metrics_history: Dict[str, deque]) -> Dict[str, Any]:
        """Perform predictive analysis."""
        predictions = {}
        
        # Simple prediction logic
        for metric_name, data in metrics_history.items():
            if len(data) > 10:
                values = [entry['value'] for entry in data]
                
                # Simple trend-based prediction
                recent_values = values[-10:]
                trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                
                predictions[metric_name] = {
                    'predicted_value': recent_values[-1] + trend,
                    'confidence': 0.8,  # Simplified confidence
                    'trend': trend
                }
        
        return predictions

class AnomalyDetector:
    """Anomaly detection system."""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_threshold = 2.0  # Standard deviations
    
    def detect(self, metrics_history: Dict[str, deque]) -> Dict[str, Any]:
        """Detect anomalies in metrics."""
        anomalies = {}
        
        for metric_name, data in metrics_history.items():
            if len(data) > 20:
                values = [entry['value'] for entry in data]
                
                # Calculate baseline
                if metric_name not in self.baseline_metrics:
                    self.baseline_metrics[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
                
                baseline = self.baseline_metrics[metric_name]
                
                # Detect anomalies
                latest_value = values[-1]
                z_score = abs(latest_value - baseline['mean']) / baseline['std']
                
                if z_score > self.anomaly_threshold:
                    anomalies[metric_name] = {
                        'value': latest_value,
                        'z_score': z_score,
                        'severity': 'high' if z_score > 3.0 else 'medium'
                    }
        
        return anomalies

class TrendAnalyzer:
    """Trend analysis system."""
    
    def analyze(self, metrics_history: Dict[str, deque]) -> Dict[str, Any]:
        """Analyze trends in metrics."""
        trends = {}
        
        for metric_name, data in metrics_history.items():
            if len(data) > 5:
                values = [entry['value'] for entry in data]
                
                # Calculate trend
                x = np.arange(len(values))
                slope, intercept = np.polyfit(x, values, 1)
                
                trends[metric_name] = {
                    'slope': slope,
                    'intercept': intercept,
                    'trend': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                    'strength': abs(slope) / np.std(values) if np.std(values) > 0 else 0
                }
        
        return trends

class CorrelationAnalyzer:
    """Correlation analysis system."""
    
    def analyze(self, metrics_history: Dict[str, deque]) -> Dict[str, Any]:
        """Analyze correlations between metrics."""
        correlations = {}
        
        metric_names = list(metrics_history.keys())
        
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i+1:]:
                if metric1 in metrics_history and metric2 in metrics_history:
                    data1 = list(metrics_history[metric1])
                    data2 = list(metrics_history[metric2])
                    
                    if len(data1) > 5 and len(data2) > 5:
                        values1 = [entry['value'] for entry in data1]
                        values2 = [entry['value'] for entry in data2]
                        
                        # Calculate correlation
                        correlation = np.corrcoef(values1, values2)[0, 1]
                        
                        if not np.isnan(correlation):
                            correlations[f"{metric1}_vs_{metric2}"] = {
                                'correlation': correlation,
                                'strength': abs(correlation),
                                'type': 'positive' if correlation > 0 else 'negative'
                            }
        
        return correlations

class AdvancedProfiler:
    """Advanced profiling system."""
    
    def __init__(self):
        self.profiling_data = {}
    
    def collect_profiling_data(self) -> Dict[str, Any]:
        """Collect profiling data."""
        # This would collect detailed profiling data
        return {
            'timestamp': time.time(),
            'profiling_data': 'placeholder'
        }

# Factory functions
def create_advanced_performance_monitor(config: PerformanceConfig = None) -> AdvancedPerformanceMonitor:
    """Create an advanced performance monitor."""
    if config is None:
        config = PerformanceConfig()
    return AdvancedPerformanceMonitor(config)

def create_performance_config(**kwargs) -> PerformanceConfig:
    """Create a performance configuration."""
    return PerformanceConfig(**kwargs)


