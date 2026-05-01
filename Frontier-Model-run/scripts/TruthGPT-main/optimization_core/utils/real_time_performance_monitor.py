"""
Real-Time Performance Monitoring and Analytics System
====================================================

A comprehensive system for real-time performance monitoring, analytics,
and intelligent insights for TruthGPT optimization systems.

Author: TruthGPT Optimization Team
Version: 41.2.0-REAL-TIME-PERFORMANCE-MONITORING
"""

import asyncio
import logging
import time
import threading
import queue
import json
import numpy as np
import torch
import psutil
import GPUtil
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from collections import defaultdict, deque
import pickle
from datetime import datetime, timedelta
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Metric type enumeration"""
    PERFORMANCE = "performance"
    MEMORY = "memory"
    ENERGY = "energy"
    QUALITY = "quality"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    UTILIZATION = "utilization"
    TEMPERATURE = "temperature"
    POWER = "power"
    CUSTOM = "custom"

class AlertLevel(Enum):
    """Alert level enumeration"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class PerformanceMetric(BaseModel):
    """Performance metric data structure"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    metric_id: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PerformanceAlert(BaseModel):
    """Performance alert data structure"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    alert_id: str
    metric_id: str
    alert_level: AlertLevel
    message: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class PerformanceInsight(BaseModel):
    """Performance insight data structure"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    insight_id: str
    insight_type: str
    description: str
    confidence: float
    impact: str
    recommendations: List[str]
    timestamp: datetime
    metrics_analyzed: List[str]

class RealTimePerformanceMonitor:
    """
    Real-Time Performance Monitoring and Analytics System
    
    Provides comprehensive monitoring, analytics, and insights for
    TruthGPT optimization systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Real-Time Performance Monitor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.metrics_buffer = deque(maxlen=10000)
        self.alerts_buffer = deque(maxlen=1000)
        self.insights_buffer = deque(maxlen=500)
        
        # Database connection
        self.db_path = self.config.get('db_path', 'performance_monitor.db')
        self._init_database()
        
        # Monitoring threads
        self.monitoring_active = True
        self.monitoring_threads = []
        
        # Alert thresholds
        self.alert_thresholds = self._init_alert_thresholds()
        
        # Performance tracking
        self.performance_stats = defaultdict(list)
        self.optimizer_performance = defaultdict(dict)
        
        # Analytics models
        self.anomaly_detector = None
        self.trend_analyzer = None
        self.correlation_analyzer = None
        
        # Start monitoring
        self._start_monitoring()
        
        logger.info("Real-Time Performance Monitor initialized")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_id TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    tags TEXT,
                    metadata TEXT
                )
            ''')
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    metric_id TEXT NOT NULL,
                    alert_level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    current_value REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_time DATETIME
                )
            ''')
            
            # Create insights table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight_id TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    impact TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    metrics_analyzed TEXT NOT NULL
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_insights_timestamp ON insights(timestamp)')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def _init_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize alert thresholds"""
        return {
            'performance': {
                'warning': 0.8,
                'error': 0.6,
                'critical': 0.4
            },
            'memory': {
                'warning': 80.0,
                'error': 90.0,
                'critical': 95.0
            },
            'energy': {
                'warning': 0.7,
                'error': 0.5,
                'critical': 0.3
            },
            'quality': {
                'warning': 0.85,
                'error': 0.75,
                'critical': 0.65
            },
            'latency': {
                'warning': 1000.0,  # ms
                'error': 2000.0,
                'critical': 5000.0
            },
            'cpu_usage': {
                'warning': 80.0,
                'error': 90.0,
                'critical': 95.0
            },
            'gpu_usage': {
                'warning': 85.0,
                'error': 95.0,
                'critical': 98.0
            }
        }
    
    def _start_monitoring(self):
        """Start monitoring threads"""
        # System metrics monitoring
        system_thread = threading.Thread(
            target=self._monitor_system_metrics,
            daemon=True
        )
        system_thread.start()
        self.monitoring_threads.append(system_thread)
        
        # Alert processing
        alert_thread = threading.Thread(
            target=self._process_alerts,
            daemon=True
        )
        alert_thread.start()
        self.monitoring_threads.append(alert_thread)
        
        # Analytics processing
        analytics_thread = threading.Thread(
            target=self._process_analytics,
            daemon=True
        )
        analytics_thread.start()
        self.monitoring_threads.append(analytics_thread)
        
        # Database cleanup
        cleanup_thread = threading.Thread(
            target=self._cleanup_database,
            daemon=True
        )
        cleanup_thread.start()
        self.monitoring_threads.append(cleanup_thread)
    
    def _monitor_system_metrics(self):
        """Monitor system metrics continuously"""
        while self.monitoring_active:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_count = psutil.cpu_count()
                cpu_freq = psutil.cpu_freq()
                
                self.record_metric(
                    metric_id="cpu_usage",
                    metric_type=MetricType.UTILIZATION,
                    value=cpu_percent,
                    tags={"component": "cpu", "type": "usage"}
                )
                
                # Memory metrics
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                
                self.record_metric(
                    metric_id="memory_usage",
                    metric_type=MetricType.MEMORY,
                    value=memory.percent,
                    tags={"component": "memory", "type": "usage"}
                )
                
                self.record_metric(
                    metric_id="memory_available",
                    metric_type=MetricType.MEMORY,
                    value=memory.available / (1024**3),  # GB
                    tags={"component": "memory", "type": "available"}
                )
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                disk_io = psutil.disk_io_counters()
                
                self.record_metric(
                    metric_id="disk_usage",
                    metric_type=MetricType.UTILIZATION,
                    value=disk.percent,
                    tags={"component": "disk", "type": "usage"}
                )
                
                # Network metrics
                network_io = psutil.net_io_counters()
                
                self.record_metric(
                    metric_id="network_bytes_sent",
                    metric_type=MetricType.THROUGHPUT,
                    value=network_io.bytes_sent,
                    tags={"component": "network", "type": "bytes_sent"}
                )
                
                self.record_metric(
                    metric_id="network_bytes_recv",
                    metric_type=MetricType.THROUGHPUT,
                    value=network_io.bytes_recv,
                    tags={"component": "network", "type": "bytes_recv"}
                )
                
                # GPU metrics
                try:
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        self.record_metric(
                            metric_id=f"gpu_{i}_usage",
                            metric_type=MetricType.UTILIZATION,
                            value=gpu.load * 100,
                            tags={"component": "gpu", "gpu_id": str(i), "type": "usage"}
                        )
                        
                        self.record_metric(
                            metric_id=f"gpu_{i}_memory",
                            metric_type=MetricType.MEMORY,
                            value=gpu.memoryUtil * 100,
                            tags={"component": "gpu", "gpu_id": str(i), "type": "memory"}
                        )
                        
                        self.record_metric(
                            metric_id=f"gpu_{i}_temperature",
                            metric_type=MetricType.TEMPERATURE,
                            value=gpu.temperature,
                            tags={"component": "gpu", "gpu_id": str(i), "type": "temperature"}
                        )
                except Exception as e:
                    logger.debug(f"GPU monitoring error: {e}")
                
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Error monitoring system metrics: {e}")
                time.sleep(10)
    
    def _process_alerts(self):
        """Process alerts based on metrics"""
        while self.monitoring_active:
            try:
                # Check recent metrics for alert conditions
                recent_metrics = list(self.metrics_buffer)[-100:]  # Last 100 metrics
                
                for metric in recent_metrics:
                    self._check_alert_conditions(metric)
                
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error processing alerts: {e}")
                time.sleep(30)
    
    def _check_alert_conditions(self, metric: PerformanceMetric):
        """Check if metric triggers alert conditions"""
        metric_type = metric.metric_type.value
        value = metric.value
        
        if metric_type in self.alert_thresholds:
            thresholds = self.alert_thresholds[metric_type]
            
            # Check each threshold level
            for level, threshold in thresholds.items():
                if self._should_trigger_alert(value, threshold, metric_type):
                    alert_level = AlertLevel(level.upper())
                    self._create_alert(metric, alert_level, threshold)
    
    def _should_trigger_alert(self, value: float, threshold: float, metric_type: str) -> bool:
        """Determine if alert should be triggered"""
        # For utilization metrics, alert if above threshold
        if metric_type in ['cpu_usage', 'memory', 'gpu_usage', 'disk_usage']:
            return value > threshold
        
        # For performance metrics, alert if below threshold
        elif metric_type in ['performance', 'energy', 'quality']:
            return value < threshold
        
        # For latency metrics, alert if above threshold
        elif metric_type in ['latency']:
            return value > threshold
        
        return False
    
    def _create_alert(self, metric: PerformanceMetric, alert_level: AlertLevel, threshold: float):
        """Create a new alert"""
        alert_id = f"{metric.metric_id}_{alert_level.value}_{int(time.time())}"
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            metric_id=metric.metric_id,
            alert_level=alert_level,
            message=f"{metric.metric_id} {alert_level.value}: {metric.value:.2f} (threshold: {threshold:.2f})",
            threshold=threshold,
            current_value=metric.value,
            timestamp=datetime.now()
        )
        
        self.alerts_buffer.append(alert)
        self._store_alert(alert)
        
        logger.warning(f"Alert created: {alert.message}")
    
    def _process_analytics(self):
        """Process analytics and generate insights"""
        while self.monitoring_active:
            try:
                # Generate insights every 5 minutes
                if len(self.metrics_buffer) >= 100:
                    insights = self._generate_insights()
                    for insight in insights:
                        self.insights_buffer.append(insight)
                        self._store_insight(insight)
                
                time.sleep(300)  # Process every 5 minutes
            except Exception as e:
                logger.error(f"Error processing analytics: {e}")
                time.sleep(600)
    
    def _generate_insights(self) -> List[PerformanceInsight]:
        """Generate performance insights"""
        insights = []
        
        # Get recent metrics
        recent_metrics = list(self.metrics_buffer)[-1000:]  # Last 1000 metrics
        
        if len(recent_metrics) < 50:
            return insights
        
        # Group metrics by type
        metrics_by_type = defaultdict(list)
        for metric in recent_metrics:
            metrics_by_type[metric.metric_type].append(metric.value)
        
        # Performance trend analysis
        if MetricType.PERFORMANCE in metrics_by_type:
            performance_values = metrics_by_type[MetricType.PERFORMANCE]
            trend_insight = self._analyze_performance_trend(performance_values)
            if trend_insight:
                insights.append(trend_insight)
        
        # Memory usage analysis
        if MetricType.MEMORY in metrics_by_type:
            memory_values = metrics_by_type[MetricType.MEMORY]
            memory_insight = self._analyze_memory_usage(memory_values)
            if memory_insight:
                insights.append(memory_insight)
        
        # Correlation analysis
        correlation_insights = self._analyze_correlations(recent_metrics)
        insights.extend(correlation_insights)
        
        # Anomaly detection
        anomaly_insights = self._detect_anomalies(recent_metrics)
        insights.extend(anomaly_insights)
        
        return insights
    
    def _analyze_performance_trend(self, values: List[float]) -> Optional[PerformanceInsight]:
        """Analyze performance trend"""
        if len(values) < 10:
            return None
        
        # Calculate trend using linear regression
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        trend_direction = "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable"
        confidence = abs(r_value)
        
        if confidence > 0.7:  # Strong correlation
            recommendations = []
            if trend_direction == "declining":
                recommendations.extend([
                    "Consider switching to a more efficient optimizer",
                    "Check for resource constraints",
                    "Review optimization parameters"
                ])
            elif trend_direction == "improving":
                recommendations.extend([
                    "Current optimization strategy is working well",
                    "Consider maintaining current configuration"
                ])
            
            return PerformanceInsight(
                insight_id=f"performance_trend_{int(time.time())}",
                insight_type="trend_analysis",
                description=f"Performance trend is {trend_direction} (slope: {slope:.4f})",
                confidence=confidence,
                impact="medium",
                recommendations=recommendations,
                timestamp=datetime.now(),
                metrics_analyzed=["performance"]
            )
        
        return None
    
    def _analyze_memory_usage(self, values: List[float]) -> Optional[PerformanceInsight]:
        """Analyze memory usage patterns"""
        if len(values) < 10:
            return None
        
        avg_memory = np.mean(values)
        max_memory = np.max(values)
        memory_variance = np.var(values)
        
        insights = []
        recommendations = []
        
        if avg_memory > 80:
            insights.append("High average memory usage detected")
            recommendations.append("Consider using memory-efficient optimizers")
            recommendations.append("Implement memory pooling strategies")
        
        if max_memory > 95:
            insights.append("Memory usage spikes detected")
            recommendations.append("Monitor for memory leaks")
            recommendations.append("Implement memory cleanup routines")
        
        if memory_variance > 100:  # High variance
            insights.append("High memory usage variability")
            recommendations.append("Implement memory usage smoothing")
            recommendations.append("Consider adaptive memory management")
        
        if insights:
            return PerformanceInsight(
                insight_id=f"memory_analysis_{int(time.time())}",
                insight_type="memory_analysis",
                description="; ".join(insights),
                confidence=0.8,
                impact="high" if avg_memory > 80 else "medium",
                recommendations=recommendations,
                timestamp=datetime.now(),
                metrics_analyzed=["memory"]
            )
        
        return None
    
    def _analyze_correlations(self, metrics: List[PerformanceMetric]) -> List[PerformanceInsight]:
        """Analyze correlations between metrics"""
        insights = []
        
        # Group metrics by time windows
        time_windows = defaultdict(list)
        for metric in metrics:
            # Round to nearest minute
            time_key = metric.timestamp.replace(second=0, microsecond=0)
            time_windows[time_key].append(metric)
        
        # Analyze correlations within time windows
        for time_window, window_metrics in time_windows.items():
            if len(window_metrics) < 5:
                continue
            
            # Extract values by metric type
            metric_values = defaultdict(list)
            for metric in window_metrics:
                metric_values[metric.metric_type].append(metric.value)
            
            # Check for correlations
            metric_types = list(metric_values.keys())
            for i, type1 in enumerate(metric_types):
                for type2 in metric_types[i+1:]:
                    values1 = metric_values[type1]
                    values2 = metric_values[type2]
                    
                    if len(values1) == len(values2) and len(values1) >= 3:
                        correlation, p_value = stats.pearsonr(values1, values2)
                        
                        if abs(correlation) > 0.7 and p_value < 0.05:  # Strong correlation
                            correlation_type = "positive" if correlation > 0 else "negative"
                            insight = PerformanceInsight(
                                insight_id=f"correlation_{type1.value}_{type2.value}_{int(time.time())}",
                                insight_type="correlation_analysis",
                                description=f"Strong {correlation_type} correlation between {type1.value} and {type2.value} (r={correlation:.3f})",
                                confidence=abs(correlation),
                                impact="medium",
                                recommendations=[
                                    f"Monitor {type1.value} when {type2.value} changes",
                                    f"Consider joint optimization of {type1.value} and {type2.value}"
                                ],
                                timestamp=datetime.now(),
                                metrics_analyzed=[type1.value, type2.value]
                            )
                            insights.append(insight)
        
        return insights
    
    def _detect_anomalies(self, metrics: List[PerformanceMetric]) -> List[PerformanceInsight]:
        """Detect anomalies in metrics"""
        insights = []
        
        # Group by metric type
        metrics_by_type = defaultdict(list)
        for metric in metrics:
            metrics_by_type[metric.metric_type].append(metric.value)
        
        # Detect anomalies using statistical methods
        for metric_type, values in metrics_by_type.items():
            if len(values) < 10:
                continue
            
            # Use Z-score for anomaly detection
            z_scores = np.abs(stats.zscore(values))
            anomalies = np.where(z_scores > 2.5)[0]  # Z-score > 2.5
            
            if len(anomalies) > 0:
                anomaly_rate = len(anomalies) / len(values)
                
                if anomaly_rate > 0.05:  # More than 5% anomalies
                    insight = PerformanceInsight(
                        insight_id=f"anomaly_{metric_type.value}_{int(time.time())}",
                        insight_type="anomaly_detection",
                        description=f"Detected {len(anomalies)} anomalies in {metric_type.value} ({anomaly_rate:.1%} of data)",
                        confidence=0.9,
                        impact="high" if anomaly_rate > 0.1 else "medium",
                        recommendations=[
                            "Investigate root cause of anomalies",
                            "Consider implementing anomaly-resistant optimizers",
                            "Monitor system stability"
                        ],
                        timestamp=datetime.now(),
                        metrics_analyzed=[metric_type.value]
                    )
                    insights.append(insight)
        
        return insights
    
    def record_metric(self, metric_id: str, metric_type: MetricType, value: float, 
                     tags: Optional[Dict[str, str]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            metric_id=metric_id,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.metrics_buffer.append(metric)
        self._store_metric(metric)
        
        # Update performance stats
        self.performance_stats[metric_id].append(value)
    
    def _store_metric(self, metric: PerformanceMetric):
        """Store metric in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO metrics (metric_id, metric_type, value, timestamp, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                metric.metric_id,
                metric.metric_type.value,
                metric.value,
                metric.timestamp,
                json.dumps(metric.tags),
                json.dumps(metric.metadata)
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store metric: {e}")
    
    def _store_alert(self, alert: PerformanceAlert):
        """Store alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts (alert_id, metric_id, alert_level, message, threshold, 
                                  current_value, timestamp, resolved, resolution_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.metric_id,
                alert.alert_level.value,
                alert.message,
                alert.threshold,
                alert.current_value,
                alert.timestamp,
                alert.resolved,
                alert.resolution_time
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    def _store_insight(self, insight: PerformanceInsight):
        """Store insight in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO insights (insight_id, insight_type, description, confidence, 
                                    impact, recommendations, timestamp, metrics_analyzed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                insight.insight_id,
                insight.insight_type,
                insight.description,
                insight.confidence,
                insight.impact,
                json.dumps(insight.recommendations),
                insight.timestamp,
                json.dumps(insight.metrics_analyzed)
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store insight: {e}")
    
    def _cleanup_database(self):
        """Clean up old database records"""
        while self.monitoring_active:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Delete records older than 30 days
                cutoff_date = datetime.now() - timedelta(days=30)
                
                cursor.execute('DELETE FROM metrics WHERE timestamp < ?', (cutoff_date,))
                cursor.execute('DELETE FROM alerts WHERE timestamp < ?', (cutoff_date,))
                cursor.execute('DELETE FROM insights WHERE timestamp < ?', (cutoff_date,))
                
                conn.commit()
                conn.close()
                
                time.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Error cleaning up database: {e}")
                time.sleep(7200)  # Retry in 2 hours
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {
            'total_metrics': len(self.metrics_buffer),
            'total_alerts': len(self.alerts_buffer),
            'total_insights': len(self.insights_buffer),
            'active_alerts': len([a for a in self.alerts_buffer if not a.resolved]),
            'recent_metrics': list(self.metrics_buffer)[-10:] if self.metrics_buffer else [],
            'recent_alerts': list(self.alerts_buffer)[-5:] if self.alerts_buffer else [],
            'recent_insights': list(self.insights_buffer)[-3:] if self.insights_buffer else []
        }
        
        return summary
    
    def get_optimizer_performance(self, optimizer_name: str) -> Dict[str, Any]:
        """Get performance statistics for a specific optimizer"""
        if optimizer_name not in self.optimizer_performance:
            return {}
        
        stats = self.optimizer_performance[optimizer_name]
        
        return {
            'total_runs': stats.get('total_runs', 0),
            'avg_performance': stats.get('avg_performance', 0.0),
            'avg_memory_usage': stats.get('avg_memory_usage', 0.0),
            'avg_energy_efficiency': stats.get('avg_energy_efficiency', 0.0),
            'success_rate': stats.get('success_rate', 0.0),
            'recent_trend': stats.get('recent_trend', 'stable')
        }
    
    def generate_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate performance report for a time period"""
        # This would query the database for metrics in the time range
        # For now, return a basic report structure
        return {
            'period': {
                'start': start_time,
                'end': end_time
            },
            'summary': self.get_performance_summary(),
            'recommendations': [
                "Monitor system resources regularly",
                "Implement adaptive optimization strategies",
                "Use performance insights for optimization decisions"
            ]
        }
    
    def shutdown(self):
        """Shutdown the performance monitor"""
        self.monitoring_active = False
        
        # Wait for threads to finish
        for thread in self.monitoring_threads:
            thread.join(timeout=5)
        
        logger.info("Performance monitor shutdown complete")

# Factory function
def create_real_time_performance_monitor(config: Optional[Dict[str, Any]] = None) -> RealTimePerformanceMonitor:
    """
    Create a Real-Time Performance Monitor instance
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RealTimePerformanceMonitor instance
    """
    return RealTimePerformanceMonitor(config)

# Example usage
if __name__ == "__main__":
    # Create performance monitor
    monitor = create_real_time_performance_monitor()
    
    # Record some example metrics
    monitor.record_metric("test_performance", MetricType.PERFORMANCE, 0.85)
    monitor.record_metric("test_memory", MetricType.MEMORY, 75.0)
    monitor.record_metric("test_energy", MetricType.ENERGY, 0.9)
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    print(f"Performance summary: {summary}")
    
    # Shutdown
    monitor.shutdown()

