"""
Performance Tracking Utility
===========================

Track and analyze performance metrics for adaptive learning systems.
"""
import time
import logging
import numpy as np
from typing import Dict, Any, List
from collections import deque
from .config import AdaptiveLearningConfig

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Track and analyze performance metrics"""
    
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.metrics_history = deque(maxlen=1000)
        self.performance_trends = {}
        self.baseline_performance = None
        
        logger.info("✅ Performance Tracker initialized")
    
    def record_metric(self, metric_name: str, value: float, timestamp: float = None):
        """Record a performance metric"""
        if timestamp is None:
            timestamp = time.time()
        
        self.metrics_history.append({
            'metric_name': metric_name,
            'value': value,
            'timestamp': timestamp
        })
        
        # Update trends
        self._update_trends(metric_name, value)
    
    def _update_trends(self, metric_name: str, value: float):
        """Update performance trends"""
        if metric_name not in self.performance_trends:
            self.performance_trends[metric_name] = {
                'values': deque(maxlen=100),
                'trend': 'stable',
                'improvement_rate': 0.0
            }
        
        trend_data = self.performance_trends[metric_name]
        trend_data['values'].append(value)
        
        # Calculate trend
        if len(trend_data['values']) >= 10:
            recent_values = list(trend_data['values'])[-10:]
            trend_data['trend'] = self._calculate_trend(recent_values)
            trend_data['improvement_rate'] = self._calculate_improvement_rate(recent_values)
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_improvement_rate(self, values: List[float]) -> float:
        """Calculate improvement rate"""
        if len(values) < 2:
            return 0.0
        
        # Calculate percentage improvement
        start_value = values[0]
        end_value = values[-1]
        
        if start_value == 0:
            return 0.0
        
        return (end_value - start_value) / abs(start_value)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {
            'total_metrics': len(self.metrics_history),
            'trends': {},
            'overall_trend': 'stable'
        }
        
        # Analyze trends
        improving_count = 0
        declining_count = 0
        
        for metric_name, trend_data in self.performance_trends.items():
            summary['trends'][metric_name] = {
                'trend': trend_data['trend'],
                'improvement_rate': trend_data['improvement_rate'],
                'recent_values': list(trend_data['values'])[-5:]
            }
            
            if trend_data['trend'] == 'improving':
                improving_count += 1
            elif trend_data['trend'] == 'declining':
                declining_count += 1
        
        # Determine overall trend
        if improving_count > declining_count:
            summary['overall_trend'] = 'improving'
        elif declining_count > improving_count:
            summary['overall_trend'] = 'declining'
        
        return summary

