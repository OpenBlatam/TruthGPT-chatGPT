"""
GPU Performance Monitoring
"""
import time
import logging
import numpy as np
import torch
from typing import Dict, Any, List

from .config import GPUAcceleratorConfig
from ...common.base_advanced_system import BaseAdvancedSystem

logger = logging.getLogger(__name__)

class GPUPerformanceMonitor(BaseAdvancedSystem):
    """Monitor GPU performance in real-time."""
    
    def __init__(self, config: GPUAcceleratorConfig):
        super().__init__(config, "GPUPerformanceMonitor")
        self.metrics_history: List[Dict[str, Any]] = []
        self.current_metrics: Dict[str, Any] = {}
        
        if self.config.enable_monitoring:
            self.start_monitoring()
        
        self.logger.info("✅ GPU Performance Monitor initialized")
    
    def update_metrics(self):
        """Collect and update GPU performance metrics (called by BaseAdvancedSystem monitor loop)."""
        metrics = self._collect_metrics()
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 entries
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)
        
        # Feed into BaseAdvancedMetrics
        mem_pct = metrics.get('memory_usage_percent', 0.0)
        self.metrics.throughput = 1.0 - mem_pct  # Inverse: less memory used = more available throughput
        self.metrics.efficiency = 1.0 if mem_pct < self.config.memory_threshold else 0.5
        self.metrics.stability = 1.0
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect GPU performance metrics."""
        metrics = {
            'timestamp': time.time(),
            'device_id': self.config.device_id
        }
        
        if torch.cuda.is_available() and self.config.device == "cuda":
            total_mem = torch.cuda.get_device_properties(self.config.device_id).total_memory
            alloc_mem = torch.cuda.memory_allocated(self.config.device_id)
            metrics.update({
                'memory_allocated': alloc_mem,
                'memory_cached': torch.cuda.memory_reserved(self.config.device_id),
                'memory_usage_percent': alloc_mem / total_mem if total_mem > 0 else 0.0,
                'utilization': 0.0,  # Would use pynvml in practice
                'temperature': 0.0,  # Would use pynvml in practice
                'power_usage': 0.0   # Would use pynvml in practice
            })
        
        return metrics
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.current_metrics.copy()
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get performance metrics history."""
        return self.metrics_history.copy()
    
    def get_average_metrics(self) -> Dict[str, Any]:
        """Get average performance metrics."""
        if not self.metrics_history:
            return {}
        
        avg_metrics = {}
        # Assumes dict keys are uniform
        for key in self.metrics_history[0].keys():
            if isinstance(self.metrics_history[0][key], (int, float)):
                values = [m[key] for m in self.metrics_history]
                avg_metrics[f'avg_{key}'] = float(np.mean(values))
                avg_metrics[f'min_{key}'] = float(np.min(values))
                avg_metrics[f'max_{key}'] = float(np.max(values))
        
        return avg_metrics

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get combined GPU stats including BaseAdvancedSystem base stats."""
        base_stats = self.get_stats()
        return {
            **base_stats,
            "current_metrics": self.current_metrics,
            "history_length": len(self.metrics_history),
        }
