"""
TruthGPT Monitoring Utilities
Advanced monitoring and analytics for TruthGPT models
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pydantic import Field
from contextlib import contextmanager
import time
from pathlib import Path
import json
import threading
import queue
from enum import Enum

logger = logging.getLogger(__name__)

from .base import BaseOptimizationModel, system_metrics_collector


class TruthGPTMetrics(BaseOptimizationModel):
    """TruthGPT metrics container."""
    # Performance metrics
    inference_time: float = 0.0
    throughput: float = 0.0
    latency: float = 0.0
    
    # Memory metrics
    memory_used_mb: float = 0.0
    memory_peak_mb: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    
    # Model metrics
    model_size_mb: float = 0.0
    parameter_count: int = 0
    trainable_parameters: int = 0
    
    # Quality metrics
    accuracy: float = 0.0
    loss: float = 0.0
    perplexity: float = 0.0
    
    # System metrics
    cpu_usage: float = 0.0
    gpu_utilization: float = 0.0
    temperature: float = 0.0
    
    # Metadata
    timestamp: float = Field(default_factory=time.time)
    model_name: str = "truthgpt"
    device: str = "cpu"

class TruthGPTMonitor:
    """TruthGPT monitoring utilities."""
    
    def __init__(self, model_name: str = "truthgpt", enable_gpu_monitoring: bool = True):
        self.model_name = model_name
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Metrics storage
        self.metrics_history: List[TruthGPTMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
        
        # Performance tracking
        self.performance_tracker = {}
        self.memory_tracker = {}
        self.gpu_tracker = {}
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start TruthGPT monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("🔍 TruthGPT monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop TruthGPT monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("🔍 TruthGPT monitoring stopped")
    
    def _monitor_loop(self, interval: float) -> None:
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                self.metrics_queue.put(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> TruthGPTMetrics:
        """Collect TruthGPT metrics."""
        # Use helper for standardized system metrics
        sys_metrics = system_metrics_collector()
        
        # GPU metrics
        gpu_utilization = 0.0
        gpu_memory_used = 0.0
        gpu_memory_peak = 0.0
        temperature = 0.0
        
        if self.enable_gpu_monitoring:
            gpu_memory_used = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_memory_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            if hasattr(torch.cuda, 'utilization'):
                gpu_utilization = torch.cuda.utilization()
        
        return TruthGPTMetrics(
            memory_used_mb=sys_metrics["memory_used_gb"] * 1024,
            memory_peak_mb=sys_metrics["memory_used_gb"] * 1024,
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_peak_mb=gpu_memory_peak,
            cpu_usage=sys_metrics["cpu_percent"],
            gpu_utilization=gpu_utilization,
            temperature=temperature,
            model_name=self.model_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def monitor_model_inference(self, model: nn.Module, input_tensor: torch.Tensor) -> TruthGPTMetrics:
        """Monitor model inference."""
        self.logger.info("📊 Monitoring TruthGPT model inference")
        
        # Get model metrics
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        # Measure inference time
        model.eval()
        with torch.no_grad():
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            output = model(input_tensor)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
        
        # Calculate metrics
        inference_time = end_time - start_time
        throughput = input_tensor.size(0) / inference_time
        latency = inference_time * 1000  # Convert to milliseconds
        
        # Get memory usage
        memory_used = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        memory_peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        
        metrics = TruthGPTMetrics(
            inference_time=inference_time,
            throughput=throughput,
            latency=latency,
            memory_used_mb=memory_used,
            memory_peak_mb=memory_peak,
            gpu_memory_used_mb=memory_used,
            gpu_memory_peak_mb=memory_peak,
            model_size_mb=(param_size + buffer_size) / (1024 * 1024),
            parameter_count=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            model_name=self.model_name,
            device=str(input_tensor.device)
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_metrics_history(self) -> List[TruthGPTMetrics]:
        """Get metrics history."""
        return self.metrics_history
    
    def get_latest_metrics(self) -> Optional[TruthGPTMetrics]:
        """Get latest metrics."""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None
    
    def save_metrics(self, filepath: str) -> None:
        """Save metrics to file."""
        metrics_data = [metrics.model_dump() for metrics in self.metrics_history]
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        self.logger.info(f"📊 Metrics saved to {filepath}")
    
    def load_metrics(self, filepath: str) -> None:
        """Load metrics from file."""
        with open(filepath, 'r') as f:
            metrics_data = json.load(f)
        
        self.metrics_history = []
        for data in metrics_data:
            metrics = TruthGPTMetrics(**data)
            self.metrics_history.append(metrics)
        self.logger.info(f"📊 Metrics loaded from {filepath}")

class TruthGPTAnalytics:
    """TruthGPT analytics utilities."""
    
    def __init__(self, monitor: TruthGPTMonitor):
        self.monitor = monitor
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.analytics_results = {}
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze TruthGPT performance."""
        self.logger.info("📈 Analyzing TruthGPT performance")
        
        if not self.monitor.metrics_history:
            return {}
        
        # Extract performance metrics
        inference_times = [m.inference_time for m in self.monitor.metrics_history]
        throughputs = [m.throughput for m in self.monitor.metrics_history]
        latencies = [m.latency for m in self.monitor.metrics_history]
        
        # Calculate statistics
        performance_stats = {
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'avg_throughput': np.mean(throughputs),
            'std_throughput': np.std(throughputs),
            'min_throughput': np.min(throughputs),
            'max_throughput': np.max(throughputs),
            'avg_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies)
        }
        
        self.analytics_results['performance'] = performance_stats
        return performance_stats
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze TruthGPT memory usage."""
        self.logger.info("🧠 Analyzing TruthGPT memory usage")
        
        if not self.monitor.metrics_history:
            return {}
        
        # Extract memory metrics
        memory_used = [m.memory_used_mb for m in self.monitor.metrics_history]
        memory_peak = [m.memory_peak_mb for m in self.monitor.metrics_history]
        gpu_memory_used = [m.gpu_memory_used_mb for m in self.monitor.metrics_history]
        gpu_memory_peak = [m.gpu_memory_peak_mb for m in self.monitor.metrics_history]
        
        # Calculate statistics
        memory_stats = {
            'avg_memory_used_mb': np.mean(memory_used),
            'std_memory_used_mb': np.std(memory_used),
            'max_memory_used_mb': np.max(memory_used),
            'avg_memory_peak_mb': np.mean(memory_peak),
            'std_memory_peak_mb': np.std(memory_peak),
            'max_memory_peak_mb': np.max(memory_peak),
            'avg_gpu_memory_used_mb': np.mean(gpu_memory_used),
            'std_gpu_memory_used_mb': np.std(gpu_memory_used),
            'max_gpu_memory_used_mb': np.max(gpu_memory_used),
            'avg_gpu_memory_peak_mb': np.mean(gpu_memory_peak),
            'std_gpu_memory_peak_mb': np.std(gpu_memory_peak),
            'max_gpu_memory_peak_mb': np.max(gpu_memory_peak)
        }
        
        self.analytics_results['memory'] = memory_stats
        return memory_stats
    
    def analyze_system_metrics(self) -> Dict[str, Any]:
        """Analyze system metrics."""
        self.logger.info("🖥️ Analyzing system metrics")
        
        if not self.monitor.metrics_history:
            return {}
        
        # Extract system metrics
        cpu_usage = [m.cpu_usage for m in self.monitor.metrics_history]
        gpu_utilization = [m.gpu_utilization for m in self.monitor.metrics_history]
        temperature = [m.temperature for m in self.monitor.metrics_history]
        
        # Calculate statistics
        system_stats = {
            'avg_cpu_usage': np.mean(cpu_usage),
            'std_cpu_usage': np.std(cpu_usage),
            'max_cpu_usage': np.max(cpu_usage),
            'avg_gpu_utilization': np.mean(gpu_utilization),
            'std_gpu_utilization': np.std(gpu_utilization),
            'max_gpu_utilization': np.max(gpu_utilization),
            'avg_temperature': np.mean(temperature),
            'std_temperature': np.std(temperature),
            'max_temperature': np.max(temperature)
        }
        
        self.analytics_results['system'] = system_stats
        return system_stats
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        self.logger.info("📊 Generating TruthGPT analytics report")
        
        # Run all analyses
        performance_analysis = self.analyze_performance()
        memory_analysis = self.analyze_memory_usage()
        system_analysis = self.analyze_system_metrics()
        
        # Generate report
        report = {
            'timestamp': time.time(),
            'model_name': self.monitor.model_name,
            'total_metrics': len(self.monitor.metrics_history),
            'performance': performance_analysis,
            'memory': memory_analysis,
            'system': system_analysis,
            'summary': self._generate_summary(performance_analysis, memory_analysis, system_analysis)
        }
        
        self.analytics_results['report'] = report
        return report
    
    def _generate_summary(self, performance: Dict[str, Any], 
                         memory: Dict[str, Any], 
                         system: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of analytics."""
        return {
            'performance_score': self._calculate_performance_score(performance),
            'memory_efficiency': self._calculate_memory_efficiency(memory),
            'system_health': self._calculate_system_health(system),
            'overall_score': self._calculate_overall_score(performance, memory, system)
        }
    
    def _calculate_performance_score(self, performance: Dict[str, Any]) -> float:
        """Calculate performance score."""
        if not performance:
            return 0.0
        
        # Higher throughput and lower latency = better score
        throughput_score = min(performance.get('avg_throughput', 0) / 100, 1.0)
        latency_score = max(0, 1.0 - performance.get('avg_latency', 1000) / 1000)
        
        return (throughput_score + latency_score) / 2.0
    
    def _calculate_memory_efficiency(self, memory: Dict[str, Any]) -> float:
        """Calculate memory efficiency score."""
        if not memory:
            return 0.0
        
        # Lower memory usage = better efficiency
        memory_score = max(0, 1.0 - memory.get('avg_memory_used_mb', 1000) / 10000)
        return memory_score
    
    def _calculate_system_health(self, system: Dict[str, Any]) -> float:
        """Calculate system health score."""
        if not system:
            return 0.0
        
        # Lower CPU usage and temperature = better health
        cpu_score = max(0, 1.0 - system.get('avg_cpu_usage', 100) / 100)
        temp_score = max(0, 1.0 - system.get('avg_temperature', 100) / 100)
        
        return (cpu_score + temp_score) / 2.0
    
    def _calculate_overall_score(self, performance: Dict[str, Any], 
                                memory: Dict[str, Any], 
                                system: Dict[str, Any]) -> float:
        """Calculate overall score."""
        perf_score = self._calculate_performance_score(performance)
        mem_score = self._calculate_memory_efficiency(memory)
        sys_score = self._calculate_system_health(system)
        
        return (perf_score + mem_score + sys_score) / 3.0
    
    def get_analytics_results(self) -> Dict[str, Any]:
        """Get analytics results."""
        return self.analytics_results

class TruthGPTDashboard:
    """TruthGPT dashboard utilities."""
    
    def __init__(self, monitor: TruthGPTMonitor, analytics: TruthGPTAnalytics):
        self.monitor = monitor
        self.analytics = analytics
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.dashboard_data = {}
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate dashboard data."""
        self.logger.info("📊 Generating TruthGPT dashboard data")
        
        # Get latest metrics
        latest_metrics = self.monitor.get_latest_metrics()
        
        # Generate analytics report
        analytics_report = self.analytics.generate_report()
        
        # Create dashboard data
        dashboard_data = {
            'timestamp': time.time(),
            'model_name': self.monitor.model_name,
            'latest_metrics': latest_metrics.model_dump() if latest_metrics else {},
            'analytics_report': analytics_report,
            'metrics_history': [m.model_dump() for m in self.monitor.metrics_history[-100:]],  # Last 100 metrics
            'summary': self._generate_dashboard_summary(latest_metrics, analytics_report)
        }
        
        self.dashboard_data = dashboard_data
        return dashboard_data
    
    def _generate_dashboard_summary(self, latest_metrics: Optional[TruthGPTMetrics], 
                                  analytics_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dashboard summary."""
        if not latest_metrics:
            return {}
        
        summary = {
            'status': 'healthy' if latest_metrics.cpu_usage < 80 else 'warning',
            'performance': {
                'inference_time': latest_metrics.inference_time,
                'throughput': latest_metrics.throughput,
                'latency': latest_metrics.latency
            },
            'memory': {
                'memory_used_mb': latest_metrics.memory_used_mb,
                'gpu_memory_used_mb': latest_metrics.gpu_memory_used_mb
            },
            'system': {
                'cpu_usage': latest_metrics.cpu_usage,
                'gpu_utilization': latest_metrics.gpu_utilization,
                'temperature': latest_metrics.temperature
            }
        }
        
        return summary
    
    def save_dashboard_data(self, filepath: str) -> None:
        """Save dashboard data to file."""
        dashboard_data = self.generate_dashboard_data()
        with open(filepath, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        self.logger.info(f"📊 Dashboard data saved to {filepath}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data."""
        return self.dashboard_data

# Factory functions
def create_truthgpt_monitoring_suite(model_name: str = "truthgpt", 
                                   enable_gpu_monitoring: bool = True) -> Tuple[TruthGPTMonitor, TruthGPTAnalytics, TruthGPTDashboard]:
    """Create TruthGPT monitoring suite."""
    monitor = TruthGPTMonitor(model_name, enable_gpu_monitoring)
    analytics = TruthGPTAnalytics(monitor)
    dashboard = TruthGPTDashboard(monitor, analytics)
    
    return monitor, analytics, dashboard

def quick_truthgpt_monitoring_setup(model: nn.Module, 
                                  input_tensor: torch.Tensor,
                                  model_name: str = "truthgpt") -> Tuple[TruthGPTMonitor, TruthGPTMetrics]:
    """Quick TruthGPT monitoring setup."""
    monitor = TruthGPTMonitor(model_name)
    metrics = monitor.monitor_model_inference(model, input_tensor)
    
    return monitor, metrics

# Context managers
@contextmanager
def truthgpt_monitoring_context(model: nn.Module, input_tensor: torch.Tensor, model_name: str = "truthgpt"):
    """Context manager for TruthGPT monitoring."""
    monitor = TruthGPTMonitor(model_name)
    try:
        yield monitor
    finally:
        # Monitor inference
        monitor.monitor_model_inference(model, input_tensor)

# Example usage
if __name__ == "__main__":
    # Example TruthGPT monitoring
    print("📊 TruthGPT Monitoring Demo")
    print("=" * 50)
    
    # Create a sample TruthGPT-style model
    class TruthGPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10000, 768)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(768, 12, 3072, dropout=0.1),
                num_layers=12
            )
            self.lm_head = nn.Linear(768, 10000)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            x = self.lm_head(x)
            return x
    
    # Create model and input
    model = TruthGPTModel()
    input_tensor = torch.randint(0, 10000, (32, 512))
    
    # Setup monitoring
    monitor, analytics, dashboard = create_truthgpt_monitoring_suite("demo_truthgpt")
    
    # Monitor inference
    with truthgpt_monitoring_context(model, input_tensor, "demo_truthgpt") as monitor:
        # Simulate inference
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
    
    # Generate analytics
    analytics_report = analytics.generate_report()
    print(f"Analytics report: {analytics_report}")
    
    # Generate dashboard
    dashboard_data = dashboard.generate_dashboard_data()
    print(f"Dashboard data: {dashboard_data}")
    
    print("✅ TruthGPT monitoring completed!")
