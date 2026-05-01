"""
TruthGPT Monitoring Module
Advanced monitoring and profiling utilities for TruthGPT models
"""

import torch
import torch.nn as nn
import psutil
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
from collections import deque
import json
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TruthGPTMonitoringConfig:
    """Configuration for TruthGPT monitoring."""
    # Monitoring settings
    enable_system_monitoring: bool = True
    enable_gpu_monitoring: bool = True
    enable_model_monitoring: bool = True
    enable_performance_monitoring: bool = True
    
    # Monitoring intervals
    system_monitoring_interval: float = 1.0
    gpu_monitoring_interval: float = 0.5
    model_monitoring_interval: float = 1.0
    performance_monitoring_interval: float = 0.1
    
    # Data retention
    max_history_size: int = 1000
    enable_data_compression: bool = True
    
    # Logging and alerts
    log_level: str = "INFO"
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 90.0,
        'memory_usage': 90.0,
        'gpu_usage': 90.0,
        'gpu_memory_usage': 90.0
    })
    
    # Export settings
    enable_export: bool = True
    export_format: str = "json"  # json, csv, pickle
    export_path: str = "./monitoring_data"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enable_system_monitoring': self.enable_system_monitoring,
            'enable_gpu_monitoring': self.enable_gpu_monitoring,
            'enable_model_monitoring': self.enable_model_monitoring,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'system_monitoring_interval': self.system_monitoring_interval,
            'gpu_monitoring_interval': self.gpu_monitoring_interval,
            'model_monitoring_interval': self.model_monitoring_interval,
            'performance_monitoring_interval': self.performance_monitoring_interval,
            'max_history_size': self.max_history_size,
            'enable_data_compression': self.enable_data_compression,
            'log_level': self.log_level,
            'enable_alerts': self.enable_alerts,
            'alert_thresholds': self.alert_thresholds,
            'enable_export': self.enable_export,
            'export_format': self.export_format,
            'export_path': self.export_path
        }

@dataclass
class TruthGPTMonitoringData:
    """Container for TruthGPT monitoring data."""
    # System metrics
    timestamp: float = field(default_factory=time.time)
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    
    # GPU metrics
    gpu_usage_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_temperature_c: float = 0.0
    
    # Model metrics
    model_parameters: int = 0
    model_size_mb: float = 0.0
    model_inference_time_ms: float = 0.0
    model_throughput: float = 0.0
    
    # Performance metrics
    training_loss: float = 0.0
    validation_loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_percent': self.memory_usage_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'gpu_usage_percent': self.gpu_usage_percent,
            'gpu_memory_used_mb': self.gpu_memory_used_mb,
            'gpu_memory_total_mb': self.gpu_memory_total_mb,
            'gpu_temperature_c': self.gpu_temperature_c,
            'model_parameters': self.model_parameters,
            'model_size_mb': self.model_size_mb,
            'model_inference_time_ms': self.model_inference_time_ms,
            'model_throughput': self.model_throughput,
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss,
            'learning_rate': self.learning_rate,
            'gradient_norm': self.gradient_norm,
            'custom_metrics': self.custom_metrics
        }

class TruthGPTMonitor:
    """Advanced monitor for TruthGPT models."""
    
    def __init__(self, config: TruthGPTMonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Monitoring state
        self.monitoring_data = deque(maxlen=config.max_history_size)
        self.monitoring_threads = {}
        self.monitoring_active = False
        
        # Alert system
        self.alert_history = []
        self.alert_callbacks = []
        
        self.logger.info("TruthGPT Monitor initialized")
    
    def start_monitoring(self, model: Optional[nn.Module] = None) -> None:
        """Start monitoring TruthGPT model."""
        self.logger.info("🚀 Starting TruthGPT monitoring")
        
        self.monitoring_active = True
        
        # Start system monitoring
        if self.config.enable_system_monitoring:
            self._start_system_monitoring()
        
        # Start GPU monitoring
        if self.config.enable_gpu_monitoring and torch.cuda.is_available():
            self._start_gpu_monitoring()
        
        # Start model monitoring
        if self.config.enable_model_monitoring and model is not None:
            self._start_model_monitoring(model)
        
        # Start performance monitoring
        if self.config.enable_performance_monitoring:
            self._start_performance_monitoring()
        
        self.logger.info("✅ TruthGPT monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.logger.info("🛑 Stopping TruthGPT monitoring")
        
        self.monitoring_active = False
        
        # Stop all monitoring threads
        for thread_name, thread in self.monitoring_threads.items():
            if thread.is_alive():
                thread.join()
        
        self.monitoring_threads.clear()
        
        self.logger.info("✅ TruthGPT monitoring stopped")
    
    def _start_system_monitoring(self) -> None:
        """Start system monitoring thread."""
        def system_monitor():
            while self.monitoring_active:
                try:
                    # Get system metrics
                    cpu_usage = psutil.cpu_percent(interval=None)
                    memory = psutil.virtual_memory()
                    
                    # Create monitoring data
                    data = TruthGPTMonitoringData(
                        cpu_usage_percent=cpu_usage,
                        memory_usage_percent=memory.percent,
                        memory_used_mb=memory.used / (1024 * 1024),
                        memory_available_mb=memory.available / (1024 * 1024)
                    )
                    
                    # Add to history
                    self.monitoring_data.append(data)
                    
                    # Check alerts
                    self._check_alerts(data)
                    
                except Exception as e:
                    self.logger.error(f"System monitoring error: {e}")
                
                time.sleep(self.config.system_monitoring_interval)
        
        thread = threading.Thread(target=system_monitor, daemon=True)
        thread.start()
        self.monitoring_threads['system'] = thread
    
    def _start_gpu_monitoring(self) -> None:
        """Start GPU monitoring thread."""
        def gpu_monitor():
            while self.monitoring_active:
                try:
                    if torch.cuda.is_available():
                        # Get GPU metrics
                        gpu_memory_allocated = torch.cuda.memory_allocated()
                        gpu_memory_reserved = torch.cuda.memory_reserved()
                        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                        
                        # Calculate GPU usage (simplified)
                        gpu_usage = (gpu_memory_allocated / gpu_memory_total) * 100
                        
                        # Create monitoring data
                        data = TruthGPTMonitoringData(
                            gpu_usage_percent=gpu_usage,
                            gpu_memory_used_mb=gpu_memory_allocated / (1024 * 1024),
                            gpu_memory_total_mb=gpu_memory_total / (1024 * 1024)
                        )
                        
                        # Add to history
                        self.monitoring_data.append(data)
                        
                        # Check alerts
                        self._check_alerts(data)
                    
                except Exception as e:
                    self.logger.error(f"GPU monitoring error: {e}")
                
                time.sleep(self.config.gpu_monitoring_interval)
        
        thread = threading.Thread(target=gpu_monitor, daemon=True)
        thread.start()
        self.monitoring_threads['gpu'] = thread
    
    def _start_model_monitoring(self, model: nn.Module) -> None:
        """Start model monitoring thread."""
        def model_monitor():
            while self.monitoring_active:
                try:
                    # Get model metrics
                    model_parameters = sum(p.numel() for p in model.parameters())
                    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
                    
                    # Create monitoring data
                    data = TruthGPTMonitoringData(
                        model_parameters=model_parameters,
                        model_size_mb=model_size
                    )
                    
                    # Add to history
                    self.monitoring_data.append(data)
                    
                except Exception as e:
                    self.logger.error(f"Model monitoring error: {e}")
                
                time.sleep(self.config.model_monitoring_interval)
        
        thread = threading.Thread(target=model_monitor, daemon=True)
        thread.start()
        self.monitoring_threads['model'] = thread
    
    def _start_performance_monitoring(self) -> None:
        """Start performance monitoring thread."""
        def performance_monitor():
            while self.monitoring_active:
                try:
                    # Get performance metrics
                    # This would typically measure training/inference performance
                    # For now, we'll create placeholder data
                    
                    data = TruthGPTMonitoringData(
                        model_inference_time_ms=0.0,
                        model_throughput=0.0
                    )
                    
                    # Add to history
                    self.monitoring_data.append(data)
                    
                except Exception as e:
                    self.logger.error(f"Performance monitoring error: {e}")
                
                time.sleep(self.config.performance_monitoring_interval)
        
        thread = threading.Thread(target=performance_monitor, daemon=True)
        thread.start()
        self.monitoring_threads['performance'] = thread
    
    def _check_alerts(self, data: TruthGPTMonitoringData) -> None:
        """Check for alert conditions."""
        if not self.config.enable_alerts:
            return
        
        # Check CPU usage
        if data.cpu_usage_percent > self.config.alert_thresholds['cpu_usage']:
            self._trigger_alert('high_cpu_usage', data.cpu_usage_percent)
        
        # Check memory usage
        if data.memory_usage_percent > self.config.alert_thresholds['memory_usage']:
            self._trigger_alert('high_memory_usage', data.memory_usage_percent)
        
        # Check GPU usage
        if data.gpu_usage_percent > self.config.alert_thresholds['gpu_usage']:
            self._trigger_alert('high_gpu_usage', data.gpu_usage_percent)
        
        # Check GPU memory usage
        if data.gpu_memory_used_mb > 0 and (data.gpu_memory_used_mb / data.gpu_memory_total_mb) * 100 > self.config.alert_thresholds['gpu_memory_usage']:
            self._trigger_alert('high_gpu_memory_usage', (data.gpu_memory_used_mb / data.gpu_memory_total_mb) * 100)
    
    def _trigger_alert(self, alert_type: str, value: float) -> None:
        """Trigger an alert."""
        alert = {
            'type': alert_type,
            'value': value,
            'timestamp': time.time(),
            'threshold': self.config.alert_thresholds.get(alert_type, 0.0)
        }
        
        self.alert_history.append(alert)
        self.logger.warning(f"🚨 Alert: {alert_type} = {value:.2f}% (threshold: {alert['threshold']:.2f}%)")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_monitoring_data(self) -> List[TruthGPTMonitoringData]:
        """Get monitoring data."""
        return list(self.monitoring_data)
    
    def get_latest_data(self) -> Optional[TruthGPTMonitoringData]:
        """Get latest monitoring data."""
        return self.monitoring_data[-1] if self.monitoring_data else None
    
    def get_alert_history(self) -> List[Dict[str, Any]]:
        """Get alert history."""
        return self.alert_history
    
    def export_monitoring_data(self, filepath: Optional[str] = None) -> str:
        """Export monitoring data."""
        if not self.config.enable_export:
            return ""
        
        # Determine filepath
        if filepath is None:
            export_path = Path(self.config.export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            filepath = export_path / f"monitoring_data_{int(time.time())}.{self.config.export_format}"
        
        # Export data
        if self.config.export_format == "json":
            with open(filepath, 'w') as f:
                json.dump([data.to_dict() for data in self.monitoring_data], f, indent=2)
        elif self.config.export_format == "csv":
            import pandas as pd
            df = pd.DataFrame([data.to_dict() for data in self.monitoring_data])
            df.to_csv(filepath, index=False)
        elif self.config.export_format == "pickle":
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(list(self.monitoring_data), f)
        
        self.logger.info(f"Monitoring data exported to {filepath}")
        return str(filepath)

class TruthGPTProfiler:
    """Advanced profiler for TruthGPT models."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.profiling_data = {}
    
    def profile_model(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Profile TruthGPT model."""
        self.logger.info("🔍 Profiling TruthGPT model")
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Profile forward pass
        start_time = time.time()
        with torch.no_grad():
            output = model(dummy_input)
        forward_time = time.time() - start_time
        
        # Calculate model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        # Calculate FLOPs (simplified)
        flops = self._calculate_flops(model, input_shape)
        
        # Create profiling data
        profiling_data = {
            'model_size_mb': model_size,
            'forward_time_ms': forward_time * 1000,
            'flops': flops,
            'input_shape': input_shape,
            'output_shape': output.shape,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        self.profiling_data = profiling_data
        self.logger.info(f"✅ Model profiling completed - Size: {model_size:.2f}MB, Time: {forward_time*1000:.2f}ms")
        
        return profiling_data
    
    def _calculate_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """Calculate FLOPs for model."""
        # Simplified FLOP calculation
        # In practice, you would use a proper FLOP counter
        total_params = sum(p.numel() for p in model.parameters())
        return total_params * 2  # Rough estimate
    
    def get_profiling_data(self) -> Dict[str, Any]:
        """Get profiling data."""
        return self.profiling_data

class TruthGPTLogger:
    """Advanced logger for TruthGPT models."""
    
    def __init__(self, name: str = "TruthGPT", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Setup formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Setup file handler
        file_handler = logging.FileHandler('truthgpt.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_training_step(self, step: int, loss: float, lr: float, **kwargs) -> None:
        """Log training step."""
        self.logger.info(f"Step {step}: Loss={loss:.4f}, LR={lr:.2e}, {kwargs}")
    
    def log_evaluation(self, metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics."""
        self.logger.info(f"Evaluation: {metrics}")
    
    def log_inference(self, input_text: str, output_text: str, metrics: Dict[str, Any]) -> None:
        """Log inference results."""
        self.logger.info(f"Inference: Input='{input_text[:50]}...', Output='{output_text[:50]}...', {metrics}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log error."""
        self.logger.error(f"Error in {context}: {error}")

# Factory functions
def create_truthgpt_monitor(config: TruthGPTMonitoringConfig) -> TruthGPTMonitor:
    """Create TruthGPT monitor."""
    return TruthGPTMonitor(config)

def create_truthgpt_profiler() -> TruthGPTProfiler:
    """Create TruthGPT profiler."""
    return TruthGPTProfiler()

def create_truthgpt_logger(name: str = "TruthGPT", level: str = "INFO") -> TruthGPTLogger:
    """Create TruthGPT logger."""
    return TruthGPTLogger(name, level)

# Example usage
if __name__ == "__main__":
    # Example TruthGPT monitoring
    print("🚀 TruthGPT Monitoring Demo")
    print("=" * 50)
    
    # Create a sample TruthGPT-style model
    class TruthGPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 10000)
        
        def forward(self, x):
            return self.linear(x)
    
    # Create model
    model = TruthGPTModel()
    
    # Create monitoring configuration
    config = TruthGPTMonitoringConfig(
        enable_system_monitoring=True,
        enable_gpu_monitoring=True,
        enable_model_monitoring=True,
        system_monitoring_interval=1.0,
        gpu_monitoring_interval=0.5,
        model_monitoring_interval=1.0
    )
    
    # Create monitor
    monitor = create_truthgpt_monitor(config)
    
    # Start monitoring
    monitor.start_monitoring(model)
    
    # Let it run for a bit
    time.sleep(5)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Get monitoring data
    data = monitor.get_monitoring_data()
    print(f"Collected {len(data)} monitoring samples")
    
    # Export data
    if config.enable_export:
        export_path = monitor.export_monitoring_data()
        print(f"Monitoring data exported to {export_path}")
    
    print("✅ TruthGPT monitoring completed!")



