"""
Utility functions - Refactored utility components
"""

import torch
import torch.nn as nn
import psutil
import time
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import threading
from contextlib import contextmanager
import gc

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    cpu_usage: float
    memory_usage: float
    gpu_memory_usage: float
    gpu_utilization: float
    disk_usage: float
    timestamp: float

class PerformanceUtils:
    """Performance measurement utilities."""
    
    @staticmethod
    def measure_time(func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure execution time of a function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    @staticmethod
    def get_system_metrics() -> PerformanceMetrics:
        """Get current system performance metrics."""
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
        
        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_memory_usage=gpu_memory_usage,
            gpu_utilization=gpu_utilization,
            disk_usage=disk_usage,
            timestamp=time.time()
        )
    
    @staticmethod
    def benchmark_model_forward(model: nn.Module, input_tensor: torch.Tensor, 
                              iterations: int = 100, warmup: int = 10) -> Dict[str, float]:
        """Benchmark model forward pass."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_tensor)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                start_time = time.time()
                _ = model(input_tensor)
                times.append(time.time() - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput': 1.0 / np.mean(times)
        }
    
    @staticmethod
    def profile_memory_usage(func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """Profile memory usage of a function."""
        # Get initial memory
        initial_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final memory
        final_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        
        return result, {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_delta_mb': final_memory - initial_memory
        }

class MemoryUtils:
    """Memory management utilities."""
    
    @staticmethod
    def get_model_memory_usage(model: nn.Module) -> Dict[str, float]:
        """Get memory usage of a model."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        return {
            'parameters_mb': param_size / (1024 * 1024),
            'buffers_mb': buffer_size / (1024 * 1024),
            'total_mb': (param_size + buffer_size) / (1024 * 1024)
        }
    
    @staticmethod
    def get_parameter_count(model: nn.Module) -> Dict[str, int]:
        """Get parameter count statistics."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
    
    @staticmethod
    def estimate_model_size(model: nn.Module, precision: str = 'fp32') -> float:
        """Estimate model size in MB."""
        bytes_per_param = {
            'fp32': 4,
            'fp16': 2,
            'int8': 1,
            'int4': 0.5
        }
        
        param_count = sum(p.numel() for p in model.parameters())
        bytes_per_element = bytes_per_param.get(precision, 4)
        
        return (param_count * bytes_per_element) / (1024 * 1024)
    
    @staticmethod
    def clear_gpu_cache():
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """Get GPU memory information."""
        if not torch.cuda.is_available():
            return {'available': False}
        
        return {
            'available': True,
            'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'cached_mb': torch.cuda.memory_reserved() / (1024 * 1024),
            'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
            'utilization_percent': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
        }

class GPUUtils:
    """GPU-specific utilities."""
    
    @staticmethod
    def get_gpu_count() -> int:
        """Get number of available GPUs."""
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    @staticmethod
    def get_current_device() -> int:
        """Get current GPU device."""
        return torch.cuda.current_device() if torch.cuda.is_available() else -1
    
    @staticmethod
    def set_device(device_id: int):
        """Set GPU device."""
        if torch.cuda.is_available() and 0 <= device_id < torch.cuda.device_count():
            torch.cuda.set_device(device_id)
        else:
            logger.warning(f"GPU device {device_id} not available")
    
    @staticmethod
    def get_device_properties(device_id: int = None) -> Dict[str, Any]:
        """Get GPU device properties."""
        if not torch.cuda.is_available():
            return {'available': False}
        
        if device_id is None:
            device_id = torch.cuda.current_device()
        
        if device_id >= torch.cuda.device_count():
            return {'available': False}
        
        props = torch.cuda.get_device_properties(device_id)
        
        return {
            'available': True,
            'device_id': device_id,
            'name': props.name,
            'total_memory_mb': props.total_memory / (1024 * 1024),
            'major': props.major,
            'minor': props.minor,
            'multi_processor_count': props.multi_processor_count
        }
    
    @staticmethod
    def optimize_for_gpu(model: nn.Module, device: str = 'cuda') -> nn.Module:
        """Optimize model for GPU execution."""
        if torch.cuda.is_available():
            model = model.to(device)
            
            # Enable optimizations
            if hasattr(model, 'half'):
                model = model.half()  # Use FP16
            
            # Enable cuDNN optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        return model

class ThreadUtils:
    """Threading utilities."""
    
    @staticmethod
    def run_in_thread(func, *args, **kwargs) -> threading.Thread:
        """Run function in a separate thread."""
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread
    
    @staticmethod
    def run_with_timeout(func, timeout: float, *args, **kwargs) -> Tuple[bool, Any]:
        """Run function with timeout."""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            return False, None  # Timeout
        
        if exception[0]:
            raise exception[0]
        
        return True, result[0]

class LoggingUtils:
    """Logging utilities."""
    
    @staticmethod
    def setup_logger(name: str, level: str = 'INFO', 
                    log_file: Optional[str] = None) -> logging.Logger:
        """Setup logger with proper formatting."""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def log_performance(logger: logging.Logger, operation: str, 
                       duration: float, **metrics):
        """Log performance metrics."""
        logger.info(f"Performance - {operation}: {duration:.3f}s")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

class ValidationUtils:
    """Validation utilities."""
    
    @staticmethod
    def validate_tensor(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Validate tensor properties."""
        if not isinstance(tensor, torch.Tensor):
            logger.error(f"{name} is not a torch.Tensor")
            return False
        
        if torch.isnan(tensor).any():
            logger.error(f"{name} contains NaN values")
            return False
        
        if torch.isinf(tensor).any():
            logger.error(f"{name} contains Inf values")
            return False
        
        return True
    
    @staticmethod
    def validate_model_shapes(model: nn.Module, input_shape: Tuple[int, ...]) -> bool:
        """Validate model can handle input shape."""
        try:
            test_input = torch.randn(1, *input_shape)
            with torch.no_grad():
                _ = model(test_input)
            return True
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

# Context managers
@contextmanager
def performance_context(operation_name: str, logger: Optional[logging.Logger] = None):
    """Context manager for performance measurement."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    start_time = time.time()
    start_memory = psutil.virtual_memory().used / (1024 * 1024)
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        end_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        logger.info(f"Performance - {operation_name}: {duration:.3f}s")
        logger.info(f"Memory delta: {end_memory - start_memory:.1f}MB")

@contextmanager
def gpu_context(device_id: int = 0):
    """Context manager for GPU operations."""
    if torch.cuda.is_available():
        old_device = torch.cuda.current_device()
        torch.cuda.set_device(device_id)
        try:
            yield
        finally:
            torch.cuda.set_device(old_device)
    else:
        yield

@contextmanager
def memory_context():
    """Context manager for memory cleanup."""
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Factory functions
def create_performance_utils() -> PerformanceUtils:
    """Create performance utils instance."""
    return PerformanceUtils()

def create_memory_utils() -> MemoryUtils:
    """Create memory utils instance."""
    return MemoryUtils()

def create_gpu_utils() -> GPUUtils:
    """Create GPU utils instance."""
    return GPUUtils()

