"""
GPU Accelerator Main System
"""
import time
import logging
from contextlib import contextmanager
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.cuda.amp as amp

from .config import GPUAcceleratorConfig
from .device import GPUDeviceManager
from .memory import GPUMemoryManager
from .optimizer import CUDAOptimizer
from .streams import GPUStreamManager
from .monitor import GPUPerformanceMonitor

logger = logging.getLogger(__name__)

class GPUAccelerator:
    """Ultra-advanced GPU accelerator with comprehensive features."""
    
    def __init__(self, config: GPUAcceleratorConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.device_manager = GPUDeviceManager(config)
        self.memory_manager = GPUMemoryManager(config)
        self.cuda_optimizer = CUDAOptimizer(config)
        
        # Only use GradScaler if AMP is enabled AND device is CUDA
        self.scaler = amp.GradScaler() if config.enable_amp and config.device == "cuda" else None
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'gpu_operations': 0,
            'optimization_time': 0.0,
            'peak_memory': 0.0
        }
        
        self.logger.info("✅ GPU Accelerator initialized")
    
    def optimize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor for GPU processing."""
        start_time = time.time()
        
        optimized = self.cuda_optimizer.optimize_tensor(tensor)
        
        # Update stats
        self.performance_stats['total_operations'] += 1
        self.performance_stats['gpu_operations'] += 1
        self.performance_stats['optimization_time'] += time.time() - start_time
        
        return optimized
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Optimize model for GPU execution."""
        start_time = time.time()
        
        optimized = self.cuda_optimizer.optimize_model(model)
        
        self.performance_stats['optimization_time'] += time.time() - start_time
        
        self.logger.info("Model optimized for GPU execution")
        return optimized
    
    def benchmark(self, model: nn.Module, input_tensor: torch.Tensor, 
                 num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark GPU performance."""
        model.eval()
        is_cuda = (self.config.device == "cuda" and torch.cuda.is_available())
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Benchmark
        if is_cuda:
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        if is_cuda:
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        
        stats = {
            'total_time': total_time,
            'average_time': total_time / num_runs if num_runs > 0 else 0,
            'throughput': num_runs / total_time if total_time > 0 else 0,
        }
        
        if is_cuda:
            stats['memory_allocated'] = torch.cuda.memory_allocated(self.config.device_id)
            stats['memory_cached'] = torch.cuda.memory_reserved(self.config.device_id)
        else:
            stats['memory_allocated'] = 0
            stats['memory_cached'] = 0
            
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'performance': self.performance_stats,
            'memory': self.memory_manager.get_stats(),
            'device': self.device_manager.get_device_info()
        }
    
    def cleanup(self):
        """Cleanup GPU resources."""
        self.memory_manager.clear_pool()
        
        if self.config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("GPU Accelerator cleanup completed")


class EnhancedGPUAccelerator(GPUAccelerator):
    """Enhanced GPU accelerator with streaming and monitoring."""
    
    def __init__(self, config: GPUAcceleratorConfig):
        super().__init__(config)
        
        # Initialize advanced features
        self.stream_manager = GPUStreamManager(config)
        self.performance_monitor = GPUPerformanceMonitor(config)
        
        # Initialize optimizer history
        self.optimization_history = []
        
        self.logger.info("✅ Enhanced GPU Accelerator initialized")
    
    def optimize_model_async(self, model: nn.Module, stream_index: int = 0) -> nn.Module:
        """Optimize model asynchronously using CUDA streams."""
        stream = self.stream_manager.get_stream(stream_index)
        
        if stream:
            with torch.cuda.stream(stream):
                optimized = self.optimize_model(model)
        else:
            optimized = self.optimize_model(model)
        
        return optimized
    
    def process_batch_async(self, batch: torch.Tensor, stream_index: int = 0) -> torch.Tensor:
        """Process batch asynchronously using CUDA streams."""
        stream = self.stream_manager.get_stream(stream_index)
        
        if stream:
            with torch.cuda.stream(stream):
                result = self.optimize_tensor(batch)
        else:
            result = self.optimize_tensor(batch)
        
        return result
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced performance statistics."""
        base_stats = super().get_stats()
        
        return {
            **base_stats,
            'current_metrics': self.performance_monitor.get_current_metrics(),
            'average_metrics': self.performance_monitor.get_average_metrics(),
            'optimization_history': self.optimization_history
        }
    
    def cleanup(self):
        """Cleanup enhanced GPU accelerator."""
        # Stop monitoring
        if self.performance_monitor.monitoring:
            self.performance_monitor.stop_monitoring()
        
        # Synchronize streams
        self.stream_manager.synchronize_all()
        
        # Call parent cleanup
        super().cleanup()
        
        self.logger.info("✅ Enhanced GPU Accelerator cleanup completed")

