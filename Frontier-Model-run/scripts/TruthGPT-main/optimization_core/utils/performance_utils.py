"""
Performance utilities for TruthGPT Optimization Core
Provides benchmarking, profiling, and optimization utilities

This module re-exports functionality from modules.base.core_system.core.performance_utils for backward compatibility.
New code should import directly from optimization_core.core.performance_utils.
"""

# Re-export from core module for backward compatibility
try:
    from optimization_core.core.performance_utils import (
        PerformanceMetrics,
        BenchmarkResult,
        PerformanceProfiler,
        profile_operation,
        benchmark_model,
        profile_model,
        optimize_model,
        create_optimization_report,
        measure_memory_usage,
    )
except ImportError:
    # Fallback implementation if core module not available
    import time
    import torch
    import torch.nn as nn
    import logging
    from typing import Dict, Any, Optional, List, Tuple, Union, Callable
    from contextlib import contextmanager
    import psutil
    import gc
    
    logger = logging.getLogger(__name__)

from .base import BaseOptimizationModel


class PerformanceMetrics(BaseOptimizationModel):
    """Performance metrics container."""
    inference_time: float
    memory_usage: float
    throughput: float
    latency: float
    gpu_utilization: Optional[float] = None
    cpu_utilization: Optional[float] = None


class BenchmarkResult(BaseOptimizationModel):
    """Benchmark result container."""

    model_name: str
    metrics: PerformanceMetrics
    configuration: Dict[str, Any]
    timestamp: float

class PerformanceProfiler:
    """Performance profiler for models and operations."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize performance profiler.
        
        Args:
            device: Device to profile on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results: List[BenchmarkResult] = []
    
    def profile_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
        warmup_runs: int = 10,
        batch_size: int = 1
    ) -> BenchmarkResult:
        """
        Profile a model's performance.
        
        Args:
            model: Model to profile
            input_shape: Input tensor shape
            num_runs: Number of profiling runs
            warmup_runs: Number of warmup runs
            batch_size: Batch size for profiling
            
        Returns:
            Benchmark result
        """
        model.eval()
        model.to(self.device)
        
        # Create input tensor
        input_tensor = torch.randn(batch_size, *input_shape, device=self.device)
        
        # Warmup runs
        logger.info(f"Running {warmup_runs} warmup runs...")
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        # Synchronize if using GPU
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Profile runs
        logger.info(f"Running {num_runs} profiling runs...")
        times = []
        memory_usage = []
        
        with torch.no_grad():
            for i in range(num_runs):
                # Clear cache
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                
                # Measure time
                start_time = time.time()
                _ = model(input_tensor)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                # Measure memory
                if self.device.type == "cuda":
                    memory_usage.append(torch.cuda.memory_allocated(self.device) / 1024 / 1024)
                else:
                    memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)
        
        # Calculate metrics
        avg_time = sum(times) / len(times)
        avg_memory = sum(memory_usage) / len(memory_usage)
        throughput = batch_size / avg_time
        latency = avg_time * 1000  # Convert to milliseconds
        
        metrics = PerformanceMetrics(
            inference_time=avg_time,
            memory_usage=avg_memory,
            throughput=throughput,
            latency=latency
        )
        
        result = BenchmarkResult(
            model_name=model.__class__.__name__,
            metrics=metrics,
            configuration={
                'input_shape': input_shape,
                'batch_size': batch_size,
                'num_runs': num_runs,
                'device': str(self.device)
            },
            timestamp=time.time()
        )
        
        self.results.append(result)
        return result
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
        warmup_runs: int = 10,
        batch_size: int = 1
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare performance of multiple models.
        
        Args:
            models: Dictionary of model names to models
            input_shape: Input tensor shape
            num_runs: Number of profiling runs
            warmup_runs: Number of warmup runs
            batch_size: Batch size for profiling
            
        Returns:
            Dictionary of benchmark results
        """
        results = {}
        
        for name, model in models.items():
            logger.info(f"Profiling model: {name}")
            result = self.profile_model(
                model, input_shape, num_runs, warmup_runs, batch_size
            )
            results[name] = result
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results."""
        if not self.results:
            return {}
        
        summary = {
            'total_benchmarks': len(self.results),
            'models_tested': list(set(r.model_name for r in self.results)),
            'average_inference_time': sum(r.metrics.inference_time for r in self.results) / len(self.results),
            'average_memory_usage': sum(r.metrics.memory_usage for r in self.results) / len(self.results),
            'average_throughput': sum(r.metrics.throughput for r in self.results) / len(self.results),
            'average_latency': sum(r.metrics.latency for r in self.results) / len(self.results)
        }
        
        return summary

@contextmanager
def profile_operation(operation_name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager for profiling operations.
    
    Args:
        operation_name: Name of the operation
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    logger.info(f"Starting operation: {operation_name}")
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        logger.info(f"Completed operation: {operation_name}")
        logger.info(f"  Duration: {duration:.4f} seconds")
        logger.info(f"  Memory delta: {memory_delta:.2f} MB")

def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_runs: int = 100,
    warmup_runs: int = 10,
    batch_size: int = 1,
    device: Optional[torch.device] = None
) -> BenchmarkResult:
    """
    Benchmark a model's performance.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        batch_size: Batch size for benchmarking
        device: Device to benchmark on
        
    Returns:
        Benchmark result
    """
    profiler = PerformanceProfiler(device)
    return profiler.profile_model(model, input_shape, num_runs, warmup_runs, batch_size)

def profile_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_runs: int = 100,
    warmup_runs: int = 10,
    batch_size: int = 1,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Profile a model and return detailed metrics.
    
    Args:
        model: Model to profile
        input_shape: Input tensor shape
        num_runs: Number of profiling runs
        warmup_runs: Number of warmup runs
        batch_size: Batch size for profiling
        device: Device to profile on
        
    Returns:
        Dictionary of profiling results
    """
    result = benchmark_model(model, input_shape, num_runs, warmup_runs, batch_size, device)
    
    return {
        'model_name': result.model_name,
        'inference_time': result.metrics.inference_time,
        'memory_usage': result.metrics.memory_usage,
        'throughput': result.metrics.throughput,
        'latency': result.metrics.latency,
        'configuration': result.configuration,
        'timestamp': result.timestamp
    }

def optimize_model(
    model: nn.Module,
    optimization_level: str = "basic",
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Apply optimizations to a model.
    
    Args:
        model: Model to optimize
        optimization_level: Level of optimization
        device: Device to optimize on
        
    Returns:
        Optimized model
    """
    logger.info(f"Optimizing model with level: {optimization_level}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model = model.to(device)
    
    if optimization_level == "basic":
        # Basic optimizations
        model.eval()
        if device.type == "cuda":
            model = model.half()  # Use half precision
        
    elif optimization_level == "advanced":
        # Advanced optimizations
        model.eval()
        if device.type == "cuda":
            model = model.half()
            # Enable cuDNN benchmarking
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
    elif optimization_level == "expert":
        # Expert optimizations
        model.eval()
        if device.type == "cuda":
            model = model.half()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TensorRT optimizations if available
            try:
                import tensorrt
                # TensorRT optimization would go here
                logger.info("TensorRT optimization available")
            except ImportError:
                logger.info("TensorRT not available")
    
    logger.info(f"Model optimization completed: {optimization_level}")
    return model

def create_optimization_report(
    results: List[BenchmarkResult],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an optimization report from benchmark results.
    
    Args:
        results: List of benchmark results
        output_file: Optional output file path
        
    Returns:
        Optimization report
    """
    if not results:
        return {}
    
    # Calculate statistics
    inference_times = [r.metrics.inference_time for r in results]
    memory_usages = [r.metrics.memory_usage for r in results]
    throughputs = [r.metrics.throughput for r in results]
    latencies = [r.metrics.latency for r in results]
    
    report = {
        'summary': {
            'total_benchmarks': len(results),
            'models_tested': list(set(r.model_name for r in results)),
            'time_range': f"{min(inference_times):.4f}s - {max(inference_times):.4f}s",
            'memory_range': f"{min(memory_usages):.2f}MB - {max(memory_usages):.2f}MB",
            'throughput_range': f"{min(throughputs):.2f} - {max(throughputs):.2f} samples/s",
            'latency_range': f"{min(latencies):.2f}ms - {max(latencies):.2f}ms"
        },
        'statistics': {
            'average_inference_time': sum(inference_times) / len(inference_times),
            'average_memory_usage': sum(memory_usages) / len(memory_usages),
            'average_throughput': sum(throughputs) / len(throughputs),
            'average_latency': sum(latencies) / len(latencies)
        },
        'results': [
            {
                'model_name': r.model_name,
                'inference_time': r.metrics.inference_time,
                'memory_usage': r.metrics.memory_usage,
                'throughput': r.metrics.throughput,
                'latency': r.metrics.latency,
                'configuration': r.configuration,
                'timestamp': r.timestamp
            }
            for r in results
        ]
    }
    
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Optimization report saved to: {output_file}")
    
    return report

def measure_memory_usage(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure memory usage of a function.
    
    Args:
        func: Function to measure
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (function result, memory usage in MB)
    """
    # Clear cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Measure initial memory
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Run function
    result = func(*args, **kwargs)
    
    # Measure final memory
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    memory_usage = final_memory - initial_memory
    
    return result, memory_usage

def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Get model size information.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with size information
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    return {
        'parameter_size_mb': param_size / 1024 / 1024,
        'buffer_size_mb': buffer_size / 1024 / 1024,
        'total_size_mb': total_size / 1024 / 1024,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'buffer_count': sum(b.numel() for b in model.buffers())
    }
