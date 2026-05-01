"""
Performance profiling and benchmarking utilities for optimization_core.

This module provides comprehensive profiling, benchmarking, and optimization utilities
with support for CPU, GPU, and memory profiling using modern libraries.

Features:
- Function-level profiling with cProfile
- Memory profiling with memory_profiler (optional)
- PyTorch model profiling with torch.profiler
- Benchmarking utilities for model comparison
- Performance metrics collection and reporting
"""

from __future__ import annotations

import logging
import time
import cProfile
import pstats
import io
import gc
import sys
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, TypeVar, ContextManager
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
import statistics

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available, some memory profiling features will be limited")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

try:
    from memory_profiler import profile as memory_profile_decorator
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    memory_profile_decorator = None

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ════════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    inference_time: float
    memory_usage: float
    throughput: float
    latency: float
    gpu_utilization: Optional[float] = None
    cpu_utilization: Optional[float] = None
    peak_memory: Optional[float] = None
    std_dev: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'inference_time': self.inference_time,
            'memory_usage': self.memory_usage,
            'throughput': self.throughput,
            'latency': self.latency,
            'gpu_utilization': self.gpu_utilization,
            'cpu_utilization': self.cpu_utilization,
            'peak_memory': self.peak_memory,
            'std_dev': self.std_dev,
        }


@dataclass
class BenchmarkResult:
    """Benchmark result container."""
    model_name: str
    metrics: PerformanceMetrics
    configuration: Dict[str, Any]
    timestamp: float
    raw_times: List[float] = field(default_factory=list)
    raw_memory: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'model_name': self.model_name,
            'metrics': self.metrics.to_dict(),
            'configuration': self.configuration,
            'timestamp': self.timestamp,
        }


@dataclass
class FunctionStats:
    """Statistics for a profiled function."""
    call_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        result = {
            'call_count': self.call_count,
            'total_time': self.total_time,
            'avg_time': self.avg_time,
            'min_time': self.min_time,
            'max_time': self.max_time,
        }
        if self.std_dev is not None:
            result['std_dev'] = self.std_dev
        return result


# ════════════════════════════════════════════════════════════════════════════════
# FUNCTION PROFILER
# ════════════════════════════════════════════════════════════════════════════════

class FunctionProfiler:
    """Profiler for tracking function call statistics."""
    
    def __init__(self):
        """Initialize profiler."""
        self.call_counts: Dict[str, int] = {}
        self.total_times: Dict[str, float] = {}
        self.min_times: Dict[str, float] = {}
        self.max_times: Dict[str, float] = {}
        self.all_times: Dict[str, List[float]] = {}
    
    def record_call(
        self,
        function_name: str,
        duration: float
    ) -> None:
        """
        Record a function call.
        
        Args:
            function_name: Name of function
            duration: Call duration in seconds
        """
        if function_name not in self.call_counts:
            self.call_counts[function_name] = 0
            self.total_times[function_name] = 0.0
            self.min_times[function_name] = float('inf')
            self.max_times[function_name] = 0.0
            self.all_times[function_name] = []
        
        self.call_counts[function_name] += 1
        self.total_times[function_name] += duration
        self.min_times[function_name] = min(self.min_times[function_name], duration)
        self.max_times[function_name] = max(self.max_times[function_name], duration)
        self.all_times[function_name].append(duration)
    
    def get_stats(self, function_name: str) -> Optional[FunctionStats]:
        """
        Get statistics for a function.
        
        Args:
            function_name: Name of function
        
        Returns:
            FunctionStats or None if function not found
        """
        if function_name not in self.call_counts:
            return None
        
        count = self.call_counts[function_name]
        total = self.total_times[function_name]
        times = self.all_times[function_name]
        
        std_dev = None
        if len(times) > 1:
            try:
                std_dev = statistics.stdev(times)
            except (ValueError, statistics.StatisticsError):
                pass
        
        return FunctionStats(
            call_count=count,
            total_time=total,
            avg_time=total / count if count > 0 else 0.0,
            min_time=self.min_times[function_name] if self.min_times[function_name] != float('inf') else 0.0,
            max_time=self.max_times[function_name],
            std_dev=std_dev,
        )
    
    def get_summary(self) -> Dict[str, FunctionStats]:
        """
        Get summary of all functions.
        
        Returns:
            Dictionary mapping function names to statistics
        """
        return {
            name: self.get_stats(name)
            for name in self.call_counts.keys()
            if self.get_stats(name) is not None
        }
    
    def print_summary(self, sort_by: str = "total_time") -> None:
        """
        Print summary to console.
        
        Args:
            sort_by: Sort key (total_time, avg_time, call_count)
        """
        summary = self.get_summary()
        
        if not summary:
            print("No profiling data available.")
            return
        
        # Sort by specified key
        sort_key_map = {
            "total_time": lambda x: x[1].total_time,
            "avg_time": lambda x: x[1].avg_time,
            "call_count": lambda x: x[1].call_count,
        }
        sort_key = sort_key_map.get(sort_by, sort_key_map["total_time"])
        
        print("\n=== Performance Profile ===")
        for name, stats in sorted(summary.items(), key=sort_key, reverse=True):
            print(f"\n{name}:")
            print(f"  Calls: {stats.call_count}")
            print(f"  Total: {stats.total_time:.6f}s")
            print(f"  Avg: {stats.avg_time:.6f}s")
            print(f"  Min: {stats.min_time:.6f}s")
            print(f"  Max: {stats.max_time:.6f}s")
            if stats.std_dev is not None:
                print(f"  Std Dev: {stats.std_dev:.6f}s")
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.call_counts.clear()
        self.total_times.clear()
        self.min_times.clear()
        self.max_times.clear()
        self.all_times.clear()


# ════════════════════════════════════════════════════════════════════════════════
# CONTEXT MANAGERS
# ════════════════════════════════════════════════════════════════════════════════

@contextmanager
def profile_context(
    output_file: Optional[Union[str, Path]] = None,
    sort_by: str = "cumulative",
    lines: int = 50,
    logger_instance: Optional[logging.Logger] = None
) -> ContextManager[cProfile.Profile]:
    """
    Context manager for profiling code with cProfile.
    
    Args:
        output_file: Optional file to save profile results
        sort_by: Sort key (cumulative, time, calls, etc.)
        lines: Number of lines to display
        logger_instance: Optional logger instance
    
    Example:
        with profile_context():
            # Your code here
            result = expensive_function()
    """
    profiler = cProfile.Profile()
    profiler.enable()
    log = logger_instance or logger
    
    try:
        yield profiler
    finally:
        profiler.disable()
        
        # Create stats
        stats = pstats.Stats(profiler)
        stats.sort_stats(sort_by)
        
        # Print to console
        output = io.StringIO()
        stats.print_stats(lines, stream=output)
        output_str = output.getvalue()
        log.info(f"Profile results:\n{output_str}")
        
        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stats.dump_stats(str(output_path))
            log.info(f"Profile saved to {output_path}")


@contextmanager
def profile_operation(
    operation_name: str,
    logger_instance: Optional[logging.Logger] = None,
    track_memory: bool = True
) -> ContextManager[None]:
    """
    Context manager for profiling operations with timing and memory.
    
    Args:
        operation_name: Name of the operation
        logger_instance: Optional logger instance
        track_memory: Whether to track memory usage
    
    Example:
        with profile_operation("data_processing"):
            process_data()
    """
    log = logger_instance or logger
    start_time = time.perf_counter()
    
    start_memory = 0.0
    if track_memory and PSUTIL_AVAILABLE:
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    log.info(f"Starting operation: {operation_name}")
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        if track_memory and PSUTIL_AVAILABLE:
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory
            log.info(
                f"Completed operation: {operation_name} | "
                f"Duration: {duration:.6f}s | "
                f"Memory delta: {memory_delta:.2f} MB"
            )
        else:
            log.info(
                f"Completed operation: {operation_name} | "
                f"Duration: {duration:.6f}s"
            )


# ════════════════════════════════════════════════════════════════════════════════
# FUNCTION PROFILING
# ════════════════════════════════════════════════════════════════════════════════

def profile_function(
    func: Callable[..., T],
    *args: Any,
    output_file: Optional[Union[str, Path]] = None,
    **kwargs: Any
) -> T:
    """
    Profile a function call.
    
    Args:
        func: Function to profile
        *args: Positional arguments
        output_file: Optional file to save profile
        **kwargs: Keyword arguments
    
    Returns:
        Function result
    """
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        result = func(*args, **kwargs)
        return result
    finally:
        profiler.disable()
        
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        
        output = io.StringIO()
        stats.print_stats(20, stream=output)
        logger.info(f"Profile for {func.__name__}:\n{output.getvalue()}")
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stats.dump_stats(str(output_path))
            logger.info(f"Profile saved to {output_path}")


def profile_decorator(
    output_file: Optional[Union[str, Path]] = None,
    enabled: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for profiling functions.
    
    Args:
        output_file: Optional file to save profile
        enabled: Whether profiling is enabled
    
    Example:
        @profile_decorator()
        def my_function():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if not enabled:
            return func
        
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return profile_function(func, *args, output_file=output_file, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# MEMORY PROFILING
# ════════════════════════════════════════════════════════════════════════════════

def measure_memory_usage(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any
) -> Tuple[T, float]:
    """
    Measure memory usage of a function.
    
    Args:
        func: Function to measure
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        Tuple of (function result, memory usage in MB)
    """
    if not PSUTIL_AVAILABLE:
        logger.warning("psutil not available, memory measurement will be inaccurate")
        result = func(*args, **kwargs)
        return result, 0.0
    
    # Clear cache
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Measure initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    # Run function
    result = func(*args, **kwargs)
    
    # Force garbage collection
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Measure final memory
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_usage = final_memory - initial_memory
    
    return result, memory_usage


# ════════════════════════════════════════════════════════════════════════════════
# PYTORCH MODEL PROFILING
# ════════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    
    class TorchModelProfiler:
        """Profiler for PyTorch models using torch.profiler."""
        
        def __init__(self, log_dir: Optional[Union[str, Path]] = None):
            """
            Initialize profiler.
            
            Args:
                log_dir: Directory for profile logs
            """
            self.log_dir = Path(log_dir) if log_dir else None
            if self.log_dir:
                self.log_dir.mkdir(parents=True, exist_ok=True)
        
        def profile(
            self,
            model: nn.Module,
            input_shape: Tuple[int, ...],
            num_runs: int = 10,
            warmup_runs: int = 3,
            activities: Optional[List] = None,
        ) -> Dict[str, Any]:
            """
            Profile model performance using torch.profiler.
            
            Args:
                model: Model to profile
                input_shape: Input tensor shape
                num_runs: Number of profiling runs
                warmup_runs: Number of warmup runs
                activities: Profiling activities
            
            Returns:
                Dictionary with profiling results
            """
            if activities is None:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
            
            # Create dummy input
            dummy_input = torch.zeros(input_shape)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                model = model.cuda()
            
            model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Profile
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                with torch.no_grad():
                    for _ in range(num_runs):
                        _ = model(dummy_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            # Process results
            if self.log_dir:
                trace_path = self.log_dir / "trace.json"
                prof.export_chrome_trace(str(trace_path))
                logger.info(f"Chrome trace exported to {trace_path}")
            
            # Get statistics
            key_averages = prof.key_averages()
            
            stats = {
                "cpu_time_total": sum(e.cpu_time_total for e in key_averages),
                "cuda_time_total": sum(e.cuda_time_total for e in key_averages) if torch.cuda.is_available() else 0,
                "cpu_memory_usage": sum(e.cpu_memory_usage for e in key_averages),
                "cuda_memory_usage": sum(e.cuda_memory_usage for e in key_averages) if torch.cuda.is_available() else 0,
            }
            
            logger.info("Model profiling completed")
            return stats
    
    
    class PerformanceProfiler:
        """Performance profiler for PyTorch models and operations."""
        
        def __init__(self, device: Optional[torch.device] = None):
            """
            Initialize performance profiler.
            
            Args:
                device: Device to profile on
            """
            self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
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
            times: List[float] = []
            memory_usage: List[float] = []
            
            with torch.no_grad():
                for i in range(num_runs):
                    # Clear cache
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    
                    # Measure time
                    start_time = time.perf_counter()
                    _ = model(input_tensor)
                    
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                    
                    # Measure memory
                    if self.device.type == "cuda":
                        memory_usage.append(torch.cuda.memory_allocated(self.device) / 1024 / 1024)
                    elif PSUTIL_AVAILABLE:
                        memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)
            
            # Calculate metrics
            avg_time = statistics.mean(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
            avg_memory = statistics.mean(memory_usage) if memory_usage else 0.0
            peak_memory = max(memory_usage) if memory_usage else 0.0
            throughput = batch_size / avg_time
            latency = avg_time * 1000  # Convert to milliseconds
            
            metrics = PerformanceMetrics(
                inference_time=avg_time,
                memory_usage=avg_memory,
                throughput=throughput,
                latency=latency,
                peak_memory=peak_memory,
                std_dev=std_dev,
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
                timestamp=time.time(),
                raw_times=times,
                raw_memory=memory_usage,
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
                'average_inference_time': statistics.mean([r.metrics.inference_time for r in self.results]),
                'average_memory_usage': statistics.mean([r.metrics.memory_usage for r in self.results]),
                'average_throughput': statistics.mean([r.metrics.throughput for r in self.results]),
                'average_latency': statistics.mean([r.metrics.latency for r in self.results])
            }
            
            return summary
    
    
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
        return result.to_dict()
    
    
    def optimize_model(
        model: nn.Module,
        optimization_level: str = "basic",
        device: Optional[torch.device] = None
    ) -> nn.Module:
        """
        Apply optimizations to a model.
        
        Args:
            model: Model to optimize
            optimization_level: Level of optimization (basic, advanced, expert)
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
            model.eval()
            if device.type == "cuda":
                model = model.half()  # Use half precision
        
        elif optimization_level == "advanced":
            model.eval()
            if device.type == "cuda":
                model = model.half()
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
        
        elif optimization_level == "expert":
            model.eval()
            if device.type == "cuda":
                model = model.half()
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                # Enable TensorRT optimizations if available
                try:
                    import tensorrt  # noqa: F401
                    logger.info("TensorRT optimization available")
                except ImportError:
                    logger.info("TensorRT not available")
        
        logger.info(f"Model optimization completed: {optimization_level}")
        return model
    
    
    def create_optimization_report(
        results: List[BenchmarkResult],
        output_file: Optional[Union[str, Path]] = None
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
                'average_inference_time': statistics.mean(inference_times),
                'average_memory_usage': statistics.mean(memory_usages),
                'average_throughput': statistics.mean(throughputs),
                'average_latency': statistics.mean(latencies)
            },
            'results': [r.to_dict() for r in results]
        }
        
        if output_file:
            import json
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Optimization report saved to: {output_path}")
        
        return report

else:
    # Stub classes when torch is not available
    TorchModelProfiler = None  # type: ignore
    PerformanceProfiler = None  # type: ignore
    
    def benchmark_model(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("PyTorch is required for model benchmarking")
    
    def profile_model(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("PyTorch is required for model profiling")
    
    def optimize_model(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("PyTorch is required for model optimization")
    
    def create_optimization_report(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("PyTorch is required for optimization reports")


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Data classes
    'PerformanceMetrics',
    'BenchmarkResult',
    'FunctionStats',
    # Profilers
    'FunctionProfiler',
    'PerformanceProfiler',
    'TorchModelProfiler',
    # Context managers
    'profile_context',
    'profile_operation',
    # Functions
    'profile_function',
    'profile_decorator',
    'measure_memory_usage',
    'benchmark_model',
    'profile_model',
    'optimize_model',
    'create_optimization_report',
]












