"""
Profiling and Performance Monitoring for polyglot_core.

Provides utilities for profiling operations, tracking performance metrics,
and monitoring resource usage across all backends.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from time import perf_counter
import time
import threading
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for an operation."""
    operation_name: str
    duration_ms: float
    memory_delta_mb: float = 0.0
    cpu_percent: float = 0.0
    backend: str = ""
    iterations: int = 1
    
    @property
    def throughput(self) -> float:
        """Calculate throughput (operations per second)."""
        if self.duration_ms <= 0:
            return 0.0
        return (self.iterations * 1000.0) / self.duration_ms
    
    @property
    def avg_duration_ms(self) -> float:
        """Average duration per iteration."""
        return self.duration_ms / self.iterations if self.iterations > 0 else 0.0


@dataclass
class ResourceUsage:
    """System resource usage."""
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    timestamp: float = field(default_factory=time.time)


class Profiler:
    """
    Performance profiler for polyglot_core operations.
    
    Tracks execution time, memory usage, and CPU usage.
    
    Example:
        >>> profiler = Profiler()
        >>> with profiler.profile("attention"):
        ...     output = attention.forward(q, k, v, batch, seq)
        >>> metrics = profiler.get_metrics("attention")
        >>> print(f"Time: {metrics.duration_ms:.2f}ms")
    """
    
    def __init__(self):
        self._metrics: Dict[str, List[PerformanceMetrics]] = {}
        self._resource_history: List[ResourceUsage] = []
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    @contextmanager
    def profile(
        self,
        operation_name: str,
        backend: str = "",
        track_memory: bool = True,
        track_cpu: bool = True
    ):
        """
        Context manager for profiling an operation.
        
        Args:
            operation_name: Name of the operation
            backend: Backend name
            track_memory: Track memory usage
            track_cpu: Track CPU usage
        """
        process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
        
        # Initial state
        start_time = perf_counter()
        start_memory = (process.memory_info().rss / (1024 * 1024) 
                       if track_memory and process else 0)
        start_cpu = (process.cpu_percent() if track_cpu and process else 0)
        
        try:
            yield
        finally:
            # Final state
            end_time = perf_counter()
            end_memory = (process.memory_info().rss / (1024 * 1024) 
                         if track_memory and process else 0)
            end_cpu = (process.cpu_percent() if track_cpu and process else 0)
            
            duration_ms = (end_time - start_time) * 1000
            memory_delta = end_memory - start_memory
            cpu_avg = (start_cpu + end_cpu) / 2 if track_cpu else 0
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                duration_ms=duration_ms,
                memory_delta_mb=memory_delta,
                cpu_percent=cpu_avg,
                backend=backend
            )
            
            with self._lock:
                if operation_name not in self._metrics:
                    self._metrics[operation_name] = []
                self._metrics[operation_name].append(metrics)
    
    def profile_function(
        self,
        func: Callable,
        operation_name: Optional[str] = None,
        iterations: int = 1,
        backend: str = ""
    ) -> PerformanceMetrics:
        """
        Profile a function call.
        
        Args:
            func: Function to profile
            operation_name: Name (defaults to function name)
            iterations: Number of iterations
            backend: Backend name
            
        Returns:
            PerformanceMetrics
        """
        if operation_name is None:
            operation_name = getattr(func, '__name__', 'unknown')
        
        process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
        start_time = perf_counter()
        start_memory = (process.memory_info().rss / (1024 * 1024) 
                       if process else 0)
        start_cpu = (process.cpu_percent() if process else 0)
        
        # Warmup
        if iterations > 1:
            func()
        
        # Actual run
        for _ in range(iterations):
            func()
        
        end_time = perf_counter()
        end_memory = (process.memory_info().rss / (1024 * 1024) 
                     if process else 0)
        end_cpu = (process.cpu_percent() if process else 0)
        
        duration_ms = (end_time - start_time) * 1000
        memory_delta = end_memory - start_memory
        cpu_avg = (start_cpu + end_cpu) / 2
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            duration_ms=duration_ms,
            memory_delta_mb=memory_delta,
            cpu_percent=cpu_avg,
            backend=backend,
            iterations=iterations
        )
        
        with self._lock:
            if operation_name not in self._metrics:
                self._metrics[operation_name] = []
            self._metrics[operation_name].append(metrics)
        
        return metrics
    
    def get_metrics(self, operation_name: str) -> Optional[PerformanceMetrics]:
        """Get latest metrics for an operation."""
        with self._lock:
            if operation_name in self._metrics and self._metrics[operation_name]:
                return self._metrics[operation_name][-1]
        return None
    
    def get_all_metrics(self, operation_name: str) -> List[PerformanceMetrics]:
        """Get all metrics for an operation."""
        with self._lock:
            return self._metrics.get(operation_name, []).copy()
    
    def get_average_metrics(self, operation_name: str) -> Optional[PerformanceMetrics]:
        """Get average metrics across all runs."""
        all_metrics = self.get_all_metrics(operation_name)
        if not all_metrics:
            return None
        
        avg_duration = sum(m.duration_ms for m in all_metrics) / len(all_metrics)
        avg_memory = sum(m.memory_delta_mb for m in all_metrics) / len(all_metrics)
        avg_cpu = sum(m.cpu_percent for m in all_metrics) / len(all_metrics)
        
        return PerformanceMetrics(
            operation_name=operation_name,
            duration_ms=avg_duration,
            memory_delta_mb=avg_memory,
            cpu_percent=avg_cpu,
            backend=all_metrics[0].backend if all_metrics else "",
            iterations=len(all_metrics)
        )
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        
        def monitor_loop():
            if not PSUTIL_AVAILABLE:
                return
            process = psutil.Process(os.getpid())
            while self._monitoring:
                usage = ResourceUsage(
                    cpu_percent=process.cpu_percent(),
                    memory_mb=process.memory_info().rss / (1024 * 1024),
                    memory_percent=process.memory_percent()
                )
                with self._lock:
                    self._resource_history.append(usage)
                time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
    
    def get_resource_history(self) -> List[ResourceUsage]:
        """Get resource usage history."""
        with self._lock:
            return self._resource_history.copy()
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        history = self.get_resource_history()
        if not history:
            return 0.0
        return max(r.memory_mb for r in history)
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._resource_history.clear()
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        
        with self._lock:
            for op_name, metrics_list in self._metrics.items():
                if metrics_list:
                    avg = self.get_average_metrics(op_name)
                    summary[op_name] = {
                        'runs': len(metrics_list),
                        'avg_duration_ms': avg.duration_ms if avg else 0,
                        'avg_memory_delta_mb': avg.memory_delta_mb if avg else 0,
                        'avg_cpu_percent': avg.cpu_percent if avg else 0,
                        'throughput': avg.throughput if avg else 0,
                        'backend': metrics_list[0].backend if metrics_list else ""
                    }
        
        return summary
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.summary()
        
        if not summary:
            print("No metrics recorded.")
            return
        
        print("\n" + "=" * 80)
        print("Performance Summary")
        print("=" * 80)
        print(f"{'Operation':<30} {'Runs':<8} {'Avg Time (ms)':<15} {'Throughput':<15} {'Backend':<10}")
        print("-" * 80)
        
        for op_name, stats in summary.items():
            print(f"{op_name:<30} {stats['runs']:<8} {stats['avg_duration_ms']:<15.2f} "
                  f"{stats['throughput']:<15.2f} {stats['backend']:<10}")
        
        print("=" * 80)


# Global profiler instance
_global_profiler = Profiler()


def get_profiler() -> Profiler:
    """Get global profiler instance."""
    return _global_profiler


def profile(operation_name: str, backend: str = ""):
    """Convenience decorator for profiling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with _global_profiler.profile(operation_name, backend):
                return func(*args, **kwargs)
        return wrapper
    return decorator


