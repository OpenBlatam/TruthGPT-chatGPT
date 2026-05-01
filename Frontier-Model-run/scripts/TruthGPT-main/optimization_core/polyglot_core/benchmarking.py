"""
Benchmarking utilities for polyglot_core.

Provides comprehensive benchmarking tools to compare backends and operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
import time
import statistics
import json
from pathlib import Path

from .profiling import Profiler, PerformanceMetrics
from .backend import Backend, get_available_backends


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    name: str
    backend: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    stddev_ms: float
    throughput: float
    memory_peak_mb: float = 0.0
    memory_avg_mb: float = 0.0
    success: bool = True
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'backend': self.backend,
            'iterations': self.iterations,
            'total_time_ms': self.total_time_ms,
            'avg_time_ms': self.avg_time_ms,
            'min_time_ms': self.min_time_ms,
            'max_time_ms': self.max_time_ms,
            'stddev_ms': self.stddev_ms,
            'throughput': self.throughput,
            'memory_peak_mb': self.memory_peak_mb,
            'memory_avg_mb': self.memory_avg_mb,
            'success': self.success,
            'error': self.error
        }


class Benchmark:
    """
    Comprehensive benchmarking tool.
    
    Example:
        >>> benchmark = Benchmark()
        >>> result = benchmark.run("kv_cache_get", cache.get, (0, 0), iterations=1000)
        >>> print(f"Throughput: {result.throughput:.0f} ops/s")
    """
    
    def __init__(self, warmup_iterations: int = 3):
        """
        Initialize benchmark.
        
        Args:
            warmup_iterations: Number of warmup iterations
        """
        self.warmup_iterations = warmup_iterations
        self.profiler = Profiler()
    
    def run(
        self,
        name: str,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[Dict] = None,
        iterations: int = 100,
        backend: str = ""
    ) -> BenchmarkResult:
        """
        Run a benchmark.
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            args: Positional arguments
            kwargs: Keyword arguments
            iterations: Number of iterations
            backend: Backend name
            
        Returns:
            BenchmarkResult
        """
        if kwargs is None:
            kwargs = {}
        
        times = []
        memory_samples = []
        
        try:
            # Warmup
            for _ in range(self.warmup_iterations):
                func(*args, **kwargs)
            
            # Actual benchmark
            for _ in range(iterations):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            # Calculate statistics
            total_time = sum(times)
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            stddev = statistics.stdev(times) if len(times) > 1 else 0.0
            throughput = (iterations * 1000.0) / total_time if total_time > 0 else 0.0
            
            return BenchmarkResult(
                name=name,
                backend=backend,
                iterations=iterations,
                total_time_ms=total_time,
                avg_time_ms=avg_time,
                min_time_ms=min_time,
                max_time_ms=max_time,
                stddev_ms=stddev,
                throughput=throughput,
                success=True
            )
        except Exception as e:
            return BenchmarkResult(
                name=name,
                backend=backend,
                iterations=iterations,
                total_time_ms=0.0,
                avg_time_ms=0.0,
                min_time_ms=0.0,
                max_time_ms=0.0,
                stddev_ms=0.0,
                throughput=0.0,
                success=False,
                error=str(e)
            )
    
    def compare_backends(
        self,
        name: str,
        func_factory: Callable[[Backend], Callable],
        args: tuple = (),
        kwargs: Optional[Dict] = None,
        iterations: int = 100,
        backends: Optional[List[Backend]] = None
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare performance across multiple backends.
        
        Args:
            name: Benchmark name
            func_factory: Function that takes Backend and returns callable
            args: Positional arguments
            kwargs: Keyword arguments
            iterations: Number of iterations
            backends: Backends to test (default: all available)
            
        Returns:
            Dict mapping backend name to BenchmarkResult
        """
        if backends is None:
            backends = [b.backend for b in get_available_backends() if b.available]
        
        results = {}
        
        for backend in backends:
            try:
                func = func_factory(backend)
                result = self.run(
                    name,
                    func,
                    args,
                    kwargs,
                    iterations,
                    backend.name
                )
                results[backend.name] = result
            except Exception as e:
                results[backend.name] = BenchmarkResult(
                    name=name,
                    backend=backend.name,
                    iterations=iterations,
                    total_time_ms=0.0,
                    avg_time_ms=0.0,
                    min_time_ms=0.0,
                    max_time_ms=0.0,
                    stddev_ms=0.0,
                    throughput=0.0,
                    success=False,
                    error=str(e)
                )
        
        return results
    
    def benchmark_suite(
        self,
        benchmarks: List[Dict[str, Any]]
    ) -> Dict[str, BenchmarkResult]:
        """
        Run a suite of benchmarks.
        
        Args:
            benchmarks: List of benchmark configs with 'name', 'func', 'args', etc.
            
        Returns:
            Dict mapping benchmark name to result
        """
        results = {}
        
        for config in benchmarks:
            name = config['name']
            func = config['func']
            args = config.get('args', ())
            kwargs = config.get('kwargs', {})
            iterations = config.get('iterations', 100)
            backend = config.get('backend', '')
            
            result = self.run(name, func, args, kwargs, iterations, backend)
            results[name] = result
        
        return results
    
    def print_comparison(self, results: Dict[str, BenchmarkResult]):
        """Print formatted comparison of results."""
        if not results:
            print("No results to compare.")
            return
        
        print("\n" + "=" * 100)
        print("Benchmark Comparison")
        print("=" * 100)
        print(f"{'Backend':<15} {'Iterations':<12} {'Avg Time (ms)':<15} {'Throughput':<15} {'Status':<10}")
        print("-" * 100)
        
        for backend, result in results.items():
            status = "✓" if result.success else f"✗ {result.error[:20]}"
            print(f"{backend:<15} {result.iterations:<12} {result.avg_time_ms:<15.2f} "
                  f"{result.throughput:<15.2f} {status:<10}")
        
        # Find fastest
        successful = {k: v for k, v in results.items() if v.success}
        if successful:
            fastest = max(successful.items(), key=lambda x: x[1].throughput)
            print(f"\nFastest: {fastest[0]} ({fastest[1].throughput:.0f} ops/s)")
        
        print("=" * 100)
    
    def save_results(
        self,
        results: Union[BenchmarkResult, Dict[str, BenchmarkResult]],
        filepath: Union[str, Path]
    ):
        """Save benchmark results to JSON file."""
        if isinstance(results, BenchmarkResult):
            data = {'results': [results.to_dict()]}
        else:
            data = {'results': [r.to_dict() for r in results.values()]}
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_results(self, filepath: Union[str, Path]) -> Dict[str, BenchmarkResult]:
        """Load benchmark results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results = {}
        for r_dict in data['results']:
            result = BenchmarkResult(**r_dict)
            results[result.name] = result
        
        return results


# Convenience functions
def benchmark(
    name: str,
    func: Callable,
    args: tuple = (),
    kwargs: Optional[Dict] = None,
    iterations: int = 100
) -> BenchmarkResult:
    """Quick benchmark function."""
    bench = Benchmark()
    return bench.run(name, func, args, kwargs, iterations)


def compare_backends_quick(
    name: str,
    func_factory: Callable[[Backend], Callable],
    args: tuple = (),
    iterations: int = 100
) -> Dict[str, BenchmarkResult]:
    """Quick backend comparison."""
    bench = Benchmark()
    return bench.compare_backends(name, func_factory, args, {}, iterations)













