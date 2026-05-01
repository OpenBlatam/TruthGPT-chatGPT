"""
Benchmark Helper Functions

Utilities for benchmarking and performance testing.
"""
import time
import statistics
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    avg_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    throughput: float = 0.0
    num_runs: int = 0
    errors: List[str] = field(default_factory=list)

def run_benchmark(
    func: Callable,
    *args,
    num_runs: int = 10,
    warmup_runs: int = 3,
    name: Optional[str] = None,
    **kwargs
) -> BenchmarkResult:
    """
    Run benchmark on a function.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments for function
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        name: Name of benchmark
        **kwargs: Keyword arguments for function
    
    Returns:
        BenchmarkResult with statistics
    """
    name = name or func.__name__
    
    # Warmup
    for _ in range(warmup_runs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    # Benchmark
    times = []
    errors = []
    
    for i in range(num_runs):
        try:
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        except Exception as e:
            errors.append(f"Run {i+1}: {str(e)}")
            logger.warning(f"Benchmark run {i+1} failed: {e}")
    
    if not times:
        return BenchmarkResult(
            name=name,
            num_runs=num_runs,
            errors=errors
        )
    
    sorted_times = sorted(times)
    n = len(sorted_times)
    
    return BenchmarkResult(
        name=name,
        avg_ms=statistics.mean(times),
        min_ms=min(times),
        max_ms=max(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        p50_ms=sorted_times[n // 2],
        p95_ms=sorted_times[int(n * 0.95)] if n > 0 else 0.0,
        p99_ms=sorted_times[int(n * 0.99)] if n > 0 else 0.0,
        throughput=1000.0 / statistics.mean(times),
        num_runs=len(times),
        errors=errors,
    )

def compare_benchmarks(
    results: Dict[str, BenchmarkResult],
    baseline: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple benchmark results.
    
    Args:
        results: Dictionary of {name: BenchmarkResult}
        baseline: Name of baseline result (default: first result)
    
    Returns:
        Dictionary with comparison metrics
    """
    if not results:
        return {}
    
    baseline_name = baseline or list(results.keys())[0]
    baseline_result = results[baseline_name]
    
    comparison = {}
    
    for name, result in results.items():
        if name == baseline_name:
            comparison[name] = {
                "is_baseline": True,
                "avg_ms": result.avg_ms,
                "throughput": result.throughput,
            }
        else:
            speedup = baseline_result.avg_ms / result.avg_ms if result.avg_ms > 0 else 0.0
            comparison[name] = {
                "is_baseline": False,
                "avg_ms": result.avg_ms,
                "throughput": result.throughput,
                "speedup_vs_baseline": speedup,
                "improvement_pct": (1.0 - result.avg_ms / baseline_result.avg_ms) * 100 if baseline_result.avg_ms > 0 else 0.0,
            }
    
    return comparison

def format_benchmark_result(result: BenchmarkResult) -> str:
    """Format benchmark result as string."""
    lines = [
        f"Benchmark: {result.name}",
        f"  Runs: {result.num_runs}",
        f"  Avg: {result.avg_ms:.2f}ms",
        f"  Min: {result.min_ms:.2f}ms",
        f"  Max: {result.max_ms:.2f}ms",
        f"  Std: {result.std_ms:.2f}ms",
        f"  P50: {result.p50_ms:.2f}ms",
        f"  P95: {result.p95_ms:.2f}ms",
        f"  P99: {result.p99_ms:.2f}ms",
        f"  Throughput: {result.throughput:.2f} ops/s",
    ]
    
    if result.errors:
        lines.append(f"  Errors: {len(result.errors)}")
    
    return "\n".join(lines)

def benchmark_backends(
    func: Callable,
    backends: List[str],
    *args,
    num_runs: int = 10,
    warmup_runs: int = 3,
    **kwargs
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark function across multiple backends.
    
    Args:
        func: Function to benchmark
        backends: List of backend names
        *args: Positional arguments
        num_runs: Number of runs per backend
        warmup_runs: Number of warmup runs
        **kwargs: Keyword arguments
    
    Returns:
        Dictionary of {backend: BenchmarkResult}
    """
    results = {}
    
    for backend in backends:
        try:
            result = run_benchmark(
                func,
                *args,
                num_runs=num_runs,
                warmup_runs=warmup_runs,
                name=f"{func.__name__}_{backend}",
                backend=backend,
                **kwargs
            )
            results[backend] = result
        except Exception as e:
            logger.warning(f"Benchmark failed for backend {backend}: {e}")
            results[backend] = BenchmarkResult(
                name=f"{func.__name__}_{backend}",
                errors=[str(e)]
            )
    
    return results

__all__ = [
    "BenchmarkResult",
    "run_benchmark",
    "compare_benchmarks",
    "format_benchmark_result",
    "benchmark_backends",
]













