"""
Benchmarking utilities for optimization_core.

Provides utilities for performance benchmarking, comparison, and analysis.
"""
from .benchmark_runner import (
    BenchmarkRunner,
    BenchmarkResult,
    run_benchmark,
    compare_benchmarks,
)
from .performance_metrics import (
    PerformanceMetrics,
    MetricsCollector,
    collect_metrics,
    analyze_performance,
)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "run_benchmark",
    "compare_benchmarks",
    "PerformanceMetrics",
    "MetricsCollector",
    "collect_metrics",
    "analyze_performance",
]

