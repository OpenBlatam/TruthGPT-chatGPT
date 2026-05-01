"""
Benchmarking modules for polyglot_core.

Benchmarking and reporting.
"""

from ..benchmarking import (
    Benchmark,
    BenchmarkResult,
    benchmark,
    compare_backends_quick,
)

from ..reporting import (
    ReportGenerator,
    PerformanceReport,
    ReportSection,
    generate_benchmark_report,
)

__all__ = [
    # Benchmarking
    "Benchmark",
    "BenchmarkResult",
    "benchmark",
    "compare_backends_quick",
    # Reporting
    "ReportGenerator",
    "PerformanceReport",
    "ReportSection",
    "generate_benchmark_report",
]













