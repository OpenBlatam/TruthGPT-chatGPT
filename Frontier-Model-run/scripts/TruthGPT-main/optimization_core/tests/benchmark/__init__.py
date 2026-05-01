"""
Benchmark Test Suite - Modular Organization

This package contains refactored benchmark tests organized by component.
"""

from .benchmark_models import BenchmarkResult, ClosedSourceResult, BenchmarkReport
from .benchmark_utils import generate_summary, print_report
from .module_detector import POLYGLOT_MODULES, detect_polyglot_modules

# Lazy imports for benchmarkers (to avoid circular dependencies)
def get_polyglot_benchmarker():
    """Get PolyglotBenchmarker (lazy import)."""
    from .polyglot_benchmarker import PolyglotBenchmarker
    return PolyglotBenchmarker

def get_closed_source_benchmarker():
    """Get ClosedSourceBenchmarker (lazy import)."""
    from .closed_source_benchmarker import ClosedSourceBenchmarker
    return ClosedSourceBenchmarker

__all__ = [
    "BenchmarkResult",
    "ClosedSourceResult",
    "BenchmarkReport",
    "generate_summary",
    "print_report",
    "POLYGLOT_MODULES",
    "detect_polyglot_modules",
    "get_polyglot_benchmarker",
    "get_closed_source_benchmarker",
]


