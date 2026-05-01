"""
Base Test Classes

Shared base classes for all tests.
"""
import unittest
import time
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BasePolyglotTest(unittest.TestCase):
    """Base class for polyglot tests."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.backend_availability = self._check_backends()
        self.test_results = []
    
    def _check_backends(self) -> Dict[str, bool]:
        """Check which backends are available."""
        availability = {
            "rust": False,
            "cpp": False,
            "julia": False,
            "python": True,
        }
        
        try:
            from truthgpt_rust import PyKVCache
            availability["rust"] = True
        except ImportError:
            pass
        
        try:
            import _cpp_core as cpp_core
            availability["cpp"] = True
        except ImportError:
            pass
        
        try:
            from julia import TruthGPTCore
            availability["julia"] = True
        except ImportError:
            pass
        
        return availability
    
    def skip_if_backend_unavailable(self, backend: str):
        """Skip test if backend is unavailable."""
        if not self.backend_availability.get(backend, False):
            self.skipTest(f"Backend '{backend}' not available")
    
    def measure_time(self, func, *args, **kwargs) -> tuple:
        """Measure function execution time."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms
    
    def assert_performance_improvement(
        self,
        baseline_ms: float,
        improved_ms: float,
        min_improvement: float = 1.1
    ):
        """Assert that performance improved."""
        improvement = baseline_ms / improved_ms
        self.assertGreaterEqual(
            improvement,
            min_improvement,
            f"Expected {min_improvement}x improvement, got {improvement:.2f}x"
        )

class BaseBenchmarkTest(BasePolyglotTest):
    """Base class for benchmark tests."""
    
    def setUp(self):
        """Setup benchmark test."""
        super().setUp()
        self.benchmark_results = []
        self.num_runs = 10
        self.warmup_runs = 3
    
    def run_benchmark(
        self,
        func,
        *args,
        num_runs: Optional[int] = None,
        warmup_runs: Optional[int] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Run benchmark and return statistics."""
        num_runs = num_runs or self.num_runs
        warmup_runs = warmup_runs or self.warmup_runs
        
        # Warmup
        for _ in range(warmup_runs):
            func(*args, **kwargs)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            _, elapsed_ms = self.measure_time(func, *args, **kwargs)
            times.append(elapsed_ms)
        
        return {
            "avg_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
            "throughput": 1000.0 / (sum(times) / len(times)),
        }
    
    def compare_backends(
        self,
        func,
        backends: List[str],
        *args,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """Compare performance across backends."""
        results = {}
        
        for backend in backends:
            self.skip_if_backend_unavailable(backend)
            stats = self.run_benchmark(
                func,
                *args,
                backend=backend,
                **kwargs
            )
            results[backend] = stats
        
        return results

class BaseIntegrationTest(BasePolyglotTest):
    """Base class for integration tests."""
    
    def setUp(self):
        """Setup integration test."""
        super().setUp()
        self.integration_results = []
    
    def test_end_to_end_flow(self):
        """Test complete end-to-end flow."""
        raise NotImplementedError("Subclasses must implement test_end_to_end_flow")

class BasePerformanceTest(BasePolyglotTest):
    """Base class for performance tests."""
    
    def setUp(self):
        """Setup performance test."""
        super().setUp()
        self.performance_metrics = {}
    
    def assert_meets_performance_target(
        self,
        metric_name: str,
        actual_value: float,
        target_value: float,
        tolerance: float = 0.1
    ):
        """Assert that performance meets target."""
        if actual_value < target_value * (1 - tolerance):
            self.fail(
                f"{metric_name}: {actual_value:.2f} < target {target_value:.2f} "
                f"(tolerance: {tolerance:.1%})"
            )













