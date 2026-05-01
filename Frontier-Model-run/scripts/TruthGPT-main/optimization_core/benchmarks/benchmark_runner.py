"""
Benchmark runner for performance testing.

Provides utilities for running benchmarks and comparing results.
"""
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    duration: float
    throughput: float
    memory_usage: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "duration": self.duration,
            "throughput": self.throughput,
            "memory_usage": self.memory_usage,
            "metrics": self.metrics,
            "error": self.error,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class BenchmarkRunner:
    """Runner for executing benchmarks."""
    
    def __init__(
        self,
        warmup_runs: int = 3,
        num_runs: int = 10,
        collect_memory: bool = True
    ):
        """
        Initialize benchmark runner.
        
        Args:
            warmup_runs: Number of warmup runs
            num_runs: Number of benchmark runs
            collect_memory: Whether to collect memory usage
        """
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs
        self.collect_memory = collect_memory
    
    def run(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a benchmark.
        
        Args:
            name: Name of benchmark
            func: Function to benchmark
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            BenchmarkResult
        """
        logger.info(f"Running benchmark: {name}")
        
        # Warmup
        for _ in range(self.warmup_runs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Warmup run failed: {e}")
        
        # Benchmark runs
        times = []
        memory_usages = []
        
        for i in range(self.num_runs):
            try:
                # Collect memory before
                memory_before = self._get_memory_usage() if self.collect_memory else None
                
                # Run function
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                
                # Collect memory after
                memory_after = self._get_memory_usage() if self.collect_memory else None
                
                duration = end_time - start_time
                times.append(duration)
                
                if memory_before is not None and memory_after is not None:
                    memory_usages.append(memory_after - memory_before)
                
            except Exception as e:
                logger.error(f"Benchmark run {i+1} failed: {e}", exc_info=True)
                return BenchmarkResult(
                    name=name,
                    duration=0.0,
                    throughput=0.0,
                    error=str(e)
                )
        
        # Calculate statistics
        avg_duration = sum(times) / len(times)
        min_duration = min(times)
        max_duration = max(times)
        throughput = 1.0 / avg_duration if avg_duration > 0 else 0.0
        
        avg_memory = sum(memory_usages) / len(memory_usages) if memory_usages else None
        
        return BenchmarkResult(
            name=name,
            duration=avg_duration,
            throughput=throughput,
            memory_usage=avg_memory,
            metrics={
                "min_duration": min_duration,
                "max_duration": max_duration,
                "num_runs": self.num_runs,
                "warmup_runs": self.warmup_runs,
            }
        )
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return None
    
    def compare(
        self,
        benchmarks: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """
        Compare multiple benchmark results.
        
        Args:
            benchmarks: List of benchmark results
        
        Returns:
            Comparison dictionary
        """
        if not benchmarks:
            return {}
        
        # Find best and worst
        best = min(benchmarks, key=lambda b: b.duration)
        worst = max(benchmarks, key=lambda b: b.duration)
        
        # Calculate improvements
        improvements = {}
        for benchmark in benchmarks:
            if benchmark != best:
                improvement = ((best.duration - benchmark.duration) / benchmark.duration) * 100
                improvements[benchmark.name] = improvement
        
        return {
            "best": best.name,
            "worst": worst.name,
            "best_duration": best.duration,
            "worst_duration": worst.duration,
            "improvements": improvements,
            "results": [b.to_dict() for b in benchmarks],
        }


def run_benchmark(
    name: str,
    func: Callable,
    *args,
    warmup_runs: int = 3,
    num_runs: int = 10,
    **kwargs
) -> BenchmarkResult:
    """
    Convenience function to run a benchmark.
    
    Args:
        name: Name of benchmark
        func: Function to benchmark
        *args: Positional arguments
        warmup_runs: Number of warmup runs
        num_runs: Number of benchmark runs
        **kwargs: Keyword arguments
    
    Returns:
        BenchmarkResult
    """
    runner = BenchmarkRunner(warmup_runs=warmup_runs, num_runs=num_runs)
    return runner.run(name, func, *args, **kwargs)


def compare_benchmarks(
    results: List[BenchmarkResult]
) -> Dict[str, Any]:
    """
    Compare multiple benchmark results.
    
    Args:
        results: List of benchmark results
    
    Returns:
        Comparison dictionary
    """
    runner = BenchmarkRunner()
    return runner.compare(results)













