"""
Production Testing System - Comprehensive testing and benchmarking framework
Provides unit tests, integration tests, performance benchmarks, and regression testing
"""

import torch
import torch.nn as nn
import time
import json
import logging
import threading
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import psutil
from enum import Enum
import traceback
from contextlib import contextmanager
import unittest
from unittest.mock import Mock, patch
import tempfile
import shutil

logger = logging.getLogger(__name__)

class TestType(Enum):
    """Types of tests."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    REGRESSION = "regression"
    STRESS = "stress"
    LOAD = "load"

class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    """Test execution result."""
    test_name: str
    test_type: TestType
    status: TestStatus
    duration: float
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0

@dataclass
class BenchmarkResult:
    """Benchmark execution result."""
    benchmark_name: str
    duration: float
    throughput: float
    memory_usage: float
    cpu_usage: float
    gpu_memory_usage: float
    accuracy: Optional[float] = None
    error_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ProductionTestSuite:
    """Production-grade testing framework."""
    
    def __init__(self, 
                 test_directory: str = "./tests",
                 enable_parallel: bool = True,
                 max_workers: int = 4,
                 enable_benchmarking: bool = True,
                 enable_regression_testing: bool = True):
        
        self.test_directory = Path(test_directory)
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.enable_benchmarking = enable_benchmarking
        self.enable_regression_testing = enable_regression_testing
        
        # Test state
        self.tests = []
        self.benchmarks = []
        self.test_results = []
        self.benchmark_results = []
        self.regression_baselines = {}
        
        # Performance monitoring
        self.performance_monitor = None
        self.test_lock = threading.Lock()
        
        # Setup test directory
        self._setup_test_directory()
        
        # Load regression baselines
        if self.enable_regression_testing:
            self._load_regression_baselines()
        
        logger.info("🧪 Production Test Suite initialized")
    
    def _setup_test_directory(self):
        """Setup test directory structure."""
        self.test_directory.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.test_directory / "unit").mkdir(exist_ok=True)
        (self.test_directory / "integration").mkdir(exist_ok=True)
        (self.test_directory / "performance").mkdir(exist_ok=True)
        (self.test_directory / "results").mkdir(exist_ok=True)
        (self.test_directory / "baselines").mkdir(exist_ok=True)
    
    def _load_regression_baselines(self):
        """Load regression testing baselines."""
        baselines_file = self.test_directory / "baselines" / "regression_baselines.json"
        
        if baselines_file.exists():
            try:
                with open(baselines_file, 'r') as f:
                    self.regression_baselines = json.load(f)
                logger.info(f"📊 Loaded {len(self.regression_baselines)} regression baselines")
            except Exception as e:
                logger.warning(f"Failed to load regression baselines: {e}")
    
    def add_test(self, 
                 test_func: Callable,
                 test_name: str,
                 test_type: TestType = TestType.UNIT,
                 timeout: float = 300.0,
                 retry_count: int = 0):
        """Add a test to the test suite."""
        test_info = {
            'func': test_func,
            'name': test_name,
            'type': test_type,
            'timeout': timeout,
            'retry_count': retry_count
        }
        
        with self.test_lock:
            self.tests.append(test_info)
        
        logger.info(f"➕ Added {test_type.value} test: {test_name}")
    
    def add_benchmark(self,
                     benchmark_func: Callable,
                     benchmark_name: str,
                     iterations: int = 10,
                     warmup_iterations: int = 2):
        """Add a benchmark to the test suite."""
        benchmark_info = {
            'func': benchmark_func,
            'name': benchmark_name,
            'iterations': iterations,
            'warmup_iterations': warmup_iterations
        }
        
        self.benchmarks.append(benchmark_info)
        logger.info(f"📊 Added benchmark: {benchmark_name}")
    
    def run_tests(self, 
                  test_types: Optional[List[TestType]] = None,
                  test_names: Optional[List[str]] = None,
                  parallel: bool = None) -> List[TestResult]:
        """Run tests with specified filters."""
        if parallel is None:
            parallel = self.enable_parallel
        
        # Filter tests
        filtered_tests = self._filter_tests(test_types, test_names)
        
        if not filtered_tests:
            logger.warning("No tests found matching criteria")
            return []
        
        logger.info(f"🚀 Running {len(filtered_tests)} tests")
        
        if parallel and len(filtered_tests) > 1:
            return self._run_tests_parallel(filtered_tests)
        else:
            return self._run_tests_sequential(filtered_tests)
    
    def _filter_tests(self, 
                      test_types: Optional[List[TestType]] = None,
                      test_names: Optional[List[str]] = None) -> List[Dict]:
        """Filter tests based on criteria."""
        filtered = []
        
        for test in self.tests:
            # Filter by type
            if test_types and test['type'] not in test_types:
                continue
            
            # Filter by name
            if test_names and test['name'] not in test_names:
                continue
            
            filtered.append(test)
        
        return filtered
    
    def _run_tests_sequential(self, tests: List[Dict]) -> List[TestResult]:
        """Run tests sequentially."""
        results = []
        
        for test in tests:
            result = self._execute_test(test)
            results.append(result)
            
            # Log result
            status_emoji = "✅" if result.status == TestStatus.PASSED else "❌"
            logger.info(f"{status_emoji} {result.test_name}: {result.status.value} ({result.duration:.2f}s)")
        
        return results
    
    def _run_tests_parallel(self, tests: List[Dict]) -> List[TestResult]:
        """Run tests in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tests
            future_to_test = {
                executor.submit(self._execute_test, test): test 
                for test in tests
            }
            
            # Collect results
            for future in as_completed(future_to_test):
                test = future_to_test[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Log result
                    status_emoji = "✅" if result.status == TestStatus.PASSED else "❌"
                    logger.info(f"{status_emoji} {result.test_name}: {result.status.value} ({result.duration:.2f}s)")
                    
                except Exception as e:
                    logger.error(f"Test execution failed for {test['name']}: {e}")
                    results.append(TestResult(
                        test_name=test['name'],
                        test_type=test['type'],
                        status=TestStatus.ERROR,
                        duration=0.0,
                        error_message=str(e)
                    ))
        
        return results
    
    def _execute_test(self, test: Dict) -> TestResult:
        """Execute a single test."""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        start_gpu_memory = self._get_gpu_memory_usage()
        
        result = TestResult(
            test_name=test['name'],
            test_type=test['type'],
            status=TestStatus.RUNNING,
            duration=0.0
        )
        
        try:
            # Execute test with timeout
            test_func = test['func']
            timeout = test['timeout']
            
            # Run test
            test_func()
            
            # Calculate metrics
            duration = time.time() - start_time
            end_memory = psutil.virtual_memory().used
            end_gpu_memory = self._get_gpu_memory_usage()
            
            result.status = TestStatus.PASSED
            result.duration = duration
            result.memory_usage = (end_memory - start_memory) / (1024 * 1024)  # MB
            result.gpu_memory_usage = end_gpu_memory - start_gpu_memory
            
        except Exception as e:
            duration = time.time() - start_time
            result.status = TestStatus.FAILED
            result.duration = duration
            result.error_message = str(e)
            result.performance_metrics = {
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
        
        return result
    
    def run_benchmarks(self, 
                      benchmark_names: Optional[List[str]] = None) -> List[BenchmarkResult]:
        """Run performance benchmarks."""
        if not self.enable_benchmarking:
            logger.warning("Benchmarking is disabled")
            return []
        
        # Filter benchmarks
        filtered_benchmarks = self.benchmarks
        if benchmark_names:
            filtered_benchmarks = [
                b for b in self.benchmarks 
                if b['name'] in benchmark_names
            ]
        
        if not filtered_benchmarks:
            logger.warning("No benchmarks found")
            return []
        
        logger.info(f"📊 Running {len(filtered_benchmarks)} benchmarks")
        
        results = []
        for benchmark in filtered_benchmarks:
            result = self._execute_benchmark(benchmark)
            results.append(result)
            
            logger.info(f"📈 {result.benchmark_name}: {result.throughput:.2f} ops/s ({result.duration:.2f}s)")
        
        return results
    
    def _execute_benchmark(self, benchmark: Dict) -> BenchmarkResult:
        """Execute a single benchmark."""
        benchmark_func = benchmark['func']
        iterations = benchmark['iterations']
        warmup_iterations = benchmark['warmup_iterations']
        
        # Warmup
        for _ in range(warmup_iterations):
            try:
                benchmark_func()
            except Exception:
                pass
        
        # Benchmark
        times = []
        memory_usage = []
        gpu_memory_usage = []
        
        for i in range(iterations):
            start_time = time.time()
            start_memory = psutil.virtual_memory().used
            start_gpu_memory = self._get_gpu_memory_usage()
            
            try:
                benchmark_func()
                duration = time.time() - start_time
                end_memory = psutil.virtual_memory().used
                end_gpu_memory = self._get_gpu_memory_usage()
                
                times.append(duration)
                memory_usage.append((end_memory - start_memory) / (1024 * 1024))
                gpu_memory_usage.append(end_gpu_memory - start_gpu_memory)
                
            except Exception as e:
                logger.warning(f"Benchmark iteration {i} failed: {e}")
        
        # Calculate results
        if not times:
            return BenchmarkResult(
                benchmark_name=benchmark['name'],
                duration=0.0,
                throughput=0.0,
                memory_usage=0.0,
                cpu_usage=0.0,
                gpu_memory_usage=0.0,
                error_rate=1.0
            )
        
        total_duration = sum(times)
        avg_duration = np.mean(times)
        throughput = iterations / total_duration if total_duration > 0 else 0
        
        return BenchmarkResult(
            benchmark_name=benchmark['name'],
            duration=total_duration,
            throughput=throughput,
            memory_usage=np.mean(memory_usage) if memory_usage else 0.0,
            cpu_usage=psutil.cpu_percent(),
            gpu_memory_usage=np.mean(gpu_memory_usage) if gpu_memory_usage else 0.0,
            error_rate=1.0 - (len(times) / iterations),
            metadata={
                'iterations': iterations,
                'warmup_iterations': warmup_iterations,
                'avg_iteration_time': avg_duration,
                'min_time': np.min(times),
                'max_time': np.max(times),
                'std_time': np.std(times)
            }
        )
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0
    
    def run_regression_tests(self) -> Dict[str, Any]:
        """Run regression tests against baselines."""
        if not self.enable_regression_testing:
            logger.warning("Regression testing is disabled")
            return {}
        
        logger.info("🔄 Running regression tests")
        
        regression_results = {
            'passed': 0,
            'failed': 0,
            'new_tests': 0,
            'improvements': [],
            'regressions': []
        }
        
        # Run current benchmarks
        current_results = self.run_benchmarks()
        
        for result in current_results:
            baseline_key = result.benchmark_name
            
            if baseline_key in self.regression_baselines:
                baseline = self.regression_baselines[baseline_key]
                
                # Compare performance
                performance_change = (result.throughput - baseline['throughput']) / baseline['throughput']
                memory_change = (result.memory_usage - baseline['memory_usage']) / baseline['memory_usage']
                
                # Check for regressions (performance decrease > 10% or memory increase > 20%)
                if performance_change < -0.1 or memory_change > 0.2:
                    regression_results['regressions'].append({
                        'benchmark': result.benchmark_name,
                        'performance_change': performance_change,
                        'memory_change': memory_change
                    })
                    regression_results['failed'] += 1
                else:
                    regression_results['passed'] += 1
                    
                    if performance_change > 0.05:  # 5% improvement
                        regression_results['improvements'].append({
                            'benchmark': result.benchmark_name,
                            'performance_improvement': performance_change
                        })
            else:
                regression_results['new_tests'] += 1
                # Add to baselines
                self.regression_baselines[baseline_key] = {
                    'throughput': result.throughput,
                    'memory_usage': result.memory_usage,
                    'timestamp': time.time()
                }
        
        # Save updated baselines
        self._save_regression_baselines()
        
        return regression_results
    
    def _save_regression_baselines(self):
        """Save regression testing baselines."""
        baselines_file = self.test_directory / "baselines" / "regression_baselines.json"
        
        try:
            with open(baselines_file, 'w') as f:
                json.dump(self.regression_baselines, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save regression baselines: {e}")
    
    def generate_test_report(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not results:
            return {}
        
        # Calculate statistics
        total_tests = len(results)
        passed_tests = len([r for r in results if r.status == TestStatus.PASSED])
        failed_tests = len([r for r in results if r.status == TestStatus.FAILED])
        error_tests = len([r for r in results if r.status == TestStatus.ERROR])
        
        # Performance metrics
        durations = [r.duration for r in results if r.duration > 0]
        memory_usage = [r.memory_usage for r in results if r.memory_usage > 0]
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'errors': error_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0
            },
            'performance': {
                'total_duration': sum(durations),
                'avg_duration': np.mean(durations) if durations else 0,
                'max_duration': np.max(durations) if durations else 0,
                'min_duration': np.min(durations) if durations else 0,
                'total_memory_usage': sum(memory_usage),
                'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0
            },
            'test_results': [
                {
                    'name': r.test_name,
                    'type': r.test_type.value,
                    'status': r.status.value,
                    'duration': r.duration,
                    'memory_usage': r.memory_usage,
                    'error_message': r.error_message
                } for r in results
            ]
        }
        
        return report
    
    def save_test_results(self, results: List[TestResult], filepath: str):
        """Save test results to file."""
        report = self.generate_test_report(results)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📊 Test results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")
    
    def cleanup(self):
        """Cleanup test resources."""
        # Cleanup temporary files
        # Force garbage collection
        gc.collect()
        
        logger.info("🧹 Test suite cleanup completed")

# Factory functions
def create_production_test_suite(**kwargs) -> ProductionTestSuite:
    """Create a production test suite instance."""
    return ProductionTestSuite(**kwargs)

# Context manager
@contextmanager
def production_testing_context(**kwargs):
    """Context manager for production testing."""
    test_suite = create_production_test_suite(**kwargs)
    try:
        yield test_suite
    finally:
        test_suite.cleanup()

# Example test functions
def test_optimization_basic():
    """Basic optimization test."""
    import torch
    import torch.nn as nn
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    
    # Test optimization
    from production_optimizer import create_production_optimizer
    
    optimizer = create_production_optimizer()
    optimized_model = optimizer.optimize_model(model)
    
    # Verify model still works
    test_input = torch.randn(1, 10)
    with torch.no_grad():
        output = optimized_model(test_input)
    
    assert output.shape == (1, 1), f"Expected output shape (1, 1), got {output.shape}"

def benchmark_optimization_performance():
    """Benchmark optimization performance."""
    import torch
    import torch.nn as nn
    
    # Create larger model for benchmarking
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 25),
        nn.ReLU(),
        nn.Linear(25, 10)
    )
    
    from production_optimizer import create_production_optimizer
    
    optimizer = create_production_optimizer()
    optimized_model = optimizer.optimize_model(model)
    
    # Benchmark forward pass
    test_input = torch.randn(32, 100)
    with torch.no_grad():
        _ = optimized_model(test_input)

if __name__ == "__main__":
    print("🧪 Production Testing System")
    print("=" * 40)
    
    # Example usage
    with production_testing_context() as test_suite:
        print("✅ Production test suite created")
        
        # Add tests
        test_suite.add_test(test_optimization_basic, "basic_optimization", TestType.UNIT)
        test_suite.add_benchmark(benchmark_optimization_performance, "optimization_performance")
        
        # Run tests
        test_results = test_suite.run_tests()
        print(f"🧪 Ran {len(test_results)} tests")
        
        # Run benchmarks
        benchmark_results = test_suite.run_benchmarks()
        print(f"📊 Ran {len(benchmark_results)} benchmarks")
        
        # Generate report
        report = test_suite.generate_test_report(test_results)
        print(f"📈 Test success rate: {report['summary']['success_rate']:.2%}")
        
        print("✅ Production testing example completed")

