"""
Test utilities for TruthGPT optimization core testing
"""

import torch
import time
import psutil
import gc
from typing import Dict, Any, List, Optional, Callable
import numpy as np
from contextlib import contextmanager

class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def assert_tensor_close(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                          rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Assert two tensors are close within tolerance"""
        return torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    
    @staticmethod
    def assert_shape_equal(tensor: torch.Tensor, expected_shape: tuple) -> bool:
        """Assert tensor has expected shape"""
        return tensor.shape == expected_shape
    
    @staticmethod
    def assert_dtype_equal(tensor: torch.Tensor, expected_dtype: torch.dtype) -> bool:
        """Assert tensor has expected dtype"""
        return tensor.dtype == expected_dtype
    
    @staticmethod
    def create_test_config(**kwargs) -> Dict[str, Any]:
        """Create test configuration with defaults"""
        default_config = {
            'batch_size': 2,
            'seq_len': 128,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'max_epochs': 10,
            'device': 'cpu'
        }
        default_config.update(kwargs)
        return default_config
    
    @staticmethod
    def measure_execution_time(func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Measure execution time of a function"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'start_memory': start_memory,
            'end_memory': end_memory
        }
    
    @staticmethod
    def compare_performance(baseline_func: Callable, optimized_func: Callable, 
                           *args, **kwargs) -> Dict[str, Any]:
        """Compare performance between baseline and optimized functions"""
        baseline_metrics = TestUtils.measure_execution_time(baseline_func, *args, **kwargs)
        optimized_metrics = TestUtils.measure_execution_time(optimized_func, *args, **kwargs)
        
        speedup = baseline_metrics['execution_time'] / optimized_metrics['execution_time']
        memory_improvement = (baseline_metrics['memory_used'] - optimized_metrics['memory_used']) / baseline_metrics['memory_used']
        
        return {
            'baseline': baseline_metrics,
            'optimized': optimized_metrics,
            'speedup': speedup,
            'memory_improvement': memory_improvement,
            'is_faster': speedup > 1.0,
            'uses_less_memory': memory_improvement > 0
        }

class PerformanceProfiler:
    """Performance profiler for testing"""
    
    def __init__(self):
        self.profiles = []
        self.current_profile = None
        
    def start_profile(self, name: str):
        """Start profiling a section"""
        self.current_profile = {
            'name': name,
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024,
            'start_cpu': psutil.cpu_percent()
        }
    
    def end_profile(self) -> Dict[str, Any]:
        """End profiling and return metrics"""
        if self.current_profile is None:
            return {}
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        profile = {
            'name': self.current_profile['name'],
            'execution_time': end_time - self.current_profile['start_time'],
            'memory_used': end_memory - self.current_profile['start_memory'],
            'cpu_usage': end_cpu - self.current_profile['start_cpu'],
            'timestamp': end_time
        }
        
        self.profiles.append(profile)
        self.current_profile = None
        
        return profile
    
    def get_all_profiles(self) -> List[Dict[str, Any]]:
        """Get all profiling results"""
        return self.profiles.copy()
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get profiling summary"""
        if not self.profiles:
            return {}
        
        total_time = sum(p['execution_time'] for p in self.profiles)
        total_memory = sum(p['memory_used'] for p in self.profiles)
        
        return {
            'total_profiles': len(self.profiles),
            'total_execution_time': total_time,
            'total_memory_used': total_memory,
            'average_execution_time': total_time / len(self.profiles),
            'average_memory_used': total_memory / len(self.profiles)
        }

class MemoryTracker:
    """Memory tracking utility for testing"""
    
    def __init__(self):
        self.memory_snapshots = []
        self.peak_memory = 0
        
    def take_snapshot(self, label: str = ""):
        """Take memory snapshot"""
        memory_info = psutil.Process().memory_info()
        snapshot = {
            'label': label,
            'timestamp': time.time(),
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'peak_memory': memory_info.rss / 1024 / 1024  # MB
        }
        
        self.memory_snapshots.append(snapshot)
        self.peak_memory = max(self.peak_memory, snapshot['rss'])
        
        return snapshot
    
    def get_memory_growth(self) -> List[Dict[str, Any]]:
        """Get memory growth between snapshots"""
        if len(self.memory_snapshots) < 2:
            return []
        
        growth = []
        for i in range(1, len(self.memory_snapshots)):
            prev = self.memory_snapshots[i-1]
            curr = self.memory_snapshots[i]
            
            growth.append({
                'from': prev['label'],
                'to': curr['label'],
                'memory_growth': curr['rss'] - prev['rss'],
                'time_elapsed': curr['timestamp'] - prev['timestamp']
            })
        
        return growth
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary"""
        if not self.memory_snapshots:
            return {}
        
        rss_values = [s['rss'] for s in self.memory_snapshots]
        
        return {
            'snapshots_taken': len(self.memory_snapshots),
            'peak_memory': self.peak_memory,
            'min_memory': min(rss_values),
            'max_memory': max(rss_values),
            'average_memory': sum(rss_values) / len(rss_values),
            'memory_growth': self.get_memory_growth()
        }
    
    @contextmanager
    def track_memory(self, label: str = ""):
        """Context manager for memory tracking"""
        self.take_snapshot(f"{label}_start")
        try:
            yield
        finally:
            self.take_snapshot(f"{label}_end")

class TestAssertions:
    """Custom assertions for testing"""
    
    @staticmethod
    def assert_performance_improvement(baseline_time: float, optimized_time: float, 
                                     min_speedup: float = 1.1) -> bool:
        """Assert performance improvement"""
        speedup = baseline_time / optimized_time
        return speedup >= min_speedup
    
    @staticmethod
    def assert_memory_efficiency(baseline_memory: float, optimized_memory: float, 
                               max_memory_increase: float = 1.1) -> bool:
        """Assert memory efficiency"""
        memory_ratio = optimized_memory / baseline_memory
        return memory_ratio <= max_memory_increase
    
    @staticmethod
    def assert_numerical_stability(tensor: torch.Tensor, max_nan: int = 0, 
                                 max_inf: int = 0) -> bool:
        """Assert numerical stability"""
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        
        return nan_count <= max_nan and inf_count <= max_inf
    
    @staticmethod
    def assert_gradient_flow(gradients: List[torch.Tensor], 
                           min_grad_norm: float = 1e-6) -> bool:
        """Assert gradient flow"""
        total_grad_norm = 0
        for grad in gradients:
            if grad is not None:
                total_grad_norm += grad.norm().item() ** 2
        
        total_grad_norm = total_grad_norm ** 0.5
        return total_grad_norm >= min_grad_norm

class TestCoverageTracker:
    """Track test coverage and create reports"""
    
    def __init__(self):
        self.coverage_data = {}
        self.test_results = []
        
    def record_test(self, test_name: str, passed: bool, duration: float, 
                   coverage: float = 0.0):
        """Record a test result"""
        self.test_results.append({
            'name': test_name,
            'passed': passed,
            'duration': duration,
            'coverage': coverage
        })
        
    def calculate_total_coverage(self) -> Dict[str, Any]:
        """Calculate total test coverage"""
        if not self.test_results:
            return {'total_coverage': 0.0, 'total_tests': 0}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for t in self.test_results if t['passed'])
        avg_coverage = sum(t['coverage'] for t in self.test_results) / total_tests
        total_duration = sum(t['duration'] for t in self.test_results)
        
        return {
            'total_coverage': avg_coverage,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests) * 100,
            'total_duration': total_duration,
            'avg_duration': total_duration / total_tests
        }

class AdvancedTestDecorators:
    """Advanced decorators for testing"""
    
    @staticmethod
    def retry(max_attempts: int = 3, delay: float = 1.0):
        """Decorator to retry flaky tests"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            raise
                        time.sleep(delay)
                return None
            return wrapper
        return decorator
    
    @staticmethod
    def timeout(seconds: int):
        """Decorator to add timeout to tests"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                import signal
                
                def handler(signum, frame):
                    raise TimeoutError(f"Test timed out after {seconds} seconds")
                
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(seconds)
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def performance_test(baseline_time: float):
        """Decorator to compare against baseline performance"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                actual_time = end_time - start_time
                if actual_time > baseline_time * 1.2:  # 20% tolerance
                    raise AssertionError(
                        f"Performance regression: {actual_time:.3f}s vs baseline {baseline_time:.3f}s"
                    )
                
                return result
            return wrapper
        return decorator

class ParallelTestRunner:
    """Run tests in parallel for speed"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        
    def run_tests_parallel(self, test_functions: List[Callable]) -> List[Any]:
        """Run multiple test functions in parallel"""
        try:
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(test_func) for test_func in test_functions]
                results = [future.result() for future in futures]
            return results
        except ImportError:
            # Fallback to sequential execution
            return [test_func() for test_func in test_functions]

class TestVisualizer:
    """Visualize test results and metrics"""
    
    @staticmethod
    def create_results_summary(results: Dict[str, Any]) -> str:
        """Create a visual summary of test results"""
        summary = []
        summary.append("=" * 80)
        summary.append("TruthGPT Optimization Core Test Results")
        summary.append("=" * 80)
        summary.append("")
        
        # Test statistics
        if 'total_tests' in results:
            summary.append(f"Total Tests: {results['total_tests']}")
            summary.append(f"Passed: {results.get('total_tests', 0) - results.get('total_failures', 0) - results.get('total_errors', 0)}")
            summary.append(f"Failed: {results.get('total_failures', 0)}")
            summary.append(f"Errors: {results.get('total_errors', 0)}")
            summary.append(f"Success Rate: {results.get('success_rate', 0):.1f}%")
            summary.append("")
        
        # Performance metrics
        if 'performance_metrics' in results:
            summary.append("Performance Metrics:")
            summary.append(f"  Execution Time: {results['performance_metrics'].get('total_execution_time', 0):.2f}s")
            summary.append(f"  Memory Used: {results['performance_metrics'].get('total_memory_used', 0):.2f}MB")
            summary.append("")
        
        # Visual bars
        if 'total_tests' in results and results['total_tests'] > 0:
            passed = results['total_tests'] - results.get('total_failures', 0) - results.get('total_errors', 0)
            failed = results.get('total_failures', 0) + results.get('total_errors', 0)
            
            total = results['total_tests']
            passed_bars = int((passed / total) * 40)
            failed_bars = 40 - passed_bars
            
            summary.append("Status: " + "█" * passed_bars + "░" * failed_bars)
            summary.append("")
        
        summary.append("=" * 80)
        
        return "\n".join(summary)
    
    @staticmethod
    def create_performance_graph(profiles: List[Dict[str, Any]]) -> str:
        """Create ASCII performance graph"""
        if not profiles:
            return "No performance data available"
        
        # Simple ASCII bar chart
        graph = []
        graph.append("Performance Profile (Execution Time in seconds)")
        graph.append("")
        
        max_time = max(p.get('execution_time', 0) for p in profiles)
        
        for profile in profiles:
            name = profile.get('name', 'Unknown')
            time = profile.get('execution_time', 0)
            bars = int((time / max_time) * 40) if max_time > 0 else 0
            graph.append(f"{name:30s} {time:8.3f}s {'█' * bars}")
        
        return "\n".join(graph)



