"""
Automatic optimization for polyglot_core.

Provides automatic backend selection optimization, performance tuning,
and adaptive configuration.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import time
import numpy as np


@dataclass
class OptimizationResult:
    """Result of optimization."""
    best_backend: str
    performance_gain: float
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class AutoOptimizer:
    """
    Automatic optimizer for polyglot_core.
    
    Automatically selects best backends and configurations based on
    performance profiling and benchmarking.
    """
    
    def __init__(self, sample_size: int = 100):
        """
        Initialize optimizer.
        
        Args:
            sample_size: Number of samples for optimization
        """
        self.sample_size = sample_size
        self._cache: Dict[str, OptimizationResult] = {}
    
    def optimize_backend(
        self,
        feature: str,
        operation: Callable,
        backends: List[str],
        iterations: int = 10
    ) -> OptimizationResult:
        """
        Optimize backend selection for a feature.
        
        Args:
            feature: Feature name
            operation: Operation function (takes backend as argument)
            backends: List of backends to test
            iterations: Number of iterations per backend
            
        Returns:
            OptimizationResult
        """
        if feature in self._cache:
            return self._cache[feature]
        
        results = {}
        
        for backend in backends:
            try:
                times = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    operation(backend)
                    times.append((time.perf_counter() - start) * 1000)  # ms
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                results[backend] = {
                    'avg_time_ms': avg_time,
                    'std_time_ms': std_time,
                    'min_time_ms': np.min(times),
                    'max_time_ms': np.max(times)
                }
            except Exception as e:
                results[backend] = {
                    'error': str(e),
                    'avg_time_ms': float('inf')
                }
        
        # Find best backend
        valid_results = {
            k: v for k, v in results.items()
            if 'error' not in v
        }
        
        if not valid_results:
            raise RuntimeError("No valid backends found")
        
        best_backend = min(valid_results, key=lambda x: valid_results[x]['avg_time_ms'])
        best_time = valid_results[best_backend]['avg_time_ms']
        
        # Calculate performance gain vs baseline (Python)
        baseline_time = valid_results.get('python', {}).get('avg_time_ms', best_time)
        if baseline_time > 0:
            performance_gain = (baseline_time - best_time) / baseline_time * 100
        else:
            performance_gain = 0.0
        
        # Generate recommendations
        recommendations = []
        if performance_gain > 50:
            recommendations.append(f"Using {best_backend} provides {performance_gain:.1f}% speedup")
        elif performance_gain < 0:
            recommendations.append(f"Consider using Python backend (current best is slower)")
        
        result = OptimizationResult(
            best_backend=best_backend,
            performance_gain=performance_gain,
            recommendations=recommendations,
            metrics=results
        )
        
        self._cache[feature] = result
        return result
    
    def optimize_cache_config(
        self,
        max_size_range: List[int],
        test_operation: Callable
    ) -> Dict[str, Any]:
        """
        Optimize cache configuration.
        
        Args:
            max_size_range: Range of max_size values to test
            test_operation: Test operation (takes max_size as argument)
            
        Returns:
            Optimal configuration
        """
        results = {}
        
        for max_size in max_size_range:
            try:
                times = []
                for _ in range(5):
                    start = time.perf_counter()
                    test_operation(max_size)
                    times.append((time.perf_counter() - start) * 1000)
                
                results[max_size] = {
                    'avg_time_ms': np.mean(times),
                    'std_time_ms': np.std(times)
                }
            except Exception as e:
                results[max_size] = {'error': str(e)}
        
        # Find optimal (balance between size and performance)
        valid_results = {
            k: v for k, v in results.items()
            if 'error' not in v
        }
        
        if not valid_results:
            return {'max_size': max_size_range[0]}
        
        # Choose size with good performance (not necessarily fastest)
        optimal = min(
            valid_results.items(),
            key=lambda x: x[1]['avg_time_ms'] / x[0]  # Time per size unit
        )
        
        return {
            'max_size': optimal[0],
            'expected_time_ms': optimal[1]['avg_time_ms']
        }
    
    def optimize_attention_config(
        self,
        d_model_range: List[int],
        n_heads_range: List[int],
        test_operation: Callable
    ) -> Dict[str, Any]:
        """
        Optimize attention configuration.
        
        Args:
            d_model_range: Range of d_model values
            n_heads_range: Range of n_heads values
            test_operation: Test operation (takes d_model, n_heads)
            
        Returns:
            Optimal configuration
        """
        results = {}
        
        for d_model in d_model_range:
            for n_heads in n_heads_range:
                if d_model % n_heads != 0:
                    continue
                
                try:
                    times = []
                    for _ in range(3):
                        start = time.perf_counter()
                        test_operation(d_model, n_heads)
                        times.append((time.perf_counter() - start) * 1000)
                    
                    results[(d_model, n_heads)] = {
                        'avg_time_ms': np.mean(times)
                    }
                except Exception as e:
                    results[(d_model, n_heads)] = {'error': str(e)}
        
        valid_results = {
            k: v for k, v in results.items()
            if 'error' not in v
        }
        
        if not valid_results:
            return {'d_model': d_model_range[0], 'n_heads': n_heads_range[0]}
        
        best = min(valid_results.items(), key=lambda x: x[1]['avg_time_ms'])
        d_model, n_heads = best[0]
        
        return {
            'd_model': d_model,
            'n_heads': n_heads,
            'expected_time_ms': best[1]['avg_time_ms']
        }
    
    def clear_cache(self):
        """Clear optimization cache."""
        self._cache.clear()


# Global optimizer
_global_optimizer = AutoOptimizer()


def get_optimizer() -> AutoOptimizer:
    """Get global optimizer."""
    return _global_optimizer


def optimize_backend(
    feature: str,
    operation: Callable,
    backends: List[str],
    iterations: int = 10
) -> OptimizationResult:
    """Convenience function to optimize backend."""
    return _global_optimizer.optimize_backend(feature, operation, backends, iterations)













