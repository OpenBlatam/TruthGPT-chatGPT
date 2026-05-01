"""
Profiling utilities for optimization_core.

Provides utilities for code profiling and performance analysis.

This module re-exports functionality from modules.base.core_system.core.performance_utils for backward compatibility.
New code should import directly from optimization_core.core.performance_utils.
"""

# Re-export from core module for backward compatibility
try:
    from optimization_core.core.performance_utils import (
        profile_context,
        profile_function,
        profile_decorator,
        FunctionProfiler as PerformanceProfiler,
    )
except ImportError:
    # Fallback implementation if core module not available
    import logging
    import time
    import cProfile
    import pstats
    import io
    from typing import Dict, Any, Optional, Callable
    from contextlib import contextmanager
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    
    @contextmanager
    def profile_context(
        output_file: Optional[Path] = None,
        sort_by: str = "cumulative",
        lines: int = 50
    ):
        """Context manager for profiling code."""
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            yield profiler
        finally:
            profiler.disable()
            
            stats = pstats.Stats(profiler)
            stats.sort_stats(sort_by)
            stats.print_stats(lines)
            
            if output_file:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                stats.dump_stats(str(output_file))
                logger.info(f"Profile saved to {output_file}")
    
    def profile_function(
        func: Callable,
        *args,
        output_file: Optional[Path] = None,
        **kwargs
    ) -> Any:
        """Profile a function call."""
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.disable()
            
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumulative")
            stats.print_stats(20)
            
            if output_file:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                stats.dump_stats(str(output_file))
    
    class PerformanceProfiler:
        """Performance profiler for tracking function calls."""
        
        def __init__(self):
            """Initialize profiler."""
            self.call_counts: Dict[str, int] = {}
            self.total_times: Dict[str, float] = {}
            self.min_times: Dict[str, float] = {}
            self.max_times: Dict[str, float] = {}
        
        def record_call(self, function_name: str, duration: float):
            """Record a function call."""
            if function_name not in self.call_counts:
                self.call_counts[function_name] = 0
                self.total_times[function_name] = 0.0
                self.min_times[function_name] = float('inf')
                self.max_times[function_name] = 0.0
            
            self.call_counts[function_name] += 1
            self.total_times[function_name] += duration
            self.min_times[function_name] = min(self.min_times[function_name], duration)
            self.max_times[function_name] = max(self.max_times[function_name], duration)
        
        def get_stats(self, function_name: str) -> Dict[str, Any]:
            """Get statistics for a function."""
            if function_name not in self.call_counts:
                return {}
            
            count = self.call_counts[function_name]
            total = self.total_times[function_name]
            
            return {
                "call_count": count,
                "total_time": total,
                "avg_time": total / count if count > 0 else 0.0,
                "min_time": self.min_times[function_name] if self.min_times[function_name] != float('inf') else 0.0,
                "max_time": self.max_times[function_name],
            }
        
        def get_summary(self) -> Dict[str, Dict[str, Any]]:
            """Get summary of all functions."""
            return {
                name: self.get_stats(name)
                for name in self.call_counts.keys()
            }
        
        def print_summary(self):
            """Print summary to console."""
            summary = self.get_summary()
            
            print("\n=== Performance Profile ===")
            for name, stats in sorted(summary.items(), key=lambda x: x[1]["total_time"], reverse=True):
                print(f"\n{name}:")
                print(f"  Calls: {stats['call_count']}")
                print(f"  Total: {stats['total_time']:.3f}s")
                print(f"  Avg: {stats['avg_time']:.3f}s")
                print(f"  Min: {stats['min_time']:.3f}s")
                print(f"  Max: {stats['max_time']:.3f}s")
        
        def reset(self):
            """Reset all statistics."""
            self.call_counts.clear()
            self.total_times.clear()
            self.min_times.clear()
            self.max_times.clear()
    
    def profile_decorator(output_file: Optional[Path] = None):
        """Decorator for profiling functions."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                return profile_function(func, *args, output_file=output_file, **kwargs)
            return wrapper
        return decorator


__all__ = [
    'profile_context',
    'profile_function',
    'profile_decorator',
    'PerformanceProfiler',
]

