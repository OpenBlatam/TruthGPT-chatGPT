"""
Polyglot Utility Functions

Shared utilities for polyglot integration.
"""
from typing import Optional, Dict, Any, List, Tuple
import logging
import time
import functools

logger = logging.getLogger(__name__)

def measure_time(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"{func.__name__} took {elapsed_ms:.2f}ms")
        return result
    return wrapper

def get_backend_preference() -> List[str]:
    """Get backend preference order based on availability."""
    try:
        from optimization_core.polyglot import get_available_backends
        backends = get_available_backends()
        
        preference = []
        if backends.get("cpp"):
            preference.append("cpp")
        if backends.get("rust"):
            preference.append("rust")
        if backends.get("julia"):
            preference.append("julia")
        preference.append("python")
        
        return preference
    except Exception:
        return ["python"]

def select_best_backend(
    available_backends: List[str],
    preference: Optional[List[str]] = None
) -> Optional[str]:
    """Select best backend from available options."""
    if preference is None:
        preference = get_backend_preference()
    
    for backend in preference:
        if backend in available_backends:
            return backend
    
    return None

def format_performance_stats(
    stats: Dict[str, Any],
    include_details: bool = False
) -> str:
    """Format performance statistics for logging."""
    lines = []
    
    if "throughput" in stats:
        lines.append(f"Throughput: {stats['throughput']:.2f} ops/s")
    
    if "latency_ms" in stats:
        lines.append(f"Latency: {stats['latency_ms']:.2f}ms")
    
    if "cache_hit_rate" in stats:
        lines.append(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
    
    if include_details and "details" in stats:
        lines.append(f"Details: {stats['details']}")
    
    return " | ".join(lines)

def create_backend_fallback(
    primary_backend: str,
    fallback_backend: str,
    func_name: str
):
    """Create a function with automatic fallback between backends."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"{func_name} failed with {primary_backend}: {e}, "
                    f"falling back to {fallback_backend}"
                )
                kwargs["backend"] = fallback_backend
                return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_backend(backend: str, available: List[str]) -> bool:
    """Validate that backend is available."""
    if backend not in available:
        logger.warning(
            f"Backend '{backend}' not available. "
            f"Available: {', '.join(available)}"
        )
        return False
    return True

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
        }
    except ImportError:
        logger.warning("psutil not available for memory stats")
        return {"rss_mb": 0.0, "vms_mb": 0.0}

def benchmark_backends(
    func,
    backends: List[str],
    *args,
    num_runs: int = 10,
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """Benchmark function across multiple backends."""
    results = {}
    
    for backend in backends:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            func(*args, backend=backend, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        
        results[backend] = {
            "avg_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
        }
    
    return results

__all__ = [
    "measure_time",
    "get_backend_preference",
    "select_best_backend",
    "format_performance_stats",
    "create_backend_fallback",
    "validate_backend",
    "get_memory_usage",
    "benchmark_backends",
]













