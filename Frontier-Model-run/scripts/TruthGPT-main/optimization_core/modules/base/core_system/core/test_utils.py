"""
Common testing utilities for optimization_core.

Provides reusable testing helpers, fixtures, and utilities
for consistent testing across all modules.
"""

import logging
import time
import gc
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, AsyncMock, patch

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from .types import DictStrAny, Number

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ════════════════════════════════════════════════════════════════════════════════
# TEST CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

def create_test_config(**kwargs) -> DictStrAny:
    """
    Create test configuration with defaults.
    
    Args:
        **kwargs: Override default values
    
    Returns:
        Test configuration dictionary
    
    Example:
        >>> config = create_test_config(batch_size=4, seq_len=256)
    """
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


# ════════════════════════════════════════════════════════════════════════════════
# TENSOR/NUMPY ASSERTIONS
# ════════════════════════════════════════════════════════════════════════════════

def assert_tensor_close(
    tensor1: Any,
    tensor2: Any,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> bool:
    """
    Assert two tensors/arrays are close within tolerance.
    
    Args:
        tensor1: First tensor/array
        tensor2: Second tensor/array
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        True if close, raises AssertionError otherwise
    """
    if TORCH_AVAILABLE and isinstance(tensor1, torch.Tensor):
        if not isinstance(tensor2, torch.Tensor):
            tensor2 = torch.tensor(tensor2)
        assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), \
            f"Tensors not close: {tensor1} vs {tensor2}"
        return True
    elif NUMPY_AVAILABLE:
        if not isinstance(tensor1, np.ndarray):
            tensor1 = np.array(tensor1)
        if not isinstance(tensor2, np.ndarray):
            tensor2 = np.array(tensor2)
        np.testing.assert_allclose(tensor1, tensor2, rtol=rtol, atol=atol)
        return True
    else:
        # Fallback comparison
        assert abs(tensor1 - tensor2) < atol, \
            f"Values not close: {tensor1} vs {tensor2}"
        return True


def assert_shape_equal(tensor: Any, expected_shape: Tuple[int, ...]) -> bool:
    """
    Assert tensor/array has expected shape.
    
    Args:
        tensor: Tensor/array to check
        expected_shape: Expected shape tuple
    
    Returns:
        True if shape matches, raises AssertionError otherwise
    """
    if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        assert tensor.shape == expected_shape, \
            f"Shape mismatch: {tensor.shape} != {expected_shape}"
    elif NUMPY_AVAILABLE:
        if not isinstance(tensor, np.ndarray):
            tensor = np.array(tensor)
        assert tensor.shape == expected_shape, \
            f"Shape mismatch: {tensor.shape} != {expected_shape}"
    else:
        # Fallback for lists
        if isinstance(tensor, list):
            assert len(tensor) == expected_shape[0], \
                f"Length mismatch: {len(tensor)} != {expected_shape[0]}"
    return True


def assert_dtype_equal(tensor: Any, expected_dtype: Any) -> bool:
    """
    Assert tensor/array has expected dtype.
    
    Args:
        tensor: Tensor/array to check
        expected_dtype: Expected dtype
    
    Returns:
        True if dtype matches, raises AssertionError otherwise
    """
    if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        assert tensor.dtype == expected_dtype, \
            f"Dtype mismatch: {tensor.dtype} != {expected_dtype}"
    elif NUMPY_AVAILABLE:
        if not isinstance(tensor, np.ndarray):
            tensor = np.array(tensor)
        assert tensor.dtype == expected_dtype, \
            f"Dtype mismatch: {tensor.dtype} != {expected_dtype}"
    return True


# ════════════════════════════════════════════════════════════════════════════════
# PERFORMANCE MEASUREMENT
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class PerformanceMetrics:
    """Performance metrics from function execution."""
    execution_time: float
    memory_used: float = 0.0
    start_memory: float = 0.0
    end_memory: float = 0.0
    result: Any = None


def measure_execution_time(
    func: Callable,
    *args,
    **kwargs
) -> PerformanceMetrics:
    """
    Measure execution time and memory usage of a function.
    
    Args:
        func: Function to measure
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        PerformanceMetrics object
    
    Example:
        >>> metrics = measure_execution_time(my_function, arg1, arg2)
        >>> print(f"Time: {metrics.execution_time:.3f}s")
    """
    start_time = time.perf_counter()
    start_memory = 0.0
    
    if PSUTIL_AVAILABLE:
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    result = func(*args, **kwargs)
    
    end_time = time.perf_counter()
    end_memory = 0.0
    
    if PSUTIL_AVAILABLE:
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    return PerformanceMetrics(
        execution_time=end_time - start_time,
        memory_used=end_memory - start_memory,
        start_memory=start_memory,
        end_memory=end_memory,
        result=result
    )


def compare_performance(
    baseline_func: Callable,
    optimized_func: Callable,
    *args,
    **kwargs
) -> DictStrAny:
    """
    Compare performance between baseline and optimized functions.
    
    Args:
        baseline_func: Baseline function
        optimized_func: Optimized function
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        Comparison dictionary with metrics
    
    Example:
        >>> comparison = compare_performance(baseline, optimized, data)
        >>> print(f"Speedup: {comparison['speedup']:.2f}x")
    """
    baseline_metrics = measure_execution_time(baseline_func, *args, **kwargs)
    optimized_metrics = measure_execution_time(optimized_func, *args, **kwargs)
    
    speedup = baseline_metrics.execution_time / optimized_metrics.execution_time \
        if optimized_metrics.execution_time > 0 else float('inf')
    
    memory_improvement = 0.0
    if baseline_metrics.memory_used > 0:
        memory_improvement = (
            (baseline_metrics.memory_used - optimized_metrics.memory_used) /
            baseline_metrics.memory_used
        )
    
    return {
        'baseline': {
            'execution_time': baseline_metrics.execution_time,
            'memory_used': baseline_metrics.memory_used,
        },
        'optimized': {
            'execution_time': optimized_metrics.execution_time,
            'memory_used': optimized_metrics.memory_used,
        },
        'speedup': speedup,
        'memory_improvement': memory_improvement,
        'is_faster': speedup > 1.0,
        'uses_less_memory': memory_improvement > 0,
    }


# ════════════════════════════════════════════════════════════════════════════════
# MEMORY TRACKING
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class MemorySnapshot:
    """Memory snapshot at a point in time."""
    label: str
    timestamp: float
    rss: float  # MB
    vms: float = 0.0  # MB


class MemoryTracker:
    """Memory tracking utility for testing."""
    
    def __init__(self):
        self.memory_snapshots: List[MemorySnapshot] = []
        self.peak_memory: float = 0.0
    
    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """
        Take memory snapshot.
        
        Args:
            label: Label for this snapshot
        
        Returns:
            MemorySnapshot object
        """
        if not PSUTIL_AVAILABLE:
            return MemorySnapshot(label=label, timestamp=time.time(), rss=0.0)
        
        memory_info = psutil.Process().memory_info()
        snapshot = MemorySnapshot(
            label=label,
            timestamp=time.time(),
            rss=memory_info.rss / 1024 / 1024,  # MB
            vms=memory_info.vms / 1024 / 1024,  # MB
        )
        
        self.memory_snapshots.append(snapshot)
        self.peak_memory = max(self.peak_memory, snapshot.rss)
        
        return snapshot
    
    @contextmanager
    def track_memory(self, label: str = ""):
        """
        Context manager for memory tracking.
        
        Args:
            label: Label for this tracking session
        
        Example:
            >>> tracker = MemoryTracker()
            >>> with tracker.track_memory("test_operation"):
            ...     result = expensive_operation()
        """
        self.take_snapshot(f"{label}_start")
        try:
            yield
        finally:
            self.take_snapshot(f"{label}_end")
    
    def get_memory_summary(self) -> DictStrAny:
        """
        Get memory usage summary.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.memory_snapshots:
            return {}
        
        rss_values = [s.rss for s in self.memory_snapshots]
        
        return {
            'snapshots_taken': len(self.memory_snapshots),
            'peak_memory': self.peak_memory,
            'min_memory': min(rss_values),
            'max_memory': max(rss_values),
            'average_memory': sum(rss_values) / len(rss_values),
        }


# ════════════════════════════════════════════════════════════════════════════════
# MOCK HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def create_mock_service(service_class: type, **methods) -> Mock:
    """
    Create a mock service with specified methods.
    
    Args:
        service_class: Service class to mock
        **methods: Methods to configure on the mock
    
    Returns:
        Mock service instance
    
    Example:
        >>> mock_service = create_mock_service(
        ...     MyService,
        ...     process=lambda x: x * 2,
        ...     validate=lambda x: True
        ... )
    """
    mock = Mock(spec=service_class)
    for method_name, return_value in methods.items():
        if callable(return_value):
            setattr(mock, method_name, Mock(side_effect=return_value))
        else:
            setattr(mock, method_name, Mock(return_value=return_value))
    return mock


def create_async_mock_service(service_class: type, **methods) -> AsyncMock:
    """
    Create an async mock service with specified methods.
    
    Args:
        service_class: Service class to mock
        **methods: Methods to configure on the mock
    
    Returns:
        AsyncMock service instance
    """
    mock = AsyncMock(spec=service_class)
    for method_name, return_value in methods.items():
        if callable(return_value):
            setattr(mock, method_name, AsyncMock(side_effect=return_value))
        else:
            setattr(mock, method_name, AsyncMock(return_value=return_value))
    return mock


# ════════════════════════════════════════════════════════════════════════════════
# TEST DECORATORS
# ════════════════════════════════════════════════════════════════════════════════

def retry_test(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry flaky tests.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between retries in seconds
    
    Example:
        >>> @retry_test(max_attempts=3, delay=1.0)
        >>> def test_flaky_operation():
        ...     assert operation() == expected
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Test {func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Test {func.__name__} failed after {max_attempts} attempts: {e}"
                        )
            raise last_exception
        return wrapper
    return decorator


def skip_if_unavailable(*modules: str):
    """
    Decorator to skip test if required modules are unavailable.
    
    Args:
        *modules: Module names to check
    
    Example:
        >>> @skip_if_unavailable('torch', 'numpy')
        >>> def test_with_dependencies():
        ...     ...
    """
    def decorator(func: Callable) -> Callable:
        import importlib
        missing = []
        for module_name in modules:
            try:
                importlib.import_module(module_name)
            except ImportError:
                missing.append(module_name)
        
        if missing:
            import pytest
            return pytest.mark.skip(reason=f"Missing modules: {', '.join(missing)}")(func)
        return func
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# CONTEXT MANAGERS
# ════════════════════════════════════════════════════════════════════════════════

@contextmanager
def performance_context(operation_name: str):
    """
    Context manager for performance monitoring.
    
    Args:
        operation_name: Name of the operation
    
    Example:
        >>> with performance_context("data_processing"):
        ...     result = process_data(data)
    """
    start_time = time.perf_counter()
    tracker = MemoryTracker()
    tracker.take_snapshot(f"{operation_name}_start")
    
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        tracker.take_snapshot(f"{operation_name}_end")
        summary = tracker.get_memory_summary()
        
        logger.info(
            f"{operation_name}: {elapsed:.3f}s, "
            f"memory: {summary.get('max_memory', 0):.1f}MB"
        )


@contextmanager
def cleanup_context():
    """
    Context manager for cleanup (GC, etc.).
    
    Example:
        >>> with cleanup_context():
        ...     result = create_large_object()
    """
    try:
        yield
    finally:
        gc.collect()


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Configuration
    "create_test_config",
    # Assertions
    "assert_tensor_close",
    "assert_shape_equal",
    "assert_dtype_equal",
    # Performance
    "measure_execution_time",
    "compare_performance",
    "PerformanceMetrics",
    # Memory
    "MemoryTracker",
    "MemorySnapshot",
    # Mocks
    "create_mock_service",
    "create_async_mock_service",
    # Decorators
    "retry_test",
    "skip_if_unavailable",
    # Context managers
    "performance_context",
    "cleanup_context",
]













