"""
Context management for polyglot_core.

Provides context managers for operations, profiling, and resource management.
"""

from contextlib import contextmanager
from typing import Optional, Dict, Any
import time


@contextmanager
def operation_context(operation_name: str, backend: Optional[str] = None, **kwargs):
    """
    Context manager for operations.
    
    Provides automatic profiling, logging, and event emission.
    
    Example:
        with operation_context("cache_get", backend="rust"):
            result = cache.get(layer=0, position=0)
    """
    start_time = time.perf_counter()
    
    # Emit start event
    try:
        from .events import EventType, emit_event
        emit_event(
            EventType.OPERATION_STARTED,
            data={'operation': operation_name, 'backend': backend, **kwargs},
            source=operation_name
        )
    except Exception:
        pass
    
    try:
        # Start profiling
        profiler = None
        try:
            from .profiling import get_profiler
            profiler = get_profiler()
            profiler_context = profiler.profile(operation_name, backend=backend)
            profiler_context.__enter__()
        except Exception:
            profiler_context = None
        
        try:
            yield
        finally:
            # Stop profiling
            if profiler_context:
                try:
                    profiler_context.__exit__(None, None, None)
                except Exception:
                    pass
        
        # Emit completion event
        duration_ms = (time.perf_counter() - start_time) * 1000
        try:
            from .events import EventType, emit_event
            emit_event(
                EventType.OPERATION_COMPLETED,
                data={
                    'operation': operation_name,
                    'backend': backend,
                    'duration_ms': duration_ms,
                    **kwargs
                },
                source=operation_name
            )
        except Exception:
            pass
        
        # Log operation
        try:
            from .logging import get_logger
            logger = get_logger()
            logger.log_operation(operation_name, backend=backend, duration_ms=duration_ms)
        except Exception:
            pass
        
        # Record metrics
        try:
            from .metrics import get_metrics_collector
            collector = get_metrics_collector()
            collector.record_latency(operation_name, duration_ms, backend=backend or "")
        except Exception:
            pass
    
    except Exception as e:
        # Emit failure event
        duration_ms = (time.perf_counter() - start_time) * 1000
        try:
            from .events import EventType, emit_event
            emit_event(
                EventType.OPERATION_FAILED,
                data={
                    'operation': operation_name,
                    'backend': backend,
                    'error': str(e),
                    'duration_ms': duration_ms,
                    **kwargs
                },
                source=operation_name
            )
        except Exception:
            pass
        
        # Log error
        try:
            from .logging import get_logger
            logger = get_logger()
            logger.log_error_with_context(e, operation_name, backend=backend)
        except Exception:
            pass
        
        raise


@contextmanager
def backend_context(backend_name: str):
    """
    Context manager for backend operations.
    
    Ensures backend is available and emits events.
    
    Example:
        with backend_context("rust"):
            cache = KVCache(max_size=10000, backend=Backend.RUST)
    """
    try:
        from .backend import is_backend_available
        if not is_backend_available(backend_name):
            from .errors import BackendNotAvailableError
            raise BackendNotAvailableError(f"Backend {backend_name} is not available")
    except ImportError:
        pass
    
    try:
        from .events import EventType, emit_event
        emit_event(EventType.BACKEND_SELECTED, data={'backend': backend_name})
    except Exception:
        pass
    
    yield
    
    # Backend context cleanup if needed
    pass


@contextmanager
def performance_context(threshold_ms: Optional[float] = None):
    """
    Context manager for performance monitoring.
    
    Warns if operation exceeds threshold.
    
    Example:
        with performance_context(threshold_ms=100.0):
            # Slow operation
            pass
    """
    start_time = time.perf_counter()
    
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        if threshold_ms and duration_ms > threshold_ms:
            try:
                from .events import EventType, emit_event
                emit_event(
                    EventType.PERFORMANCE_THRESHOLD,
                    data={
                        'duration_ms': duration_ms,
                        'threshold_ms': threshold_ms
                    }
                )
            except Exception:
                pass
            
            try:
                from .logging import get_logger
                logger = get_logger()
                logger.warning(
                    f"Operation exceeded threshold: {duration_ms:.2f}ms > {threshold_ms:.2f}ms"
                )
            except Exception:
                pass


@contextmanager
def resource_context(max_memory_mb: Optional[float] = None):
    """
    Context manager for resource monitoring.
    
    Monitors memory usage during operation.
    
    Example:
        with resource_context(max_memory_mb=1024):
            # Memory-intensive operation
            pass
    """
    initial_memory = None
    
    if max_memory_mb:
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        except Exception:
            pass
    
    try:
        yield
    finally:
        if max_memory_mb and initial_memory is not None:
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = final_memory - initial_memory
                
                if memory_used > max_memory_mb:
                    try:
                        from .logging import get_logger
                        logger = get_logger()
                        logger.warning(
                            f"Memory usage exceeded threshold: {memory_used:.2f}MB > {max_memory_mb:.2f}MB"
                        )
                    except Exception:
                        pass
            except Exception:
                pass













