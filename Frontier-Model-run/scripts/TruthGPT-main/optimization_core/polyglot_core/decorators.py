"""
Useful decorators for polyglot_core.

Provides decorators for profiling, validation, error handling, and more.
"""

from functools import wraps
from typing import Callable, Any, Optional, Dict
import time
import logging


def profile_operation(operation_name: Optional[str] = None, backend: Optional[str] = None):
    """
    Decorator to profile an operation.
    
    Example:
        @profile_operation("cache_get", backend="rust")
        def get_from_cache(self, key):
            return self.cache[key]
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            
            try:
                from .profiling import get_profiler
                profiler = get_profiler()
                
                with profiler.profile(op_name, backend=backend):
                    result = func(*args, **kwargs)
                
                return result
            except Exception:
                # If profiling fails, just run the function
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_inputs(**validators):
    """
    Decorator to validate function inputs.
    
    Example:
        @validate_inputs(
            tensor=lambda x: validate_tensor(x, name="input"),
            max_size=lambda x: validate_positive(x, name="max_size")
        )
        def create_cache(tensor, max_size):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    bound.arguments[param_name] = validator(bound.arguments[param_name])
            
            return func(*bound.args, **bound.kwargs)
        
        return wrapper
    return decorator


def handle_errors(
    fallback: Optional[Callable] = None,
    log_errors: bool = True,
    reraise: bool = False
):
    """
    Decorator to handle errors gracefully.
    
    Args:
        fallback: Fallback function to call on error
        log_errors: Whether to log errors
        reraise: Whether to re-raise the exception
        
    Example:
        @handle_errors(fallback=lambda: None, log_errors=True)
        def risky_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    try:
                        from .logging import get_logger
                        logger = get_logger()
                        logger.log_error_with_context(
                            e,
                            func.__name__,
                            context={'args': str(args), 'kwargs': str(kwargs)}
                        )
                    except Exception:
                        logging.error(f"Error in {func.__name__}: {e}", exc_info=True)
                
                if fallback:
                    return fallback(*args, **kwargs)
                
                if reraise:
                    raise
                
                return None
        
        return wrapper
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry on failure.
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
        
    Example:
        @retry_on_failure(max_retries=3, delay=0.1)
        def network_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def cache_result(cache_key: Optional[Callable] = None, ttl: Optional[float] = None):
    """
    Decorator to cache function results.
    
    Args:
        cache_key: Function to generate cache key from args/kwargs
        ttl: Time to live in seconds (None for no expiration)
        
    Example:
        @cache_result(cache_key=lambda args, kwargs: f"key_{args[0]}")
        def expensive_operation(key):
            ...
    """
    _cache: Dict[str, tuple] = {}  # (result, timestamp)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key:
                key = cache_key(args, kwargs)
            else:
                key = str((args, tuple(sorted(kwargs.items()))))
            
            # Check cache
            if key in _cache:
                result, timestamp = _cache[key]
                if ttl is None or (time.time() - timestamp) < ttl:
                    return result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            _cache[key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator


def log_operation(operation_name: Optional[str] = None, level: int = logging.INFO):
    """
    Decorator to log operation execution.
    
    Example:
        @log_operation("cache_operation")
        def cache_op():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            
            try:
                from .logging import get_logger
                logger = get_logger()
                
                start = time.perf_counter()
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                
                logger.log_operation(op_name, duration_ms=duration_ms)
                
                return result
            except Exception:
                # If logging fails, just run the function
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def measure_performance(operation_name: Optional[str] = None):
    """
    Decorator to measure and record performance.
    
    Example:
        @measure_performance("attention_forward")
        def forward(self, q, k, v):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            
            try:
                from .metrics import get_metrics_collector
                collector = get_metrics_collector()
                
                start = time.perf_counter()
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                
                # Try to get backend from self if available
                backend = None
                if args and hasattr(args[0], 'backend'):
                    backend = str(args[0].backend.name) if hasattr(args[0].backend, 'name') else str(args[0].backend)
                
                collector.record_latency(op_name, duration_ms, backend=backend or "")
                
                return result
            except Exception:
                # If metrics fail, just run the function
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def ensure_backend(backend_name: str):
    """
    Decorator to ensure specific backend is available.
    
    Example:
        @ensure_backend("rust")
        def rust_only_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                from .backend import is_backend_available
                if not is_backend_available(backend_name):
                    raise RuntimeError(f"Backend {backend_name} is not available")
            except ImportError:
                pass  # Backend module not available
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator













