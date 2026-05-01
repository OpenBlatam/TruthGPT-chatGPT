"""
Decorators for inference engines.

Provides decorators for validation, error handling, logging, and performance monitoring.
"""
import logging
import time
import functools
from typing import Callable, Any, Optional, Dict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def validate_inputs(**validators):
    """
    Decorator to validate function inputs.
    
    Args:
        **validators: Dictionary mapping parameter names to validation functions
    
    Example:
        @validate_inputs(
            max_tokens=lambda x: x > 0,
            temperature=lambda x: 0 < x <= 2.0
        )
        def generate(self, max_tokens, temperature):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate parameters
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValueError(
                            f"Invalid value for {param_name}: {value}. "
                            f"Validation failed."
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def handle_errors(
    reraise: bool = True,
    default_return: Any = None,
    log_level: int = logging.ERROR
):
    """
    Decorator to handle errors with logging and optional recovery.
    
    Args:
        reraise: Whether to re-raise exceptions
        default_return: Default return value on error (if not reraise)
        log_level: Logging level for errors
    
    Example:
        @handle_errors(reraise=False, default_return=[])
        def process_batch(self, items):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(
                    log_level,
                    f"Error in {func.__name__}: {e}",
                    exc_info=True
                )
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


def log_execution_time(operation_name: Optional[str] = None):
    """
    Decorator to log execution time of functions.
    
    Args:
        operation_name: Custom name for the operation (defaults to function name)
    
    Example:
        @log_execution_time("model_generation")
        def generate(self, prompts):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"{op_name} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"{op_name} failed after {elapsed:.3f}s: {e}",
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry function on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry
    
    Example:
        @retry_on_failure(max_attempts=3, delay=1.0)
        def load_model(self, path):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}",
                            exc_info=True
                        )
            
            raise last_exception
        return wrapper
    return decorator


def cache_result(ttl: Optional[float] = None, max_size: int = 128):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time-to-live in seconds (None for no expiration)
        max_size: Maximum cache size
    
    Example:
        @cache_result(ttl=3600, max_size=64)
        def expensive_computation(self, input_data):
            ...
    """
    def decorator(func: Callable) -> Callable:
        cache: Dict[Any, tuple] = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            import hashlib
            import pickle
            key_data = pickle.dumps((args, sorted(kwargs.items())))
            cache_key = hashlib.md5(key_data).hexdigest()
            
            # Check cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if ttl is None or (time.time() - timestamp) < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            if len(cache) >= max_size:
                # Remove oldest entry
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
            
            cache[cache_key] = (result, time.time())
            logger.debug(f"Cached result for {func.__name__}")
            return result
        
        # Add cache management methods
        wrapper.clear_cache = lambda: cache.clear()
        wrapper.cache_size = lambda: len(cache)
        
        return wrapper
    return decorator


@contextmanager
def performance_monitor(operation_name: str):
    """
    Context manager for performance monitoring.
    
    Args:
        operation_name: Name of the operation being monitored
    
    Example:
        with performance_monitor("model_inference"):
            result = model.generate(prompts)
    """
    start_time = time.time()
    start_memory = None
    
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        pass
    
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        
        if start_memory is not None:
            try:
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = end_memory - start_memory
                logger.info(
                    f"{operation_name}: {elapsed:.3f}s, "
                    f"memory: {memory_delta:+.1f}MB"
                )
            except Exception:
                logger.info(f"{operation_name}: {elapsed:.3f}s")
        else:
            logger.info(f"{operation_name}: {elapsed:.3f}s")













