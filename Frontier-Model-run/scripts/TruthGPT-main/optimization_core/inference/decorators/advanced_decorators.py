"""
Advanced Decorators for Inference Engines
==========================================

High-level decorators for common inference patterns:
- Retry logic with exponential backoff
- Circuit breaker integration
- Rate limiting
- Metrics collection
- Caching
- Timeout handling
"""

import time
import functools
import logging
import threading
from typing import Callable, TypeVar, Optional, Any, Dict, List
from functools import wraps
import asyncio

from ..exceptions import InferenceEngineError, GenerationError
from ..helpers.engine_helpers import timing_context, format_error_details

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable)


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retry_on: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        retry_on: Tuple of exceptions to retry on
        on_retry: Optional callback called on each retry
    
    Example:
        >>> @retry(max_attempts=3, initial_delay=1.0)
        >>> def generate(self, prompt):
        ...     return self._do_generate(prompt)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        if on_retry:
                            on_retry(e, attempt + 1)
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        
                        time.sleep(delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            
            raise last_exception
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        if on_retry:
                            on_retry(e, attempt + 1)
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        
                        await asyncio.sleep(delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


def timeout(seconds: float, timeout_error: type = TimeoutError):
    """
    Timeout decorator for synchronous functions.
    
    Args:
        seconds: Timeout in seconds
        timeout_error: Exception to raise on timeout
    
    Example:
        >>> @timeout(seconds=30.0)
        >>> def generate(self, prompt):
        ...     return self._do_generate(prompt)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise timeout_error(f"Function {func.__name__} timed out after {seconds}s")
            
            # Set signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        
        return wrapper
    
    return decorator


def async_timeout(seconds: float, timeout_error: type = asyncio.TimeoutError):
    """
    Timeout decorator for async functions.
    
    Args:
        seconds: Timeout in seconds
        timeout_error: Exception to raise on timeout
    
    Example:
        >>> @async_timeout(seconds=30.0)
        >>> async def generate_async(self, prompt):
        ...     return await self._do_generate(prompt)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise timeout_error(
                    f"Function {func.__name__} timed out after {seconds}s"
                )
        
        return wrapper
    
    return decorator


def with_metrics(metric_name: Optional[str] = None, collect_args: bool = False):
    """
    Decorator to automatically collect metrics.
    
    Args:
        metric_name: Name of the metric (defaults to function name)
        collect_args: Whether to include function arguments in metric labels
    
    Example:
        >>> @with_metrics(metric_name="generation_latency")
        >>> def generate(self, prompt):
        ...     return self._do_generate(prompt)
    """
    def decorator(func: F) -> F:
        from ..metrics import get_metrics
        
        metrics = get_metrics()
        name = metric_name or func.__name__
        timer = metrics.timer(f"{name}_duration")
        counter = metrics.counter(f"{name}_total")
        errors = metrics.counter(f"{name}_errors")
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            counter.inc()
            
            try:
                with timer.time():
                    result = func(*args, **kwargs)
                return result
            except Exception as e:
                errors.inc()
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            counter.inc()
            
            try:
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                duration = time.perf_counter() - start
                timer.record(duration)
                return result
            except Exception as e:
                errors.inc()
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


def cached(
    ttl: Optional[float] = None,
    max_size: int = 128,
    key_func: Optional[Callable] = None
):
    """
    Caching decorator with TTL and size limits.
    
    Args:
        ttl: Time-to-live in seconds (None for no expiration)
        max_size: Maximum cache size
        key_func: Custom key function (defaults to args + kwargs)
    
    Example:
        >>> @cached(ttl=3600, max_size=1000)
        >>> def encode(self, text):
        ...     return self._do_encode(text)
    """
    def decorator(func: F) -> F:
        cache: Dict[str, tuple] = {}
        cache_times: Dict[str, float] = {}
        
        def make_key(*args, **kwargs) -> str:
            if key_func:
                return str(key_func(*args, **kwargs))
            return str((args, tuple(sorted(kwargs.items()))))
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = make_key(*args, **kwargs)
            now = time.time()
            
            # Check cache
            if key in cache:
                value, cached_time = cache[key]
                
                # Check TTL
                if ttl is None or (now - cached_time) < ttl:
                    return value
                else:
                    # Expired, remove
                    del cache[key]
                    del cache_times[key]
            
            # Cache miss, compute
            result = func(*args, **kwargs)
            
            # Add to cache
            if len(cache) >= max_size:
                # Remove oldest entry
                oldest_key = min(cache_times.items(), key=lambda x: x[1])[0]
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            cache[key] = (result, now)
            cache_times[key] = now
            
            return result
        
        wrapper.cache_clear = lambda: (cache.clear(), cache_times.clear())
        wrapper.cache_info = lambda: {
            "size": len(cache),
            "max_size": max_size,
            "ttl": ttl
        }
        
        return wrapper
    
    return decorator


def rate_limit(calls: int, period: float):
    """
    Rate limiting decorator.
    
    Args:
        calls: Maximum number of calls
        period: Time period in seconds
    
    Example:
        >>> @rate_limit(calls=100, period=60.0)
        >>> def generate(self, prompt):
        ...     return self._do_generate(prompt)
    """
    def decorator(func: F) -> F:
        call_times: List[float] = []
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            with lock:
                # Remove old calls
                call_times[:] = [t for t in call_times if now - t < period]
                
                # Check if limit exceeded
                if len(call_times) >= calls:
                    sleep_time = period - (now - call_times[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        now = time.time()
                        call_times[:] = [t for t in call_times if now - t < period]
                
                # Record call
                call_times.append(now)
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception
):
    """
    Circuit breaker decorator.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before trying again
        expected_exception: Exception type that counts as failure
    
    Example:
        >>> @circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
        >>> def generate(self, prompt):
        ...     return self._do_generate(prompt)
    """
    def decorator(func: F) -> F:
        failures = 0
        last_failure_time = 0.0
        circuit_open = False
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal failures, last_failure_time, circuit_open
            
            with lock:
                now = time.time()
                
                # Check if circuit should be closed
                if circuit_open:
                    if now - last_failure_time > recovery_timeout:
                        circuit_open = False
                        failures = 0
                        logger.info(f"Circuit breaker closed for {func.__name__}")
                    else:
                        raise InferenceEngineError(
                            f"Circuit breaker is open for {func.__name__}. "
                            f"Retry after {recovery_timeout - (now - last_failure_time):.1f}s"
                        )
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset failures
                with lock:
                    failures = 0
                
                return result
            except expected_exception as e:
                with lock:
                    failures += 1
                    last_failure_time = time.time()
                    
                    if failures >= failure_threshold:
                        circuit_open = True
                        logger.error(
                            f"Circuit breaker opened for {func.__name__} "
                            f"after {failures} failures"
                        )
                
                raise
        
        return wrapper
    
    return decorator


def validate_input(validator: Callable[[Any], bool], error_message: str = "Validation failed"):
    """
    Input validation decorator.
    
    Args:
        validator: Function that takes args and returns bool
        error_message: Error message if validation fails
    
    Example:
        >>> @validate_input(lambda args, kwargs: len(args) > 0)
        >>> def generate(self, prompt):
        ...     return self._do_generate(prompt)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not validator(args, kwargs):
                raise ValueError(error_message)
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def log_execution(level: int = logging.INFO, log_args: bool = False, log_result: bool = False):
    """
    Logging decorator for function execution.
    
    Args:
        level: Logging level
        log_args: Whether to log function arguments
        log_result: Whether to log function result
    
    Example:
        >>> @log_execution(level=logging.DEBUG, log_args=True)
        >>> def generate(self, prompt):
        ...     return self._do_generate(prompt)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.log(level, f"Executing {func.__name__}")
            
            if log_args:
                logger.log(level, f"  Args: {args}")
                logger.log(level, f"  Kwargs: {kwargs}")
            
            try:
                result = func(*args, **kwargs)
                
                if log_result:
                    logger.log(level, f"  Result: {result}")
                
                logger.log(level, f"Completed {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                raise
        
        return wrapper
    
    return decorator


# Composite decorators
def production_ready(
    max_retries: int = 3,
    timeout_seconds: float = 30.0,
    collect_metrics: bool = True
):
    """
    Composite decorator for production-ready functions.
    
    Combines: retry, timeout, metrics, logging
    
    Args:
        max_retries: Maximum retry attempts
        timeout_seconds: Timeout in seconds
        collect_metrics: Whether to collect metrics
    
    Example:
        >>> @production_ready(max_retries=3, timeout_seconds=30.0)
        >>> def generate(self, prompt):
        ...     return self._do_generate(prompt)
    """
    def decorator(func: F) -> F:
        # Apply decorators in order
        decorated = retry(max_attempts=max_retries)(func)
        decorated = timeout(seconds=timeout_seconds)(decorated)
        
        if collect_metrics:
            decorated = with_metrics()(decorated)
        
        decorated = log_execution()(decorated)
        
        return decorated
    
    return decorator


