"""
Common decorators for optimization_core.

Provides reusable decorators for:
- Retry logic (sync and async)
- Timeout handling
- Caching
- Rate limiting
- Circuit breaker
- Performance monitoring
- Error handling
"""

import asyncio
import functools
import hashlib
import logging
import pickle
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from .exceptions import OptimizationCoreError
from .types import CallableNoArgs, CallableOneArg, DictStrAny

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable)


# ════════════════════════════════════════════════════════════════════════════════
# RETRY DECORATORS
# ════════════════════════════════════════════════════════════════════════════════

def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retry_on: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    jitter: bool = False
) -> Callable[[F], F]:
    """
    Retry decorator with exponential backoff (sync and async).
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        retry_on: Tuple of exceptions to retry on
        on_retry: Optional callback called on each retry
        jitter: Whether to add random jitter to delay
    
    Example:
        >>> @retry(max_attempts=3, initial_delay=1.0)
        >>> def generate(self, prompt):
        ...     return self._do_generate(prompt)
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
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
                            
                            if jitter:
                                import random
                                jitter_amount = delay * 0.1 * random.uniform(-1, 1)
                                actual_delay = max(0, delay + jitter_amount)
                            else:
                                actual_delay = delay
                            
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                                f"Retrying in {actual_delay:.2f}s..."
                            )
                            
                            await asyncio.sleep(actual_delay)
                            delay = min(delay * exponential_base, max_delay)
                        else:
                            logger.error(
                                f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                            )
                
                raise last_exception
            
            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
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
                            
                            if jitter:
                                import random
                                jitter_amount = delay * 0.1 * random.uniform(-1, 1)
                                actual_delay = max(0, delay + jitter_amount)
                            else:
                                actual_delay = delay
                            
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                                f"Retrying in {actual_delay:.2f}s..."
                            )
                            
                            time.sleep(actual_delay)
                            delay = min(delay * exponential_base, max_delay)
                        else:
                            logger.error(
                                f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                            )
                
                raise last_exception
            
            return sync_wrapper  # type: ignore
    
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# TIMEOUT DECORATORS
# ════════════════════════════════════════════════════════════════════════════════

def timeout(seconds: float) -> Callable[[F], F]:
    """
    Timeout decorator (sync and async).
    
    Args:
        seconds: Timeout in seconds
    
    Example:
        >>> @timeout(seconds=30.0)
        >>> def long_operation(self):
        ...     ...
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            
            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, use threading
                result = [None]
                exception = [None]
                
                def target():
                    try:
                        result[0] = func(*args, **kwargs)
                    except Exception as e:
                        exception[0] = e
                
                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(timeout=seconds)
                
                if thread.is_alive():
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")
                
                if exception[0]:
                    raise exception[0]
                
                return result[0]
            
            return sync_wrapper  # type: ignore
    
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# CACHING DECORATORS
# ════════════════════════════════════════════════════════════════════════════════

def cache_result(
    ttl: Optional[float] = None,
    max_size: int = 128,
    key_func: Optional[Callable[..., str]] = None
) -> Callable[[F], F]:
    """
    Cache function results (sync only).
    
    Uses MemoryCache from modules.base.core_system.core.cache_utils for efficient LRU caching.
    
    Args:
        ttl: Time-to-live in seconds (None for no expiration)
        max_size: Maximum cache size
        key_func: Optional function to generate cache keys
    
    Example:
        >>> @cache_result(ttl=3600, max_size=64)
        >>> def expensive_computation(self, input_data):
        ...     ...
    """
    # Import here to avoid circular dependency
    from .cache_utils import MemoryCache
    
    cache = MemoryCache(max_size=max_size, default_ttl=ttl)
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                try:
                    key_data = pickle.dumps((args, sorted(kwargs.items())))
                    cache_key = hashlib.md5(key_data).hexdigest()
                except Exception:
                    # If pickling fails, use string representation
                    cache_key = str((args, kwargs))
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            # Compute result
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl=ttl)
            logger.debug(f"Cached result for {func.__name__}")
            return result
        
        # Add cache management methods
        wrapper.clear_cache = cache.clear
        wrapper.cache_size = cache.size
        wrapper.cache_stats = cache.get_stats
        
        return wrapper  # type: ignore
    
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# RATE LIMITING DECORATORS
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class RateLimiter:
    """Thread-safe rate limiter."""
    
    max_calls: int
    period: float
    _calls: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def acquire(self) -> bool:
        """Try to acquire a permit."""
        with self._lock:
            now = time.time()
            # Remove old calls
            self._calls = [t for t in self._calls if now - t < self.period]
            
            if len(self._calls) < self.max_calls:
                self._calls.append(now)
                return True
            return False
    
    def wait(self) -> None:
        """Wait until a permit is available."""
        while not self.acquire():
            if self._calls:
                sleep_time = self.period - (time.time() - self._calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                time.sleep(0.1)


def rate_limit(max_calls: int, period: float = 1.0) -> Callable[[F], F]:
    """
    Rate limiting decorator (sync only).
    
    Args:
        max_calls: Maximum number of calls
        period: Time period in seconds
    
    Example:
        >>> @rate_limit(max_calls=10, period=1.0)
        >>> def api_call(self):
        ...     ...
    """
    limiter = RateLimiter(max_calls=max_calls, period=period)
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter.wait()
            return func(*args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ════════════════════════════════════════════════════════════════════════════════

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker implementation."""
    
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Tuple[type, ...] = (Exception,)
    
    _state: CircuitState = CircuitState.CLOSED
    _failure_count: int = 0
    _last_failure_time: Optional[float] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_failure_time and \
                   (time.time() - self._last_failure_time) >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker: transitioning to HALF_OPEN")
                else:
                    raise OptimizationCoreError(
                        "Circuit breaker is OPEN. Service unavailable."
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info("Circuit breaker: transitioning to CLOSED")
            self._failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker: transitioning to OPEN "
                    f"(failures: {self._failure_count})"
                )


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Tuple[type, ...] = (Exception,)
) -> Callable[[F], F]:
    """
    Circuit breaker decorator.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        expected_exception: Exceptions that trigger circuit breaker
    
    Example:
        >>> @circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
        >>> def external_api_call(self):
        ...     ...
    """
    breaker = CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception
    )
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# PERFORMANCE MONITORING
# ════════════════════════════════════════════════════════════════════════════════

def log_execution_time(operation_name: Optional[str] = None) -> Callable[[F], F]:
    """
    Log execution time decorator.
    
    Args:
        operation_name: Custom name for the operation (defaults to function name)
    
    Example:
        >>> @log_execution_time("model_generation")
        >>> def generate(self, prompts):
        ...     ...
    """
    def decorator(func: F) -> F:
        op_name = operation_name or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
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
            
            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
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
            
            return sync_wrapper  # type: ignore
    
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING
# ════════════════════════════════════════════════════════════════════════════════

def handle_errors(
    reraise: bool = True,
    default_return: Any = None,
    log_level: int = logging.ERROR
) -> Callable[[F], F]:
    """
    Error handling decorator.
    
    Args:
        reraise: Whether to re-raise exceptions
        default_return: Default return value on error (if not reraise)
        log_level: Logging level for errors
    
    Example:
        >>> @handle_errors(reraise=False, default_return=[])
        >>> def process_batch(self, items):
        ...     ...
    """
    def decorator(func: F) -> F:
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
        
        return wrapper  # type: ignore
    
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Retry
    "retry",
    # Timeout
    "timeout",
    # Caching
    "cache_result",
    # Rate limiting
    "rate_limit",
    "RateLimiter",
    # Circuit breaker
    "circuit_breaker",
    "CircuitBreaker",
    "CircuitState",
    # Performance
    "log_execution_time",
    # Error handling
    "handle_errors",
]





