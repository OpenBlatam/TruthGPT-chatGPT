"""
Advanced retry utilities for optimization_core.

Provides reusable retry strategies with various backoff algorithms.
"""

import asyncio
import logging
import random
import time
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ════════════════════════════════════════════════════════════════════════════════
# BACKOFF STRATEGIES
# ════════════════════════════════════════════════════════════════════════════════

class BackoffStrategy(str, Enum):
    """Backoff strategy types."""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    POLYNOMIAL = "polynomial"


def calculate_backoff(
    attempt: int,
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
    polynomial_degree: float = 2.0
) -> float:
    """
    Calculate backoff delay based on strategy.
    
    Args:
        attempt: Current attempt number (1-indexed)
        strategy: Backoff strategy
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        multiplier: Multiplier for exponential/linear
        polynomial_degree: Degree for polynomial strategy
    
    Returns:
        Delay in seconds
    
    Example:
        >>> calculate_backoff(3, BackoffStrategy.EXPONENTIAL, base_delay=1.0)
        4.0
    """
    if strategy == BackoffStrategy.FIXED:
        delay = base_delay
    elif strategy == BackoffStrategy.LINEAR:
        delay = base_delay * attempt * multiplier
    elif strategy == BackoffStrategy.EXPONENTIAL:
        delay = base_delay * (multiplier ** (attempt - 1))
    elif strategy == BackoffStrategy.FIBONACCI:
        # Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, ...
        fib = _fibonacci(attempt)
        delay = base_delay * fib
    elif strategy == BackoffStrategy.POLYNOMIAL:
        delay = base_delay * (attempt ** polynomial_degree)
    else:
        delay = base_delay
    
    return min(delay, max_delay)


def _fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return 1
    a, b = 1, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def add_jitter(delay: float, jitter_factor: float = 0.1) -> float:
    """
    Add random jitter to delay to prevent thundering herd.
    
    Args:
        delay: Base delay
        jitter_factor: Jitter factor (0.0 to 1.0)
    
    Returns:
        Delay with jitter
    
    Example:
        >>> delay = add_jitter(5.0, jitter_factor=0.2)
        4.0 <= delay <= 6.0
    """
    jitter_amount = delay * jitter_factor * random.uniform(-1, 1)
    return max(0, delay + jitter_amount)


# ════════════════════════════════════════════════════════════════════════════════
# SYNC RETRY
# ════════════════════════════════════════════════════════════════════════════════

def retry_with_backoff(
    func: Callable[[], T],
    max_attempts: int = 3,
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_enabled: bool = True,
    retry_on: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
) -> T:
    """
    Retry function with backoff (sync).
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        strategy: Backoff strategy
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter_enabled: Enable jitter
        retry_on: Exceptions to retry on
        on_retry: Callback on retry (attempt, exception)
    
    Returns:
        Function result
    
    Raises:
        Last exception if all attempts fail
    
    Example:
        >>> result = retry_with_backoff(
        ...     lambda: risky_operation(),
        ...     max_attempts=5,
        ...     strategy=BackoffStrategy.EXPONENTIAL
        ... )
    """
    last_exception = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except retry_on as e:
            last_exception = e
            
            if attempt < max_attempts:
                delay = calculate_backoff(
                    attempt,
                    strategy,
                    base_delay,
                    max_delay
                )
                
                if jitter_enabled:
                    delay = add_jitter(delay)
                
                if on_retry:
                    on_retry(attempt, e)
                else:
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                
                time.sleep(delay)
            else:
                logger.error(f"All {max_attempts} attempts failed")
    
    raise last_exception


# ════════════════════════════════════════════════════════════════════════════════
# ASYNC RETRY
# ════════════════════════════════════════════════════════════════════════════════

async def async_retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_enabled: bool = True,
    retry_on: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
) -> T:
    """
    Retry async function with backoff.
    
    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        strategy: Backoff strategy
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter_enabled: Enable jitter
        retry_on: Exceptions to retry on
        on_retry: Callback on retry (attempt, exception)
    
    Returns:
        Function result
    
    Raises:
        Last exception if all attempts fail
    
    Example:
        >>> result = await async_retry_with_backoff(
        ...     lambda: async_risky_operation(),
        ...     max_attempts=5
        ... )
    """
    last_exception = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            return await func()
        except retry_on as e:
            last_exception = e
            
            if attempt < max_attempts:
                delay = calculate_backoff(
                    attempt,
                    strategy,
                    base_delay,
                    max_delay
                )
                
                if jitter_enabled:
                    delay = add_jitter(delay)
                
                if on_retry:
                    on_retry(attempt, e)
                else:
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_attempts} attempts failed")
    
    raise last_exception


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "BackoffStrategy",
    # Backoff calculation
    "calculate_backoff",
    "add_jitter",
    # Retry functions
    "retry_with_backoff",
    "async_retry_with_backoff",
]













