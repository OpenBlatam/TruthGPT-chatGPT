"""
Advanced retry utilities for optimization_core.

Provides utilities for retry logic with exponential backoff and jitter.
"""
import logging
import time
import random
from typing import Callable, Optional, Type, Tuple, List, Any
from functools import wraps

logger = logging.getLogger(__name__)


from .base import BaseOptimizationModel


class RetryConfig(BaseOptimizationModel):
    """Retry configuration."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: Tuple[Type[Exception], ...] = (Exception,)
    retry_on_result: Optional[Callable] = None


class RetryHandler:
    """Handler for retry logic."""
    
    def __init__(self, config: RetryConfig):
        """
        Initialize retry handler.
        
        Args:
            config: Retry configuration
        """
        self.config = config
    
    def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                
                # Check if result should trigger retry
                if self.config.retry_on_result and self.config.retry_on_result(result):
                    if attempt < self.config.max_attempts:
                        delay = self._calculate_delay(attempt)
                        logger.warning(
                            f"Retry {attempt}/{self.config.max_attempts} "
                            f"after {delay:.2f}s (result check failed)"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise ValueError("Result check failed after all retries")
                
                return result
            
            except self.config.retry_on as e:
                last_exception = e
                
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Retry {attempt}/{self.config.max_attempts} "
                        f"after {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.config.max_attempts} attempts failed: {e}",
                        exc_info=True
                    )
                    raise
        
        if last_exception:
            raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry.
        
        Args:
            attempt: Attempt number (1-indexed)
        
        Returns:
            Delay in seconds
        """
        delay = self.config.initial_delay * (
            self.config.exponential_base ** (attempt - 1)
        )
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)
        
        return delay


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    retry_on_result: Optional[Callable] = None
):
    """
    Decorator for retry logic.
    
    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter
        retry_on: Exception types to retry on
        retry_on_result: Optional function to check result
    
    Returns:
        Decorator function
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retry_on=retry_on,
        retry_on_result=retry_on_result
    )
    
    def decorator(func: Callable):
        """Retry decorator."""
        handler = RetryHandler(config)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function."""
            return handler.execute(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


def with_retry(
    func: Callable,
    config: Optional[RetryConfig] = None
) -> Callable:
    """
    Wrap function with retry logic.
    
    Args:
        func: Function to wrap
        config: Optional retry configuration
    
    Returns:
        Wrapped function
    """
    if config is None:
        config = RetryConfig()
    
    handler = RetryHandler(config)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function."""
        return handler.execute(func, *args, **kwargs)
    
    return wrapper













